"""Tracking tangan + jari + gesture thumbs-up pakai kamera + MediaPipe Tasks.

Cara jalan:
    pip install opencv-python mediapipe pillow
    python hand_tracking.py
    # q = keluar, f = ganti flip. Acungkan jempol ke atas = emoji 👍
"""

import argparse
import time
import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    HandLandmarker,
    HandLandmarkerOptions,
    RunningMode,
)

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
MODEL_PATH = Path(__file__).with_name("hand_landmarker.task")

# Batas model: 1..10 tangan (multi-orang). Default 4 biar >2 tangan langsung bisa.
DEFAULT_MAX_HANDS = 4
HARD_MAX_HANDS = 10

# WHY: flip 1 sumbu = cermin (tukar kirality) -> label model tertukar.
# Flip 2 sumbu (-1) = rotasi 180° -> label tetap. Terbukti uji woman_hands.jpg:
# orig=(Left,Right), hflip/vflip=(Right,Left), hv=(Left,Right).
MIRROR_SWAP_FLIPS = (0, 1)
_SWAP_SIDE = {"Left": "Right", "Right": "Left"}
_ID_SIDE = {"Left": "Kiri", "Right": "Kanan"}
_SIDE_COLOR = {"Left": (0, 255, 255), "Right": (0, 255, 0)}  # BGR: Kiri=kuning, Kanan=hijau

# Topologi 21 landmark MediaPipe: wrist=0, jempol 1-4, telunjuk 5-8,
# tengah 9-12, manis 13-16, kelingking 17-20.
PALM_LINES = [(0, 1), (0, 5), (0, 17), (5, 9), (9, 13), (13, 17)]
FINGERS = {
    "Jempol": ([1, 2, 3, 4], (255, 0, 255)),
    "Telunjuk": ([5, 6, 7, 8], (255, 0, 0)),
    "Tengah": ([9, 10, 11, 12], (0, 165, 255)),
    "Manis": ([13, 14, 15, 16], (0, 255, 0)),
    "Kelingking": ([17, 18, 19, 20], (0, 255, 255)),
}
FINGER_TIPS = {"Jempol": 4, "Telunjuk": 8, "Tengah": 12, "Manis": 16, "Kelingking": 20}

# Sendi ukur per jari (a, sendi-b, c) utk status lurus/lipat.
FINGER_JOINTS = {
    "Jempol": (2, 3, 4),
    "Telunjuk": (5, 6, 7),
    "Tengah": (9, 10, 11),
    "Manis": (13, 14, 15),
    "Kelingking": (17, 18, 19),
}

# WHY ambang diketatkan: 4 jari lipat maks 141° (thumbs_up Manis),
# lurus min 168° (woman_hands Manis). Jempol beda anatomi: lipat 139°,
# lurus min 155° (foto thumbs-up) -> ambang sendiri 150°.
FINGER_EXT_MIN = 160.0
THUMB_STATUS_MIN = 150.0

# WHY ambang kalibrasi dari 4 foto uji (sudut di sendi PIP/IP, 180=lurus):
# thumbs_up=(Jmp155,Tjk108,Tgh124,Mns141,Klk125) + jempol paling atas, di atas wrist.
# pointing_up Tjk177, victory Tjk174/Tgh176, woman_hands semua ~170 -> tertolak.
THUMB_EXT_MIN = 140.0
FINGER_FOLD_MAX = 150.0
THUMB_TOP_MARGIN = 0.02
THUMB_WRIST_GAP = 0.03

# Font utk emoji 👍 (dicoba berurutan). cv2.Hershey tak bisa render emoji.
EMOJI_FONT_CANDIDATES = [
    "/usr/share/fonts/TTF/DejaVuSans.ttf",
    "/usr/share/fonts/noto/NotoSansSymbols-Black.ttf",
]
_EMOJI_FONT_CACHE = {}

# WHY: sistem kamera (open/flip) terpusat di camera.py biar tidak duplikat.
from camera import (
    FLIP_ARG_MAP,
    FLIP_CYCLE,
    FLIP_NAMES,
    apply_flip,
    initial_flip_index,
    next_flip_index,
    open_camera,
)


def ensure_model() -> str:
    if not MODEL_PATH.exists():
        print(f"Download model ke {MODEL_PATH} ...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    return str(MODEL_PATH)


def resolve_delegate(use_gpu: int) -> "BaseOptions.Delegate":
    """Map --gpu 0/1 ke BaseOptions.Delegate (0=CPU, 1=GPU)."""
    return BaseOptions.Delegate.GPU if int(use_gpu) == 1 else BaseOptions.Delegate.CPU


def create_hand_landmarker(model_path, max_hands, use_gpu):
    """Buat HandLandmarker; kalau GPU gagal (misal tak didukung) fallback ke CPU."""
    delegate = resolve_delegate(use_gpu)
    try:
        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path, delegate=delegate),
            running_mode=RunningMode.VIDEO,
            num_hands=max_hands,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        landmarker = HandLandmarker.create_from_options(options)
        return landmarker, delegate.name
    except Exception as e:
        if delegate == BaseOptions.Delegate.GPU:
            print(f"GPU gagal ({e}), fallback ke CPU.")
            options = HandLandmarkerOptions(
                base_options=BaseOptions(
                    model_asset_path=model_path,
                    delegate=BaseOptions.Delegate.CPU,
                ),
                running_mode=RunningMode.VIDEO,
                num_hands=max_hands,
                min_hand_detection_confidence=0.5,
                min_hand_presence_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            return HandLandmarker.create_from_options(options), "CPU(fallback)"
        raise


def correct_side(raw_side: str, flip_code) -> str:
    """Kembalikan sisi anatomis. Model melihat citra yg SUDAH di-flip,
    jadi flip cermin 1-sumbu (0/1) harus ditukar; None/-1 tetap."""
    if flip_code in MIRROR_SWAP_FLIPS:
        return _SWAP_SIDE.get(raw_side, raw_side)
    return raw_side


def extract_handedness(result, i):
    """Ambil (raw_side, score) dgn aman utk struktur list-of-list maupun datar."""
    try:
        group = result.handedness[i]
    except (IndexError, TypeError):
        return "Unknown", 0.0
    cats = group if isinstance(group, (list, tuple)) else [group]
    if not cats:
        return "Unknown", 0.0
    cat = cats[0]
    side = getattr(cat, "category_name", "Unknown") or "Unknown"
    try:
        score = float(getattr(cat, "score", 0.0) or 0.0)
    except (TypeError, ValueError):
        score = 0.0
    return side, score


def hand_center(landmarks, w, h):
    xs = [p.x for p in landmarks]
    ys = [p.y for p in landmarks]
    x1, y1 = int(min(xs) * w), int(min(ys) * h)
    x2, y2 = int(max(xs) * w), int(max(ys) * h)
    pad = 20
    x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
    x2, y2 = min(w - 1, x2 + pad), min(h - 1, y2 + pad)
    return (x1, y1, x2, y2, (x1 + x2) // 2, (y1 + y2) // 2)


def screen_side(cx, w):
    return "layar-kiri" if cx < w // 2 else "layar-kanan"


def draw_hand_box(frame, landmarks, label: str, color=(0, 255, 0)):
    """Gambar bounding box + titik tengah tangan saja, tanpa skeleton jari."""
    h, w = frame.shape[:2]
    x1, y1, x2, y2, cx, cy = hand_center(landmarks, w, h)

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.circle(frame, (cx, cy), 6, (0, 0, 255), -1)
    cv2.putText(frame, f"{label} ({cx},{cy})", (x1, max(0, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return (cx, cy)


def landmarks_to_px(landmarks, w, h):
    pts = []
    for p in landmarks:
        x = min(max(p.x, 0.0), 1.0)
        y = min(max(p.y, 0.0), 1.0)
        pts.append((int(x * (w - 1)), int(y * (h - 1))))
    return pts


def draw_fingers(frame, landmarks, tip_labels=True):
    """Gambar skeleton + status tiap jari.

    Kembalikan detail {nama: {angle, extended, tip, joints, color}}.
    """
    h, w = frame.shape[:2]
    details = track_fingers(landmarks, w, h)
    if not details:
        return {}
    pts = landmarks_to_px(landmarks, w, h)
    for a, b in PALM_LINES:
        cv2.line(frame, pts[a], pts[b], (200, 200, 200), 2)
    for name, (idxs, color) in FINGERS.items():
        d = details[name]
        chain = [0] + idxs
        for a, b in zip(chain, chain[1:]):
            cv2.line(frame, pts[a], pts[b], color, 2)
        for i in idxs:
            cv2.circle(frame, pts[i], 4, color, -1)
        tx, ty = d["tip"]
        if d["extended"]:
            cv2.circle(frame, (tx, ty), 7, color, 2)  # ring = jari tegak
        if tip_labels:
            state = "lurus" if d["extended"] else "lipat"
            cv2.putText(frame, f"{name}:{state} {d['angle']:.0f}d", (tx + 6, ty - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    cv2.circle(frame, pts[0], 5, (255, 255, 255), -1)
    return details


def finger_angle(landmarks, a, b, c):
    """Sudut (derajat) di sendi b antara titik a-b-c. 180 = lurus."""
    import math
    v1 = (landmarks[a].x - landmarks[b].x, landmarks[a].y - landmarks[b].y)
    v2 = (landmarks[c].x - landmarks[b].x, landmarks[c].y - landmarks[b].y)
    denom = math.hypot(*v1) * math.hypot(*v2) + 1e-9
    cosv = (v1[0] * v2[0] + v1[1] * v2[1]) / denom
    return math.degrees(math.acos(max(-1.0, min(1.0, cosv))))


def is_thumbs_up(landmarks):
    """True jika jempol tegak ke atas dan 4 jari lain terlipat."""
    if len(landmarks) < 21:
        return False
    if finger_angle(landmarks, 2, 3, 4) < THUMB_EXT_MIN:
        return False
    for a, b, _c in [(5, 6, 7), (9, 10, 11), (13, 14, 15), (17, 18, 19)]:
        if finger_angle(landmarks, a, b, b + 1) > FINGER_FOLD_MAX:
            return False
    tip_ys = [landmarks[i].y for i in (4, 8, 12, 16, 20)]
    if min(tip_ys) != tip_ys[0]:
        return False
    if not (landmarks[4].y + THUMB_TOP_MARGIN < min(tip_ys[1:])):
        return False
    return landmarks[4].y < landmarks[0].y - THUMB_WRIST_GAP


def track_fingers(landmarks, w, h):
    """Status tiap jari: {nama: {angle, extended, tip, joints, color}}."""
    if len(landmarks) < 21:
        return {}
    pts = landmarks_to_px(landmarks, w, h)
    details = {}
    for name, (idxs, color) in FINGERS.items():
        a, b, c = FINGER_JOINTS[name]
        angle = finger_angle(landmarks, a, b, c)
        thresh = THUMB_STATUS_MIN if name == "Jempol" else FINGER_EXT_MIN
        details[name] = {
            "angle": angle,
            "extended": angle >= thresh,
            "tip": pts[idxs[-1]],
            "joints": [pts[i] for i in idxs],
            "color": color,
        }
    return details


def overlay_emoji(frame, pos, emoji="👍", size=64):
    """Tempel emoji via PIL (fallback teks cv2 bila font tak ada). Kembalikan True bila emoji."""
    x, y = int(pos[0]), int(pos[1])
    try:
        from PIL import Image, ImageDraw, ImageFont
        if size not in _EMOJI_FONT_CACHE:
            font = None
            for path in EMOJI_FONT_CANDIDATES:
                try:
                    font = ImageFont.truetype(path, size)
                    break
                except OSError:
                    continue
            _EMOJI_FONT_CACHE[size] = font or False
        font = _EMOJI_FONT_CACHE[size]
        if font:
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            ImageDraw.Draw(img).text((x, y), emoji, font=font, fill=(255, 235, 0))
            frame[:, :] = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
            return True
    except ImportError:
        pass
    cv2.putText(frame, "THUMBS UP!", (x, y + size // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
    return False


def main(camera_index=0, flip_mode=None, max_hands=DEFAULT_MAX_HANDS, use_gpu=0):
    max_hands = max(1, min(HARD_MAX_HANDS, int(max_hands)))
    use_gpu = 1 if int(use_gpu) == 1 else 0
    model_path = ensure_model()
    device_req = "GPU" if use_gpu == 1 else "CPU"

    cap = open_camera(camera_index)
    if not cap.isOpened():
        print(f"Tidak bisa membuka kamera index {camera_index}")
        return

    flip_idx = initial_flip_index(flip_mode)

    print(f"Tracking tangan+jari jalan [{device_req}]. 'f'=flip, 'q'=keluar.")
    start = time.time()

    landmarker_obj, device_str = create_hand_landmarker(model_path, max_hands, use_gpu)
    print(f"Device aktif: {device_str}")
    with landmarker_obj as landmarker:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Gagal membaca frame.")
                break

            flip_code = FLIP_CYCLE[flip_idx]
            frame = apply_flip(frame, flip_code)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms = int((time.time() - start) * 1000)
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            n_hands = len(result.hand_landmarks)
            h_frame, w_frame = frame.shape[:2]
            # Urutkan kiri-ke-kanan layar biar nomor stabil utk >2 tangan.
            order = sorted(
                range(n_hands),
                key=lambda i: hand_center(result.hand_landmarks[i], w_frame, h_frame)[4],
            )
            for n, i in enumerate(order):
                landmarks = result.hand_landmarks[i]
                raw_side, score = extract_handedness(result, i)
                anat_side = correct_side(raw_side, flip_code)
                indo = _ID_SIDE.get(anat_side, anat_side)
                _, _, _, _, cx, _ = hand_center(landmarks, w_frame, h_frame)
                pos = screen_side(cx, w_frame)
                color = _SIDE_COLOR.get(anat_side, (255, 255, 255))
                details = draw_fingers(frame, landmarks)
                n_up = sum(1 for d in details.values() if d["extended"])
                label = f"T{n + 1} {indo} {score:.0%} {pos} | {n_up}/5"
                draw_hand_box(frame, landmarks, label, color)
                if is_thumbs_up(landmarks):
                    x1, y1, _, _, _, _ = hand_center(landmarks, w_frame, h_frame)
                    overlay_emoji(frame, (x1, max(0, y1 - 72)))
                    cv2.putText(frame, "THUMBS UP!", (x1, max(20, y1 - 78)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            corrected = "koreksi-mirror ON" if flip_code in MIRROR_SWAP_FLIPS else "koreksi-mirror OFF"
            cv2.putText(frame, f"Terdeteksi: {n_hands}/{max_hands} tangan",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (255, 0, 0), 2)
            cv2.putText(frame, f"Flip: {FLIP_NAMES[flip_code]} | {corrected} | {device_str}",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (255, 0, 0), 2)
            cv2.imshow("Hand Tracking (tangan+jari)", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("f"):
                flip_idx = next_flip_index(flip_idx)
                print(f"Mode flip: {FLIP_NAMES[FLIP_CYCLE[flip_idx]]}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tracking tangan + jari.")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--flip", choices=["normal", "h", "v", "hv"], default="normal")
    parser.add_argument("--max-hands", type=int, default=DEFAULT_MAX_HANDS,
                        help=f"Jumlah tangan maks 1-{HARD_MAX_HANDS} (default {DEFAULT_MAX_HANDS})")
    parser.add_argument("--gpu", type=int, choices=[0, 1], default=0,
                        help="0=CPU (default), 1=GPU (fallback ke CPU bila gagal)")
    args = parser.parse_args()
    initial = FLIP_ARG_MAP[args.flip]
    main(camera_index=args.camera, flip_mode=initial, max_hands=args.max_hands,
         use_gpu=args.gpu)
