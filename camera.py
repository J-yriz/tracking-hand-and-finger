import argparse

import cv2

# Urutan mode saat tekan 'f': Normal -> Horizontal (mirror) -> Vertikal -> Keduanya -> Normal
FLIP_CYCLE = [None, 1, 0, -1]
FLIP_NAMES = {None: "Normal", 1: "Horizontal/Mirror", 0: "Vertikal", -1: "Horizontal+Vertikal"}
FLIP_ARG_MAP = {"normal": None, "h": 1, "v": 0, "hv": -1}


def apply_flip(frame, flip_code):
    return cv2.flip(frame, flip_code) if flip_code is not None else frame


def open_camera(camera_index=0):
    """Buka kamera. Caller wajib cek cap.isOpened() dan cap.release()."""
    return cv2.VideoCapture(camera_index)


def initial_flip_index(flip_mode):
    try:
        return FLIP_CYCLE.index(flip_mode)
    except ValueError:
        return 0


def next_flip_index(flip_idx):
    return (flip_idx + 1) % len(FLIP_CYCLE)


def main(camera_index=0, flip_mode=None):
    cap = open_camera(camera_index)
    if not cap.isOpened():
        print(f"Tidak bisa membuka kamera index {camera_index}")
        return

    flip_idx = initial_flip_index(flip_mode)
    print("Kamera terbuka. Tekan 'f' untuk flip, 'q' untuk keluar.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Gagal membaca frame dari kamera.")
            break

        flip_code = FLIP_CYCLE[flip_idx]
        frame = apply_flip(frame, flip_code)

        cv2.putText(
            frame,
            f"Flip: {FLIP_NAMES[flip_code]} (f)",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )
        cv2.imshow("Camera", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("f"):
            flip_idx = next_flip_index(flip_idx)
            print(f"Mode flip: {FLIP_NAMES[FLIP_CYCLE[flip_idx]]}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tampilkan kamera dengan flip.")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument(
        "--flip",
        choices=["normal", "h", "v", "hv"],
        default="normal",
        help="Mode awal: normal, h=horizontal/mirror, v=vertikal, hv=keduanya",
    )
    args = parser.parse_args()
    initial = FLIP_ARG_MAP[args.flip]
    main(camera_index=args.camera, flip_mode=initial)
