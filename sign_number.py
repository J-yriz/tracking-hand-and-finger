"""Klasifikasi angka bahasa isyarat 0-10 dari status jari.

Aturan (disepakati dengan user):
- 1 tangan: angka = jumlah jari terangkat (0-5).
  Jempol saja = 1, telunjuk saja = 1, jempol + telunjuk = 2.
- 2 tangan: angka = total jari terangkat kedua tangan (0-10).

File ini TIDAK melakukan smoothing sendiri. Status "extended" sudah final
dari hand_tracking.track_fingers(). Stabilisasi output (anti kedut) dilakukan
oleh NumberStabilizer di bawah (majority vote beberapa frame terakhir).
"""

from collections import deque
from typing import Dict, List, Tuple

FINGER_NAMES = ["Jempol", "Telunjuk", "Tengah", "Manis", "Kelingking"]


class SignNumberClassifier:
    """Hitung angka dari status jari. Stateless (tanpa history per jari)."""

    def count_extended_fingers(self, details: Dict) -> int:
        """Hitung jari terangkat pada satu tangan (0-5)."""
        if not details:
            return 0
        return sum(
            1 for name in FINGER_NAMES
            if details.get(name, {}).get("extended", False)
        )

    def classify_single_hand(self, details: Dict) -> Tuple[int, float]:
        """Satu tangan: angka = jumlah jari terangkat."""
        if not details:
            return -1, 0.0
        return self.count_extended_fingers(details), 0.9

    def classify_multiple_hands(self, hands_details: List[Dict]) -> Tuple[int, float]:
        """Dua tangan (atau lebih): angka = total jari terangkat, maks 10."""
        if not hands_details:
            return -1, 0.0
        total = sum(self.count_extended_fingers(d) for d in hands_details)
        return min(total, 10), 0.9

    def classify(self, details: Dict) -> Tuple[int, float]:
        """Kompatibilitas: klasifikasi satu tangan."""
        return self.classify_single_hand(details)


class NumberStabilizer:
    """Anti kedut: angka baru dipakai kalau stabil di N frame terakhir.

    Contoh: window=5, min_agree=3 -> angka keluar kalau >=3 dari 5 frame
    terakhir sama. Kalau belum stabil, pertahankan angka lama (return None).
    """

    def __init__(self, window: int = 5, min_agree: int = 3):
        self.window = max(1, int(window))
        self.min_agree = max(1, int(min_agree))
        self.history: deque = deque(maxlen=self.window)

    def update(self, number: int):
        """Masukkan bacaan frame ini. Kembalikan angka stabil atau None."""
        if number is None or number < 0:
            return None
        self.history.append(int(number))
        if len(self.history) < self.min_agree:
            return None
        candidate = self.history[-1]
        if sum(1 for n in self.history if n == candidate) >= self.min_agree:
            return candidate
        return None

    def reset(self):
        self.history.clear()


if __name__ == "__main__":
    clf = SignNumberClassifier()

    def mock(extended_names):
        return {n: {"extended": n in extended_names} for n in FINGER_NAMES}

    cases = [
        ("jempol saja -> 1", mock(["Jempol"]), 1),
        ("telunjuk saja -> 1", mock(["Telunjuk"]), 1),
        ("jempol+telunjuk -> 2 (bug lama: tetap 1)", mock(["Jempol", "Telunjuk"]), 2),
        ("telunjuk+tengah -> 2", mock(["Telunjuk", "Tengah"]), 2),
        ("3 jari -> 3", mock(["Telunjuk", "Tengah", "Manis"]), 3),
        ("4 jari -> 4", mock(["Telunjuk", "Tengah", "Manis", "Kelingking"]), 4),
        ("5 jari -> 5", mock(FINGER_NAMES), 5),
        ("genggam -> 0", mock([]), 0),
    ]
    ok = True
    for label, details, expected in cases:
        got, _ = clf.classify_single_hand(details)
        status = "OK " if got == expected else "FAIL"
        if got != expected:
            ok = False
        print(f"[{status}] {label}: dapat {got}")

    # 2 tangan: 5 + 1 = 6, 5 + 5 = 10
    got, _ = clf.classify_multiple_hands([mock(FINGER_NAMES), mock(["Jempol"])])
    print(f"[{'OK ' if got == 6 else 'FAIL'}] 5+1 jari -> 6: dapat {got}")
    ok = ok and got == 6
    got, _ = clf.classify_multiple_hands([mock(FINGER_NAMES), mock(FINGER_NAMES)])
    print(f"[{'OK ' if got == 10 else 'FAIL'}] 5+5 jari -> 10: dapat {got}")
    ok = ok and got == 10

    # Stabilizer: belum stabil -> None, sudah stabil -> angka
    stab = NumberStabilizer(window=5, min_agree=3)
    assert stab.update(2) is None
    assert stab.update(2) is None
    assert stab.update(2) == 2
    print("[OK ] NumberStabilizer menahan angka sampai stabil")
    raise SystemExit(0 if ok else 1)
