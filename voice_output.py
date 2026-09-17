"""Output suara angka bahasa isyarat (bunyi beneran, bukan cuma teks).

Cara kerja: teks angka ("satu", "dua", ...) diubah jadi MP3 via gTTS
(butuh internet SEKALI saja per angka, hasilnya di-cache di voice_cache/),
lalu MP3 diputar via mpg123 atau ffplay (keduanya sudah ada di sistem ini).

Instalasi yang dibutuhkan:
    pip install gtts

Alternatif offline (tanpa internet, butuh akses sudo sekali):
    sudo pacman -S espeak-ng   # lalu pyttsx3 bisa dipakai
"""

import shutil
import subprocess
import threading
import time
from pathlib import Path

CACHE_DIR = Path(__file__).with_name("voice_cache")

NUMBER_WORDS = {
    0: "nol",
    1: "satu",
    2: "dua",
    3: "tiga",
    4: "empat",
    5: "lima",
    6: "enam",
    7: "tujuh",
    8: "delapan",
    9: "sembilan",
    10: "sepuluh",
}


def _find_player():
    """Kembalikan perintah pemutar MP3 yang tersedia, atau None."""
    if shutil.which("mpg123"):
        return ["mpg123", "-q"]
    if shutil.which("ffplay"):
        return ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet"]
    if shutil.which("paplay"):
        return None  # paplay tidak bisa mainkan MP3 langsung
    return None


class NumberVoiceOutput:
    """Ucapkan angka yang terdeteksi. Aman dipanggil tiap frame."""

    def __init__(self, enabled: bool = False):
        self.enabled = enabled
        self.last_spoken = None
        self.last_time = 0.0
        self.cooldown = 2.0  # detik minimal antar suara
        self._lock = threading.Lock()
        self._player = _find_player()

        try:
            import gtts  # noqa: F401
            self._gtts_ok = True
        except ImportError:
            self._gtts_ok = False

        if self._player is None:
            print("Voice: tidak ada pemutar MP3 (mpg123/ffplay). Suara jadi teks saja.")
        elif not self._gtts_ok:
            print("Voice: install dulu 'pip install gtts' agar bisa bersuara.")

    def _ensure_mp3(self, number: int):
        """Kembalikan path MP3 untuk angka; generate via gTTS bila belum ada."""
        CACHE_DIR.mkdir(exist_ok=True)
        mp3 = CACHE_DIR / f"{number}_id.mp3"
        if mp3.exists():
            return mp3
        if not self._gtts_ok:
            return None
        try:
            from gtts import gTTS
            gTTS(text=NUMBER_WORDS.get(number, str(number)), lang="id").save(str(mp3))
            return mp3
        except Exception as e:
            print(f"Voice: gagal generate suara ({e}). Cek koneksi internet.")
            return None

    def speak(self, number: int):
        """Ucapkan angka. Tidak spam: hanya bila beda angka / cooldown lewat."""
        if not self.enabled:
            return
        now = time.time()
        with self._lock:
            if self.last_spoken == number and now - self.last_time < self.cooldown:
                return
            self.last_spoken = number
            self.last_time = now
        threading.Thread(target=self._play, args=(int(number),), daemon=True).start()

    def _play(self, number: int):
        word = NUMBER_WORDS.get(number, str(number))
        mp3 = self._ensure_mp3(number)
        if mp3 is None or self._player is None:
            print(f"[SUARA-teks] {word} (pasang 'pip install gtts' untuk bunyi)")
            return
        print(f"[SUARA] {word}")
        try:
            subprocess.run([*self._player, str(mp3)],
                           stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL,
                           timeout=10)
        except Exception as e:
            print(f"Voice: gagal memutar suara ({e})")

    def toggle(self) -> bool:
        self.enabled = not self.enabled
        print(f"Voice output: {'ON' if self.enabled else 'OFF'}")
        return self.enabled


_voice_output = None


def get_voice_output(enabled: bool = False) -> NumberVoiceOutput:
    global _voice_output
    if _voice_output is None:
        _voice_output = NumberVoiceOutput(enabled)
    return _voice_output


def toggle_voice_output() -> bool:
    return get_voice_output().toggle()
