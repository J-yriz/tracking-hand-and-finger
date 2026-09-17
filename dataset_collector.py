"""Tool koleksi dataset untuk bahasa isyarat angka 1-10."""

import argparse
import csv
import json
import time
from datetime import datetime
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    HandLandmarker,
    HandLandmarkerOptions,
    RunningMode,
)

# Import dari proyek utama
try:
    from hand_tracking import ensure_model, draw_fingers, track_fingers
    from camera import open_camera
except ImportError:
    print("Error: Pastikan hand_tracking.py dan camera.py ada di direktori yang sama")
    exit(1)


class GestureDatasetCollector:
    """Koleksi dataset gesture angka 1-10."""
    
    def __init__(self, output_dir="dataset_numbers"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Model path
        self.model_path = ensure_model()
        
        # Setup landmarker
        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=self.model_path),
            running_mode=RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.landmarker = HandLandmarker.create_from_options(options)
        
        # State
        self.current_number = 1
        self.sample_count = 0
        self.max_samples = 10  # Sampel per angka
        self.collecting = False
        self.samples = []
        
        # File output
        self.csv_path = self.output_dir / "dataset.csv"
        self.metadata_path = self.output_dir / "metadata.json"
        
        # Init CSV jika belum ada
        if not self.csv_path.exists():
            with open(self.csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                # Header: number, timestamp, landmark_coordinates (21*3=63), finger_states (5)
                header = ['number', 'timestamp']
                header += [f'landmark_{i}_{coord}' for i in range(21) for coord in ['x', 'y', 'z']]
                header += ['thumb', 'index', 'middle', 'ring', 'pinky']
                writer.writerow(header)
    
    def start_collection(self, number: int):
        """Mulai koleksi untuk angka tertentu."""
        self.current_number = number
        self.sample_count = 0
        self.collecting = True
        self.samples = []
        print(f"Mulai koleksi untuk angka {number}. Tekan SPACE untuk capture.")
    
    def capture_sample(self, landmarks, details):
        """Capture sampel gesture."""
        if not landmarks or len(landmarks) < 21:
            print("Error: Landmark tidak valid")
            return False
        
        # Ekstrak koordinat landmark
        landmark_data = []
        for lm in landmarks:
            landmark_data.extend([lm.x, lm.y, lm.z])
        
        # Ekstrak status jari
        finger_states = {}
        for finger_name in ["Jempol", "Telunjuk", "Tengah", "Manis", "Kelingking"]:
            if finger_name in details:
                finger_states[finger_name] = details[finger_name]["extended"]
            else:
                finger_states[finger_name] = False
        
        # Buat sampel
        sample = {
            'number': self.current_number,
            'timestamp': datetime.now().isoformat(),
            'landmarks': landmark_data,
            'thumb': int(finger_states["Jempol"]),
            'index': int(finger_states["Telunjuk"]),
            'middle': int(finger_states["Tengah"]),
            'ring': int(finger_states["Manis"]),
            'pinky': int(finger_states["Kelingking"]),
        }
        
        self.samples.append(sample)
        self.sample_count += 1
        
        print(f"Sample {self.sample_count}/{self.max_samples} captured untuk angka {self.current_number}")
        
        if self.sample_count >= self.max_samples:
            self.save_batch()
            return True
        
        return False
    
    def save_batch(self):
        """Simpan batch sampel ke CSV."""
        if not self.samples:
            return
        
        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            for sample in self.samples:
                row = [
                    sample['number'],
                    sample['timestamp'],
                    *sample['landmarks'],
                    sample['thumb'],
                    sample['index'],
                    sample['middle'],
                    sample['ring'],
                    sample['pinky']
                ]
                writer.writerow(row)
        
        print(f"Saved {len(self.samples)} samples untuk angka {self.current_number} ke {self.csv_path}")
        self.samples = []
        self.collecting = False
    
    def save_metadata(self):
        """Simpan metadata dataset."""
        metadata = {
            'created': datetime.now().isoformat(),
            'numbers_collected': list(range(1, 11)),
            'samples_per_number': self.max_samples,
            'total_samples': self.max_samples * 10,
            'landmark_format': '21 landmarks, each with (x, y, z) normalized coordinates',
            'finger_states': '0=lipat, 1=lurus',
        }
        
        with open(self.metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Metadata saved to {self.metadata_path}")
    
    def run(self, camera_index=0):
        """Run koleksi dataset."""
        cap = open_camera(camera_index)
        if not cap.isOpened():
            print(f"Tidak bisa membuka kamera index {camera_index}")
            return
        
        print("\n=== Gesture Dataset Collector ===\n")
        print("Petunjuk:")
        print("  1-0: Pilih angka (1-10, 0 untuk angka 0)")
        print("  SPACE: Capture sampel saat ini")
        print("  s: Simpan batch dan lanjut ke angka berikutnya")
        print("  q: Keluar dan simpan semua data")
        print(f"\nDataset akan disimpan di: {self.output_dir}")
        
        start_time = time.time()
        current_number = None
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Gagal membaca frame")
                break
            
            # Process dengan MediaPipe
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms = int((time.time() - start_time) * 1000)
            result = self.landmarker.detect_for_video(mp_image, timestamp_ms)
            
            # Draw UI
            h, w = frame.shape[:2]
            
            # Info status
            status_text = f"Angka: {current_number if current_number is not None else '-'}"
            if self.collecting:
                status_text += f" | Samples: {self.sample_count}/{self.max_samples}"
            
            cv2.putText(frame, status_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            cv2.putText(frame, "1-0: Pilih angka | SPACE: Capture | s: Save | q: Quit", 
                       (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            # Process tangan jika ada
            if result.hand_landmarks:
                landmarks = result.hand_landmarks[0]
                details = draw_fingers(frame, landmarks, tip_labels=False)
                
                # Draw bounding box info
                x1, y1, x2, y2, cx, cy = self._hand_center(landmarks, w, h)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Tampilkan status jari
                finger_info = ""
                if details:
                    for finger_name in ["Jempol", "Telunjuk", "Tengah", "Manis", "Kelingking"]:
                        if finger_name in details:
                            state = "LURUS" if details[finger_name]["extended"] else "LIPAT"
                            finger_info += f"{finger_name[0]}:{state} "
                
                cv2.putText(frame, finger_info.strip(), (x1, max(0, y1 - 10)),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            cv2.imshow("Gesture Dataset Collector", frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                if self.samples:
                    self.save_batch()
                self.save_metadata()
                print("Dataset collection selesai.")
                break
            
            elif ord('1') <= key <= ord('9'):
                number = key - ord('0')
                current_number = number
                self.start_collection(number)
            
            elif key == ord('0'):
                current_number = 0
                self.start_collection(0)
            
            elif key == ord(' ') and self.collecting and result.hand_landmarks:
                # Capture sampel
                landmarks = result.hand_landmarks[0]
                details = track_fingers(landmarks, w, h)
                completed = self.capture_sample(landmarks, details)
                
                if completed:
                    current_number = None
            
            elif key == ord('s') and self.collecting:
                # Simpan batch sekarang
                if self.samples:
                    self.save_batch()
                    current_number = None
        
        cap.release()
        cv2.destroyAllWindows()
        self.landmarker.close()
    
    def _hand_center(self, landmarks, w, h):
        """Helper: hitung center tangan."""
        xs = [p.x for p in landmarks]
        ys = [p.y for p in landmarks]
        x1, y1 = int(min(xs) * w), int(min(ys) * h)
        x2, y2 = int(max(xs) * w), int(max(ys) * h)
        return (x1, y1, x2, y2, (x1 + x2) // 2, (y1 + y2) // 2)


def main():
    parser = argparse.ArgumentParser(description="Koleksi dataset bahasa isyarat angka.")
    parser.add_argument("--camera", type=int, default=0, help="Index kamera")
    parser.add_argument("--output", type=str, default="dataset_numbers", 
                       help="Direktori output untuk dataset")
    parser.add_argument("--samples", type=int, default=10,
                       help="Jumlah sampel per angka")
    
    args = parser.parse_args()
    
    collector = GestureDatasetCollector(args.output)
    collector.max_samples = args.samples
    collector.run(args.camera)


if __name__ == "__main__":
    main()