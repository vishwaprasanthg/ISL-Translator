from pathlib import Path
import sys
import numpy as np
import mediapipe as mp
import time

# ---------------------------------------------------------
# Project root
# ---------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]

DATASET_DIR = ROOT / "ml" / "datasets" / "Indian"
MODEL_PATH = ROOT / "frontend" / "public" / "models" / "hand_landmarker.task"
OUTPUT_PATH = ROOT / "ml" / "datasets" / "full_features.npz"

# Make project imports available
sys.path.insert(0, str(ROOT))

from ml.preprocessing.landmark_normalizer import create_feature_vector


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

# ---------------------------------------------------------
# MediaPipe
# ---------------------------------------------------------
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
    running_mode=RunningMode.IMAGE,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)

# ---------------------------------------------------------
# Statistics
# ---------------------------------------------------------
total_raw_images = 0
total_valid = 0
total_discarded = 0
zero_hands = 0
failed = 0

class_stats = {}
features = []
labels = []

# ---------------------------------------------------------
# Extract
# ---------------------------------------------------------
print("=" * 60)
print("FULL LANDMARK EXTRACTION")
print("=" * 60)

if not DATASET_DIR.exists():
    raise FileNotFoundError(f"Dataset directory not found: {DATASET_DIR}")
if not MODEL_PATH.exists():
    raise FileNotFoundError(f"MediaPipe model not found: {MODEL_PATH}")

classes = sorted(
    [d for d in DATASET_DIR.iterdir() if d.is_dir()],
    key=lambda x: x.name
)

print(f"Classes found: {len(classes)}\n")

start_time = time.time()

with HandLandmarker.create_from_options(options) as detector:

    for class_dir in classes:
        
        # Collect and sort image files numerically if possible
        image_files = []
        for f in class_dir.iterdir():
            if f.is_file() and f.suffix.lower() in VALID_EXTENSIONS:
                image_files.append(f)
                
        def sort_key(f):
            stem = f.stem
            return (0, int(stem)) if stem.isdigit() else (1, stem)
            
        image_files = sorted(image_files, key=sort_key)

        class_name = class_dir.name
        class_raw = len(image_files)
        class_valid = 0
        class_discarded = 0

        print(f"Processing Class {class_name} ({class_raw} images)...")

        for image_path in image_files:
            total_raw_images += 1

            try:
                image = mp.Image.create_from_file(str(image_path))
                result = detector.detect(image)

                detected = len(result.hand_landmarks)

                if detected == 0:
                    zero_hands += 1
                    class_discarded += 1
                    total_discarded += 1
                    continue  # CRITICAL: Discard sample if no hands detected

                left_hand = None
                right_hand = None

                for i, hand_landmarks in enumerate(result.hand_landmarks):
                    if i >= len(result.handedness):
                        continue
                    handedness = result.handedness[i]
                    if not handedness:
                        continue
                    
                    category = handedness[0]
                    label = category.category_name

                    if label == "Left":
                        left_hand = hand_landmarks
                    elif label == "Right":
                        right_hand = hand_landmarks

                def convert_landmarks(hand):
                    if hand is None: return None
                    return [{"x": lm.x, "y": lm.y, "z": lm.z} for lm in hand]

                left = convert_landmarks(left_hand)
                right = convert_landmarks(right_hand)

                vector = create_feature_vector(left_hand_landmarks=left, right_hand_landmarks=right)

                if vector.shape != (126,) or vector.dtype != np.float32 or not np.all(np.isfinite(vector)):
                    class_discarded += 1
                    total_discarded += 1
                    failed += 1
                    continue
                    
                # Discard if it's all zeros (which shouldn't happen here since detected > 0)
                if np.all(vector == 0):
                    class_discarded += 1
                    total_discarded += 1
                    failed += 1
                    continue

                features.append(vector)
                labels.append(class_name)
                class_valid += 1
                total_valid += 1

            except Exception as exc:
                failed += 1
                class_discarded += 1
                total_discarded += 1

        class_stats[class_name] = {
            "raw": class_raw,
            "valid": class_valid,
            "discarded": class_discarded
        }

# ---------------------------------------------------------
# Save
# ---------------------------------------------------------
X = np.asarray(features, dtype=np.float32)
y = np.asarray(labels)

np.savez_compressed(OUTPUT_PATH, X=X, y=y)

elapsed = time.time() - start_time

# ---------------------------------------------------------
# Final report
# ---------------------------------------------------------
print("\n" + "=" * 60)
print("DATASET EXTRACTION REPORT")
print("=" * 60)
print(f"Time taken          : {elapsed:.2f} seconds")
print(f"Total raw images    : {total_raw_images}")
print(f"Total valid samples : {total_valid}")
print(f"Total discarded     : {total_discarded} (Zero hands: {zero_hands}, Failed/Invalid: {failed})")
print(f"Discard percentage  : {(total_discarded / total_raw_images * 100) if total_raw_images > 0 else 0:.2f}%\n")

print("Per-Class Report:")
print(f"{'Class':<6} | {'Raw':<8} | {'Valid':<8} | {'Discarded':<10}")
print("-" * 40)
for c, stats in class_stats.items():
    print(f"{c:<6} | {stats['raw']:<8} | {stats['valid']:<8} | {stats['discarded']:<10}")

print("\nDataset dimensions:")
print(f"Feature matrix : {X.shape}")
print(f"Labels         : {y.shape}")
print(f"Saved to       : {OUTPUT_PATH}")
