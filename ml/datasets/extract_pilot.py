from pathlib import Path
import sys
import numpy as np
import mediapipe as mp

# ---------------------------------------------------------
# Project root
# ---------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]

DATASET_DIR = ROOT / "ml" / "datasets" / "Indian"

MODEL_PATH = ROOT / "frontend" / "public" / "models" / "hand_landmarker.task"

OUTPUT_PATH = ROOT / "ml" / "datasets" / "pilot_100_features.npz"

# Make project imports available
sys.path.insert(0, str(ROOT))

from ml.preprocessing.landmark_normalizer import create_feature_vector


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------
IMAGES_PER_CLASS = 100

VALID_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp"
}


# ---------------------------------------------------------
# MediaPipe
# ---------------------------------------------------------
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode


options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=str(MODEL_PATH)
    ),
    running_mode=RunningMode.IMAGE,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)


# ---------------------------------------------------------
# Statistics
# ---------------------------------------------------------
total_images = 0
one_hand = 0
two_hands = 0
zero_hands = 0
failed = 0

features = []
labels = []


# ---------------------------------------------------------
# Extract
# ---------------------------------------------------------
print("=" * 60)
print("PHASE 7 PILOT LANDMARK EXTRACTION")
print("=" * 60)

print(f"Dataset : {DATASET_DIR}")
print(f"Model   : {MODEL_PATH}")
print(f"Output  : {OUTPUT_PATH}")
print()


if not DATASET_DIR.exists():
    raise FileNotFoundError(
        f"Dataset directory not found: {DATASET_DIR}"
    )

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"MediaPipe model not found: {MODEL_PATH}"
    )


classes = sorted(
    [
        directory
        for directory in DATASET_DIR.iterdir()
        if directory.is_dir()
    ],
    key=lambda x: x.name
)

print(f"Classes found: {len(classes)}")
print()


with HandLandmarker.create_from_options(options) as detector:

    for class_dir in classes:

        image_files = sorted(
            [
                file
                for file in class_dir.iterdir()
                if file.is_file()
                and file.suffix.lower() in VALID_EXTENSIONS
            ]
        )

        image_files = image_files[:IMAGES_PER_CLASS]

        print(
            f"Class {class_dir.name}: "
            f"processing {len(image_files)} images"
        )

        for image_path in image_files:

            total_images += 1

            try:

                image = mp.Image.create_from_file(
                    str(image_path)
                )

                result = detector.detect(image)

                detected = len(result.hand_landmarks)

                if detected == 0:
                    zero_hands += 1

                elif detected == 1:
                    one_hand += 1

                else:
                    two_hands += 1

                # -------------------------------------------------
                # Extract Left / Right hands
                # -------------------------------------------------
                left_hand = None
                right_hand = None

                for i, hand_landmarks in enumerate(
                    result.hand_landmarks
                ):

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

                # -------------------------------------------------
                # Convert MediaPipe landmarks into dictionaries
                # so they use the existing Phase 5 normalizer.
                # -------------------------------------------------
                def convert_landmarks(hand):

                    if hand is None:
                        return None

                    return [
                        {
                            "x": landmark.x,
                            "y": landmark.y,
                            "z": landmark.z
                        }
                        for landmark in hand
                    ]

                left = convert_landmarks(left_hand)
                right = convert_landmarks(right_hand)

                vector = create_feature_vector(
                    left_hand_landmarks=left,
                    right_hand_landmarks=right
                )

                # Safety checks
                if vector.shape != (126,):
                    raise ValueError(
                        f"Invalid feature shape: {vector.shape}"
                    )

                if vector.dtype != np.float32:
                    raise ValueError(
                        f"Invalid dtype: {vector.dtype}"
                    )

                if not np.all(np.isfinite(vector)):
                    raise ValueError(
                        "Feature vector contains NaN/Infinity"
                    )

                features.append(vector)

                labels.append(class_dir.name)

            except Exception as exc:

                failed += 1

                print(
                    f"  ERROR: {image_path.name}: {exc}"
                )


# ---------------------------------------------------------
# Save
# ---------------------------------------------------------
X = np.asarray(
    features,
    dtype=np.float32
)

y = np.asarray(
    labels
)


np.savez_compressed(
    OUTPUT_PATH,
    X=X,
    y=y
)


# ---------------------------------------------------------
# Final report
# ---------------------------------------------------------
print()
print("=" * 60)
print("PILOT EXTRACTION COMPLETE")
print("=" * 60)

print(f"Total images      : {total_images}")
print(f"0 hands           : {zero_hands}")
print(f"1 hand            : {one_hand}")
print(f"2 hands           : {two_hands}")
print(f"Failed            : {failed}")

print()
print(f"Feature matrix    : {X.shape}")
print(f"Feature dtype     : {X.dtype}")
print(f"Labels            : {y.shape}")

if len(X) > 0:

    print()
    print(
        f"Min feature value : {X.min():.6f}"
    )

    print(
        f"Max feature value : {X.max():.6f}"
    )

    print(
        f"NaN present       : {np.isnan(X).any()}"
    )

    print(
        f"Infinity present  : {np.isinf(X).any()}"
    )

print()
print(f"Saved to: {OUTPUT_PATH}")