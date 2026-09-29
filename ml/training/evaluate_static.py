from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

from ml.models.static_model import StaticHandCNN


# ============================================================
# Paths
# ============================================================

DATASET_PATH = ROOT / "ml" / "datasets" / "pilot_100_features.npz"
MODEL_PATH = ROOT / "backend" / "models" / "static_alphabet_numbers.pt"


# ============================================================
# Configuration
# ============================================================

BATCH_SIZE = 64
RANDOM_STATE = 42

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


print("=" * 60)
print("PHASE 7 STATIC CNN EVALUATION")
print("=" * 60)

print(f"Dataset : {DATASET_PATH}")
print(f"Model   : {MODEL_PATH}")
print(f"Device  : {DEVICE}")
print()


# ============================================================
# Load dataset
# ============================================================

data = np.load(DATASET_PATH)

X = data["X"]
y_text = data["y"]

classes = sorted(np.unique(y_text).tolist())

label_to_index = {
    label: i
    for i, label in enumerate(classes)
}

index_to_label = {
    i: label
    for label, i in label_to_index.items()
}

y = np.array(
    [label_to_index[label] for label in y_text],
    dtype=np.int64
)


# ============================================================
# Recreate EXACT test split used during training
# ============================================================

X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=RANDOM_STATE,
    stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=RANDOM_STATE,
    stratify=y_temp
)


print(f"Total samples : {len(X)}")
print(f"Test samples  : {len(X_test)}")
print()


# ============================================================
# Load model
# ============================================================

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)

model = StaticHandCNN(
    num_classes=checkpoint["num_classes"]
).to(DEVICE)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()


# ============================================================
# Test dataset
# ============================================================

test_dataset = TensorDataset(
    torch.from_numpy(X_test.astype(np.float32)),
    torch.from_numpy(y_test)
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# ============================================================
# Prediction
# ============================================================

all_predictions = []
all_labels = []
all_confidences = []

with torch.no_grad():

    for features, labels in test_loader:

        features = features.to(DEVICE)

        logits = model(features)

        probabilities = torch.softmax(
            logits,
            dim=1
        )

        confidence, predictions = (
            probabilities.max(dim=1)
        )

        all_predictions.extend(
            predictions.cpu().numpy()
        )

        all_labels.extend(
            labels.numpy()
        )

        all_confidences.extend(
            confidence.cpu().numpy()
        )


all_predictions = np.asarray(
    all_predictions
)

all_labels = np.asarray(
    all_labels
)

all_confidences = np.asarray(
    all_confidences
)


# ============================================================
# Overall accuracy
# ============================================================

accuracy = accuracy_score(
    all_labels,
    all_predictions
)

print("=" * 60)
print("OVERALL RESULTS")
print("=" * 60)

print(
    f"Accuracy    : {accuracy * 100:.2f}%"
)

print(
    f"Avg confidence : "
    f"{all_confidences.mean() * 100:.2f}%"
)

print()


# ============================================================
# Classification report
# ============================================================

print("=" * 60)
print("PER-CLASS RESULTS")
print("=" * 60)

print(
    classification_report(
        all_labels,
        all_predictions,
        labels=list(range(len(classes))),
        target_names=classes,
        digits=4,
        zero_division=0
    )
)


# ============================================================
# Confusion matrix
# ============================================================

cm = confusion_matrix(
    all_labels,
    all_predictions,
    labels=list(range(len(classes)))
)


print("=" * 60)
print("CONFUSION MATRIX")
print("=" * 60)

print("Rows = Actual")
print("Columns = Predicted")
print()

print("     " + " ".join(
    f"{c:>3}"
    for c in classes
))

for i, row in enumerate(cm):

    print(
        f"{classes[i]:>3}  " +
        " ".join(
            f"{value:>3}"
            for value in row
        )
    )


# ============================================================
# Most confused pairs
# ============================================================

confusions = []

for i in range(len(classes)):

    for j in range(len(classes)):

        if i != j and cm[i, j] > 0:

            confusions.append(
                (
                    int(cm[i, j]),
                    classes[i],
                    classes[j]
                )
            )

confusions.sort(
    reverse=True
)


print()
print("=" * 60)
print("TOP CONFUSIONS")
print("=" * 60)

if confusions:

    for count, actual, predicted in confusions[:15]:

        print(
            f"{actual} -> {predicted} : "
            f"{count} samples"
        )

else:

    print("No incorrect predictions.")


# ============================================================
# Save evaluation results
# ============================================================

results_path = (
    ROOT / "ml" / "datasets" /
    "static_evaluation.txt"
)

with open(
    results_path,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        f"Accuracy: {accuracy * 100:.2f}%\n"
    )

    f.write(
        f"Average confidence: "
        f"{all_confidences.mean() * 100:.2f}%\n"
    )

    f.write("\nClassification Report\n")
    f.write(
        classification_report(
            all_labels,
            all_predictions,
            labels=list(range(len(classes))),
            target_names=classes,
            digits=4,
            zero_division=0
        )
    )

print()
print(f"Results saved to: {results_path}")

print()
print("=" * 60)
print("EVALUATION COMPLETE")
print("=" * 60)