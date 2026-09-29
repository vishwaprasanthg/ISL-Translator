import sys
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix, f1_score

from ml.models.static_model import StaticHandCNN

# ============================================================
# Paths
# ============================================================

DATASET_PATH = ROOT / "ml" / "datasets" / "full_features.npz"
MODEL_DIR = ROOT / "backend" / "models"
LABEL_DIR = ROOT / "ml" / "label_maps"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
LABEL_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# Configuration
# ============================================================

BATCH_SIZE = 64
EPOCHS = 100
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
PATIENCE = 10  # For early stopping
RANDOM_STATE = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 60)
print("PHASE 8 ROBUST STATIC CNN TRAINING")
print("=" * 60)
print(f"Dataset : {DATASET_PATH}")
print(f"Device  : {DEVICE}")
print()

# ============================================================
# Load dataset
# ============================================================

if not DATASET_PATH.exists():
    raise FileNotFoundError(f"Dataset not found at {DATASET_PATH}")

data = np.load(DATASET_PATH)
X_raw = data["X"]
y_raw = data["y"]

print(f"Total Loaded Samples : {X_raw.shape}")

# Deduplication
unique_samples, unique_indices = np.unique(X_raw, axis=0, return_index=True)
X = X_raw[unique_indices]
y_text = y_raw[unique_indices]

print(f"After deduplication  : {X.shape}")

if X.shape[1] != 126:
    raise ValueError(f"Expected 126 features, got {X.shape[1]}")

if X.dtype != np.float32:
    X = X.astype(np.float32)

if not np.all(np.isfinite(X)):
    raise ValueError("Dataset contains NaN or Infinity")

# Create label mapping
classes = sorted(np.unique(y_text).tolist())
label_to_index = {label: index for index, label in enumerate(classes)}
index_to_label = {index: label for label, index in label_to_index.items()}

y = np.array([label_to_index[label] for label in y_text], dtype=np.int64)

print(f"Classes : {len(classes)}")

# ============================================================
# Contiguous Block Split (Prevent Video Data Leakage)
# ============================================================

X_train_list, y_train_list = [], []
X_val_list, y_val_list = [], []
X_test_list, y_test_list = [], []

for c_idx, c_name in index_to_label.items():
    idx = np.where(y == c_idx)[0]
    # Since extraction was ordered, indices within a class correspond to chronological frames
    n = len(idx)
    n_train = int(n * 0.70)
    n_val = int(n * 0.15)
    
    train_idx = idx[:n_train]
    val_idx = idx[n_train:n_train + n_val]
    test_idx = idx[n_train + n_val:]
    
    X_train_list.append(X[train_idx])
    y_train_list.append(y[train_idx])
    
    X_val_list.append(X[val_idx])
    y_val_list.append(y[val_idx])
    
    X_test_list.append(X[test_idx])
    y_test_list.append(y[test_idx])

X_train = np.vstack(X_train_list)
y_train = np.concatenate(y_train_list)

X_val = np.vstack(X_val_list)
y_val = np.concatenate(y_val_list)

X_test = np.vstack(X_test_list)
y_test = np.concatenate(y_test_list)

print("\nDataset split (Contiguous to prevent leakage):")
print(f"Train      : {len(X_train)}")
print(f"Validation : {len(X_val)}")
print(f"Test       : {len(X_test)}")
print()

# ============================================================
# Dataset and Augmentation
# ============================================================

class AugmentedISLDataset(Dataset):
    def __init__(self, X, y, augment=False):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)
        self.augment = augment

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx].clone()
        if self.augment:
            # Gaussian Noise (Jitter)
            noise = torch.randn_like(x) * 0.005  # small jitter
            x += noise
            
            # Small scale variation (0.95 to 1.05)
            scale = 0.95 + 0.1 * torch.rand(1).item()
            x *= scale
            
        return x, self.y[idx]

train_dataset = AugmentedISLDataset(X_train, y_train, augment=True)
val_dataset = AugmentedISLDataset(X_val, y_val, augment=False)
test_dataset = AugmentedISLDataset(X_test, y_test, augment=False)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ============================================================
# Model
# ============================================================

model = StaticHandCNN(num_classes=len(classes)).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, verbose=True)

# ============================================================
# Evaluation function
# ============================================================

def evaluate(loader):
    model.eval()
    all_preds = []
    all_labels = []
    running_loss = 0.0

    with torch.no_grad():
        for features, labels in loader:
            features, labels = features.to(DEVICE), labels.to(DEVICE)
            outputs = model(features)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * labels.size(0)
            
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    avg_loss = running_loss / len(loader.dataset)
    macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    acc = np.mean(np.array(all_preds) == np.array(all_labels))
    return avg_loss, acc, macro_f1, all_preds, all_labels

# ============================================================
# Training
# ============================================================

best_val_f1 = 0.0
best_state = None
epochs_without_improvement = 0

for epoch in range(1, EPOCHS + 1):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for features, labels in train_loader:
        features, labels = features.to(DEVICE), labels.to(DEVICE)
        
        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * labels.size(0)
        predictions = outputs.argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    train_loss = running_loss / total
    train_accuracy = correct / total

    val_loss, val_acc, val_f1, _, _ = evaluate(val_loader)
    
    # Step scheduler based on validation F1
    scheduler.step(val_f1)

    print(
        f"Epoch {epoch:02d}/{EPOCHS} | "
        f"Loss: {train_loss:.4f} | "
        f"Train Acc: {train_accuracy:.4f} | "
        f"Val Acc: {val_acc:.4f} | "
        f"Val F1: {val_f1:.4f}"
    )

    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        best_state = {key: value.cpu().clone() for key, value in model.state_dict().items()}
        epochs_without_improvement = 0
    else:
        epochs_without_improvement += 1
        
    if epochs_without_improvement >= PATIENCE:
        print(f"\nEarly stopping triggered after {epoch} epochs.")
        break

# ============================================================
# Test & Reporting
# ============================================================

if best_state is None:
    raise RuntimeError("No model checkpoint was produced")

model.load_state_dict(best_state)
model.to(DEVICE)

test_loss, test_acc, test_f1, test_preds, test_labels = evaluate(test_loader)

print("\n" + "=" * 60)
print("FINAL RESULTS")
print("=" * 60)
print(f"Best Val Macro F1 : {best_val_f1 * 100:.2f}%")
print(f"Test Accuracy     : {test_acc * 100:.2f}%")
print(f"Test Macro F1     : {test_f1 * 100:.2f}%")

print("\nPer-Class Metrics:")
report = classification_report(test_labels, test_preds, target_names=classes, zero_division=0)
print(report)

print("\nConfusion Matrix:")
cm = confusion_matrix(test_labels, test_preds)
print(cm)

print("\nWorst Confusion Pairs:")
# Find high off-diagonal elements
pairs = []
for i in range(len(classes)):
    for j in range(len(classes)):
        if i != j and cm[i, j] > 0:
            pairs.append((classes[i], classes[j], cm[i, j]))

pairs.sort(key=lambda x: x[2], reverse=True)
for pair in pairs[:15]:
    if pair[2] > 0:
        print(f"Actual {pair[0]} -> Predicted {pair[1]} : {pair[2]} times")

# ============================================================
# Save checkpoint
# ============================================================

checkpoint_path = MODEL_DIR / "static_alphabet_numbers.pt"
torch.save(
    {
        "model_state_dict": model.state_dict(),
        "num_classes": len(classes),
        "input_features": 126,
        "classes": classes,
    },
    checkpoint_path
)

label_map_path = LABEL_DIR / "static.json"
with open(label_map_path, "w", encoding="utf-8") as f:
    json.dump(index_to_label, f, indent=2)

print("\n" + "=" * 60)
print(f"Checkpoint saved to : {checkpoint_path}")
print(f"Label map saved to  : {label_map_path}")
print("TRAINING COMPLETE")
print("=" * 60)
