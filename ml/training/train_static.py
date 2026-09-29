"""
ISL Separate Model Training Pipeline

Trains EITHER the Alphabet CNN (26 classes) or Number CNN (9 classes)
using session-aware data splitting and landmark-level augmentation.

Usage:
    python train_static.py --mode alphabet
    python train_static.py --mode numbers
"""

import sys
import argparse
import json
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from pathlib import Path
from collections import defaultdict
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)

# ============================================================
# Paths
# ============================================================

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from ml.models.static_model import StaticHandCNN

DATASET_ROOT = ROOT / "ml" / "datasets" / "manual"
MODEL_DIR = ROOT / "backend" / "models"
LABEL_DIR = ROOT / "ml" / "label_maps"
EVAL_DIR = ROOT / "ml" / "datasets" / "manual" / "evaluation"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
LABEL_DIR.mkdir(parents=True, exist_ok=True)
EVAL_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# Constants
# ============================================================

TOTAL_FEATURES = 126
RANDOM_STATE = 42

ALPHABET_CLASSES = [chr(i) for i in range(ord("A"), ord("Z") + 1)]
NUMBER_CLASSES = [str(i) for i in range(1, 10)]

MODE_CONFIG = {
    "alphabet": {
        "dir": DATASET_ROOT / "alphabet",
        "classes": ALPHABET_CLASSES,
        "checkpoint": MODEL_DIR / "static_alphabet.pt",
        "label_map": LABEL_DIR / "alphabet.json",
        "num_classes": 26,
    },
    "numbers": {
        "dir": DATASET_ROOT / "numbers",
        "classes": NUMBER_CLASSES,
        "checkpoint": MODEL_DIR / "static_numbers.pt",
        "label_map": LABEL_DIR / "numbers.json",
        "num_classes": 9,
    },
}


# ============================================================
# Augmentation (training only)
# ============================================================

class LandmarkAugmenter:
    """Applies small realistic perturbations to 126-D landmark features."""

    def __init__(self, noise_std=0.02, scale_range=(0.95, 1.05)):
        self.noise_std = noise_std
        self.scale_range = scale_range

    def __call__(self, features):
        """
        Args:
            features: (126,) numpy array or tensor
        Returns:
            augmented: (126,) tensor
        """
        if isinstance(features, np.ndarray):
            features = torch.from_numpy(features).float()

        # Gaussian jitter
        noise = torch.randn_like(features) * self.noise_std
        features = features + noise

        # Small scale variation (applied per-hand)
        scale = torch.empty(1).uniform_(*self.scale_range).item()
        features = features * scale

        return features


# ============================================================
# Dataset
# ============================================================

class ISLLandmarkDataset(Dataset):
    """
    Dataset for ISL landmark features with optional augmentation.
    """

    def __init__(self, features, labels, augmenter=None):
        self.features = torch.from_numpy(features).float()
        self.labels = torch.from_numpy(labels).long()
        self.augmenter = augmenter

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        x = self.features[idx].clone()
        y = self.labels[idx]

        if self.augmenter is not None:
            x = self.augmenter(x)

        return x, y


# ============================================================
# Data Loading
# ============================================================

def load_dataset(mode: str):
    """
    Load all data for a mode, returning features, labels, and group indices.
    
    Since each class has only 1 session, we create pseudo-sessions by splitting
    the single session into 5 temporal chunks to enable session-aware splitting.
    """
    config = MODE_CONFIG[mode]
    base_dir = config["dir"]
    classes = config["classes"]

    label_to_idx = {cls: i for i, cls in enumerate(classes)}

    all_features = []
    all_labels = []
    all_groups = []  # Group index for session-aware splitting

    group_counter = 0

    for cls in classes:
        cls_dir = base_dir / cls
        session_files = sorted(cls_dir.glob("session_*.npz"))

        for npz_path in session_files:
            data = np.load(npz_path, allow_pickle=True)
            features = data["features"].astype(np.float32)

            # Validate
            valid_mask = np.all(np.isfinite(features), axis=1) & ~np.all(features == 0, axis=1)
            features = features[valid_mask]

            n = len(features)
            if n == 0:
                continue

            labels = np.full(n, label_to_idx[cls], dtype=np.int64)

            # Split single session into 5 pseudo-sessions (temporal chunks)
            # This enables session-aware train/val/test splitting
            n_chunks = 5
            chunk_size = n // n_chunks
            remainder = n % n_chunks

            start = 0
            for chunk_idx in range(n_chunks):
                end = start + chunk_size + (1 if chunk_idx < remainder else 0)
                chunk_features = features[start:end]
                chunk_labels = labels[start:end]

                all_features.append(chunk_features)
                all_labels.append(chunk_labels)
                all_groups.extend([group_counter] * len(chunk_features))
                group_counter += 1
                start = end

    X = np.concatenate(all_features, axis=0)
    y = np.concatenate(all_labels, axis=0)
    groups = np.array(all_groups, dtype=np.int64)

    return X, y, groups, classes


def session_aware_split(X, y, groups, train_ratio=0.70, val_ratio=0.15):
    """
    Split data by group (session) membership.
    Groups are assigned to train/val/test ensuring no group spans sets.
    
    For each class, assigns groups proportionally:
      - ~70% groups -> train
      - ~15% groups -> val  
      - ~15% groups -> test
    """
    np.random.seed(RANDOM_STATE)

    unique_groups = np.unique(groups)
    num_classes = len(np.unique(y))

    # Build class -> groups mapping
    class_groups = defaultdict(list)
    for g in unique_groups:
        mask = groups == g
        cls = y[mask][0]  # All samples in a group have same class
        class_groups[cls].append(g)

    train_groups = set()
    val_groups = set()
    test_groups = set()

    for cls in sorted(class_groups.keys()):
        g_list = class_groups[cls]
        np.random.shuffle(g_list)
        n = len(g_list)

        n_train = max(1, int(n * train_ratio))
        n_val = max(1, int(n * val_ratio))
        n_test = n - n_train - n_val

        if n_test < 1:
            n_test = 1
            n_train = n - n_val - n_test

        train_groups.update(g_list[:n_train])
        val_groups.update(g_list[n_train:n_train + n_val])
        test_groups.update(g_list[n_train + n_val:])

    train_mask = np.isin(groups, list(train_groups))
    val_mask = np.isin(groups, list(val_groups))
    test_mask = np.isin(groups, list(test_groups))

    return (
        X[train_mask], y[train_mask],
        X[val_mask], y[val_mask],
        X[test_mask], y[test_mask],
    )


# ============================================================
# Training
# ============================================================

def train_model(mode: str, config: dict):
    """Train a single model (alphabet or numbers)."""

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BATCH_SIZE = 64
    MAX_EPOCHS = 100
    PATIENCE = 15
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4

    print("=" * 60)
    print(f"  TRAINING: {mode.upper()} MODEL")
    print("=" * 60)
    print(f"  Device       : {DEVICE}")
    print(f"  Classes      : {config['num_classes']}")
    print(f"  Max epochs   : {MAX_EPOCHS}")
    print(f"  Patience     : {PATIENCE}")
    print(f"  Batch size   : {BATCH_SIZE}")
    print(f"  LR           : {LEARNING_RATE}")
    print()

    # Load data
    print("Loading dataset...")
    X, y, groups, classes = load_dataset(mode)
    print(f"  Total samples: {len(X)}")
    print(f"  Feature dim  : {X.shape[1]}")
    print(f"  Classes      : {len(classes)}")
    print(f"  Groups       : {len(np.unique(groups))}")

    # Split
    print("\nSession-aware splitting...")
    X_train, y_train, X_val, y_val, X_test, y_test = session_aware_split(X, y, groups)

    print(f"  Train      : {len(X_train)} ({len(X_train)/len(X)*100:.1f}%)")
    print(f"  Validation : {len(X_val)} ({len(X_val)/len(X)*100:.1f}%)")
    print(f"  Test       : {len(X_test)} ({len(X_test)/len(X)*100:.1f}%)")

    # Per-class distribution
    print("\n  Per-class split:")
    for i, cls in enumerate(classes):
        n_tr = (y_train == i).sum()
        n_va = (y_val == i).sum()
        n_te = (y_test == i).sum()
        print(f"    {cls}: train={n_tr}, val={n_va}, test={n_te}")

    # Class weights
    class_counts = np.bincount(y_train, minlength=len(classes))
    if class_counts.min() > 0:
        class_weights = 1.0 / class_counts.astype(np.float32)
        class_weights = class_weights / class_weights.sum() * len(classes)
        class_weights_tensor = torch.from_numpy(class_weights).float().to(DEVICE)
    else:
        class_weights_tensor = None

    # Datasets
    augmenter = LandmarkAugmenter(noise_std=0.02, scale_range=(0.95, 1.05))
    train_dataset = ISLLandmarkDataset(X_train, y_train, augmenter=augmenter)
    val_dataset = ISLLandmarkDataset(X_val, y_val, augmenter=None)
    test_dataset = ISLLandmarkDataset(X_test, y_test, augmenter=None)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Model
    model = StaticHandCNN(num_classes=config["num_classes"]).to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n  Model params: {total_params:,}")

    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS, eta_min=1e-6)

    # Training loop
    best_val_f1 = 0.0
    best_state = None
    best_epoch = 0
    patience_counter = 0

    history = {
        "train_loss": [], "train_acc": [],
        "val_loss": [], "val_acc": [], "val_f1": [],
        "lr": [],
    }

    print(f"\n{'Epoch':>6} {'TrLoss':>8} {'TrAcc':>7} {'VaLoss':>8} {'VaAcc':>7} {'VaF1':>7} {'LR':>10} {'Time':>6}")
    print("-" * 68)

    for epoch in range(1, MAX_EPOCHS + 1):
        epoch_start = time.time()

        # --- Train ---
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for features, labels in train_loader:
            features = features.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * labels.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss / total
        train_acc = correct / total

        # --- Validate ---
        model.eval()
        val_running_loss = 0.0
        val_correct = 0
        val_total = 0
        all_val_preds = []
        all_val_labels = []

        with torch.no_grad():
            for features, labels in val_loader:
                features = features.to(DEVICE)
                labels = labels.to(DEVICE)

                outputs = model(features)
                loss = criterion(outputs, labels)

                val_running_loss += loss.item() * labels.size(0)
                preds = outputs.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

                all_val_preds.extend(preds.cpu().numpy())
                all_val_labels.extend(labels.cpu().numpy())

        val_loss = val_running_loss / val_total
        val_acc = val_correct / val_total

        # Macro F1
        _, _, val_f1, _ = precision_recall_fscore_support(
            all_val_labels, all_val_preds, average="macro", zero_division=0
        )

        current_lr = optimizer.param_groups[0]["lr"]
        epoch_time = time.time() - epoch_start

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_f1"].append(val_f1)
        history["lr"].append(current_lr)

        print(
            f"{epoch:>6} {train_loss:>8.4f} {train_acc:>7.4f} "
            f"{val_loss:>8.4f} {val_acc:>7.4f} {val_f1:>7.4f} "
            f"{current_lr:>10.6f} {epoch_time:>5.1f}s"
        )

        # Checkpoint
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= PATIENCE:
            print(f"\nEarly stopping at epoch {epoch} (patience={PATIENCE})")
            break

        scheduler.step()

    # ============================================================
    # Restore best model
    # ============================================================

    if best_state is None:
        raise RuntimeError("No checkpoint was produced!")

    model.load_state_dict(best_state)
    model.to(DEVICE)

    print(f"\nBest epoch: {best_epoch} (Val Macro F1: {best_val_f1:.4f})")

    # ============================================================
    # Test evaluation
    # ============================================================

    print(f"\n{'=' * 60}")
    print(f"  TEST EVALUATION: {mode.upper()}")
    print(f"{'=' * 60}")

    model.eval()
    all_test_preds = []
    all_test_labels = []
    all_test_probs = []

    with torch.no_grad():
        for features, labels in test_loader:
            features = features.to(DEVICE)
            outputs = model(features)
            probs = F.softmax(outputs, dim=1)
            preds = outputs.argmax(dim=1)

            all_test_preds.extend(preds.cpu().numpy())
            all_test_labels.extend(labels.cpu().numpy())
            all_test_probs.extend(probs.cpu().numpy())

    all_test_preds = np.array(all_test_preds)
    all_test_labels = np.array(all_test_labels)

    test_acc = accuracy_score(all_test_labels, all_test_preds)
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        all_test_labels, all_test_preds, average="macro", zero_division=0
    )
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(
        all_test_labels, all_test_preds, average="weighted", zero_division=0
    )

    print(f"\n  Test Accuracy     : {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"  Macro Precision   : {macro_p:.4f}")
    print(f"  Macro Recall      : {macro_r:.4f}")
    print(f"  Macro F1          : {macro_f1:.4f}")
    print(f"  Weighted F1       : {weighted_f1:.4f}")

    # Per-class report
    print(f"\n  Per-class metrics:")
    report = classification_report(
        all_test_labels, all_test_preds,
        target_names=classes, output_dict=True, zero_division=0
    )
    report_str = classification_report(
        all_test_labels, all_test_preds,
        target_names=classes, zero_division=0
    )
    print(report_str)

    # Confusion matrix
    cm = confusion_matrix(all_test_labels, all_test_preds)
    print(f"\n  Confusion Matrix:")
    print(f"  {'':>4}", end="")
    for cls in classes:
        print(f"{cls:>4}", end="")
    print()
    for i, cls in enumerate(classes):
        print(f"  {cls:>4}", end="")
        for j in range(len(classes)):
            print(f"{cm[i][j]:>4}", end="")
        print()

    # Top confusion pairs
    print(f"\n  Top confusion pairs:")
    confusion_pairs = []
    for i in range(len(classes)):
        for j in range(len(classes)):
            if i != j and cm[i][j] > 0:
                confusion_pairs.append((classes[i], classes[j], cm[i][j]))

    confusion_pairs.sort(key=lambda x: x[2], reverse=True)
    for actual, predicted, count in confusion_pairs[:15]:
        print(f"    {actual} -> {predicted} : {count}")

    # ============================================================
    # Save confusion matrix as image
    # ============================================================

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(max(8, len(classes) * 0.5), max(6, len(classes) * 0.4)))
        im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
        ax.set_title(f"{mode.upper()} Model - Confusion Matrix")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_xticks(range(len(classes)))
        ax.set_yticks(range(len(classes)))
        ax.set_xticklabels(classes, fontsize=7)
        ax.set_yticklabels(classes, fontsize=7)
        plt.colorbar(im, ax=ax)
        plt.tight_layout()

        cm_path = EVAL_DIR / f"{mode}_confusion_matrix.png"
        plt.savefig(cm_path, dpi=150)
        plt.close()
        print(f"\n  Confusion matrix saved: {cm_path}")
    except ImportError:
        print("\n  [!] matplotlib not available, skipping confusion matrix image")

    # ============================================================
    # Save checkpoint
    # ============================================================

    checkpoint_path = config["checkpoint"]

    checkpoint = {
        "model_state_dict": best_state,
        "num_classes": config["num_classes"],
        "input_features": TOTAL_FEATURES,
        "classes": classes,
        "best_epoch": best_epoch,
        "best_val_f1": best_val_f1,
        "test_accuracy": test_acc,
        "test_macro_f1": macro_f1,
    }

    torch.save(checkpoint, checkpoint_path)
    print(f"\n  Checkpoint saved: {checkpoint_path}")

    # ============================================================
    # Save metrics report
    # ============================================================

    metrics_report = {
        "mode": mode,
        "num_classes": config["num_classes"],
        "classes": classes,
        "total_params": total_params,
        "dataset": {
            "train": int(len(X_train)),
            "val": int(len(X_val)),
            "test": int(len(X_test)),
        },
        "best_epoch": best_epoch,
        "best_val_f1": float(best_val_f1),
        "test": {
            "accuracy": float(test_acc),
            "macro_precision": float(macro_p),
            "macro_recall": float(macro_r),
            "macro_f1": float(macro_f1),
            "weighted_f1": float(weighted_f1),
        },
        "top_confusions": [
            {"actual": a, "predicted": p, "count": int(c)}
            for a, p, c in confusion_pairs[:15]
        ],
        "history": {
            "train_loss": [float(v) for v in history["train_loss"]],
            "val_f1": [float(v) for v in history["val_f1"]],
        },
    }

    report_path = EVAL_DIR / f"{mode}_metrics.json"
    with open(report_path, "w") as f:
        json.dump(metrics_report, f, indent=2)
    print(f"  Metrics saved: {report_path}")

    # Save text report
    report_txt_path = EVAL_DIR / f"{mode}_report.txt"
    with open(report_txt_path, "w") as f:
        f.write(f"{mode.upper()} MODEL EVALUATION REPORT\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Classes: {config['num_classes']}\n")
        f.write(f"Parameters: {total_params:,}\n")
        f.write(f"Best epoch: {best_epoch}\n")
        f.write(f"Train samples: {len(X_train)}\n")
        f.write(f"Val samples: {len(X_val)}\n")
        f.write(f"Test samples: {len(X_test)}\n\n")
        f.write(f"Test Accuracy: {test_acc:.4f}\n")
        f.write(f"Macro Precision: {macro_p:.4f}\n")
        f.write(f"Macro Recall: {macro_r:.4f}\n")
        f.write(f"Macro F1: {macro_f1:.4f}\n")
        f.write(f"Weighted F1: {weighted_f1:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report_str)
        f.write("\n\nTop Confusion Pairs:\n")
        for a, p, c in confusion_pairs[:15]:
            f.write(f"  {a} -> {p} : {c}\n")
    print(f"  Report saved: {report_txt_path}")

    print(f"\n{'=' * 60}")
    print(f"  {mode.upper()} MODEL TRAINING COMPLETE")
    print(f"{'=' * 60}")

    return {
        "test_acc": test_acc,
        "macro_f1": macro_f1,
        "best_epoch": best_epoch,
    }


# ============================================================
# Inference test
# ============================================================

def test_inference(mode: str, config: dict, n_samples=10):
    """Load the saved model and test on random samples."""
    print(f"\n{'=' * 60}")
    print(f"  INFERENCE TEST: {mode.upper()}")
    print(f"{'=' * 60}")

    checkpoint_path = config["checkpoint"]
    if not checkpoint_path.exists():
        print(f"  [X] Checkpoint not found: {checkpoint_path}")
        return

    # Load model
    model = StaticHandCNN(num_classes=config["num_classes"])
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Load label map
    label_map_path = config["label_map"]
    with open(label_map_path) as f:
        idx_to_label = {int(k): v for k, v in json.load(f).items()}

    # Load test data
    X, y, _, classes = load_dataset(mode)

    np.random.seed(RANDOM_STATE + 1)
    indices = np.random.choice(len(X), min(n_samples, len(X)), replace=False)

    print(f"\n  {'Expected':>10} {'Predicted':>10} {'Confidence':>12} {'Result':>8}")
    print(f"  {'-' * 44}")

    correct = 0
    for idx in indices:
        features = torch.from_numpy(X[idx:idx+1]).float()
        with torch.no_grad():
            logits = model(features)
            probs = F.softmax(logits, dim=1)
            conf, pred_idx = probs.max(dim=1)

        expected = classes[y[idx]]
        predicted = idx_to_label.get(pred_idx.item(), "?")
        confidence = conf.item()
        match = "PASS" if expected == predicted else "FAIL"
        if expected == predicted:
            correct += 1

        print(f"  {expected:>10} {predicted:>10} {confidence:>12.4f} {match:>8}")

    print(f"\n  Inference test: {correct}/{len(indices)} correct")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Train ISL static sign model")
    parser.add_argument("--mode", required=True, choices=["alphabet", "numbers"],
                        help="Which model to train")
    args = parser.parse_args()

    if args.mode not in MODE_CONFIG:
        print(f"Unknown mode: {args.mode}")
        return 1

    config = MODE_CONFIG[args.mode]

    # Train
    result = train_model(args.mode, config)

    # Inference test
    test_inference(args.mode, config)

    return 0


if __name__ == "__main__":
    sys.exit(main())