"""
ISL Manual Dataset Collector — Flask Backend

Serves the collection UI and saves validated 126-D landmark feature vectors
organized by mode (alphabet/numbers), class, and session.
"""

import os
import json
import time
import numpy as np
from pathlib import Path
from flask import Flask, render_template, request, jsonify

# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_ROOT = PROJECT_ROOT / "ml" / "datasets" / "manual"
LABEL_MAPS_DIR = PROJECT_ROOT / "ml" / "label_maps"

# ============================================================
# Constants
# ============================================================

TOTAL_FEATURES = 126
TARGET_PER_CLASS = 500

ALPHABET_CLASSES = [chr(i) for i in range(ord("A"), ord("Z") + 1)]  # A-Z
NUMBER_CLASSES = [str(i) for i in range(1, 10)]  # 1-9

# ============================================================
# Flask App
# ============================================================

app = Flask(__name__)


def get_dataset_dir(mode: str) -> Path:
    """Return the dataset directory for the given mode."""
    if mode == "alphabet":
        return DATASET_ROOT / "alphabet"
    elif mode == "numbers":
        return DATASET_ROOT / "numbers"
    else:
        raise ValueError(f"Unknown mode: {mode}")


def get_classes(mode: str) -> list:
    """Return the class list for the given mode."""
    if mode == "alphabet":
        return ALPHABET_CLASSES
    elif mode == "numbers":
        return NUMBER_CLASSES
    else:
        raise ValueError(f"Unknown mode: {mode}")


def get_session_dir(mode: str, class_name: str, session_id: str) -> Path:
    """Return the path to a session .npz file."""
    return get_dataset_dir(mode) / class_name / f"{session_id}.npz"


def load_existing_session(npz_path: Path) -> dict:
    """Load existing session data or return empty arrays."""
    if npz_path.exists():
        data = np.load(npz_path, allow_pickle=True)
        return {
            "features": data["features"].tolist(),
            "timestamps": data["timestamps"].tolist(),
            "sample_ids": data["sample_ids"].tolist(),
        }
    return {
        "features": [],
        "timestamps": [],
        "sample_ids": [],
    }


def save_session(npz_path: Path, session_data: dict):
    """Save session data to .npz file."""
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        npz_path,
        features=np.array(session_data["features"], dtype=np.float32),
        timestamps=np.array(session_data["timestamps"], dtype=np.float64),
        sample_ids=np.array(session_data["sample_ids"], dtype="U50"),
    )


def validate_features(features: list) -> tuple:
    """
    Validate a 126-D feature vector.
    Returns (is_valid, reason).
    """
    if not isinstance(features, list):
        return False, "features is not a list"

    if len(features) != TOTAL_FEATURES:
        return False, f"expected {TOTAL_FEATURES} features, got {len(features)}"

    arr = np.array(features, dtype=np.float32)

    if not np.all(np.isfinite(arr)):
        return False, "contains NaN or Infinity"

    if np.all(arr == 0):
        return False, "all-zero vector"

    return True, "ok"


def count_class_samples(mode: str, class_name: str) -> dict:
    """Count total valid samples across all sessions for a class."""
    class_dir = get_dataset_dir(mode) / class_name
    total = 0
    sessions = {}

    if class_dir.exists():
        for npz_file in sorted(class_dir.glob("session_*.npz")):
            try:
                data = np.load(npz_file, allow_pickle=True)
                count = len(data["features"])
                sessions[npz_file.stem] = count
                total += count
            except Exception:
                pass

    return {"total": total, "sessions": sessions}


def get_current_session_id(mode: str, class_name: str) -> str:
    """Determine the current (latest) session ID for a class."""
    class_dir = get_dataset_dir(mode) / class_name
    if not class_dir.exists():
        return "session_001"

    existing = sorted(class_dir.glob("session_*.npz"))
    if not existing:
        return "session_001"

    # Return the latest session
    latest = existing[-1].stem
    return latest


def next_session_id(mode: str, class_name: str) -> str:
    """Create the next session ID for a class."""
    class_dir = get_dataset_dir(mode) / class_name
    if not class_dir.exists():
        return "session_001"

    existing = sorted(class_dir.glob("session_*.npz"))
    if not existing:
        return "session_001"

    # Parse the highest session number and increment
    latest_num = 0
    for f in existing:
        try:
            num = int(f.stem.split("_")[1])
            latest_num = max(latest_num, num)
        except (IndexError, ValueError):
            pass

    return f"session_{latest_num + 1:03d}"


# ============================================================
# Routes
# ============================================================

@app.route("/")
def index():
    """Serve the collector UI."""
    return render_template("collector.html")


@app.route("/api/config", methods=["GET"])
def get_config():
    """Return collector configuration."""
    return jsonify({
        "alphabet_classes": ALPHABET_CLASSES,
        "number_classes": NUMBER_CLASSES,
        "target_per_class": TARGET_PER_CLASS,
        "total_features": TOTAL_FEATURES,
    })


@app.route("/api/progress", methods=["GET"])
def get_progress():
    """Return collection progress for all classes in both modes."""
    progress = {}

    for mode in ["alphabet", "numbers"]:
        classes = get_classes(mode)
        mode_progress = {}

        for cls in classes:
            info = count_class_samples(mode, cls)
            mode_progress[cls] = {
                "total": info["total"],
                "sessions": info["sessions"],
                "complete": info["total"] >= TARGET_PER_CLASS,
            }

        progress[mode] = mode_progress

    return jsonify(progress)


@app.route("/api/progress/<mode>/<class_name>", methods=["GET"])
def get_class_progress(mode, class_name):
    """Return progress for a specific class."""
    classes = get_classes(mode)
    if class_name not in classes:
        return jsonify({"error": f"Invalid class: {class_name}"}), 400

    info = count_class_samples(mode, class_name)
    session_id = get_current_session_id(mode, class_name)

    return jsonify({
        "class": class_name,
        "mode": mode,
        "total": info["total"],
        "sessions": info["sessions"],
        "current_session": session_id,
        "complete": info["total"] >= TARGET_PER_CLASS,
        "target": TARGET_PER_CLASS,
    })


@app.route("/api/new_session", methods=["POST"])
def new_session():
    """Start a new collection session for a class."""
    data = request.json
    mode = data.get("mode")
    class_name = data.get("class_name")

    if mode not in ["alphabet", "numbers"]:
        return jsonify({"error": f"Invalid mode: {mode}"}), 400

    classes = get_classes(mode)
    if class_name not in classes:
        return jsonify({"error": f"Invalid class: {class_name}"}), 400

    session_id = next_session_id(mode, class_name)

    # Create the empty session directory
    session_path = get_session_dir(mode, class_name, session_id)
    session_path.parent.mkdir(parents=True, exist_ok=True)

    return jsonify({
        "session_id": session_id,
        "class": class_name,
        "mode": mode,
    })


@app.route("/api/save_sample", methods=["POST"])
def save_sample():
    """
    Save a single validated 126-D feature vector.
    
    Expects JSON:
    {
        "mode": "alphabet" | "numbers",
        "class_name": "A" | "1" | ...,
        "session_id": "session_001",
        "features": [126 float values]
    }
    """
    data = request.json

    mode = data.get("mode")
    class_name = data.get("class_name")
    session_id = data.get("session_id")
    features = data.get("features")

    # Validate mode
    if mode not in ["alphabet", "numbers"]:
        return jsonify({"error": f"Invalid mode: {mode}", "saved": False}), 400

    # Validate class
    classes = get_classes(mode)
    if class_name not in classes:
        return jsonify({"error": f"Invalid class: {class_name}", "saved": False}), 400

    # Validate session
    if not session_id or not session_id.startswith("session_"):
        return jsonify({"error": f"Invalid session: {session_id}", "saved": False}), 400

    # Validate features
    is_valid, reason = validate_features(features)
    if not is_valid:
        return jsonify({
            "saved": False,
            "valid": False,
            "reason": reason,
        })

    # Load existing session data
    npz_path = get_session_dir(mode, class_name, session_id)
    session_data = load_existing_session(npz_path)

    # Create sample metadata
    timestamp = time.time()
    sample_id = f"{class_name}_{session_id}_{timestamp:.4f}"

    # Append new sample
    session_data["features"].append(features)
    session_data["timestamps"].append(timestamp)
    session_data["sample_ids"].append(sample_id)

    # Save
    try:
        save_session(npz_path, session_data)
    except Exception as e:
        return jsonify({
            "saved": False,
            "valid": True,
            "reason": f"Save failed: {str(e)}",
        }), 500

    # Count total for this class
    total_info = count_class_samples(mode, class_name)

    return jsonify({
        "saved": True,
        "valid": True,
        "sample_id": sample_id,
        "session_samples": len(session_data["features"]),
        "total_samples": total_info["total"],
        "complete": total_info["total"] >= TARGET_PER_CLASS,
    })


@app.route("/api/save_batch", methods=["POST"])
def save_batch():
    """
    Save a batch of validated 126-D feature vectors at once.
    More efficient than individual saves during rapid collection.
    
    Expects JSON:
    {
        "mode": "alphabet" | "numbers",
        "class_name": "A" | "1" | ...,
        "session_id": "session_001",
        "samples": [
            [126 float values],
            [126 float values],
            ...
        ]
    }
    """
    data = request.json

    mode = data.get("mode")
    class_name = data.get("class_name")
    session_id = data.get("session_id")
    samples = data.get("samples", [])

    # Validate mode
    if mode not in ["alphabet", "numbers"]:
        return jsonify({"error": f"Invalid mode: {mode}", "saved": 0}), 400

    # Validate class
    classes = get_classes(mode)
    if class_name not in classes:
        return jsonify({"error": f"Invalid class: {class_name}", "saved": 0}), 400

    # Validate session
    if not session_id or not session_id.startswith("session_"):
        return jsonify({"error": f"Invalid session: {session_id}", "saved": 0}), 400

    # Load existing session data
    npz_path = get_session_dir(mode, class_name, session_id)
    session_data = load_existing_session(npz_path)

    saved_count = 0
    invalid_count = 0

    for features in samples:
        is_valid, reason = validate_features(features)
        if not is_valid:
            invalid_count += 1
            continue

        timestamp = time.time()
        sample_id = f"{class_name}_{session_id}_{timestamp:.4f}"

        session_data["features"].append(features)
        session_data["timestamps"].append(timestamp)
        session_data["sample_ids"].append(sample_id)
        saved_count += 1

    # Save if anything was added
    if saved_count > 0:
        try:
            save_session(npz_path, session_data)
        except Exception as e:
            return jsonify({
                "saved": 0,
                "invalid": invalid_count,
                "reason": f"Save failed: {str(e)}",
            }), 500

    total_info = count_class_samples(mode, class_name)

    return jsonify({
        "saved": saved_count,
        "invalid": invalid_count,
        "session_samples": len(session_data["features"]),
        "total_samples": total_info["total"],
        "complete": total_info["total"] >= TARGET_PER_CLASS,
    })


@app.route("/api/reset_class", methods=["POST"])
def reset_class():
    """
    Reset (delete) all sessions for a specific class.
    Requires confirmation field.
    """
    data = request.json
    mode = data.get("mode")
    class_name = data.get("class_name")
    confirm = data.get("confirm", False)

    if not confirm:
        return jsonify({"error": "Confirmation required"}), 400

    if mode not in ["alphabet", "numbers"]:
        return jsonify({"error": f"Invalid mode: {mode}"}), 400

    classes = get_classes(mode)
    if class_name not in classes:
        return jsonify({"error": f"Invalid class: {class_name}"}), 400

    class_dir = get_dataset_dir(mode) / class_name
    deleted = 0

    if class_dir.exists():
        for npz_file in class_dir.glob("session_*.npz"):
            npz_file.unlink()
            deleted += 1

    return jsonify({
        "reset": True,
        "class": class_name,
        "mode": mode,
        "sessions_deleted": deleted,
    })


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    # Ensure dataset directories exist
    for mode in ["alphabet", "numbers"]:
        for cls in get_classes(mode):
            (get_dataset_dir(mode) / cls).mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  ISL MANUAL DATASET COLLECTOR")
    print("=" * 60)
    print(f"  Dataset root : {DATASET_ROOT}")
    print(f"  Alphabet     : {len(ALPHABET_CLASSES)} classes (A-Z)")
    print(f"  Numbers      : {len(NUMBER_CLASSES)} classes (1-9)")
    print(f"  Target       : {TARGET_PER_CLASS} valid samples per class")
    print(f"  Total target : {len(ALPHABET_CLASSES) * TARGET_PER_CLASS + len(NUMBER_CLASSES) * TARGET_PER_CLASS}")
    print("=" * 60)
    print()
    print("  Open http://localhost:5001 in your browser")
    print()

    app.run(host="0.0.0.0", port=5001, debug=False)
