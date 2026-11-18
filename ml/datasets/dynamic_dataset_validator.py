# Dynamic ISL Dataset Validator
# Validates ml/datasets/manual/dynamic/ for training readiness.
# Usage: python dynamic_dataset_validator.py [--quick]

import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime, timezone

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DYNAMIC_CLASSES = [
    "HELLO", "THANK_YOU", "SORRY", "PLEASE", "YES", "NO", "HELP", "WATER",
    "FOOD_EAT", "DRINK", "SLEEP", "COME", "GO", "STOP", "WAIT",
    "AGAIN_REPEAT", "UNDERSTAND", "I_ME", "YOU", "MOTHER", "FATHER",
    "FRIEND", "WHAT", "WHERE", "HOW", "GOOD", "BAD", "MORE",
    "LESS_LITTLE", "FINISH_DONE",
]

MIN_FRAMES = 10
MAX_FRAMES = 120
VECTOR_DIM = 140
EXPECTED_FEATURE_VERSION = "140d"
MIN_SEQUENCES_ERROR = 20
MIN_SEQUENCES_WARN = 50
DUPLICATE_COSINE_THRESH = 0.995
BALANCE_RATIO_THRESH = 3.0

# Landmark index ranges (0-based in the 126-D vector)
# Right wrist  → indices 63, 64, 65  (hand landmark 0, x/y/z)
# Left wrist   → indices  0,  1,  2  (hand landmark 0, x/y/z)
RIGHT_WRIST_IDX = [63, 64, 65]
LEFT_WRIST_IDX  = [0,  1,  2]

DATASET_ROOT = Path(__file__).parent / "manual" / "dynamic"
REPORT_PATH  = DATASET_ROOT / "validation_report.json"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Return cosine similarity between two 1-D vectors."""
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _seq_fingerprint(frames: np.ndarray, target_frames: int = 30) -> np.ndarray:
    """Resample sequence to a fixed temporal length and normalize into a 1-D fingerprint for dup detection."""
    n_frames = len(frames)
    if n_frames == 0:
        return np.zeros(target_frames * (frames.shape[1] if frames.ndim > 1 else 140), dtype=np.float32)
    if n_frames == 1:
        resampled = np.tile(frames, (target_frames, 1))
    else:
        orig_indices = np.linspace(0, n_frames - 1, num=n_frames)
        target_indices = np.linspace(0, n_frames - 1, num=target_frames)
        resampled = np.empty((target_frames, frames.shape[1]), dtype=np.float32)
        for col in range(frames.shape[1]):
            resampled[:, col] = np.interp(target_indices, orig_indices, frames[:, col])
    flat = resampled.flatten().astype(np.float32)
    n = np.linalg.norm(flat)
    return flat / n if n > 0 else flat


def _motion_score(frames: np.ndarray) -> float:
    """
    Average frame-to-frame Euclidean displacement of active non-wrist landmarks
    across consecutive frames.

    Landmarks 0 and 21 (wrists) are normalized to (0,0,0) by design, so displacement
    is measured on articulated landmarks (1..20 for left hand, 22..41 for right hand)
    for whichever hands are detected across consecutive frames.
    """
    if len(frames) < 2:
        return 0.0

    handshape = frames[:, 0:126]
    frames_3d = handshape.reshape(len(handshape), 42, 3)
    diff = np.diff(frames_3d, axis=0)  # (N-1, 42, 3)
    disp = np.linalg.norm(diff, axis=2)  # (N-1, 42)

    # Exclude wrists: landmark 0 (left wrist) and landmark 21 (right wrist)
    non_wrist_mask = np.ones(42, dtype=bool)
    non_wrist_mask[0] = False
    non_wrist_mask[21] = False

    # A landmark is active if it was detected (non-zero) in BOTH consecutive frames
    present = np.any(frames_3d != 0, axis=2)  # (N, 42)
    active_transitions = present[:-1] & present[1:] & non_wrist_mask[None, :]

    hand_motion = float(disp[active_transitions].mean()) if np.any(active_transitions) else 0.0

    # If 140-D features present, also inspect global wrist trajectory
    wrist_motion = 0.0
    if frames.shape[1] >= 140:
        lw_diff = np.diff(frames[:, 126:129], axis=0)
        rw_diff = np.diff(frames[:, 130:133], axis=0)
        lp = frames[:-1, 138] == 1.0
        rp = frames[:-1, 139] == 1.0

        w_disps = []
        if np.any(lp):
            w_disps.append(float(np.linalg.norm(lw_diff[lp], axis=1).mean()))
        if np.any(rp):
            w_disps.append(float(np.linalg.norm(rw_diff[rp], axis=1).mean()))
        if w_disps:
            wrist_motion = float(np.mean(w_disps))

    return max(hand_motion, wrist_motion)


# ---------------------------------------------------------------------------
# Per-sequence validation
# ---------------------------------------------------------------------------

def validate_sequence(npz_path: Path, expected_label: str):
    """
    Load a single .npz file and return (errors, warnings, motion_score).

    errors   – list of str, critical problems
    warnings – list of str, non-critical notices
    motion   – float or None
    """
    errors, warnings = [], []
    motion = None

    try:
        data = np.load(str(npz_path), allow_pickle=True)
    except Exception as exc:
        errors.append(f"Cannot load file: {exc}  →  Re-record or delete this file.")
        return errors, warnings, motion

    # --- frames ---
    if "frames" not in data:
        errors.append(
            "Missing 'frames' key.  →  Re-record; every .npz must contain a 'frames' array."
        )
        return errors, warnings, motion

    frames = data["frames"]

    if frames.ndim != 2 or frames.shape[1] != VECTOR_DIM:
        errors.append(
            f"frames shape is {frames.shape}, expected (N, {VECTOR_DIM}).  "
            f"→  Check your recording pipeline – ensure you export all 42 landmarks × 3 coords."
        )
        return errors, warnings, motion

    n_frames = frames.shape[0]
    if n_frames < MIN_FRAMES:
        errors.append(
            f"Only {n_frames} frames (minimum {MIN_FRAMES}).  "
            f"→  Re-record: perform the sign more slowly or extend the recording window."
        )
    elif n_frames > MAX_FRAMES:
        warnings.append(
            f"{n_frames} frames exceeds maximum {MAX_FRAMES}.  "
            f"→  Consider trimming padding from start/end of the recording."
        )

    if not np.all(np.isfinite(frames)):
        errors.append(
            "frames contain NaN or Inf values.  "
            "→  Check landmark extraction; drop frames where the hand is not detected instead of filling with NaN."
        )

    if np.all(frames == 0):
        errors.append(
            "frames are all-zero.  "
            "→  Recording captured no landmark data – check camera and MediaPipe hand detection."
        )

    # --- feature version & vector dim ---
    version = str(data["feature_version"]) if "feature_version" in data else "unknown"
    if version != EXPECTED_FEATURE_VERSION:
        warnings.append(
            f"feature_version is '{version}', expected '{EXPECTED_FEATURE_VERSION}'."
        )

    # --- spatial features consistency check (for 140-D) ---
    if frames.shape[1] == 140:
        lp = frames[:, 138]
        rp = frames[:, 139]
        if not np.all(np.isin(lp, [0.0, 1.0])) or not np.all(np.isin(rp, [0.0, 1.0])):
            errors.append("presence mask (indices 138, 139) contains values other than 0.0 and 1.0.")

    # --- label ---
    if "label" not in data:
        warnings.append(
            "Missing 'label' key.  →  Add label='{expected_label}' when saving the .npz."
        )
    else:
        label = str(data["label"])
        if label != expected_label:
            errors.append(
                f"label '{label}' does not match directory '{expected_label}'.  "
                f"→  Re-save with label='{expected_label}', or move the file to the correct directory."
            )

    # --- motion score (only when no critical frame errors) ---
    if not errors and frames.shape[0] >= 2:
        motion = _motion_score(frames)

    return errors, warnings, motion


# ---------------------------------------------------------------------------
# Main validator
# ---------------------------------------------------------------------------

def run_validation(quick: bool = False):
    print("=" * 70)
    print("  Dynamic ISL Dataset Validator")
    print(f"  Dataset root : {DATASET_ROOT}")
    print(f"  Mode         : {'quick (coverage + count only)' if quick else 'full'}")
    print(f"  Run at       : {datetime.now(timezone.utc).isoformat()} UTC")
    print("=" * 70)

    report = {
        "run_at": datetime.now(timezone.utc).isoformat(),
        "mode": "quick" if quick else "full",
        "dataset_root": str(DATASET_ROOT),
        "classes": {},
        "summary": {
            "critical_errors": 0,
            "warnings": 0,
            "missing_classes": [],
            "low_count_classes": [],
            "ready_for_training": False,
        },
    }

    if not DATASET_ROOT.exists():
        msg = (
            f"Dataset directory not found: {DATASET_ROOT}\n"
            f"→  Create it and add class subdirectories before running the validator."
        )
        print(f"\n[CRITICAL] {msg}\n")
        report["summary"]["critical_errors"] += 1
        _save_report(report)
        return 1

    # -----------------------------------------------------------------------
    # 1. Class coverage
    # -----------------------------------------------------------------------
    print("\n── 1. Class coverage ──────────────────────────────────────────────")
    missing_classes = []
    for cls in DYNAMIC_CLASSES:
        cls_dir = DATASET_ROOT / cls
        if not cls_dir.is_dir():
            missing_classes.append(cls)
            print(f"  [MISSING] {cls}  →  mkdir {cls_dir} and add ≥50 session .npz files.")

    if missing_classes:
        report["summary"]["missing_classes"] = missing_classes
        report["summary"]["critical_errors"] += len(missing_classes)
        print(f"\n  {len(missing_classes)} class director(ies) missing (see above).")
    else:
        print("  All 30 classes present. ✓")

    # -----------------------------------------------------------------------
    # 2–7. Per-class checks
    # -----------------------------------------------------------------------
    class_seq_counts = {}   # cls → count of valid sequences
    all_fingerprints = {}   # cls → list of (path_str, fingerprint)

    for cls in DYNAMIC_CLASSES:
        cls_dir = DATASET_ROOT / cls
        if not cls_dir.is_dir():
            class_seq_counts[cls] = 0
            report["classes"][cls] = {"error": "directory missing", "sessions": 0}
            continue

        npz_files = sorted(cls_dir.glob("*.npz"))
        total = len(npz_files)
        cls_errors, cls_warnings = [], []
        seq_motions = []
        fingerprints = []
        valid_count = 0

        for npz_path in npz_files:
            errs, warns, motion = validate_sequence(npz_path, cls)
            if errs:
                for e in errs:
                    cls_errors.append(f"{npz_path.name}: {e}")
            if warns:
                for w in warns:
                    cls_warnings.append(f"{npz_path.name}: {w}")
            if not errs:
                valid_count += 1
                if motion is not None:
                    seq_motions.append(motion)
                if not quick:
                    try:
                        data = np.load(str(npz_path), allow_pickle=True)
                        fp = _seq_fingerprint(data["frames"])
                        fingerprints.append((str(npz_path), fp))
                    except Exception:
                        pass

        all_fingerprints[cls] = fingerprints
        class_seq_counts[cls] = valid_count

        avg_motion = float(np.mean(seq_motions)) if seq_motions else None
        cls_report = {
            "total_files": total,
            "valid_sequences": valid_count,
            "errors": cls_errors,
            "warnings": cls_warnings,
            "avg_motion_score": avg_motion,
        }
        report["classes"][cls] = cls_report
        report["summary"]["critical_errors"] += len(cls_errors)
        report["summary"]["warnings"] += len(cls_warnings)

    # -----------------------------------------------------------------------
    # 3. Count thresholds
    # -----------------------------------------------------------------------
    print("\n── 2. Sequence counts ─────────────────────────────────────────────")
    for cls in DYNAMIC_CLASSES:
        cnt = class_seq_counts[cls]
        if cnt < MIN_SEQUENCES_ERROR:
            tag = "[ERROR]  "
            report["summary"]["low_count_classes"].append(cls)
            report["summary"]["critical_errors"] += 1
            fix = f"Record at least {MIN_SEQUENCES_ERROR - cnt} more sessions for {cls}."
        elif cnt < MIN_SEQUENCES_WARN:
            tag = "[WARN]   "
            report["summary"]["low_count_classes"].append(cls)
            report["summary"]["warnings"] += 1
            fix = f"Aim for ≥50 sequences; record {MIN_SEQUENCES_WARN - cnt} more for {cls}."
        else:
            tag = "[OK]     "
            fix = ""
        line = f"  {tag}{cls:<20} {cnt:>4} sequences"
        if fix:
            line += f"  →  {fix}"
        print(line)

    # -----------------------------------------------------------------------
    # 4. Duplicate detection (full mode only)
    # -----------------------------------------------------------------------
    dup_report = {}
    if not quick:
        print("\n── 3. Near-duplicate detection ────────────────────────────────────")
        total_dups = 0
        for cls in DYNAMIC_CLASSES:
            fps = all_fingerprints.get(cls, [])
            dups = []
            for i in range(len(fps)):
                for j in range(i + 1, len(fps)):
                    sim = _cosine_sim(fps[i][1], fps[j][1])
                    if sim > DUPLICATE_COSINE_THRESH:
                        dups.append({
                            "file_a": Path(fps[i][0]).name,
                            "file_b": Path(fps[j][0]).name,
                            "cosine_similarity": round(sim, 6),
                        })
            if dups:
                total_dups += len(dups)
                print(f"  [WARN] {cls}: {len(dups)} near-duplicate pair(s)")
                for d in dups:
                    print(
                        f"         {d['file_a']} ↔ {d['file_b']}  "
                        f"(sim={d['cosine_similarity']:.4f})"
                        f"  →  Delete one; possible copy-paste error."
                    )
                report["summary"]["warnings"] += len(dups)
            dup_report[cls] = dups
        if total_dups == 0:
            print("  No near-duplicates found. ✓")
        report["duplicate_pairs"] = dup_report

    # -----------------------------------------------------------------------
    # 5. Class balance
    # -----------------------------------------------------------------------
    print("\n── 4. Class balance ───────────────────────────────────────────────")
    valid_counts = [v for v in class_seq_counts.values() if v > 0]
    if valid_counts:
        min_cnt = min(valid_counts)
        max_cnt = max(valid_counts)
        ratio = max_cnt / min_cnt if min_cnt > 0 else float("inf")
        print(f"  Min sequences : {min_cnt}   Max : {max_cnt}   Ratio : {ratio:.2f}x")
        if ratio > BALANCE_RATIO_THRESH:
            print(
                f"  [WARN] Imbalance ratio {ratio:.2f}x exceeds {BALANCE_RATIO_THRESH}x threshold.\n"
                f"         →  Collect more data for under-represented classes, or apply class-weighted loss."
            )
            report["summary"]["warnings"] += 1
        else:
            print("  Balance is acceptable. ✓")
        report["summary"]["balance_ratio"] = round(ratio, 3)
    else:
        print("  No valid sequences found across any class.")

    # -----------------------------------------------------------------------
    # 6. Session distribution
    # -----------------------------------------------------------------------
    print("\n── 5. Session distribution ────────────────────────────────────────")
    for cls in DYNAMIC_CLASSES:
        cnt = class_seq_counts[cls]
        bar = "█" * min(cnt // 2, 40)
        print(f"  {cls:<20} {cnt:>4}  {bar}")

    # -----------------------------------------------------------------------
    # 7. Motion score report (full mode only)
    # -----------------------------------------------------------------------
    if not quick:
        print("\n── 6. Motion score report (avg landmark displacement per frame) ────")
        for cls in DYNAMIC_CLASSES:
            score = report["classes"].get(cls, {}).get("avg_motion_score")
            if score is None:
                print(f"  {cls:<20}  n/a")
            else:
                bar = "▸" * min(int(score * 200), 40)
                flag = "  [LOW – check if sign is truly dynamic]" if score < 0.002 else ""
                print(f"  {cls:<20}  {score:.5f}  {bar}{flag}")

    # -----------------------------------------------------------------------
    # Final summary + exit code
    # -----------------------------------------------------------------------
    all_classes_present = len(missing_classes) == 0
    all_counts_ok = all(class_seq_counts[c] >= MIN_SEQUENCES_WARN for c in DYNAMIC_CLASSES)
    no_critical_errors = report["summary"]["critical_errors"] == 0

    ready = all_classes_present and all_counts_ok and no_critical_errors
    report["summary"]["ready_for_training"] = ready

    print("\n── Summary ────────────────────────────────────────────────────────")
    print(f"  Critical errors : {report['summary']['critical_errors']}")
    print(f"  Warnings        : {report['summary']['warnings']}")
    print(f"  Missing classes : {len(missing_classes)}")
    print(f"  Low-count classes: {len(report['summary']['low_count_classes'])}")
    print(f"  Ready for training: {'YES ✓' if ready else 'NO ✗'}")

    if not ready:
        print("\n  To make this dataset ready for training:")
        if missing_classes:
            print(f"  • Add directories + ≥50 sessions for: {', '.join(missing_classes)}")
        low = [c for c in DYNAMIC_CLASSES if class_seq_counts[c] < MIN_SEQUENCES_WARN]
        if low:
            for c in low:
                needed = MIN_SEQUENCES_WARN - class_seq_counts[c]
                print(f"  • Record {needed} more sessions for {c} (currently {class_seq_counts[c]})")
        if report["summary"]["critical_errors"] > 0:
            print("  • Fix all [ERROR] items listed above.")

    _save_report(report)
    print(f"\n  Full report saved → {REPORT_PATH}")
    print("=" * 70)

    return 0 if ready else 1


def _save_report(report: dict):
    try:
        REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(REPORT_PATH, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, default=str)
    except Exception as exc:
        print(f"  [WARN] Could not save JSON report: {exc}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Validate the dynamic ISL gesture sequence dataset."
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run only coverage and count checks (skip duplicate detection and motion analysis).",
    )
    args = parser.parse_args()
    sys.exit(run_validation(quick=args.quick))
