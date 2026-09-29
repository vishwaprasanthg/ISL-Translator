"""
ISL Dataset Inspector — Comprehensive validation and statistics.
Produces a detailed report of the manually collected dataset.
"""

import sys
import numpy as np
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_ROOT = PROJECT_ROOT / "ml" / "datasets" / "manual"

TOTAL_FEATURES = 126
TARGET_PER_CLASS = 500

ALPHABET_CLASSES = [chr(i) for i in range(ord("A"), ord("Z") + 1)]
NUMBER_CLASSES = [str(i) for i in range(1, 10)]


def inspect_mode(mode_name, base_dir, classes):
    """Inspect a dataset mode and return detailed stats."""
    print(f"\n{'=' * 60}")
    print(f"  {mode_name.upper()} DATASET INSPECTION")
    print(f"{'=' * 60}")

    total_valid = 0
    total_invalid = 0
    total_sessions = 0
    class_stats = {}
    all_issues = []

    for cls in classes:
        cls_dir = base_dir / cls
        stats = {
            "sessions": {},
            "total_valid": 0,
            "total_raw": 0,
            "nan_count": 0,
            "inf_count": 0,
            "zero_count": 0,
            "wrong_dim": 0,
            "issues": [],
        }

        if not cls_dir.exists():
            stats["issues"].append("Directory missing")
            all_issues.append(f"{cls}: directory missing")
            class_stats[cls] = stats
            continue

        session_files = sorted(cls_dir.glob("session_*.npz"))

        if not session_files:
            stats["issues"].append("No session files")
            all_issues.append(f"{cls}: no session files")
            class_stats[cls] = stats
            continue

        for npz_path in session_files:
            sess = npz_path.stem
            try:
                data = np.load(npz_path, allow_pickle=True)
                features = data["features"]

                n = features.shape[0]
                stats["total_raw"] += n
                total_sessions += 1

                # Dimension check
                if features.ndim != 2 or features.shape[1] != TOTAL_FEATURES:
                    stats["wrong_dim"] += n
                    stats["sessions"][sess] = {"raw": n, "valid": 0, "issue": "wrong dim"}
                    all_issues.append(f"{cls}/{sess}: wrong dimension {features.shape}")
                    continue

                # Per-sample validation
                nan_mask = np.any(np.isnan(features), axis=1)
                inf_mask = np.any(np.isinf(features), axis=1)
                zero_mask = np.all(features == 0, axis=1)

                n_nan = int(nan_mask.sum())
                n_inf = int(inf_mask.sum())
                n_zero = int(zero_mask.sum())

                valid_mask = ~nan_mask & ~inf_mask & ~zero_mask
                n_valid = int(valid_mask.sum())

                stats["nan_count"] += n_nan
                stats["inf_count"] += n_inf
                stats["zero_count"] += n_zero
                stats["total_valid"] += n_valid

                stats["sessions"][sess] = {
                    "raw": n,
                    "valid": n_valid,
                    "nan": n_nan,
                    "inf": n_inf,
                    "zero": n_zero,
                }

                if n_nan > 0:
                    all_issues.append(f"{cls}/{sess}: {n_nan} NaN samples")
                if n_inf > 0:
                    all_issues.append(f"{cls}/{sess}: {n_inf} Inf samples")
                if n_zero > 0:
                    all_issues.append(f"{cls}/{sess}: {n_zero} zero vectors")

            except Exception as e:
                stats["issues"].append(f"{sess}: load error: {e}")
                all_issues.append(f"{cls}/{sess}: load error: {e}")

        total_valid += stats["total_valid"]
        total_invalid += (stats["total_raw"] - stats["total_valid"])
        class_stats[cls] = stats

    # Print table
    print(f"\n  {'Class':<6} {'Valid':>6} {'Raw':>6} {'Sessions':>9} {'NaN':>5} {'Inf':>5} {'Zero':>5} {'Status':>10}")
    print(f"  {'-' * 58}")

    under_target = []
    for cls in classes:
        s = class_stats[cls]
        n_sess = len(s["sessions"])
        status = "OK" if s["total_valid"] >= TARGET_PER_CLASS else f"NEED {TARGET_PER_CLASS - s['total_valid']}"
        if s["total_valid"] < TARGET_PER_CLASS:
            under_target.append((cls, s["total_valid"]))
        print(f"  {cls:<6} {s['total_valid']:>6} {s['total_raw']:>6} {n_sess:>9} {s['nan_count']:>5} {s['inf_count']:>5} {s['zero_count']:>5} {status:>10}")

    print(f"  {'-' * 58}")
    print(f"  {'TOTAL':<6} {total_valid:>6} {total_valid + total_invalid:>6} {total_sessions:>9}")

    # Session distribution
    print(f"\n  Session distribution:")
    for cls in classes:
        s = class_stats[cls]
        sess_str = ", ".join(
            f"{k}({v['valid']})" for k, v in sorted(s["sessions"].items())
        )
        print(f"    {cls}: {sess_str}")

    return {
        "total_valid": total_valid,
        "total_invalid": total_invalid,
        "total_sessions": total_sessions,
        "class_stats": class_stats,
        "issues": all_issues,
        "under_target": under_target,
    }


def main():
    print("=" * 60)
    print("  ISL DATASET VALIDATION REPORT")
    print("=" * 60)
    print(f"  Dataset: {DATASET_ROOT}")
    print(f"  Target: {TARGET_PER_CLASS} valid samples per class")

    alpha_report = inspect_mode(
        "alphabet",
        DATASET_ROOT / "alphabet",
        ALPHABET_CLASSES,
    )

    num_report = inspect_mode(
        "numbers",
        DATASET_ROOT / "numbers",
        NUMBER_CLASSES,
    )

    # Final summary
    print(f"\n{'=' * 60}")
    print(f"  FINAL SUMMARY")
    print(f"{'=' * 60}")
    print(f"  Alphabet valid samples : {alpha_report['total_valid']}")
    print(f"  Number valid samples   : {num_report['total_valid']}")
    print(f"  Total valid            : {alpha_report['total_valid'] + num_report['total_valid']}")
    print(f"  Total invalid          : {alpha_report['total_invalid'] + num_report['total_invalid']}")
    print(f"  Total sessions         : {alpha_report['total_sessions'] + num_report['total_sessions']}")

    all_issues = alpha_report["issues"] + num_report["issues"]
    if all_issues:
        print(f"\n  Issues ({len(all_issues)}):")
        for iss in all_issues[:30]:
            print(f"    [!] {iss}")

    ready = True
    if alpha_report["under_target"]:
        ready = False
        print(f"\n  [X] Alphabet classes under target:")
        for cls, count in alpha_report["under_target"]:
            print(f"    {cls}: {count}/{TARGET_PER_CLASS}")

    if num_report["under_target"]:
        ready = False
        print(f"\n  [X] Number classes under target:")
        for cls, count in num_report["under_target"]:
            print(f"    {cls}: {count}/{TARGET_PER_CLASS}")

    print()
    if ready:
        print("  [OK] READY FOR TRAINING: YES")
    else:
        print("  [X] READY FOR TRAINING: NO")
    print("=" * 60)

    return 0 if ready else 1


if __name__ == "__main__":
    sys.exit(main())
