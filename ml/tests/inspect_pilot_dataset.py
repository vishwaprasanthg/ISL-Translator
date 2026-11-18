import numpy as np
from pathlib import Path

root = Path("ml/datasets/manual/dynamic")
classes = ["HELLO", "THANK_YOU", "SORRY", "PLEASE", "YES"]

print("=" * 78)
print(f"{'CLASS':<12} | {'COUNT':<5} | {'FRAMES (min/avg/max)':<20} | {'MOTION (min/avg/max)':<20} | {'DIM':<5}")
print("=" * 78)

total_sequences = 0
all_frame_counts = []
all_motions = []

for cls in classes:
    files = sorted((root / cls).glob("*.npz"))
    if not files:
        continue
    total_sequences += len(files)
    frame_counts = []
    motions = []
    dims = set()
    versions = set()
    for f in files:
        d = np.load(str(f), allow_pickle=True)
        frames = d["frames"]
        frame_counts.append(len(frames))
        motions.append(float(d["motion_score"]))
        dims.add(frames.shape[1])
        if "feature_version" in d:
            versions.add(str(d["feature_version"]))

    all_frame_counts.extend(frame_counts)
    all_motions.extend(motions)
    fc_str = f"{min(frame_counts)} / {np.mean(frame_counts):.1f} / {max(frame_counts)}"
    m_str = f"{min(motions):.3f} / {np.mean(motions):.3f} / {max(motions):.3f}"
    dim_str = str(list(dims)[0]) if dims else "N/A"
    print(f"{cls:<12} | {len(files):<5} | {fc_str:<20} | {m_str:<20} | {dim_str:<5}")

print("=" * 78)
print(f"TOTAL VALID SEQUENCES : {total_sequences}")
print(f"AVERAGE FRAME COUNT   : {np.mean(all_frame_counts):.1f} frames (range: {min(all_frame_counts)}–{max(all_frame_counts)})")
print(f"AVERAGE MOTION SCORE  : {np.mean(all_motions):.3f} (min: {min(all_motions):.3f}, max: {max(all_motions):.3f})")
print("=" * 78)

# Check detailed HELLO sessions
print("\n--- HELLO SESSION DETAILS ---")
hello_files = sorted((root / "HELLO").glob("*.npz"))
for f in hello_files:
    d = np.load(str(f), allow_pickle=True)
    frames = d["frames"]
    m = float(d["motion_score"])
    rec_at = float(d["recorded_at"]) if "recorded_at" in d else 0
    print(f"  {f.name}: {len(frames)} frames, motion={m:.4f}, recorded_at={rec_at:.1f}")
