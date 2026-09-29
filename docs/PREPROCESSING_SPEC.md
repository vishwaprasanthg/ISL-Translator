# Preprocessing Specification

This is the authoritative specification for transforming MediaPipe hand landmarks into the canonical 126-dimensional feature vector. The frontend JavaScript preprocessing and Python ML preprocessing MUST implement identical mathematics.

## Canonical Representation

Every frame is represented as exactly 126 features.

1.  **Left Hand:** 21 landmarks × 3 coordinates (x, y, z) = 63 features
2.  **Right Hand:** 21 landmarks × 3 coordinates (x, y, z) = 63 features

**Canonical Ordering:**
The final vector MUST ALWAYS follow: `LEFT → RIGHT`, regardless of MediaPipe detection order.
```text
[
  left_landmark_0_x,
  left_landmark_0_y,
  left_landmark_0_z,
  ...
  left_landmark_20_z,
  right_landmark_0_x,
  ...
  right_landmark_20_z
]
```

## Normalization Steps (Per Hand)
1. **Missing Data:** If a hand is not detected, insert 63 zeros for its section. No NaNs or infinities.
2. **Translation:** Translate all landmarks relative to the wrist (landmark 0). `normalized_x = landmark.x - wrist.x`
3. **Scale Normalization:** Calculate a reference distance `scale = EuclideanDistance(wrist, landmark_9)`. Divide all translated coordinates by this `scale`.
4. **Numerical Stability:** If `scale` is less than `1e-6`, use `scale = 1.0` to avoid division by zero.
5. **Z Coordinate:** The MediaPipe Z coordinate MUST be preserved and scaled identically to X and Y.
6. **Data Type:** Output must be a flat sequence of exactly 126 floating-point values (`Float32Array` in JS, `numpy.ndarray(shape=(126,), dtype=float32)` in Python).

## Landmark Index Documentation
- 0 = Wrist
- Thumb: 1-4 (CMC, MCP, IP, TIP)
- Index: 5-8 (MCP, PIP, DIP, TIP)
- Middle: 9-12 (MCP, PIP, DIP, TIP) (9 used for scale reference)
- Ring: 13-16 (MCP, PIP, DIP, TIP)
- Pinky: 17-20 (MCP, PIP, DIP, TIP)
