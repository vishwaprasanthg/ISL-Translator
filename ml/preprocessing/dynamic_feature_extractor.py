import numpy as np
from typing import Optional, List, Dict, Any, Union

from .landmark_normalizer import (
    NUM_LANDMARKS_PER_HAND,
    COORDINATES_PER_LANDMARK,
    FEATURES_PER_HAND,
    TOTAL_FEATURES as STATIC_FEATURE_DIM,
    EPSILON,
    normalize_hand_landmarks,
    create_feature_vector,
)

# ============================================================
# Constants & Slices for 140-D Dynamic Feature Representation
# ============================================================

GLOBAL_FEATURE_DIM = 14
DYNAMIC_FEATURE_DIM = STATIC_FEATURE_DIM + GLOBAL_FEATURE_DIM  # 126 + 14 = 140

# Slices for modular feature access
HANDSHAPE_SLICE = slice(0, 126)               # Existing 126-D normalised handshape
LEFT_HANDSHAPE_SLICE = slice(0, 63)           # Left hand normalised landmarks (0..20)
RIGHT_HANDSHAPE_SLICE = slice(63, 126)        # Right hand normalised landmarks (0..20)

GLOBAL_SLICE = slice(126, 140)                # All 14 global & relational features
LEFT_WRIST_SLICE = slice(126, 129)            # (x, y, z) left wrist camera position
LEFT_SCALE_IDX = 129                          # Left hand apparent scale (wrist to middle MCP)
RIGHT_WRIST_SLICE = slice(130, 133)           # (x, y, z) right wrist camera position
RIGHT_SCALE_IDX = 133                         # Right hand apparent scale (wrist to middle MCP)
INTER_WRIST_VEC_SLICE = slice(134, 137)       # (dx, dy, dz) = right_wrist - left_wrist
INTER_WRIST_DIST_IDX = 137                    # Euclidean distance between wrists
PRESENCE_MASK_SLICE = slice(138, 140)         # [left_present, right_present] in {0.0, 1.0}
LEFT_PRESENT_IDX = 138
RIGHT_PRESENT_IDX = 139


def _get_coords(landmarks: Union[List[Dict[str, float]], List[Any], np.ndarray]) -> Optional[np.ndarray]:
    """Helper to convert landmark objects or dicts into a (21, 3) float32 numpy array."""
    if not landmarks or len(landmarks) != NUM_LANDMARKS_PER_HAND:
        return None
    
    if isinstance(landmarks, np.ndarray) and landmarks.shape == (NUM_LANDMARKS_PER_HAND, 3):
        return landmarks.astype(np.float32)

    coords = np.array(
        [[lm['x'], lm['y'], lm['z']] if isinstance(lm, dict) else [lm.x, lm.y, lm.z] for lm in landmarks],
        dtype=np.float32
    )
    return coords


def extract_dynamic_features(left_hand_landmarks=None, right_hand_landmarks=None) -> np.ndarray:
    """
    Extracts the canonical 140-D dynamic feature vector:
      [0..125]   : Existing 126-D wrist-relative & scale-normalised handshape features
      [126..128] : Left wrist camera position (x, y, z) in [0, 1]^2 x R (or 0 if absent)
      [129]      : Left hand apparent scale in image space (or 0 if absent)
      [130..132] : Right wrist camera position (x, y, z) in [0, 1]^2 x R (or 0 if absent)
      [133]      : Right hand apparent scale in image space (or 0 if absent)
      [134..136] : Inter-wrist displacement vector (right - left) (or 0 if either absent)
      [137]      : Inter-wrist Euclidean distance (or 0 if either absent)
      [138..139] : Hand presence mask [p_L, p_R] in {0.0, 1.0}

    Returns:
        np.ndarray: Exactly 140 features with dtype float32.
    """
    vector = np.zeros(DYNAMIC_FEATURE_DIM, dtype=np.float32)

    # 1. Base 126-D normalised handshape features (completely identical to static pipeline)
    static_126 = create_feature_vector(left_hand_landmarks, right_hand_landmarks)
    vector[HANDSHAPE_SLICE] = static_126

    # 2. Extract raw coordinates to compute global spatial features
    left_coords = _get_coords(left_hand_landmarks)
    right_coords = _get_coords(right_hand_landmarks)

    left_present = left_coords is not None
    right_present = right_coords is not None

    left_wrist = np.zeros(3, dtype=np.float32)
    left_scale = 0.0
    right_wrist = np.zeros(3, dtype=np.float32)
    right_scale = 0.0

    if left_present:
        left_wrist = left_coords[0]
        left_middle_mcp = left_coords[9]
        left_scale = float(np.linalg.norm(left_middle_mcp - left_wrist))
        vector[LEFT_WRIST_SLICE] = left_wrist
        vector[LEFT_SCALE_IDX] = left_scale
        vector[LEFT_PRESENT_IDX] = 1.0

    if right_present:
        right_wrist = right_coords[0]
        right_middle_mcp = right_coords[9]
        right_scale = float(np.linalg.norm(right_middle_mcp - right_wrist))
        vector[RIGHT_WRIST_SLICE] = right_wrist
        vector[RIGHT_SCALE_IDX] = right_scale
        vector[RIGHT_PRESENT_IDX] = 1.0

    # 3. Inter-hand spatial relationships (only meaningful when both hands are visible)
    if left_present and right_present:
        inter_wrist_vec = right_wrist - left_wrist
        inter_wrist_dist = float(np.linalg.norm(inter_wrist_vec))
        vector[INTER_WRIST_VEC_SLICE] = inter_wrist_vec
        vector[INTER_WRIST_DIST_IDX] = inter_wrist_dist

    return vector
