import numpy as np

NUM_LANDMARKS_PER_HAND = 21
COORDINATES_PER_LANDMARK = 3
FEATURES_PER_HAND = NUM_LANDMARKS_PER_HAND * COORDINATES_PER_LANDMARK
TOTAL_FEATURES = FEATURES_PER_HAND * 2
EPSILON = 1e-6

def normalize_hand_landmarks(landmarks):
    """
    Normalizes a single hand's landmarks (wrist-relative translation & scale normalization).
    
    Args:
        landmarks: List of dictionaries [{'x': x, 'y': y, 'z': z}, ...] or similar objects
                   Must be length 21.
    
    Returns:
        numpy.ndarray: Shape (63,) with dtype float32
    """
    if not landmarks or len(landmarks) != NUM_LANDMARKS_PER_HAND:
        return np.zeros(FEATURES_PER_HAND, dtype=np.float32)

    # Convert to numpy array for easier math (shape: 21, 3)
    coords = np.array([[lm['x'], lm['y'], lm['z']] if isinstance(lm, dict) else [lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32)

    wrist = coords[0]
    middle_mcp = coords[9]

    # 1. Calculate scale (distance from wrist to middle finger MCP)
    scale = np.linalg.norm(middle_mcp - wrist)
    
    # Prevent division by zero
    if scale < EPSILON:
        scale = 1.0

    # 2. Translate and scale
    normalized = (coords - wrist) / scale

    return normalized.flatten()


def create_feature_vector(left_hand_landmarks=None, right_hand_landmarks=None):
    """
    Creates the canonical 126-D feature vector.
    
    Args:
        left_hand_landmarks: 21 landmarks for left hand or None
        right_hand_landmarks: 21 landmarks for right hand or None
        
    Returns:
        numpy.ndarray: Exactly 126 features, dtype float32
    """
    feature_vector = np.zeros(TOTAL_FEATURES, dtype=np.float32)

    if left_hand_landmarks:
        left_normalized = normalize_hand_landmarks(left_hand_landmarks)
        feature_vector[0:FEATURES_PER_HAND] = left_normalized

    if right_hand_landmarks:
        right_normalized = normalize_hand_landmarks(right_hand_landmarks)
        feature_vector[FEATURES_PER_HAND:TOTAL_FEATURES] = right_normalized

    return feature_vector
