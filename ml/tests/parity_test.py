import numpy as np
import math
from pathlib import Path
import sys

# Add project root to sys.path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from ml.preprocessing.landmark_normalizer import create_feature_vector, normalize_hand_landmarks

def parity_test():
    """
    Test that the Python normalizer matches the theoretical mathematical
    expectation of the frontend Javascript normalizer.
    
    Frontend JS logic:
    1. Distance from wrist (index 0) to middle finger MCP (index 9).
    2. Subtract wrist from all landmarks.
    3. Divide all landmarks by the distance.
    """
    print("Running Preprocessing Parity Test...")

    # Mock some fake 3D landmarks for a hand
    # Wrist at (1, 1, 1)
    # Middle MCP at (1, 2, 1) -> distance is 1.0
    mock_hand = []
    for i in range(21):
        if i == 0:
            mock_hand.append({"x": 1.0, "y": 1.0, "z": 1.0})
        elif i == 9:
            mock_hand.append({"x": 1.0, "y": 2.0, "z": 1.0})
        else:
            mock_hand.append({"x": 1.0 + i, "y": 1.0 + i, "z": 1.0 + i})

    normalized = normalize_hand_landmarks(mock_hand)
    
    # Expected distance: sqrt((1-1)^2 + (2-1)^2 + (1-1)^2) = 1.0
    # So wrist should be (0, 0, 0)
    # MCP should be (0, 1, 0)
    # Landmark 2 should be ((1+2) - 1)/1 = 2, so (2, 2, 2)
    
    expected_wrist = np.array([0.0, 0.0, 0.0])
    expected_mcp = np.array([0.0, 1.0, 0.0])
    expected_lm2 = np.array([2.0, 2.0, 2.0])

    actual_wrist = normalized[0:3]
    actual_mcp = normalized[27:30]
    actual_lm2 = normalized[6:9]

    assert np.allclose(actual_wrist, expected_wrist, atol=1e-5), f"Wrist mismatch. Got: {actual_wrist}"
    assert np.allclose(actual_mcp, expected_mcp, atol=1e-5), f"MCP mismatch. Got: {actual_mcp}"
    assert np.allclose(actual_lm2, expected_lm2, atol=1e-5), f"LM2 mismatch. Got: {actual_lm2}"

    print("✔ Single hand normalization matches exact mathematical expectation.")

    # Test create_feature_vector ordering
    # Left hand is provided, Right hand is None
    vec1 = create_feature_vector(left_hand_landmarks=mock_hand, right_hand_landmarks=None)
    assert len(vec1) == 126
    assert np.all(vec1[0:63] != 0), "Left hand features should be non-zero"
    assert np.all(vec1[63:126] == 0), "Right hand features should be zero"

    # Left hand is None, Right hand is provided
    vec2 = create_feature_vector(left_hand_landmarks=None, right_hand_landmarks=mock_hand)
    assert len(vec2) == 126
    assert np.all(vec2[0:63] == 0), "Left hand features should be zero"
    assert np.all(vec2[63:126] != 0), "Right hand features should be non-zero"
    
    print("✔ Feature vector hand ordering and dimensionality verified (Left: 0-62, Right: 63-125).")
    print("\nPARITY TEST PASSED: Training preprocessing correctly matches frontend expectations.")

if __name__ == "__main__":
    parity_test()
