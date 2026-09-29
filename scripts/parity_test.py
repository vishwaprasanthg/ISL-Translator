import sys
import json
import subprocess
import os
import numpy as np

# Adjust sys.path to find ml module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from ml.preprocessing.landmark_normalizer import create_feature_vector

def generate_landmarks(offset):
    # deterministic non-zero, slightly complex floats
    landmarks = []
    for i in range(21):
        x = i * 0.123 + offset * 0.5 + 0.05
        y = i * 0.456 + offset * 0.3 - 0.01
        z = i * 0.789 + offset * 0.1 + 0.02
        landmarks.append({'x': x, 'y': y, 'z': z})
    return landmarks

def run_parity(test_name, landmarks_array, handednesses_array):
    print(f"\n--- Parity Test: {test_name} ---")
    
    # 1. Prepare JSON fixture
    fixture = {
        "landmarks": landmarks_array,
        "handednesses": handednesses_array
    }
    
    fixture_path = os.path.join(os.path.dirname(__file__), "fixture.json")
    with open(fixture_path, 'w') as f:
        json.dump(fixture, f)
        
    # 2. Run JS implementation
    js_script_path = os.path.join(os.path.dirname(__file__), '..', 'frontend', 'scripts', 'js_parity.js')
    result = subprocess.run(
        ["node", js_script_path, fixture_path],
        capture_output=True,
        text=True,
        cwd=os.path.join(os.path.dirname(__file__), '..', 'frontend')
    )
    
    if result.returncode != 0:
        print("JS Error:")
        print(result.stderr)
        sys.exit(1)
        
    js_features = np.array(json.loads(result.stdout), dtype=np.float32)
    
    # 3. Run Python implementation
    left_hand = None
    right_hand = None
    
    for i in range(len(landmarks_array)):
        hand_cat = handednesses_array[i][0]['categoryName']
        if hand_cat == 'Left' and left_hand is None:
            left_hand = landmarks_array[i]
        elif hand_cat == 'Right' and right_hand is None:
            right_hand = landmarks_array[i]
            
    py_features = create_feature_vector(left_hand, right_hand)
    
    # 4. Compare
    print(f"JS output shape: {js_features.shape}")
    print(f"Python output shape: {py_features.shape}")
    
    diff = np.abs(js_features - py_features)
    max_diff = float(np.max(diff))
    print(f"Maximum absolute difference: {max_diff:.10f}")
    
    if max_diff <= 1e-5:
        print("Parity status: PASSED")
    else:
        print("Parity status: FAILED")
        print("JS:", js_features)
        print("PY:", py_features)
        sys.exit(1)

def main():
    left_lm = generate_landmarks(0.1)
    right_lm = generate_landmarks(0.9)
    
    # Test A: Both hands, Left first
    run_parity(
        "Both Hands (Left First)",
        [left_lm, right_lm],
        [[{'categoryName': 'Left'}], [{'categoryName': 'Right'}]]
    )
    
    # Test B: Both hands, Right first (reversed MediaPipe order)
    run_parity(
        "Both Hands (Right First)",
        [right_lm, left_lm],
        [[{'categoryName': 'Right'}], [{'categoryName': 'Left'}]]
    )
    
    # Test C: Left hand only
    run_parity(
        "Left Hand Only",
        [left_lm],
        [[{'categoryName': 'Left'}]]
    )
    
    # Test D: Right hand only
    run_parity(
        "Right Hand Only",
        [right_lm],
        [[{'categoryName': 'Right'}]]
    )
    
    # Test E: No hands
    run_parity(
        "No Hands",
        [],
        []
    )
    
    print("\nALL PARITY TESTS PASSED SUCCESSFULLY.")

if __name__ == "__main__":
    main()
