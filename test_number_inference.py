import os
import sys
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "backend"))

from app.services.static_inference_service import inference_service

print("Testing direct inference on Numbers model:")

classes = [str(i) for i in range(1, 10)]

for cls in classes:
    class_dir = os.path.join("ml", "datasets", "manual", "numbers", cls)
    if not os.path.exists(class_dir):
        print(f"Directory not found: {class_dir}")
        continue
        
    # Pick the first .npz file
    files = [f for f in os.listdir(class_dir) if f.endswith(".npz")]
    if not files:
        print(f"No files in {class_dir}")
        continue
        
    data_file = os.path.join(class_dir, files[0])
    
    try:
        data = np.load(data_file)
        features = data['features'][0].tolist()
        
        # Predict
        res = inference_service.predict('numbers', features)
        
        print(f"Expected: {cls} | Predicted: {res.get('label')} | Confidence: {res.get('confidence'):.4f}")
    except Exception as e:
        print(f"Error processing {cls}: {e}")
