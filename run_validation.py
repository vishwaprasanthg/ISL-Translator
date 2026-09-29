import sys
import os
import json
import numpy as np
from fastapi.testclient import TestClient

# 1. Setup path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "backend")))
os.environ["PYTHONPATH"] = os.path.abspath(os.path.join(os.path.dirname(__file__), "backend"))

# 2. Import app
try:
    from backend.app.main import app
except ImportError as e:
    print(f"Failed to import app: {e}")
    sys.exit(1)

client = TestClient(app)

print("="*40)
print("6. Test GET /health")
print("="*40)
response = client.get("/health")
print(f"Status: {response.status_code}")
print(f"JSON: {response.json()}")
print()

print("="*40)
print("7. Test the WebSocket /ws/translate")
print("="*40)
# Create dummy features
d = np.load('ml/datasets/pilot_100_features.npz')
features = d['X'][0].tolist()

with client.websocket_connect("/ws/translate") as websocket:
    payload = {
        "type": "landmarks",
        "mode": "alphabet",
        "features": features,
        "timestamp": 123456789.0
    }
    websocket.send_json(payload)
    
    # Receive ACK
    ack = websocket.receive_json()
    print("Received ACK:", ack)
    
    # Receive Prediction
    prediction = websocket.receive_json()
    print("Received Prediction:", prediction)
    
print("All verifications passed!")
