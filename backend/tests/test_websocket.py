import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_websocket_connect_and_valid_message():
    with client.websocket_connect("/ws/translate") as websocket:
        features = [0.0] * 126
        payload = {
            "type": "landmarks",
            "mode": "alphabet",
            "features": features,
            "timestamp": 12345.0
        }
        websocket.send_json(payload)
        data = websocket.receive_json()
        assert data["type"] == "ack"
        assert data["status"] == "received"
        assert data["feature_count"] == 126
        assert data["mode"] == "alphabet"

def test_websocket_wrong_feature_count():
    with client.websocket_connect("/ws/translate") as websocket:
        features = [0.0] * 10  # Invalid count
        payload = {
            "type": "landmarks",
            "mode": "alphabet",
            "features": features,
            "timestamp": 12345.0
        }
        websocket.send_json(payload)
        data = websocket.receive_json()
        assert data["type"] == "error"
        assert data["status"] == "invalid_message"

def test_websocket_invalid_mode():
    with client.websocket_connect("/ws/translate") as websocket:
        features = [0.0] * 126
        payload = {
            "type": "landmarks",
            "mode": "unknown",
            "features": features,
            "timestamp": 12345.0
        }
        websocket.send_json(payload)
        data = websocket.receive_json()
        assert data["type"] == "error"
        assert data["status"] == "invalid_message"

def test_websocket_nan_infinity_rejected():
    with client.websocket_connect("/ws/translate") as websocket:
        features = [float('inf')] * 126
        payload = {
            "type": "landmarks",
            "mode": "alphabet",
            "features": features,
            "timestamp": 12345.0
        }
        websocket.send_json(payload)
        data = websocket.receive_json()
        assert data["type"] == "error"
        assert data["status"] == "invalid_message"

def test_websocket_malformed_json():
    with client.websocket_connect("/ws/translate") as websocket:
        websocket.send_text("NOT JSON")
        data = websocket.receive_json()
        assert data["type"] == "error"
        assert data["status"] == "invalid_message"
        assert data["message"] == "Malformed JSON"

def test_websocket_disconnect():
    with client.websocket_connect("/ws/translate") as websocket:
        pass  # Just connecting and leaving scope triggers disconnect
