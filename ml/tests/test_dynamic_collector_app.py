import os
import shutil
import tempfile
import numpy as np
import pytest
from ml.collector.dynamic_collector_app import (
    app,
    DYNAMIC_CLASSES,
    TOTAL_FEATURES,
    FEATURE_VERSION,
    MIN_MOTION_THRESHOLD,
    MIN_FRAMES,
    MAX_FRAMES,
)


@pytest.fixture
def client(tmp_path, monkeypatch):
    """Create Flask test client with an isolated temporary dynamic dataset directory."""
    temp_dynamic_dir = tmp_path / "dynamic"
    temp_dynamic_dir.mkdir(parents=True, exist_ok=True)
    for cls in DYNAMIC_CLASSES:
        (temp_dynamic_dir / cls).mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr("ml.collector.dynamic_collector_app.DATASET_ROOT", temp_dynamic_dir)
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def test_index_route(client):
    response = client.get("/")
    assert response.status_code == 200
    assert b"Dynamic ISL Dataset Collector" in response.data or b"Collector" in response.data


def test_api_config(client):
    response = client.get("/api/config")
    assert response.status_code == 200
    data = response.get_json()
    assert data["feature_version"] == FEATURE_VERSION
    assert data["total_features"] == TOTAL_FEATURES
    assert data["total_features"] == 140
    assert len(data["dynamic_classes"]) == 30
    assert data["min_frames"] == MIN_FRAMES
    assert data["max_frames"] == MAX_FRAMES


def test_api_progress_initial(client):
    response = client.get("/api/progress")
    assert response.status_code == 200
    data = response.get_json()
    for cls in DYNAMIC_CLASSES:
        assert cls in data
        assert data[cls]["total"] == 0
        assert data[cls]["complete"] is False


def test_save_sequence_invalid_class(client):
    payload = {
        "class_label": "INVALID_SIGN",
        "frames": [[0.0] * 140] * 30,
        "session_id": "session_001",
    }
    response = client.post("/api/save_sequence", json=payload)
    assert response.status_code == 400
    data = response.get_json()
    assert "Unknown class" in data["error"]


def test_save_sequence_invalid_session_id(client):
    payload = {
        "class_label": "HELLO",
        "frames": [[0.1] * 140] * 30,
        "session_id": "bad_session_name",
    }
    response = client.post("/api/save_sequence", json=payload)
    assert response.status_code == 400
    data = response.get_json()
    assert "Invalid session_id" in data["error"]


def test_save_sequence_wrong_feature_dimension(client):
    # Attempting to save legacy 126-D features must be rejected
    payload = {
        "class_label": "HELLO",
        "frames": [[0.1] * 126] * 30,
        "session_id": "session_001",
    }
    response = client.post("/api/save_sequence", json=payload)
    assert response.status_code == 200
    data = response.get_json()
    assert data["saved"] is False
    assert data["valid"] is False
    assert "expected 140 values" in data["reason"]


def test_save_sequence_insufficient_frames(client):
    payload = {
        "class_label": "HELLO",
        "frames": [[0.1] * 140] * 5,  # Less than MIN_FRAMES (10)
        "session_id": "session_001",
    }
    response = client.post("/api/save_sequence", json=payload)
    assert response.status_code == 200
    data = response.get_json()
    assert data["saved"] is False
    assert data["valid"] is False
    assert "too few frames" in data["reason"]


def test_save_sequence_insufficient_motion(client):
    # Identical stationary frames with no displacement
    identical_frame = [0.1] * 140
    payload = {
        "class_label": "HELLO",
        "frames": [identical_frame for _ in range(30)],
        "session_id": "session_001",
    }
    response = client.post("/api/save_sequence", json=payload)
    assert response.status_code == 200
    data = response.get_json()
    assert data["saved"] is False
    assert data["valid"] is False
    assert "insufficient movement" in data["reason"]


def test_save_sequence_valid_success(client):
    # Construct sequence with clear wrist and landmark motion
    frames = []
    for t in range(30):
        frame = [0.0] * 140
        # Active right hand wrist moving along X and Y
        frame[130] = 0.3 + t * 0.030  # right wrist x moves
        frame[131] = 0.5 + t * 0.010  # right wrist y moves
        frame[133] = 0.12             # right hand scale
        frame[139] = 1.0              # right hand present
        # Non-wrist right landmarks (e.g. index fingertip at index 63 + 8*3 = 87)
        frame[87] = 0.2 + t * 0.025
        frames.append(frame)

    payload = {
        "class_label": "HELLO",
        "frames": frames,
        "session_id": "session_001",
    }
    response = client.post("/api/save_sequence", json=payload)
    assert response.status_code == 200
    data = response.get_json()
    assert data["saved"] is True
    assert data["valid"] is True
    assert data["motion_score"] >= MIN_MOTION_THRESHOLD
    assert data["feature_version"] == "140d"
    assert data["vector_dim"] == 140

    # Verify progress reflects the new sequence
    prog_resp = client.get("/api/progress/HELLO")
    assert prog_resp.status_code == 200
    prog = prog_resp.get_json()
    assert prog["total"] == 1
    assert prog["sessions"]["session_001"] == 1


def test_delete_session(client):
    # First save a valid sequence
    frames = []
    for t in range(25):
        frame = [0.0] * 140
        frame[130] = 0.2 + t * 0.02
        frame[139] = 1.0
        frames.append(frame)

    save_resp = client.post("/api/save_sequence", json={
        "class_label": "STOP",
        "frames": frames,
        "session_id": "session_001",
    })
    assert save_resp.get_json()["saved"] is True

    # Delete session
    del_resp = client.post("/api/delete_session", json={
        "class_label": "STOP",
        "session_id": "session_001",
        "confirm": True,
    })
    assert del_resp.status_code == 200
    del_data = del_resp.get_json()
    assert del_data["deleted"] is True
    assert del_data["session_id"] == "session_001"

    # Check progress is back to 0
    prog_resp = client.get("/api/progress/STOP")
    assert prog_resp.get_json()["total"] == 0
