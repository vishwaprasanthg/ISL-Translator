import pytest
import numpy as np
from app.services.static_inference_service import StaticInferenceService

def test_inference_service_initialization():
    service = StaticInferenceService()
    assert service.alphabet_session is not None
    assert service.number_session is not None
    
    assert service.alphabet_session.is_available
    assert service.number_session.is_available

def test_inference_service_unavailable_response():
    # To test unavailable response, we can temporarily disable the session
    service = StaticInferenceService()
    if service.alphabet_session:
        service.alphabet_session.is_available = False
        
    dummy_features = [0.0] * 126
    response = service.predict("alphabet", dummy_features)
    
    assert response["type"] == "prediction"
    assert response["status"] == "model_unavailable"
    assert response["label"] is None

def test_inference_service_invalid_mode():
    service = StaticInferenceService()
    dummy_features = [0.0] * 126
    
    response = service.predict("vocabulary", dummy_features)
    assert response["type"] == "prediction"
    assert response["status"] == "model_unavailable"

def test_alphabet_prediction():
    service = StaticInferenceService()
    # Dummy features that aren't all zeros to avoid zero-vector checks if any
    dummy_features = [0.1] * 126
    
    response = service.predict("alphabet", dummy_features)
    assert response["type"] == "prediction"
    assert response["sign_type"] == "static"
    # Even with low confidence, it should return a structured response
    assert response["status"] in ["accepted", "low_confidence"]

def test_number_prediction():
    service = StaticInferenceService()
    dummy_features = [0.1] * 126
    
    response = service.predict("number", dummy_features)
    assert response["type"] == "prediction"
    assert response["sign_type"] == "static"
    assert response["status"] in ["accepted", "low_confidence"]
