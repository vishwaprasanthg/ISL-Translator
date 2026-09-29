from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from pydantic import ValidationError
from app.core.websocket_models import LandmarkMessage
import json

router = APIRouter()

@router.websocket("/translate")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            try:
                # Parse JSON
                payload = json.loads(data)
                
                # Validate with Pydantic
                message = LandmarkMessage(**payload)
                
                # Success ACK
                await websocket.send_json({
                    "type": "ack",
                    "status": "received",
                    "feature_count": 126,
                    "mode": message.mode,
                    "timestamp": message.timestamp
                })
                
                # Phase 7: Static Inference for alphabet/number modes
                if message.mode in ["alphabet", "number", "numbers"]:
                    from app.services.static_inference_service import inference_service
                    
                    # Execute prediction
                    prediction_response = inference_service.predict(
                        mode=message.mode, 
                        features=message.features
                    )
                    
                    # Send prediction to client
                    await websocket.send_json(prediction_response)
                
            except json.JSONDecodeError:
                await websocket.send_json({
                    "type": "error",
                    "status": "invalid_message",
                    "message": "Malformed JSON"
                })
            except ValidationError as e:
                # Extract first error message
                error_msg = e.errors()[0]['msg'] if e.errors() else "Validation failed"
                await websocket.send_json({
                    "type": "error",
                    "status": "invalid_message",
                    "message": error_msg
                })
                
    except WebSocketDisconnect:
        print("Client disconnected")
