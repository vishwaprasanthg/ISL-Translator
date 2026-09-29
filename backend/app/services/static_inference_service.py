import sys
import os
import time

# Add project root so ml/ can be imported
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../../")
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    from ml.inference.static_inference import StaticInferenceSession
except ImportError as e:
    print(f"Warning: Could not import StaticInferenceSession: {e}")
    StaticInferenceSession = None


class StaticInferenceService:

    def __init__(self):
        self.alphabet_session = None
        self.number_session = None
        self._initialize()

    def _initialize(self):

        if StaticInferenceSession is None:
            print("StaticInferenceSession unavailable")
            return

        # Alphabet model
        alphabet_ckpt = os.path.join(PROJECT_ROOT, "backend", "models", "static_alphabet.pt")
        alphabet_label = os.path.join(PROJECT_ROOT, "ml", "label_maps", "alphabet.json")
        
        print("Loading alphabet model...")
        self.alphabet_session = StaticInferenceSession(alphabet_ckpt, alphabet_label)
        if self.alphabet_session.is_available:
            print("Alphabet model loaded successfully.")
        else:
            print("Alphabet model failed to load.")

        # Number model
        number_ckpt = os.path.join(PROJECT_ROOT, "backend", "models", "static_numbers.pt")
        number_label = os.path.join(PROJECT_ROOT, "ml", "label_maps", "numbers.json")

        print("Loading number model...")
        self.number_session = StaticInferenceSession(number_ckpt, number_label)
        if self.number_session.is_available:
            print("Number model loaded successfully.")
        else:
            print("Number model failed to load.")

    def predict(self, mode: str, features: list) -> dict:

        start_time = time.perf_counter()

        # Route to appropriate session
        if mode == "alphabet":
            session = self.alphabet_session
        elif mode == "number" or mode == "numbers":
            session = self.number_session
        else:
            return self._unavailable_response()

        if session is None or not session.is_available:
            return self._unavailable_response()

        try:

            label, confidence = session.predict(features)

            latency_ms = (
                time.perf_counter() - start_time
            ) * 1000.0

            # Basic confidence threshold
            if label is None or confidence < 0.2:

                return {
                    "type": "prediction",
                    "label": None,
                    "confidence": confidence,
                    "sign_type": "static",
                    "latency_ms": round(latency_ms, 2),
                    "status": "low_confidence"
                }

            return {
                "type": "prediction",
                "label": label,
                "confidence": confidence,
                "sign_type": "static",
                "latency_ms": round(latency_ms, 2),
                "status": "accepted"
            }

        except Exception as e:

            print(f"Static inference error: {e}")

            return self._unavailable_response()

    def _unavailable_response(self):

        return {
            "type": "prediction",
            "label": None,
            "confidence": 0.0,
            "sign_type": "static",
            "latency_ms": 0.0,
            "status": "model_unavailable"
        }


# Singleton
inference_service = StaticInferenceService()