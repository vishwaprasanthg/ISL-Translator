import os
import json
import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional
from ..models.static_model import StaticHandCNN

class StaticInferenceSession:
    """
    Handles loading and inference for the StaticHandCNN model.
    """
    def __init__(self, checkpoint_path: str, label_map_path: str):
        self.checkpoint_path = checkpoint_path
        self.label_map_path = label_map_path
        self.model = None
        self.labels = {}
        self.is_available = False
        
        self._load()

    def _load(self):
        # 1. Check if files exist
        if not os.path.exists(self.checkpoint_path):
            print(f"Checkpoint not found: {self.checkpoint_path}")
            return
        if not os.path.exists(self.label_map_path):
            print(f"Label map not found: {self.label_map_path}")
            return

        # 2. Load label map
        try:
            with open(self.label_map_path, 'r', encoding='utf-8') as f:
                raw_map = json.load(f)
                # Map should be {"A": 0, "B": 1, ...} -> We need to reverse it to {0: "A", 1: "B"}
                # If it's already a list or dict where keys are indices, handle accordingly
                if isinstance(raw_map, dict):
                    # Check if keys are strings that look like ints (e.g. "0": "A")
                    if all(k.isdigit() for k in raw_map.keys()):
                        self.labels = {int(k): str(v) for k, v in raw_map.items()}
                    else:
                        # e.g., "A": 0
                        self.labels = {int(v): str(k) for k, v in raw_map.items()}
                elif isinstance(raw_map, list):
                    self.labels = {i: str(label) for i, label in enumerate(raw_map)}
        except Exception as e:
            print(f"Failed to load label map: {e}")
            return
            
        num_classes = len(self.labels)
        if num_classes == 0:
            print("Label map is empty.")
            return

        # 3. Load model checkpoint
        try:
            self.model = StaticHandCNN(num_classes=num_classes)
            # Use map_location='cpu' for inference safety
            checkpoint = torch.load(self.checkpoint_path, map_location=torch.device('cpu'), weights_only=True)
            
            # Handle potential DataParallel saved models or direct state dicts
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
                
            self.model.eval()
            self.is_available = True
            print(f"Successfully loaded static model from {self.checkpoint_path} with {num_classes} classes.")
        except Exception as e:
            print(f"Failed to load checkpoint: {e}")
            self.is_available = False

    def predict(self, features: list) -> Tuple[Optional[str], float]:
        """
        Runs inference on exactly 126 features.
        Returns: (label_str, confidence_float)
        """
        if not self.is_available or self.model is None:
            return None, 0.0

        if len(features) != 126:
            raise ValueError(f"Expected exactly 126 features, got {len(features)}")

        # Validate finiteness
        if not all(np.isfinite(f) for f in features):
            raise ValueError("Features contain NaN or Infinity")

        # Convert to tensor
        try:
            tensor = torch.tensor([features], dtype=torch.float32)
            
            with torch.no_grad():
                logits = self.model(tensor)
                probs = F.softmax(logits, dim=1)
                
                # Get max probability and corresponding class index
                max_prob, max_idx = torch.max(probs, dim=1)
                confidence = max_prob.item()
                class_idx = max_idx.item()
                
                label = self.labels.get(class_idx, "UNKNOWN")
                return label, confidence
                
        except Exception as e:
            raise RuntimeError(f"Inference failed: {e}")
