from pydantic import BaseModel, Field, field_validator
from typing import List
import math

class LandmarkMessage(BaseModel):
    type: str
    mode: str
    features: List[float]
    timestamp: float

    @field_validator('type')
    def validate_type(cls, v):
        if v != "landmarks":
            raise ValueError('Type must be "landmarks"')
        return v

    @field_validator('mode')
    def validate_mode(cls, v):
        if v not in {"alphabet", "number", "numbers", "vocabulary"}:
            raise ValueError('Mode must be one of: alphabet, numbers, vocabulary')
        return v

    @field_validator('features')
    def validate_features(cls, v):
        if len(v) != 126:
            raise ValueError('Features must contain exactly 126 values')
        for val in v:
            if not math.isfinite(val):
                raise ValueError('All features must be finite')
        return v
