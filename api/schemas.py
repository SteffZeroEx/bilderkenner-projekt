# -*- coding: utf-8 -*-
"""
Pydantic Schemas fuer die API
"""

from pydantic import BaseModel
from typing import List


class PredictionResponse(BaseModel):
    """Response-Schema fuer Vorhersagen"""
    class_id: int
    class_name: str
    confidence: float
    probabilities: List[float]


class HealthResponse(BaseModel):
    """Response-Schema fuer Health-Check"""
    status: str
    model_loaded: bool
