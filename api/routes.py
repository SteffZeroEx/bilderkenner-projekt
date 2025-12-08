# -*- coding: utf-8 -*-
"""
API Routes fuer Bildklassifikation
"""

from fastapi import APIRouter, UploadFile, File, HTTPException
from PIL import Image
import io

from api.schemas import PredictionResponse, HealthResponse
from models.predict import Predictor

router = APIRouter()

# Predictor wird beim ersten Request geladen
predictor = None


def get_predictor():
    """Lazy-Loading fuer den Predictor"""
    global predictor
    if predictor is None:
        try:
            predictor = Predictor()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Modell konnte nicht geladen werden: {str(e)}")
    return predictor


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health-Check Endpoint"""
    try:
        get_predictor()
        return HealthResponse(status="healthy", model_loaded=True)
    except Exception:
        return HealthResponse(status="unhealthy", model_loaded=False)


@router.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    """
    Klassifiziert ein hochgeladenes Bild

    - **file**: Bilddatei (JPG, PNG)

    Returns: Vorhersage mit Klasse und Konfidenz
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Datei muss ein Bild sein")

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        pred = get_predictor()
        result = pred.predict_image(image)

        return PredictionResponse(**result)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Fehler bei der Vorhersage: {str(e)}")
