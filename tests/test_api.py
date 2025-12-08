# -*- coding: utf-8 -*-
"""
Tests fuer API-Module
"""

import pytest
from fastapi.testclient import TestClient

from api.main import app
from api.schemas import PredictionResponse, HealthResponse


class TestAPI:
    """Tests fuer die FastAPI Endpoints"""

    @pytest.fixture
    def client(self):
        """Test-Client fuer die API"""
        return TestClient(app)

    def test_root_endpoint(self, client):
        """Root-Endpoint gibt Info zurueck"""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert data["message"] == "Bilderkenner API"

    def test_docs_available(self, client):
        """API-Dokumentation ist erreichbar"""
        response = client.get("/docs")
        assert response.status_code == 200


class TestSchemas:
    """Tests fuer Pydantic Schemas"""

    def test_prediction_response(self):
        """PredictionResponse Schema funktioniert"""
        response = PredictionResponse(
            class_id=0,
            class_name="Flugzeug",
            confidence=0.95,
            probabilities=[0.95, 0.01, 0.01, 0.01, 0.0, 0.0, 0.01, 0.0, 0.01, 0.0]
        )
        assert response.class_id == 0
        assert response.class_name == "Flugzeug"

    def test_health_response(self):
        """HealthResponse Schema funktioniert"""
        response = HealthResponse(status="healthy", model_loaded=True)
        assert response.status == "healthy"
        assert response.model_loaded is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
