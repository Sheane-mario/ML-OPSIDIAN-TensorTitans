"""
Integration tests for the FastAPI application.

Uses FastAPI's TestClient to test endpoints without a running server.
The database is mocked to avoid requiring PostgreSQL during testing.
"""

import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient


# Mock the database before importing the app
@pytest.fixture(autouse=True)
def mock_db():
    """Mock the database initialization for all tests."""
    with patch('api.main.init_db'):
        yield


@pytest.fixture
def client():
    """Create a test client for the FastAPI app."""
    from api.main import app
    from api.database import get_db
    
    # Properly override the dependency in FastAPI
    mock_session = MagicMock()
    app.dependency_overrides[get_db] = lambda: iter([mock_session])
    
    # Use context manager to trigger lifespan events (like model loading)
    with TestClient(app) as test_client:
        yield test_client
        
    app.dependency_overrides.clear()


class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        """Health endpoint should always return 200."""
        response = client.get("/api/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "models_loaded" in data

    def test_health_shows_models(self, client):
        """Health endpoint should list available models."""
        response = client.get("/api/health")
        data = response.json()
        assert isinstance(data["models_loaded"], list)
        assert len(data["models_loaded"]) > 0


class TestModelsEndpoint:
    def test_list_models(self, client):
        """Should return a list of available model versions."""
        response = client.get("/api/models")
        assert response.status_code == 200
        models = response.json()
        assert isinstance(models, list)
        assert any(m["version"] == "v13" for m in models)

    def test_model_info_structure(self, client):
        """Each model should have the expected fields."""
        response = client.get("/api/models")
        models = response.json()
        for model in models:
            assert "version" in model
            assert "name" in model
            assert "description" in model
            assert "is_default" in model


class TestDistrictsEndpoint:
    def test_list_districts(self, client):
        """Should return all 25 districts."""
        response = client.get("/api/districts")
        assert response.status_code == 200
        districts = response.json()
        assert len(districts) == 25

    def test_district_has_defaults(self, client):
        """Each district should have default feature values."""
        response = client.get("/api/districts")
        districts = response.json()
        for d in districts:
            assert "district" in d
            assert "defaults" in d
            assert "rainfall_7d_mm" in d["defaults"]
            assert "drainage_index" in d["defaults"]


class TestSimulateEndpoint:
    def test_simulate_with_defaults(self, client):
        """Simulation with default values should return a valid score."""
        response = client.post("/api/simulate", json={})
        assert response.status_code == 200
        data = response.json()
        assert 0.0 <= data["flood_risk_score"] <= 1.0
        assert data["risk_level"] in ["Low", "Medium", "High", "Critical"]
        assert data["inference_time_ms"] >= 0
        assert data["model_version"] == "v13"

    def test_simulate_with_custom_values(self, client):
        """Simulation with custom high-risk values should work."""
        payload = {
            "district": "Colombo",
            "model_version": "v13",
            "rainfall_7d_mm": 300.0,
            "drainage_index": 0.1,
            "elevation_m": 2.0,
            "distance_to_river_m": 50.0,
            "infrastructure_score": 0.2,
        }
        response = client.post("/api/simulate", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert 0.0 <= data["flood_risk_score"] <= 1.0

    def test_simulate_returns_feature_importance(self, client):
        """Simulation should return feature importance data."""
        response = client.post("/api/simulate", json={})
        data = response.json()
        assert "feature_importance" in data
        assert isinstance(data["feature_importance"], list)

    def test_simulate_rejects_invalid_input(self, client):
        """Invalid input should return 422."""
        payload = {"rainfall_7d_mm": -999}
        response = client.post("/api/simulate", json=payload)
        assert response.status_code == 422

    def test_simulate_different_models(self, client):
        """Should accept all model versions."""
        for version in ["v13", "v18_titan", "v20_colossus"]:
            response = client.post("/api/simulate", json={"model_version": version})
            assert response.status_code == 200
            assert response.json()["model_version"] == version
