import json
from pathlib import Path
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_predict_payload():
    p = Path(__file__).resolve().parent / "sample_payload.json"
    payload = json.loads(p.read_text())

    with TestClient(app) as client:
        r = client.post("/predict", json=payload)
        assert r.status_code == 200, r.text
        data = r.json()
        assert 0.0 <= data["probability_default"] <= 1.0
        assert data["prediction"] in [0,1]
        assert data["decision"] in ["Accepté", "Refusé"]

def test_predict_rejects_unknown_feature():
    with TestClient(app) as client:
        payload = {"features": {"FAKE_FEATURE": 123}}
        r = client.post("/predict", json=payload)
        assert r.status_code == 422

def test_predict_rejects_empty_features():
    with TestClient(app) as client:
        payload = {"features": {}}
        r = client.post("/predict", json=payload)
        assert r.status_code == 422