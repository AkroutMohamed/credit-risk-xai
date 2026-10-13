from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

def test_health_without_model():
    assert client.get("/health").status_code == 200
    assert client.get("/health").json()["status"] in ("ready","model_not_loaded")

def test_predict_without_bundle():
    result = client.post("/predict",json={"features":{"age":35},"threshold":0.5})
    assert result.status_code in (422,503)

def test_schema_validation():
    assert client.post("/predict",json={"features":{},"threshold":1.5}).status_code == 422
