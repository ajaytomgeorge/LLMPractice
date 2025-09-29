from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_sentiment_huggingface():
    response = client.get("/sentiment/huggingface/This is a great product!")
    assert response.status_code == 200
    assert "sentiment" in response.json()

def test_sentiment_custom():
    response = client.get("/sentiment/custom/This product is good!")
    assert response.status_code == 200
    assert response.json()["prediction"] == "positive"

def test_sentiment_custom_negative():
    response = client.get("/sentiment/custom/This product is bad!")
    assert response.status_code == 200
    assert response.json()["prediction"] == "negative"
