from fastapi import FastAPI
from transformers import pipeline
import torch

app = FastAPI()

# Load a simple Hugging Face model for sentiment analysis
sentiment_model = pipeline("sentiment-analysis")

# For demonstration, a simple custom model (in real scenarios, this would be a more complex model)
class CustomModel:
    def predict(self, text):
        # Simple dummy prediction
        return {"prediction": "positive" if "good" in text else "negative"}

custom_model = CustomModel()

@app.get("/sentiment/huggingface/{text}")
def get_sentiment_huggingface(text: str):
    result = sentiment_model(text)
    return {"sentiment": result}

@app.get("/sentiment/custom/{text}")
def get_sentiment_custom(text: str):
    result = custom_model.predict(text)
    return result

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
