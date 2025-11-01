from fastapi import FastAPI
from pydantic import BaseModel
import joblib
from fastapi.middleware.cors import CORSMiddleware

# Initialize app
app = FastAPI(title="Social Sentiment API", version="1.0")

# Enable CORS (to allow frontend to connect)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # you can restrict this to ["http://localhost:3000"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = joblib.load(os.path.join(BASE_DIR, "model", "sentiment_model.pkl"))
vectorizer = joblib.load(os.path.join(BASE_DIR, "model", "vectorizer.pkl"))

# Input format
class TextInput(BaseModel):
    text: str

@app.get("/")
def root():
    return {"message": "Welcome to the Sentiment Analysis API!"}

@app.post("/analyze")
def analyze_sentiment(data: TextInput):
    text_vector = vectorizer.transform([data.text])
    prediction = model.predict(text_vector)[0]
    return {"text": data.text, "sentiment": prediction}
