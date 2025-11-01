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

# Load model and vectorizer
model = joblib.load("../model/sentiment_model.pkl")
vectorizer = joblib.load("../model/vectorizer.pkl")

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
