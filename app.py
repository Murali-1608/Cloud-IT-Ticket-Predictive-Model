from fastapi import FastAPI
from pydantic import BaseModel
import tensorflow as tf
import numpy as np
import pickle
import os

# === Initialize App ===
app = FastAPI(title="Cloud IT Ticket Prediction API")

# === Define Input Schema ===
class TicketInput(BaseModel):
    description: str
    severity: int
    priority: int

# === Load Model and Tokenizer with RELATIVE PATH ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "models", "IT-Ticket-Prediction-Model-tuned.keras")
TOKENIZER_PATH = os.path.join(BASE_DIR, "models", "tokenizer.pkl")

# Load tokenizer
with open(TOKENIZER_PATH, 'rb') as f:
    tokenizer = pickle.load(f)

# Load model
model = tf.keras.models.load_model(MODEL_PATH)

# === Define Max Sequence Length (must match training) ===
MAX_SEQUENCE_LENGTH = 100

# === Prediction Endpoint ===
@app.post("/predict")
def predict_ticket(input: TicketInput):
    # Tokenize the description
    sequence = tokenizer.texts_to_sequences([input.description])
    padded_sequence = tf.keras.preprocessing.sequence.pad_sequences(
        sequence, maxlen=MAX_SEQUENCE_LENGTH, padding='post', truncating='post'
    )

    # Prepare metadata
    metadata = np.array([[input.severity, input.priority]])

    # Predict using model
    prediction = model.predict([padded_sequence, metadata])

    # Handle outputs (adjust if needed based on your model's output shape)
    category_index = int(np.argmax(prediction[0]))  # Classification output
    resolution_days = float(prediction[1][0][0])    # Regression output

    return {
        "predicted_category_index": category_index,
        "predicted_resolution_time_days": round(resolution_days, 2)
    }

# === Health Check Route ===
@app.get("/")
def home():
    return {"message": "✅ Cloud IT Ticket Prediction API is live!"}
