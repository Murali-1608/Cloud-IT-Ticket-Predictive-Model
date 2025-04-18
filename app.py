from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import os

# Load model and tokenizer using correct local path
MODEL_PATH = os.path.join("models", "IT-Ticket-Prediction-Model-tuned.keras")
TOKENIZER_PATH = os.path.join("models", "tokenizer.pkl")

# Load model
model = tf.keras.models.load_model(MODEL_PATH)

# Load tokenizer
with open(TOKENIZER_PATH, "rb") as f:
    tokenizer = pickle.load(f)

# FastAPI app setup
app = FastAPI()

# Request body definition
class TicketRequest(BaseModel):
    text: str
    severity: int
    priority: int

@app.get("/")
def read_root():
    return {"message": "✅ Cloud IT Ticket Prediction API is live!"}

@app.post("/predict")
def predict_ticket(ticket: TicketRequest):
    try:
        # Tokenize and pad input text
        sequence = tokenizer.texts_to_sequences([ticket.text])
        padded_seq = pad_sequences(sequence, maxlen=50, padding="post", truncating="post")

        # Prepare metadata input
        metadata = np.array([[ticket.severity, ticket.priority]])

        # Predict
        category_pred, time_pred = model.predict([padded_seq, metadata])
        predicted_category = int(np.argmax(category_pred[0]))
        predicted_days = float(time_pred[0][0])

        # Return predictions
        return {
            "predicted_category_index": predicted_category,
            "predicted_resolution_time_days": round(predicted_days, 2)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

