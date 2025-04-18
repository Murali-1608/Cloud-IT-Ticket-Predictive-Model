from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os

# === Path Setup ===
# Manually define BASE_DIR since __file__ may not work in Colab or certain IDEs
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_PATH = os.path.join(BASE_DIR, "models", "IT-Ticket-Prediction-Model-tuned.keras")
TOKENIZER_PATH = os.path.join(BASE_DIR, "models", "tokenizer.pkl")
ENCODER_PATH = os.path.join(BASE_DIR, "models", "label_encoders.pkl")

# === Load Resources ===
# Load tokenizer
with open(TOKENIZER_PATH, "rb") as f:
    tokenizer = pickle.load(f)

# Load label encoders (assumed as dict: {"category": LabelEncoder()})
with open(ENCODER_PATH, "rb") as f:
    label_encoders = pickle.load(f)

# Load hybrid model
model = tf.keras.models.load_model(MODEL_PATH)

# === FastAPI App ===
app = FastAPI(title="Cloud IT Ticket Predictor API")

# Input schema
class TicketRequest(BaseModel):
    description: str

# === Prediction Endpoint ===
@app.post("/predict/")
def predict_ticket(request: TicketRequest):
    try:
        text = request.description

        # Tokenize and pad input
        sequence = tokenizer.texts_to_sequences([text])
        padded = pad_sequences(sequence, maxlen=100)  # Adjust based on training

        # Get prediction
        predictions = model.predict(padded)[0]

        # Extract classification (category)
        category_encoder = label_encoders["category"]
        num_classes = len(category_encoder.classes_)
        class_probs = predictions[:num_classes]
        predicted_class_idx = np.argmax(class_probs)
        predicted_category = category_encoder.inverse_transform([predicted_class_idx])[0]

        # Extract regression (resolution time)
        predicted_time = predictions[-1]

        return {
            "predicted_category": predicted_category,
            "estimated_resolution_time_hours": round(float(predicted_time), 2)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

# === Run via terminal ===
# uvicorn src.predict:app --reload
