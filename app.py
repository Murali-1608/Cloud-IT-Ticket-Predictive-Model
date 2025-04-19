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

try:
    # Load tokenizer
    with open(TOKENIZER_PATH, 'rb') as f:
        tokenizer = pickle.load(f)
    print("✅ Tokenizer loaded successfully.")

    # Load model
    model = tf.keras.models.load_model(MODEL_PATH)
    print("✅ Model loaded successfully.")
except Exception as e:
    print("❌ Error loading model or tokenizer:", str(e))

# === Define Max Sequence Length (must match training) ===
MAX_SEQUENCE_LENGTH = 100

# === Prediction Endpoint ===
@app.post("/predict")
def predict_ticket(input: TicketInput):
    try:
        print("📥 Input received:", input.dict())

        # Tokenize the description
        sequence = tokenizer.texts_to_sequences([input.description])
        padded_sequence = tf.keras.preprocessing.sequence.pad_sequences(
            sequence, maxlen=MAX_SEQUENCE_LENGTH, padding='post', truncating='post'
        )
        print("🧠 Tokenized and padded sequence:", padded_sequence.shape)

        # Prepare metadata
        metadata = np.array([[input.severity, input.priority]])
        print("🧾 Metadata:", metadata)

        # Predict using model
        prediction = model.predict([padded_sequence, metadata])
        print("✅ Prediction raw output:", prediction)

        # Extract outputs
        category_index = int(np.argmax(prediction[0]))  # Classification
        resolution_days = float(prediction[1][0][0])    # Regression

        return {
            "predicted_category_index": category_index,
            "predicted_resolution_time_days": round(resolution_days, 2)
        }

    except Exception as e:
        print("❌ Exception during prediction:", str(e))
        return {"error": str(e)}

# === Health Check Route ===
@app.get("/")
def home():
    return {"message": "✅ Cloud IT Ticket Prediction API is live!"}

# === Required for Render Deployment ===
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))  # Get Render port or default to 8000
    uvicorn.run("app:app", host="0.0.0.0", port=port)
