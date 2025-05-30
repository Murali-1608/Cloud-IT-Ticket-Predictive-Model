from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import tensorflow as tf
import numpy as np
import pickle
import os

# === Initialize App ===
app = FastAPI(title="Cloud IT Ticket Prediction API")

# ✅ Enable CORS (allow your frontend only)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:63342"],  # 🔐 Your local frontend URL origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === Define Input Schema ===
class TicketInput(BaseModel):
    description: str
    severity: int
    priority: int

# === Dynamic Relative Paths for Model Files ===
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
    raise RuntimeError("❌ Critical error loading model or tokenizer. App cannot continue.")

# === Define Max Sequence Length (must match training) ===
MAX_SEQUENCE_LENGTH = 50

#  Prediction Endpoint
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
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

# === Health Check Route ===
@app.get("/")
def home():
    return {"message": "✅ Cloud IT Ticket Prediction API is live!"}

# === Required for Render Deployment (also works locally) ===
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port)
