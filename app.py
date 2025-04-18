from fastapi import FastAPI
from pydantic import BaseModel
import tensorflow as tf
import pickle
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ✅ Initialize FastAPI app
app = FastAPI()

# ✅ Relative paths (WORKS inside Docker)
MODEL_PATH = "models/IT-Ticket-Prediction-Model-tuned.keras"
TOKENIZER_PATH = "models/tokenizer.pkl"

print("✅ Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)
print("✅ Model loaded.")

print("✅ Loading tokenizer...")
with open(TOKENIZER_PATH, "rb") as f:
    tokenizer = pickle.load(f)
print("✅ Tokenizer loaded.")

# ✅ Define input format
class TicketRequest(BaseModel):
    text: str

# ✅ Root check endpoint
@app.get("/")
def read_root():
    return {"message": "✅ Cloud IT Ticket Prediction API is live!"}

# ✅ Prediction endpoint with debug info
@app.post("/predict")
async def predict_ticket(request: TicketRequest):
    try:
        print("🔹 Incoming request:", request)
        text = request.text
        print("🔹 Extracted text:", text)

        # Tokenize
        sequence = tokenizer.texts_to_sequences([text])
        print("🔹 Tokenized:", sequence)

        # Pad
        padded = pad_sequences(sequence, maxlen=50)
        print("🔹 Padded:", padded)

        # Predict
        prediction = model.predict(padded)
        print("🔹 Prediction:", prediction)

        return {"prediction": prediction.tolist()}

    except Exception as e:
        print(" Internal Server Error:", str(e))
        return {"error": str(e)}
