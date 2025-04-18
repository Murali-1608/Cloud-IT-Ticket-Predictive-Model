from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pickle
import tensorflow as tf
import os
import gdown

# === FILE CONFIG ===
FILES = {
    "model": {
        "file_id": "1kEoTCUDpyObXbAEWNh7-y9NZOu5Ehyxk",
        "filename": "IT-Ticket-Prediction-Model-tuned.keras"
    },
    "label_encoders": {
        "file_id": "1beEpIlgvsTOcer617HcbK8eJpS2Cu5pt",
        "filename": "label_encoders.pkl"
    },
    "tokenizer": {
        "file_id": "1lVJi28Qojfq5cBgwIUkZet6OicZ4xbM3",
        "filename": "tokenizer.pkl"
    }
}

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model")
os.makedirs(MODEL_DIR, exist_ok=True)


# === DOWNLOAD FILES ===
def download_file(file_id, filename):
    path = os.path.join(MODEL_DIR, filename)
    if not os.path.exists(path):
        print(f"⬇️ Downloading {filename}...")
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, path, quiet=False)
    return path


# Download all required files
model_path = download_file(FILES["model"]["file_id"], FILES["model"]["filename"])
encoder_path = download_file(FILES["label_encoders"]["file_id"], FILES["label_encoders"]["filename"])
tokenizer_path = download_file(FILES["tokenizer"]["file_id"], FILES["tokenizer"]["filename"])

# === LOAD MODEL AND ARTIFACTS ===
try:
    model = tf.keras.models.load_model(model_path)
except Exception as e:
    raise RuntimeError(f"❌ Failed to load model: {e}")

try:
    with open(encoder_path, "rb") as f:
        label_encoders = pickle.load(f)
except Exception as e:
    raise RuntimeError(f"❌ Failed to load label encoders: {e}")

try:
    with open(tokenizer_path, "rb") as f:
        tokenizer = pickle.load(f)
except Exception as e:
    raise RuntimeError(f"❌ Failed to load tokenizer: {e}")

# === FASTAPI SETUP ===
app = FastAPI()


class TicketInput(BaseModel):
    issue_type: str
    priority: str
    department: str
    description_length: int
    created_hour: int


@app.get("/")
def home():
    return {"message": "🚀 Cloud IT Support Predictive API is Live!"}


@app.post("/predict/")
def predict_ticket(input_data: TicketInput):
    try:
        # Prepare input array
        X = np.array([[input_data.issue_type, input_data.priority,
                       input_data.department, input_data.description_length,
                       input_data.created_hour]], dtype=object)

        # Label encode categorical values
        for i, col in enumerate(['issue_type', 'priority', 'department']):
            le = label_encoders[col]
            X[:, i] = le.transform(X[:, i])

        X = X.astype(float)

        # Normalize numeric values using tokenizer (if it contains a scaler)
        # You may replace this with a scaler if it's not a tokenizer-based normalizer
        X_scaled = tokenizer.transform(X)
        X_reshaped = X_scaled.reshape((X_scaled.shape[0], X_scaled.shape[1], 1))

        prediction = model.predict(X_reshaped)
        predicted_class = int(np.argmax(prediction, axis=1)[0])

        return {"predicted_resolution_time": predicted_class}

    except Exception as e:
        return {"error": str(e)}
