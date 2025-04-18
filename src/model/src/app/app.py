from fastapi import FastAPI
from pydantic import BaseModel
import tensorflow as tf
import pickle
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Initialize FastAPI app
app = FastAPI()

# Load the model and tokenizer
model = tf.keras.models.load_model("src/model/IT-Ticket-Prediction-Model-tuned.keras")

with open("src/model/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)


# Define Pydantic model for request body
class TicketData(BaseModel):
    text: str


# Preprocess function (adjust as needed)
def preprocess_text(text: str):
    # Tokenize and pad the input text (adjust maxlen as per your model's requirement)
    sequences = tokenizer.texts_to_sequences([text])
    padded_sequence = pad_sequences(sequences, maxlen=50)  # Assuming 50 is the maxlen used during training
    return padded_sequence


@app.post("/predict")
async def predict(ticket_data: TicketData):
    # Preprocess the input text
    processed_data = preprocess_text(ticket_data.text)

    # Make prediction
    prediction = model.predict(processed_data)

    # Return prediction (you can adjust this to return the needed format)
    return {"prediction": prediction.tolist()}
