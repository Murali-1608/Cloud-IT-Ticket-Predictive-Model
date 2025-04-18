from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import tensorflow as tf
import pickle

# Initialize FastAPI app
app = FastAPI()

# Load the model and tokenizer at startup
model_path = "C:/Users/mural/Cloud-IT-Ticket-Predictive-Model/models/IT-Ticket-Prediction-Model-tuned.keras"
tokenizer_path = "C:/Users/mural/Cloud-IT-Ticket-Predictive-Model/models/tokenizer.pkl"

# Load the model
model = tf.keras.models.load_model(model_path)

# Load the tokenizer
with open(tokenizer_path, 'rb') as f:
    tokenizer = pickle.load(f)

# Define the category labels
category_labels = ["Network", "Software", "Hardware", "Other"]  # Adjust based on your categories

# Define the input data structure for the prediction request
class TicketRequest(BaseModel):
    text: str
    severity: int  # Severity should be encoded as an integer
    priority: int  # Priority should be encoded as an integer

@app.post("/predict")
async def predict(ticket: TicketRequest):
    try:
        # Preprocess the text (tokenize and pad)
        text_sequence = tokenizer.texts_to_sequences([ticket.text])
        padded_sequence = tf.keras.preprocessing.sequence.pad_sequences(text_sequence, maxlen=50, padding="post", truncating="post")

        # Prepare other features (severity and priority)
        other_features = np.array([[ticket.severity, ticket.priority]])

        # Make the prediction
        category_pred, resolution_time_pred = model.predict([padded_sequence, other_features])

        # Get the predicted category label
        predicted_category_index = np.argmax(category_pred, axis=1)[0]  # Get the index
        predicted_category = category_labels[predicted_category_index]

        # Get the predicted resolution time
        predicted_resolution_time_days = resolution_time_pred[0][0]

        # Return the response
        return {
            "predicted_category": predicted_category,
            "predicted_resolution_time_days": predicted_resolution_time_days
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
