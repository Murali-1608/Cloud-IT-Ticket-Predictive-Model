import os
import gdown
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Google Drive file IDs
model_file_id = '1kEoTCUDpyObXbAEWNh7-y9NZOu5Ehyxk'  # Model file ID
tokenizer_file_id = '1lVJi28Qojfq5cBgwIUkZet6OicZ4xbM3'  # Tokenizer file ID

# Paths to save the model and tokenizer
model_path = 'src/model/IT-Ticket-Prediction-Model-tuned.keras'
tokenizer_path = 'src/model/tokenizer.pkl'

# Create the directories if they don't exist
os.makedirs(os.path.dirname(model_path), exist_ok=True)

# Download the model file from Google Drive
if not os.path.exists(model_path):
    print("Downloading model...")
    gdown.download(f'https://drive.google.com/uc?export=download&id={model_file_id}', model_path, quiet=False)

# Download the tokenizer file from Google Drive
if not os.path.exists(tokenizer_path):
    print("Downloading tokenizer...")
    gdown.download(f'https://drive.google.com/uc?export=download&id={tokenizer_file_id}', tokenizer_path, quiet=False)

# Load the model
print("Loading model...")
model = load_model(model_path)
print("Model loaded successfully.")

# Load the tokenizer
try:
    print("Loading tokenizer...")
    with open(tokenizer_path, 'rb') as f:
        tokenizer = pickle.load(f)
    print("Tokenizer loaded successfully.")
except FileNotFoundError:
    print("Tokenizer file not found. Please check the file path.")
    exit(1)

# Sample input data for testing
sample_text = "User cannot login to the system. They are getting authentication error."

#
