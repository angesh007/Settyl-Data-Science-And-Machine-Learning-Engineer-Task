from fastapi import FastAPI
from pydantic import BaseModel
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
import json
import numpy as np

# Load pre-trained model and tokenizer
# Note: Replace these lines with loading your actual model and tokenizer
model = None  # Load your model here
tokenizer = None  # Load your tokenizer here
label_encoder = None  # Load your label encoder here
max_length = None  # Set the maximum length for padding

app = FastAPI()

class Status(BaseModel):
    externalStatus: str

class Prediction(BaseModel):
    predicted_internal_status: str

@app.post("/predict/", response_model=Prediction)
async def predict_status(status: Status):
    text = status.externalStatus
    # Tokenize and pad the input text
    sequence = tokenizer.texts_to_sequences([text])
    padded_sequence = pad_sequences(sequence, maxlen=max_length, padding='post')
    # Make predictions
    prediction = model.predict(padded_sequence)
    predicted_label = label_encoder.inverse_transform([prediction.argmax()])[0]
    return {"predicted_internal_status": predicted_label}

# Endpoint for testing API functionality
@app.get("/test/")
async def test_api():
    return {"message": "API is running and ready for testing"}

# Endpoint for validating predictions against a validation dataset
@app.get("/validate/")
async def validate_predictions():
    # Load validation dataset
    with open('validation_dataset.json', 'r') as f:
        validation_dataset = json.load(f)
    
    # Extract external and internal statuses
    external_statuses = [data['externalStatus'] for data in validation_dataset]
    internal_statuses = [data['internalStatus'] for data in validation_dataset]

    # Encode internal statuses
    encoded_internal_statuses = label_encoder.transform(internal_statuses)

    # Tokenize and pad sequences
    sequences = tokenizer.texts_to_sequences(external_statuses)
    padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post')

    # Make predictions
    predictions = model.predict(padded_sequences)
    predicted_labels = label_encoder.inverse_transform(np.argmax(predictions, axis=1))

    # Calculate accuracy
    accuracy = np.mean(predicted_labels == internal_statuses)
    
    return {"accuracy": accuracy}
