import joblib
from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
import pandas as pd
import json
import numpy as np

# Load the trained model
model = joblib.load("trained_model.pkl")

# Initialize FastAPI app
app = FastAPI()

# Define endpoint for uploading input JSON and making predictions
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    if file.content_type != 'application/json':
        raise HTTPException(status_code=400, detail="Invalid file format. Please upload a .json file.")

    # Read and parse the JSON file
    try:
        contents = await file.read()
        input_data = json.loads(contents)
        input_df = pd.DataFrame([input_data])  # Convert dictionary to DataFrame
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading the JSON file: {e}")
    
    # Make prediction
    try:
        prediction = model.predict(input_df)
        probability = model.predict_proba(input_df) if hasattr(model, "predict_proba") else None
        response = {
            "prediction": int(prediction[0]),  # Converting to int for JSON serialization
            "probability": probability[0].tolist() if probability is not None else "Not available"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")

    return response