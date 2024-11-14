'''' Machine Learning API - HW7 '''
import joblib
import numpy as np
import pandas as pd
import json
from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
from sklearn.linear_model import LogisticRegression

# inputs = ["age", "height", "weight", "aids", "cirrhosis", "hepatic_failure", "immunosuppression", "leukemia", "lymphoma", "solid_t"]
# Create FastAPI application
app = FastAPI()

# Load the final model
model = joblib.load('trained_model.pkl')

# List of expected columns for prediction
exp_columns = ['age', 'height', 'weight', 'aids', 'cirrhosis', 'hepatic_failure', 'immunosuppression', 'leukemia', 'lymphoma', 'solid_tumor_with_metastasis']

@app.post('/predict')
async def predict(file: UploadFile = File(...)):
    # Validate that the file is of JSON type
    if file.content_type != 'application/json':
        raise HTTPException(status_code=400, detail="Invalid file format. Please upload a .json file.")
    
    # Read and process the JSON file
    try:
        contents = await file.read()
        input_data = json.loads(contents)  # Convert the content to a dictionary
        input_df = pd.DataFrame([input_data])  # Convert the dictionary to a DataFrame
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading the JSON file: {e}")
    
    # Validate if there aren't enough columns
    missing_columns = [col for col in exp_columns if col not in input_data]
    if missing_columns:
        raise HTTPException(status_code=400, detail=f"Missing columns: {', '.join(missing_columns)}")
    
     # Make the prediction
    try:
        prediction = model.predict(input_df)
        probability = model.predict_proba(input_df) if hasattr(model, "predict_proba") else None
        response = {
            "prediction": int(prediction[0]),  # Convert a int to JSON
            "probability": probability[0].tolist() if probability is not None else "Not available"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")
    
    return response
