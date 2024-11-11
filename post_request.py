import requests
import json

# Set the API endpoint URL
url = "http://127.0.0.1:8000/predict/"

# Read the input JSON file
with open("json_file.json", "r") as file:
    input_data = json.load(file)

# Send the POST request
try:
    with open("json_file.json", "r") as file:
        response = requests.post(url, files={"file": ("json_file.json", file, "application/json")})
    response.raise_for_status()  # Check for HTTP errors
    result = response.json()
    print("Prediction:", result["prediction"])
    print("Probability:", result["probability"])
except requests.exceptions.RequestException as e:
    print(f"API request error: {e}")
except json.JSONDecodeError:
    print("Error decoding the response from the API.")