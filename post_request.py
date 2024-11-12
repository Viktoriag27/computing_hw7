import requests
import json

# Set the API endpoint URL
url = "http://127.0.0.1:8000/predict/"

# Send the POST request
try:
    # Open the input JSON file
    with open("json_file.json", "r") as file:
        response = requests.post(url, files={"file": ("json_file.json", file, "application/json")})
    
    # Check for HTTP errors
    response.raise_for_status()
    
    # Parse and print the response
    result = response.json()
    print("Prediction:", result["prediction"])
    print("Probability:", result["probability"])
except requests.exceptions.RequestException as e:
    print(f"API request error: {e}")
except json.JSONDecodeError:
    print("Error decoding the response from the API.")
except KeyError as e:
    print(f"Missing expected field in the API response: {e}")
