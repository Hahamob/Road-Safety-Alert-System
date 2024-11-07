from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import joblib

# Initialize Flask application
app = Flask(__name__)
CORS(app)  # Allow cross-origin requests

# Load the saved model and preprocessor pipeline
model = joblib.load('road_safety_model_xgb_best.pkl')
preprocessor = joblib.load('preprocessor.pkl')  # Pipeline containing imputer and scaler
columns = joblib.load('columns.pkl')  # Load saved list of feature column names

@app.route('/predict', methods=['POST'])
def predict():
    # Get JSON data from request
    data = request.get_json()

    # Convert data to DataFrame format
    new_data = pd.DataFrame([data])

    # One-hot encode and ensure feature alignment
    new_data = pd.get_dummies(new_data, columns=['roadSurface', 'roadCurvature', 'weatherA', 'weatherB', 'roadCharacter', 'trafficControl'], drop_first=True)
    new_data = new_data.reindex(columns=columns, fill_value=0)

    # Process new data using the preprocessor pipeline
    new_data = preprocessor.transform(new_data)

    # Make prediction
    pred_proba = model.predict_proba(new_data)[:, 1][0]  # Get probability of danger

    # set threshold
    threshold = 0.2
    if pred_proba <= threshold:
        result = "High risk: Please slow down and be cautious!"
    else:
        result = "Low risk: Safe road conditions."

    # Return prediction result as JSON, including probability
    return jsonify({
        "alert": result,
        #"probability_of_danger": pred_proba  # Include probability in the response
    })

if __name__ == '__main__':
    app.run(debug=True)
