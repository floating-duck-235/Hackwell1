#!/usr/bin/env python3
"""
API endpoint for chronic care risk prediction
"""
import sys
import os
import json
import pandas as pd
import joblib
from flask import Flask, request, jsonify
from flask_cors import CORS

# Add the server directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'server'))

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend requests

# Load the trained model
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'server', 'models', 'chronic_care_risk_model.pkl')

try:
    # Load your trained model
    model_data = joblib.load(MODEL_PATH)
    predictor = model_data  # Adjust based on how you saved your model
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    predictor = None

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        # Get patient data from request
        data = request.get_json()
        patients = data.get('patients', [])
        
        if not patients:
            return jsonify({'error': 'No patient data provided'}), 400
        
        # Convert to DataFrame
        df = pd.DataFrame(patients)
        
        # Use your model's predict method
        # Adjust this based on your model's interface
        predictions, probabilities = predictor.predict(df)
        
        # Format results
        results = []
        for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
            # Determine risk category
            if prob < 0.3:
                risk_category = 'Low'
            elif prob < 0.7:
                risk_category = 'Medium'
            else:
                risk_category = 'High'
            
            results.append({
                'patient_id': patients[i].get('patient_id', f'Patient_{i+1}'),
                'risk_probability': float(prob),
                'risk_category': risk_category,
                'prediction': int(pred)
            })
        
        return jsonify(results)
    
    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'model_loaded': predictor is not None
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)