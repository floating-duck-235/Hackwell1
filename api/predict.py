# #!/usr/bin/env python3
# """
# API endpoint for chronic care risk prediction
# """
# import sys
# import os
# import json
# import pandas as pd
# import joblib
# from flask import Flask, request, jsonify
# from flask_cors import CORS

# # Add the server directory to Python path
# sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'server'))

# app = Flask(__name__)
# CORS(app)  # Enable CORS for frontend requests

# # Load the trained model
# MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'server', 'models', 'chronic_care_risk_model.pkl')

# try:
#     # Load your trained model
#     model_data = joblib.load(MODEL_PATH)
#     predictor = model_data  # Adjust based on how you saved your model
#     print("Model loaded successfully!")
# except Exception as e:
#     print(f"Error loading model: {e}")
#     predictor = None

# @app.route('/api/predict', methods=['POST'])
# def predict():
#     try:
#         # Get patient data from request
#         data = request.get_json()
#         patients = data.get('patients', [])
        
#         if not patients:
#             return jsonify({'error': 'No patient data provided'}), 400
        
#         # Convert to DataFrame
#         df = pd.DataFrame(patients)
        
#         # Use your model's predict method
#         # Adjust this based on your model's interface
#         predictions, probabilities = predictor.predict(df)
        
#         # Format results
#         results = []
#         for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
#             # Determine risk category
#             if prob < 0.3:
#                 risk_category = 'Low'
#             elif prob < 0.7:
#                 risk_category = 'Medium'
#             else:
#                 risk_category = 'High'
            
#             results.append({
#                 'patient_id': patients[i].get('patient_id', f'Patient_{i+1}'),
#                 'risk_probability': float(prob),
#                 'risk_category': risk_category,
#                 'prediction': int(pred)
#             })
        
#         return jsonify(results)
    
#     except Exception as e:
#         print(f"Prediction error: {e}")
#         return jsonify({'error': str(e)}), 500

# @app.route('/api/health', methods=['GET'])
# def health_check():
#     return jsonify({
#         'status': 'healthy',
#         'model_loaded': predictor is not None
#     })

# if __name__ == '__main__':
#     app.run(debug=True, port=5000)
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

# Load the trained model dictionary
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'server', 'models', 'chronic_care_risk_model.pkl')

try:
    model_data = joblib.load(MODEL_PATH)
    model = model_data["best_model"]
    scaler = model_data["scaler"]
    encoders = model_data["label_encoders"]
    feature_columns = model_data["feature_columns"]
    print(f"Model ({model_data['best_model_name']}) loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    model, scaler, encoders, feature_columns = None, None, None, None


@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        patients = data.get('patients', [])

        if not patients:
            return jsonify({'error': 'No patient data provided'}), 400

        # Convert to DataFrame
        df = pd.DataFrame(patients)

        # Ensure all expected features exist
        for col in feature_columns:
            if col not in df.columns:
                df[col] = None  # fill missing

        # Encode categorical features
        for col, le in encoders.items():
            if col in df:
                df[col] = df[col].astype(str).map(lambda x: x if x in le.classes_ else le.classes_[0])
                df[col] = le.transform(df[col])

        # Scale numeric features
        X = scaler.transform(df[feature_columns])

        # Predict
        predictions = model.predict(X)
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(X).max(axis=1)  # highest class probability
        else:
            probabilities = [0.5] * len(predictions)  # fallback if no proba

        # Format results
        results = []
        for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
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
        'model_loaded': model is not None
    })


if __name__ == '__main__':
    app.run(debug=True, port=5000)
