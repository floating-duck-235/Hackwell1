# Model Integration Guide

## Setup Instructions

1. **Place your trained model file** in `server/models/chronic_care_risk_model.pkl`

2. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the API server**:
   ```bash
   python api/predict.py
   ```

4. **Update the frontend** to use the real API by modifying `src/services/modelService.ts`

## Model File Structure

Your model should be saved using joblib and contain:
- The trained model
- Preprocessors (scalers, encoders)
- Feature columns list

Example of how to save your model:
```python
import joblib

# Save your trained predictor
joblib.dump(predictor, 'server/models/chronic_care_risk_model.pkl')
```

## API Endpoints

- `POST /api/predict` - Make predictions on patient data
- `GET /api/health` - Check if the model is loaded correctly