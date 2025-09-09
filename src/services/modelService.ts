import { PatientData } from '../utils/csvParser';

export interface PredictionResult {
  patient_id: string;
  risk_probability: number;
  risk_category: 'Low' | 'Medium' | 'High';
  prediction: number;
}

// Mock model service - In a real application, this would call your Python backend
export const runPredictionModel = async (patientData: PatientData[]): Promise<PredictionResult[]> => {
  // Simulate API call delay
  await new Promise(resolve => setTimeout(resolve, 2000));
  
  // Mock prediction logic based on patient characteristics
  const results: PredictionResult[] = patientData.map(patient => {
    // Calculate risk score based on various factors
    let riskScore = 0;
    
    // Age factor (higher age = higher risk)
    riskScore += Math.min((patient.age - 40) / 60, 0.3);
    
    // Chronic conditions
    riskScore += patient.diabetes * 0.2;
    riskScore += patient.heart_failure * 0.3;
    riskScore += patient.chronic_kidney_disease * 0.2;
    riskScore += patient.hypertension * 0.1;
    
    // Vital signs
    if (patient.systolic_BP > 140) riskScore += 0.1;
    if (patient.diastolic_BP > 90) riskScore += 0.1;
    if (patient.heart_rate > 100 || patient.heart_rate < 60) riskScore += 0.05;
    
    // Lab values
    if (patient.hba1c_level > 7) riskScore += 0.1;
    if (patient.estimated_GFR < 60) riskScore += 0.15;
    
    // Healthcare utilization
    riskScore += Math.min(patient.number_of_hospital_visits * 0.05, 0.2);
    riskScore += Math.min(patient.number_of_er_visits * 0.1, 0.2);
    
    // Medication adherence (lower adherence = higher risk)
    riskScore += (1 - patient.medication_adherence_rate) * 0.15;
    
    // Mental health
    riskScore += patient.depression_diagnosis * 0.1;
    riskScore += patient.anxiety_diagnosis * 0.05;
    
    // Lifestyle factors
    if (patient.smoking_status === 'Current') riskScore += 0.15;
    if (patient.BMI > 30) riskScore += 0.1;
    if (patient.physical_activity_level === 'Sedentary') riskScore += 0.1;
    
    // Add some randomness to simulate model uncertainty
    riskScore += (Math.random() - 0.5) * 0.2;
    
    // Ensure score is between 0 and 1
    const risk_probability = Math.max(0, Math.min(1, riskScore));
    
    // Determine risk category
    let risk_category: 'Low' | 'Medium' | 'High';
    if (risk_probability < 0.3) {
      risk_category = 'Low';
    } else if (risk_probability < 0.7) {
      risk_category = 'Medium';
    } else {
      risk_category = 'High';
    }
    
    // Binary prediction (1 = deterioration risk, 0 = stable)
    const prediction = risk_probability > 0.5 ? 1 : 0;
    
    return {
      patient_id: patient.patient_id,
      risk_probability,
      risk_category,
      prediction
    };
  });
  
  return results;
};

// Function to integrate with your actual Python model
export const callPythonModel = async (patientData: PatientData[]): Promise<PredictionResult[]> => {
  try {
    // Call your Python API endpoint
    const response = await fetch('http://localhost:5000/api/predict', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ patients: patientData }),
    });
    
    if (!response.ok) {
      throw new Error('Model prediction failed');
    }
    
    const results = await response.json();
    return results;
  } catch (error) {
    console.error('Error calling Python model:', error);
    // Fallback to mock model
    return runPredictionModel(patientData);
  }
};

// Switch between mock and real model
export const predictWithModel = async (patientData: PatientData[]): Promise<PredictionResult[]> => {
  // Try real model first, fallback to mock if it fails
  try {
    return await callPythonModel(patientData);
  } catch (error) {
    console.warn('Using mock model as fallback');
    return runPredictionModel(patientData);
  }
};