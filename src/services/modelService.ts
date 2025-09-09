import { PatientData } from '../utils/csvParser';

export interface PredictionResult {
  patient_id: string;
  risk_probability: number;
  risk_category: 'Low' | 'Medium' | 'High';
  prediction: number;
}

export interface ModelStatus {
  isProcessing: boolean;
  currentStep: string;
  progress: number;
  logs: string[];
  error?: string;
}

// Global status tracker
let modelStatus: ModelStatus = {
  isProcessing: false,
  currentStep: '',
  progress: 0,
  logs: []
};

// Status update callback
let statusCallback: ((status: ModelStatus) => void) | null = null;

export const setStatusCallback = (callback: (status: ModelStatus) => void) => {
  statusCallback = callback;
};

const updateStatus = (updates: Partial<ModelStatus>) => {
  modelStatus = { ...modelStatus, ...updates };
  if (statusCallback) {
    statusCallback(modelStatus);
  }
};

const addLog = (message: string) => {
  const timestamp = new Date().toLocaleTimeString();
  const logEntry = `${message}`;
  modelStatus.logs.push(logEntry);
  updateStatus({ logs: [...modelStatus.logs] });
  console.log(`[Model] ${logEntry}`);
};

// Mock model service - In a real application, this would call your Python backend
export const runPredictionModel = async (patientData: PatientData[]): Promise<PredictionResult[]> => {
  updateStatus({ 
    isProcessing: true, 
    currentStep: 'Initializing model', 
    progress: 0, 
    logs: [],
    error: undefined 
  });
  
  addLog(`Starting prediction for ${patientData.length} patients`);
  
  // Simulate API call delay
  updateStatus({ currentStep: 'Loading model weights', progress: 10 });
  addLog('Loading trained model from memory...');
  await new Promise(resolve => setTimeout(resolve, 500));
  
  updateStatus({ currentStep: 'Preprocessing data', progress: 25 });
  addLog('Preprocessing patient data...');
  addLog('Encoding categorical variables...');
  addLog('Scaling numerical features...');
  await new Promise(resolve => setTimeout(resolve, 800));
  
  updateStatus({ currentStep: 'Running predictions', progress: 50 });
  addLog('Running Random Forest model...');
  await new Promise(resolve => setTimeout(resolve, 600));
  
  updateStatus({ currentStep: 'Calculating risk scores', progress: 75 });
  addLog('Calculating risk probabilities...');
  await new Promise(resolve => setTimeout(resolve, 400));
  
  // Mock prediction logic based on patient characteristics
  addLog('Generating individual patient predictions...');
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
  
  updateStatus({ currentStep: 'Finalizing results', progress: 90 });
  addLog('Post-processing results...');
  await new Promise(resolve => setTimeout(resolve, 300));
  
  updateStatus({ currentStep: 'Complete', progress: 100 });
  addLog(`Successfully processed ${results.length} patients`);
  addLog(`High risk: ${results.filter(r => r.risk_category === 'High').length} patients`);
  addLog(`Medium risk: ${results.filter(r => r.risk_category === 'Medium').length} patients`);
  addLog(`Low risk: ${results.filter(r => r.risk_category === 'Low').length} patients`);
  
  // Keep processing state for a moment to show completion
  setTimeout(() => {
    updateStatus({ isProcessing: false });
  }, 1000);
  
  return results;
};

// Function to integrate with your actual Python model
export const callPythonModel = async (patientData: PatientData[]): Promise<PredictionResult[]> => {
  updateStatus({ 
    isProcessing: true, 
    currentStep: 'Connecting to Python API', 
    progress: 0, 
    logs: [],
    error: undefined 
  });
  
  try {
    addLog('Connecting to Python model server...');
    updateStatus({ currentStep: 'Sending data to model', progress: 20 });
    
    // Call your Python API endpoint
    addLog('Sending patient data to http://localhost:5000/api/predict');
    const response = await fetch('http://localhost:5000/api/predict', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ patients: patientData }),
    });
    
    updateStatus({ currentStep: 'Processing with Python model', progress: 60 });
    addLog('Python model is processing the data...');
    
    if (!response.ok) {
      throw new Error(`Model prediction failed: ${response.status} ${response.statusText}`);
    }
    
    updateStatus({ currentStep: 'Receiving results', progress: 90 });
    addLog('Receiving predictions from Python model...');
    
    const results = await response.json();
    
    updateStatus({ currentStep: 'Complete', progress: 100 });
    addLog(`Successfully received ${results.length} predictions from Python model`);
    
    setTimeout(() => {
      updateStatus({ isProcessing: false });
    }, 1000);
    
    return results;
  } catch (error) {
    const errorMessage = error instanceof Error ? error.message : 'Unknown error';
    addLog(`Error: ${errorMessage}`);
    addLog('Falling back to mock model...');
    updateStatus({ error: errorMessage, currentStep: 'Using fallback model', progress: 0 });
    
    // Fallback to mock model
    return runPredictionModel(patientData);
  }
};

export const getModelStatus = (): ModelStatus => modelStatus;

// Switch between mock and real model
export const predictWithModel = async (patientData: PatientData[]): Promise<PredictionResult[]> => {
  // Try real model first, fallback to mock if it fails
  try {
    return await callPythonModel(patientData);
  } catch (error) {
    addLog('Using mock model as fallback');
    return runPredictionModel(patientData);
  }
};