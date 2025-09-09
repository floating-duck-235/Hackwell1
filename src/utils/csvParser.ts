export interface PatientData {
  patient_id: string;
  age: number;
  sex: string;
  ethnicity: string;
  height_cm: number;
  weight_kg: number;
  BMI: number;
  systolic_BP: number;
  diastolic_BP: number;
  heart_rate: number;
  respiratory_rate: number;
  temperature_F: number;
  oxygen_saturation: number;
  smoking_status: string;
  alcohol_use: string;
  exercise_frequency_per_week: number;
  diet_quality_score: number;
  diabetes: number;
  hypertension: number;
  heart_failure: number;
  chronic_kidney_disease: number;
  obesity: number;
  hba1c_level: number;
  fasting_glucose_mg_dl: number;
  total_cholesterol_mg_dl: number;
  LDL_cholesterol_mg_dl: number;
  HDL_cholesterol_mg_dl: number;
  triglycerides_mg_dl: number;
  serum_creatinine_mg_dl: number;
  estimated_GFR: number;
  sodium_mmol_l: number;
  potassium_mmol_l: number;
  medication_adherence_rate: number;
  medication_count: number;
  number_of_hospital_visits: number;
  number_of_er_visits: number;
  days_since_last_visit: number;
  depression_diagnosis: number;
  anxiety_diagnosis: number;
  sleep_quality_score: number;
  systolic_BP_variation: number;
  diastolic_BP_variation: number;
  smoking_pack_years: number;
  alcohol_units_per_week: number;
  physical_activity_level: string;
  last_lab_date: string;
  last_clinical_visit_date: string;
}

export const parseCSV = (csvText: string): PatientData[] => {
  const lines = csvText.trim().split('\n');
  const headers = lines[0].split(',').map(h => h.trim().replace(/"/g, ''));
  
  const data: PatientData[] = [];
  
  for (let i = 1; i < lines.length; i++) {
    const values = lines[i].split(',').map(v => v.trim().replace(/"/g, ''));
    const row: any = {};
    
    headers.forEach((header, index) => {
      const value = values[index];
      
      // Convert numeric fields
      if (['age', 'height_cm', 'weight_kg', 'BMI', 'systolic_BP', 'diastolic_BP', 
           'heart_rate', 'respiratory_rate', 'temperature_F', 'oxygen_saturation',
           'exercise_frequency_per_week', 'diet_quality_score', 'diabetes', 'hypertension',
           'heart_failure', 'chronic_kidney_disease', 'obesity', 'hba1c_level',
           'fasting_glucose_mg_dl', 'total_cholesterol_mg_dl', 'LDL_cholesterol_mg_dl',
           'HDL_cholesterol_mg_dl', 'triglycerides_mg_dl', 'serum_creatinine_mg_dl',
           'estimated_GFR', 'sodium_mmol_l', 'potassium_mmol_l', 'medication_adherence_rate',
           'medication_count', 'number_of_hospital_visits', 'number_of_er_visits',
           'days_since_last_visit', 'depression_diagnosis', 'anxiety_diagnosis',
           'sleep_quality_score', 'systolic_BP_variation', 'diastolic_BP_variation',
           'smoking_pack_years', 'alcohol_units_per_week'].includes(header)) {
        row[header] = parseFloat(value) || 0;
      } else {
        row[header] = value;
      }
    });
    
    data.push(row as PatientData);
  }
  
  return data;
};

export const downloadCSV = (data: any[], filename: string) => {
  const headers = Object.keys(data[0]);
  const csvContent = [
    headers.join(','),
    ...data.map(row => headers.map(header => row[header]).join(','))
  ].join('\n');
  
  const blob = new Blob([csvContent], { type: 'text/csv' });
  const url = window.URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = filename;
  link.click();
  window.URL.revokeObjectURL(url);
};