import React, { useState } from 'react';
import { Brain, FileText, TrendingUp, Shield } from 'lucide-react';
import { FileUploader } from './components/FileUploader';
import { ResultsDisplay } from './components/ResultsDisplay';
import { LoadingSpinner } from './components/LoadingSpinner';
import { parseCSV, downloadCSV, PatientData } from './utils/csvParser';
import { predictWithModel, PredictionResult } from './services/modelService';

function App() {
  const [isLoading, setIsLoading] = useState(false);
  const [results, setResults] = useState<PredictionResult[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleFileUpload = async (file: File) => {
    setIsLoading(true);
    setError(null);
    setResults(null);

    try {
      const text = await file.text();
      const patientData: PatientData[] = parseCSV(text);
      
      if (patientData.length === 0) {
        throw new Error('No valid patient data found in the CSV file');
      }

      console.log(`Processing ${patientData.length} patients...`);
      
      // Run the prediction model
      const predictions = await predictWithModel(patientData);
      setResults(predictions);
      
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred while processing the file');
    } finally {
      setIsLoading(false);
    }
  };

  const handleDownloadResults = () => {
    if (results) {
      downloadCSV(results, 'chronic_care_predictions.csv');
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-indigo-50">
      {/* Header */}
      <header className="bg-white shadow-sm border-b border-gray-200">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
          <div className="flex items-center space-x-4">
            <div className="p-3 bg-blue-600 rounded-xl">
              <Brain className="w-8 h-8 text-white" />
            </div>
            <div>
              <h1 className="text-3xl font-bold text-gray-900">
                Chronic Care Risk Predictor
              </h1>
              <p className="text-gray-600 mt-1">
                AI-powered healthcare risk assessment and prediction system
              </p>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {!results && !isLoading && (
          <div className="space-y-8">
            {/* Feature Cards */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
              <div className="bg-white p-6 rounded-xl shadow-sm border border-gray-200">
                <div className="flex items-center space-x-3 mb-4">
                  <FileText className="w-6 h-6 text-blue-600" />
                  <h3 className="text-lg font-semibold text-gray-900">CSV Upload</h3>
                </div>
                <p className="text-gray-600 text-sm">
                  Upload patient data in CSV format for batch risk assessment
                </p>
              </div>

              <div className="bg-white p-6 rounded-xl shadow-sm border border-gray-200">
                <div className="flex items-center space-x-3 mb-4">
                  <TrendingUp className="w-6 h-6 text-green-600" />
                  <h3 className="text-lg font-semibold text-gray-900">Risk Analysis</h3>
                </div>
                <p className="text-gray-600 text-sm">
                  Advanced ML algorithms analyze 40+ health indicators
                </p>
              </div>

              <div className="bg-white p-6 rounded-xl shadow-sm border border-gray-200">
                <div className="flex items-center space-x-3 mb-4">
                  <Shield className="w-6 h-6 text-purple-600" />
                  <h3 className="text-lg font-semibold text-gray-900">Predictions</h3>
                </div>
                <p className="text-gray-600 text-sm">
                  Get detailed risk scores and actionable insights
                </p>
              </div>
            </div>

            {/* File Uploader */}
            <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-8">
              <div className="text-center mb-8">
                <h2 className="text-2xl font-bold text-gray-900 mb-2">
                  Upload Patient Data
                </h2>
                <p className="text-gray-600">
                  Upload a CSV file containing patient health data to get risk predictions
                </p>
              </div>
              
              <FileUploader onFileUpload={handleFileUpload} isLoading={isLoading} />
              
              {error && (
                <div className="mt-6 p-4 bg-red-50 border border-red-200 rounded-lg">
                  <p className="text-red-700">{error}</p>
                </div>
              )}
            </div>

            {/* Sample Data Info */}
            <div className="bg-blue-50 border border-blue-200 rounded-xl p-6">
              <h3 className="text-lg font-semibold text-blue-900 mb-3">
                Expected CSV Format
              </h3>
              <p className="text-blue-800 mb-3">
                Your CSV should include the following columns (based on your model):
              </p>
              <div className="grid grid-cols-2 md:grid-cols-3 gap-2 text-sm text-blue-700">
                <span>• patient_id</span>
                <span>• age</span>
                <span>• sex</span>
                <span>• BMI</span>
                <span>• systolic_BP</span>
                <span>• diastolic_BP</span>
                <span>• diabetes</span>
                <span>• hypertension</span>
                <span>• heart_failure</span>
                <span>• chronic_kidney_disease</span>
                <span>• hba1c_level</span>
                <span>• medication_adherence_rate</span>
                <span>• ... and more</span>
              </div>
            </div>
          </div>
        )}

        {isLoading && (
          <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-8">
            <LoadingSpinner message="Running AI model on your patient data..." />
          </div>
        )}

        {results && (
          <div className="space-y-6">
            <div className="text-center">
              <h2 className="text-2xl font-bold text-gray-900 mb-2">
                Prediction Results
              </h2>
              <p className="text-gray-600">
                Risk assessment completed for {results.length} patients
              </p>
            </div>
            
            <ResultsDisplay 
              results={results} 
              onDownload={handleDownloadResults}
            />
            
            <div className="text-center">
              <button
                onClick={() => {
                  setResults(null);
                  setError(null);
                }}
                className="px-6 py-3 bg-gray-600 text-white rounded-lg hover:bg-gray-700 transition-colors duration-200 font-medium"
              >
                Upload New Data
              </button>
            </div>
          </div>
        )}
      </main>

      {/* Footer */}
      <footer className="bg-gray-50 border-t border-gray-200 mt-16">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
          <div className="text-center text-gray-600">
            <p>Chronic Care Risk Prediction System - Powered by Advanced Machine Learning</p>
            <p className="text-sm mt-2">
              This system uses AI to assess patient risk factors and predict potential health deterioration
            </p>
          </div>
        </div>
      </footer>
    </div>
  );
}

export default App;