import React from 'react';
import { Brain, Loader2, CheckCircle, AlertCircle } from 'lucide-react';

interface LoadingSpinnerProps {
  message?: string;
  status?: 'processing' | 'success' | 'error';
  logs?: string[];
}

export const LoadingSpinner: React.FC<LoadingSpinnerProps> = ({ 
  message = "Processing your data...",
  status = 'processing',
  logs = []
}) => {
  const getStatusIcon = () => {
    switch (status) {
      case 'success':
        return <CheckCircle className="w-12 h-12 text-green-600" />;
      case 'error':
        return <AlertCircle className="w-12 h-12 text-red-600" />;
      default:
        return (
          <div className="relative">
            <Brain className="w-12 h-12 text-blue-600" />
            <Loader2 className="w-6 h-6 text-blue-400 animate-spin absolute -top-1 -right-1" />
          </div>
        );
    }
  };

  return (
    <div className="flex flex-col items-center justify-center py-12 space-y-6">
      {getStatusIcon()}
      <div className="text-center">
        <h3 className="text-lg font-semibold text-gray-900 mb-2">
          {status === 'success' ? 'Processing Complete!' : 
           status === 'error' ? 'Processing Error' : 'AI Model Processing'}
        </h3>
        <p className="text-gray-600">{message}</p>
      </div>
      
      {status === 'processing' && (
        <div className="flex space-x-1">
          <div className="w-2 h-2 bg-blue-600 rounded-full animate-bounce" style={{ animationDelay: '0ms' }}></div>
          <div className="w-2 h-2 bg-blue-600 rounded-full animate-bounce" style={{ animationDelay: '150ms' }}></div>
          <div className="w-2 h-2 bg-blue-600 rounded-full animate-bounce" style={{ animationDelay: '300ms' }}></div>
        </div>
      )}

      {/* Processing Logs */}
      {logs.length > 0 && (
        <div className="w-full max-w-2xl bg-gray-900 rounded-lg p-4 text-left">
          <div className="flex items-center space-x-2 mb-3">
            <div className="w-3 h-3 bg-green-500 rounded-full animate-pulse"></div>
            <span className="text-green-400 text-sm font-medium">Processing Log</span>
          </div>
          <div className="space-y-1 max-h-40 overflow-y-auto">
            {logs.map((log, index) => (
              <div key={index} className="text-gray-300 text-xs font-mono">
                <span className="text-gray-500">[{new Date().toLocaleTimeString()}]</span> {log}
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};