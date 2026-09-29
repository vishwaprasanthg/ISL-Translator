import React from 'react';
import Card from '../common/Card';
import { useAppContext } from '../../context/AppContext';
import { MODEL_STATUS } from '../../utils/constants';
import { Volume2 } from 'lucide-react';

export default function TranslationCard() {
  const { modelStatus, prediction } = useAppContext();

  return (
    <Card title="Translation" className="flex flex-col h-full">
      {modelStatus === MODEL_STATUS.NOT_TRAINED ? (
        <div className="flex-1 text-text-secondary text-sm flex flex-col justify-center space-y-2">
          <p>No translation available.</p>
          <p>Start the translator once the recognition model is available.</p>
        </div>
      ) : (
        <div className="flex-1 text-xl text-text-primary">
          {prediction?.label || "Waiting for recognition..."}
        </div>
      )}

      <div className="mt-4 pt-4 border-t border-gray-800">
        <button 
          disabled={!prediction?.label}
          className={`flex items-center justify-center space-x-2 w-full py-2.5 px-4 rounded-lg transition-colors font-medium text-sm ${
            prediction?.label 
              ? 'bg-accent-primary text-white hover:bg-accent-primary/90 cursor-pointer' 
              : 'bg-bg-tertiary text-gray-500 cursor-not-allowed'
          }`}
        >
          <Volume2 className="w-4 h-4" />
          <span>Speak</span>
        </button>
      </div>
    </Card>
  );
}
