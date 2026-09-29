import React from 'react';
import Card from '../common/Card';
import { useAppContext } from '../../context/AppContext';
import { MODEL_STATUS } from '../../utils/constants';

export default function PredictionCard() {
  const { modelStatus, prediction } = useAppContext();

  return (
    <Card title="Current Sign">
      {modelStatus === MODEL_STATUS.NOT_TRAINED ? (
        <div className="flex-1 flex items-center justify-center">
          <p className="text-accent-warning text-sm font-medium">Model not trained</p>
        </div>
      ) : (
        <div className="flex-1 flex flex-col justify-center space-y-4">
          <div>
            <div className="text-3xl font-bold text-text-primary mb-1">
              {prediction?.label || "—"}
            </div>
            <div className="text-sm text-text-secondary">
              {prediction ? "Prediction available" : "No prediction available"}
            </div>
          </div>
          
          <div className="grid grid-cols-2 gap-4 pt-4 border-t border-gray-800">
            <div>
              <div className="text-xs text-text-secondary uppercase tracking-wider mb-1">Confidence</div>
              <div className="text-sm font-mono text-text-primary">
                {prediction?.confidence ? `${(prediction.confidence * 100).toFixed(1)}%` : "—"}
              </div>
            </div>
            <div>
              <div className="text-xs text-text-secondary uppercase tracking-wider mb-1">Sign Type</div>
              <div className="text-sm font-mono text-text-primary">
                {prediction?.sign_type || "—"}
              </div>
            </div>
          </div>
        </div>
      )}
    </Card>
  );
}
