import React from 'react';
import Card from '../common/Card';
import { useAppContext } from '../../context/AppContext';
import { MODEL_STATUS } from '../../utils/constants';

export default function PerformancePanel() {
  const { modelStatus, metrics } = useAppContext();

  return (
    <Card title="Performance" className="h-full">
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <div className="bg-bg-tertiary/50 p-3 rounded-lg border border-gray-800">
          <div className="text-xs text-text-secondary uppercase tracking-wider mb-1">FPS</div>
          <div className="text-lg font-mono text-text-primary">{metrics?.fps || '--'}</div>
        </div>
        
        <div className="bg-bg-tertiary/50 p-3 rounded-lg border border-gray-800">
          <div className="text-xs text-text-secondary uppercase tracking-wider mb-1">Inference</div>
          <div className="text-lg font-mono text-text-primary">
            {metrics?.inferenceMs ? `${metrics.inferenceMs} ms` : '--'}
          </div>
        </div>

        <div className="bg-bg-tertiary/50 p-3 rounded-lg border border-gray-800">
          <div className="text-xs text-text-secondary uppercase tracking-wider mb-1">Network</div>
          <div className="text-lg font-mono text-text-primary">
            {metrics?.wsRttMs ? `${Math.round(metrics.wsRttMs)} ms` : '--'}
          </div>
        </div>

        <div className="bg-bg-tertiary/50 p-3 rounded-lg border border-gray-800">
          <div className="text-xs text-text-secondary uppercase tracking-wider mb-1">Total Latency</div>
          <div className="text-lg font-mono text-text-primary">
            {metrics?.totalLatencyMs ? `${metrics.totalLatencyMs} ms` : '--'}
          </div>
        </div>
      </div>
      
      <div className="mt-4 pt-4 border-t border-gray-800 flex justify-between items-center text-sm">
        <span className="text-text-secondary">Model</span>
        <span className={`font-medium ${
          modelStatus === MODEL_STATUS.READY ? 'text-accent-success' :
          modelStatus === MODEL_STATUS.NOT_TRAINED ? 'text-accent-warning' : 'text-text-primary'
        }`}>
          {modelStatus === MODEL_STATUS.READY ? 'Loaded' : 
           modelStatus === MODEL_STATUS.NOT_TRAINED ? 'Not Trained' : 
           modelStatus === MODEL_STATUS.ERROR ? 'Error' : 'Loading...'}
        </span>
      </div>
    </Card>
  );
}
