import React from 'react';
import CameraFeed from '../components/camera/CameraFeed';
import PredictionCard from '../components/dashboard/PredictionCard';
import TranslationCard from '../components/dashboard/TranslationCard';
import PerformancePanel from '../components/dashboard/PerformancePanel';
import { useAppContext } from '../context/AppContext';
import { MODES } from '../utils/constants';

export default function Dashboard() {
  const { selectedMode, setSelectedMode } = useAppContext();

  return (
    <div className="flex flex-col h-full gap-6">
      <div className="flex space-x-2">
        {Object.values(MODES).map((mode) => (
          <button
            key={mode}
            onClick={() => setSelectedMode(mode)}
            className={`px-4 py-2 rounded-lg text-sm font-medium capitalize transition-colors ${
              selectedMode === mode
                ? 'bg-accent-primary text-white'
                : 'bg-bg-tertiary text-text-secondary hover:text-text-primary'
            }`}
          >
            {mode}
          </button>
        ))}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 flex-1 min-h-0">
        <div className="lg:col-span-2 flex flex-col gap-6 h-full">
          <CameraFeed />
          <div className="h-40 shrink-0">
            <PerformancePanel />
          </div>
        </div>

        <div className="flex flex-col gap-6 h-full">
          <div className="h-48 shrink-0">
            <PredictionCard />
          </div>
          <div className="flex-1">
            <TranslationCard />
          </div>
        </div>
      </div>
    </div>
  );
}
