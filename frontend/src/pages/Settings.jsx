import React from 'react';
import Card from '../components/common/Card';
import { useAppContext } from '../context/AppContext';
import { MODES } from '../utils/constants';

export default function Settings() {
  const { 
    confidenceThreshold, 
    setConfidenceThreshold,
    selectedMode,
    setSelectedMode,
    speechEnabled,
    setSpeechEnabled
  } = useAppContext();

  return (
    <div className="flex flex-col h-full gap-6">
      <h2 className="text-2xl font-bold">Settings</h2>
      
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6 max-w-4xl">
        <Card title="Recognition">
          <div className="space-y-6">
            <div>
              <label className="block text-sm font-medium text-text-secondary mb-2">
                Confidence Threshold: {confidenceThreshold}%
              </label>
              <input 
                type="range" 
                min="10" 
                max="99" 
                value={confidenceThreshold}
                onChange={(e) => setConfidenceThreshold(Number(e.target.value))}
                className="w-full accent-accent-primary"
              />
              <p className="text-xs text-text-secondary mt-1">
                Minimum confidence required to accept a prediction.
              </p>
            </div>

            <div>
              <label className="block text-sm font-medium text-text-secondary mb-2">
                Default Mode
              </label>
              <select 
                value={selectedMode}
                onChange={(e) => setSelectedMode(e.target.value)}
                className="w-full bg-bg-tertiary border border-gray-700 rounded-lg p-2.5 text-sm text-text-primary focus:ring-accent-primary focus:border-accent-primary"
              >
                {Object.entries(MODES).map(([key, value]) => (
                  <option key={key} value={value} className="capitalize">
                    {value}
                  </option>
                ))}
              </select>
            </div>
          </div>
        </Card>

        <Card title="Preferences">
          <div className="space-y-6">
            <div className="flex items-center justify-between">
              <div>
                <label className="text-sm font-medium text-text-secondary">Text-to-Speech</label>
                <p className="text-xs text-text-secondary mt-1">Read translations aloud automatically.</p>
              </div>
              <button 
                onClick={() => setSpeechEnabled(!speechEnabled)}
                className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors ${speechEnabled ? 'bg-accent-primary' : 'bg-gray-600'}`}
              >
                <span className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${speechEnabled ? 'translate-x-6' : 'translate-x-1'}`} />
              </button>
            </div>

            <div className="pt-4 border-t border-gray-800">
              <label className="block text-sm font-medium text-text-secondary mb-2">Theme</label>
              <div className="flex space-x-2">
                <button className="px-4 py-2 bg-accent-primary text-white rounded-lg text-sm font-medium cursor-default">Dark</button>
                <button disabled className="px-4 py-2 bg-bg-tertiary text-gray-500 rounded-lg text-sm font-medium cursor-not-allowed">Light (Pro)</button>
              </div>
            </div>
          </div>
        </Card>
      </div>
    </div>
  );
}
