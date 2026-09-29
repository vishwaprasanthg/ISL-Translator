import React from 'react';
import Card from '../components/common/Card';

export default function Vocabulary() {
  return (
    <div className="flex flex-col h-full gap-6">
      <h2 className="text-2xl font-bold">Vocabulary Translator</h2>
      
      <Card title="Vocabulary Recognition" className="max-w-2xl">
        <div className="mb-6 space-y-2">
          <p className="text-text-secondary">Sequence-based recognition</p>
          <p className="text-text-secondary text-sm font-mono bg-bg-tertiary inline-block px-2 py-1 rounded">
            30-frame temporal model
          </p>
        </div>

        <div className="space-y-4 pt-4 border-t border-gray-800">
          <div>
            <span className="text-sm text-text-secondary">Model: </span>
            <span className="text-sm font-medium text-accent-warning">Not trained</span>
          </div>
          
          <div>
            <span className="text-sm text-text-secondary">Sequence: </span>
            <span className="text-sm font-mono text-text-primary">0 / 30 frames</span>
            <div className="w-full h-2 bg-bg-tertiary rounded-full mt-2 overflow-hidden">
              <div className="h-full bg-accent-primary w-0"></div>
            </div>
          </div>
        </div>
      </Card>
    </div>
  );
}
