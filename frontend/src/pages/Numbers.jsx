import React from 'react';
import Card from '../components/common/Card';

export default function Numbers() {
  return (
    <div className="flex flex-col h-full gap-6">
      <h2 className="text-2xl font-bold">Number Translator</h2>
      
      <Card title="Number Recognition" className="max-w-2xl">
        <div className="mb-6 p-4 bg-bg-tertiary rounded-lg border border-gray-800">
          <p className="text-text-secondary text-sm">
            Dataset/model classes will appear here after the verified ISL number dataset is configured.
          </p>
        </div>

        <div>
          <span className="text-sm text-text-secondary">Model Status: </span>
          <span className="text-sm font-medium text-accent-warning">Not trained</span>
        </div>
      </Card>
    </div>
  );
}
