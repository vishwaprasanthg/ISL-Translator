import React from 'react';
import Card from '../components/common/Card';
import { MODEL_STATUS } from '../utils/constants';

export default function Alphabet() {
  const alphabet = Array.from({ length: 26 }, (_, i) => String.fromCharCode(65 + i));

  return (
    <div className="flex flex-col h-full gap-6">
      <h2 className="text-2xl font-bold">Alphabet Translator</h2>
      
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6 flex-1">
        <Card title="ISL Alphabet Reference" className="flex flex-col">
          <div className="grid grid-cols-6 sm:grid-cols-8 gap-2 mt-4">
            {alphabet.map((letter) => (
              <div 
                key={letter} 
                className="aspect-square flex items-center justify-center bg-bg-tertiary rounded-lg text-lg font-bold border border-gray-800"
              >
                {letter}
              </div>
            ))}
          </div>
        </Card>

        <Card title="Alphabet Recognition" className="flex flex-col">
          <div className="mb-4">
            <span className="text-sm text-text-secondary">Model status: </span>
            <span className="text-sm font-medium text-accent-warning">Not trained</span>
          </div>
          
          <div className="flex-1 flex flex-col justify-end">
            <div className="grid grid-cols-2 gap-3 mt-8">
              <button disabled className="py-3 px-4 bg-bg-tertiary text-gray-500 rounded-lg cursor-not-allowed font-medium text-sm">Clear</button>
              <button disabled className="py-3 px-4 bg-bg-tertiary text-gray-500 rounded-lg cursor-not-allowed font-medium text-sm">Backspace</button>
              <button disabled className="py-3 px-4 bg-bg-tertiary text-gray-500 rounded-lg cursor-not-allowed font-medium text-sm">Space</button>
              <button disabled className="py-3 px-4 bg-accent-primary/50 text-white/50 rounded-lg cursor-not-allowed font-medium text-sm">Confirm Word</button>
            </div>
          </div>
        </Card>
      </div>
    </div>
  );
}
