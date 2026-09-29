import React from 'react';
import Card from '../components/common/Card';
import { APP_NAME } from '../utils/constants';

export default function About() {
  const technologies = [
    'React + Vite',
    'Tailwind CSS v4',
    'MediaPipe',
    'FastAPI',
    'PyTorch',
    'LSTM',
    '1D CNN',
    'WebSocket',
    'SQLite',
    'Web Speech API'
  ];

  return (
    <div className="flex flex-col h-full gap-6 max-w-4xl">
      <h2 className="text-2xl font-bold">About Project</h2>
      
      <Card title={APP_NAME}>
        <p className="text-text-secondary leading-relaxed mb-6">
          A real-time AI-powered system designed to recognize Indian Sign Language gestures through a webcam 
          and convert recognized signs into English text and speech.
        </p>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-8 mb-6">
          <div>
            <h4 className="text-sm font-medium text-text-primary mb-3">Project Status</h4>
            <div className="space-y-2">
              <div className="flex items-center space-x-2">
                <span className="w-2 h-2 rounded-full bg-accent-warning"></span>
                <span className="text-sm text-text-secondary">Development</span>
              </div>
              <div className="text-sm font-mono bg-bg-tertiary inline-block px-2 py-1 rounded">
                Current Phase: Phase 2
              </div>
            </div>
          </div>

          <div>
            <h4 className="text-sm font-medium text-text-primary mb-3">Architecture</h4>
            <div className="text-xs text-text-secondary font-mono bg-bg-tertiary p-3 rounded-lg border border-gray-800">
              Webcam<br/>
              ↓<br/>
              MediaPipe<br/>
              ↓<br/>
              Hand Landmarks<br/>
              ↓<br/>
              AI Recognition<br/>
              ↓<br/>
              English Text<br/>
              ↓<br/>
              Speech
            </div>
          </div>
        </div>

        <div>
          <h4 className="text-sm font-medium text-text-primary mb-3">Technologies</h4>
          <div className="flex flex-wrap gap-2">
            {technologies.map(tech => (
              <span key={tech} className="px-3 py-1 bg-accent-primary/10 text-accent-primary border border-accent-primary/20 rounded-full text-xs font-medium">
                {tech}
              </span>
            ))}
          </div>
        </div>
      </Card>
    </div>
  );
}
