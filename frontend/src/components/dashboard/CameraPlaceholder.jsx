import React from 'react';
import { Camera } from 'lucide-react';
import Card from '../common/Card';

export default function CameraPlaceholder() {
  return (
    <Card className="flex-1 flex flex-col items-center justify-center min-h-[400px]">
      <Camera className="w-16 h-16 text-gray-600 mb-6" />
      <h2 className="text-xl font-medium text-text-primary mb-2">LIVE TRANSLATOR</h2>
      <p className="text-text-secondary text-center max-w-md">
        Camera recognition will be enabled in Phase 3.
      </p>
    </Card>
  );
}
