import React from 'react';
import { Camera, Video, VideoOff } from 'lucide-react';

export default function CameraControls({
  isCameraActive,
  isLoading,
  cameraDevices,
  selectedDeviceId,
  onStart,
  onStop,
  onSwitchDevice
}) {
  return (
    <div className="flex flex-wrap items-center justify-between gap-4">
      <div className="flex items-center space-x-3">
        {!isCameraActive ? (
          <button
            onClick={onStart}
            disabled={isLoading}
            className="flex items-center space-x-2 px-5 py-2.5 bg-accent-primary hover:bg-accent-primary/90 text-white rounded-lg font-medium transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
          >
            <Video className="w-4 h-4" />
            <span>Start Camera</span>
          </button>
        ) : (
          <button
            onClick={onStop}
            className="flex items-center space-x-2 px-5 py-2.5 bg-accent-danger/20 text-accent-danger hover:bg-accent-danger/30 border border-accent-danger/30 rounded-lg font-medium transition-colors"
          >
            <VideoOff className="w-4 h-4" />
            <span>Stop Camera</span>
          </button>
        )}
      </div>

      {cameraDevices.length > 1 && (
        <div className="flex items-center space-x-2 bg-bg-tertiary px-3 py-1.5 rounded-lg border border-gray-700">
          <Camera className="w-4 h-4 text-text-secondary" />
          <select
            value={selectedDeviceId}
            onChange={(e) => onSwitchDevice(e.target.value)}
            disabled={isLoading}
            className="bg-transparent text-sm text-text-primary border-none focus:ring-0 cursor-pointer outline-none max-w-[200px] truncate disabled:opacity-50"
          >
            {cameraDevices.map((device) => (
              <option key={device.deviceId} value={device.deviceId} className="bg-bg-secondary text-text-primary">
                {device.label}
              </option>
            ))}
          </select>
        </div>
      )}
    </div>
  );
}
