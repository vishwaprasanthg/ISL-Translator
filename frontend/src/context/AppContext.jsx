import React, { createContext, useContext, useState } from 'react';
import { MODES, MODEL_STATUS, DEFAULT_CONFIDENCE_THRESHOLD } from '../utils/constants';

const AppContext = createContext();

export function AppProvider({ children }) {
  const [selectedMode, setSelectedMode] = useState(MODES.ALPHABET);
  const [modelStatus, setModelStatus] = useState(MODEL_STATUS.NOT_TRAINED);
  const [confidenceThreshold, setConfidenceThreshold] = useState(DEFAULT_CONFIDENCE_THRESHOLD);
  const [speechEnabled, setSpeechEnabled] = useState(true);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [cameraStatus, setCameraStatus] = useState('ready');
  const [prediction, setPrediction] = useState(null);
  const [metrics, setMetrics] = useState({
    fps: 0,
    inferenceMs: 0,
    normalizationMs: 0,
    wsSendMs: 0,
    wsRttMs: 0,
    backendInferenceMs: 0,
    totalLatencyMs: 0
  });

  const value = {
    selectedMode,
    setSelectedMode,
    modelStatus,
    setModelStatus,
    confidenceThreshold,
    setConfidenceThreshold,
    speechEnabled,
    setSpeechEnabled,
    sidebarOpen,
    setSidebarOpen,
    cameraStatus,
    setCameraStatus,
    prediction,
    setPrediction,
    metrics,
    setMetrics,
  };

  return <AppContext.Provider value={value}>{children}</AppContext.Provider>;
}

export function useAppContext() {
  const context = useContext(AppContext);
  if (!context) {
    throw new Error('useAppContext must be used within an AppProvider');
  }
  return context;
}
