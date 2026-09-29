import React, { useEffect, useState, useRef } from 'react';
import { Camera, AlertCircle, Loader2 } from 'lucide-react';
import Card from '../common/Card';
import CameraControls from './CameraControls';
import HandOverlay from './HandOverlay';
import { useCamera } from '../../hooks/useCamera';
import { useMediaPipe } from '../../hooks/useMediaPipe';
import { useWebSocket } from '../../hooks/useWebSocket';
import { useAppContext } from '../../context/AppContext';

export default function CameraFeed() {
  const { setCameraStatus, setModelStatus, selectedMode, setPrediction, setMetrics } = useAppContext();
  
  const {
    videoRef,
    isCameraActive,
    isLoading: isCameraLoading,
    error: cameraError,
    cameraDevices,
    selectedDeviceId,
    startCamera,
    stopCamera,
    switchCamera
  } = useCamera();

  const {
    connect: connectWs,
    disconnect: disconnectWs,
    sendFeatures,
    connectionStatus: wsStatus,
    lastMessage: wsLastMessage,
    error: wsError
  } = useWebSocket();

  const {
    isReady: isMediaPipeReady,
    isLoading: isMediaPipeLoading,
    error: mediaPipeError,
    results,
    normalizedFeatures,
    metrics,
    startDetectionLoop,
    stopDetectionLoop
  } = useMediaPipe();

  const [videoDimensions, setVideoDimensions] = useState({ width: 640, height: 480 });
  const featureCountRef = useRef(0);

  // Connect WebSocket when camera becomes active
  useEffect(() => {
    if (isCameraActive) {
      connectWs();
    } else {
      disconnectWs();
      if (setModelStatus) setModelStatus('not_trained');
    }
    
    return () => disconnectWs();
  }, [isCameraActive, connectWs, disconnectWs, setModelStatus]);

  // Sync WebSocket connection status with model status
  useEffect(() => {
    if (wsStatus === 'connected' && setModelStatus) {
      setModelStatus('ready');
    } else if (wsStatus === 'error' && setModelStatus) {
      setModelStatus('error');
    }
  }, [wsStatus, setModelStatus]);

  // Update global context so TopBar can reflect the status
  useEffect(() => {
    if (setCameraStatus) {
      if (cameraError || mediaPipeError || wsError) setCameraStatus('error');
      else if (isCameraLoading || isMediaPipeLoading || wsStatus === 'connecting') setCameraStatus('loading');
      else if (isCameraActive) setCameraStatus('active');
      else setCameraStatus('ready');
    }
  }, [isCameraActive, isCameraLoading, cameraError, isMediaPipeLoading, mediaPipeError, wsError, wsStatus, setCameraStatus]);

  // Handle incoming websocket predictions and acks
  useEffect(() => {
    if (wsLastMessage?.type === 'prediction' && setPrediction) {
      setPrediction(wsLastMessage);
    } else if (wsLastMessage?.type === 'ack' && wsLastMessage.rtt !== undefined && setMetrics) {
      setMetrics(prev => ({ ...prev, wsRttMs: wsLastMessage.rtt }));
    }
  }, [wsLastMessage, setPrediction, setMetrics]);

  // Sync MediaPipe metrics with global context
  useEffect(() => {
    if (setMetrics && metrics) {
      setMetrics(prev => ({
        ...prev,
        fps: metrics.fps,
        inferenceMs: metrics.inferenceMs,
        normalizationMs: metrics.normalizationMs,
        wsSendMs: metrics.wsSendMs,
        totalLatencyMs: metrics.totalLatencyMs,
        resolution: metrics.resolution,
        delegate: metrics.delegate,
      }));
    }
  }, [metrics, setMetrics]);

  // Handle MediaPipe lifecycle when camera state changes
  useEffect(() => {
    if (isCameraActive && isMediaPipeReady && videoRef.current) {
      const handleVideoPlay = () => {
        setVideoDimensions({
          width: videoRef.current.videoWidth,
          height: videoRef.current.videoHeight
        });
        
        // Pass the websocket sender to the loop
        startDetectionLoop(videoRef.current, (features, detectionResult) => {
          // sendFeatures handles connection check internally
          if (detectionResult?.landmarks?.length > 0) {
            const success = sendFeatures(features, selectedMode || 'alphabet');
            if (success) featureCountRef.current += 1;
          }
        });
      };

      videoRef.current.addEventListener('loadeddata', handleVideoPlay);
      
      if (videoRef.current.readyState >= 2) {
        handleVideoPlay();
      }

      return () => {
        if (videoRef.current) {
          videoRef.current.removeEventListener('loadeddata', handleVideoPlay);
        }
        stopDetectionLoop();
      };
    } else {
      stopDetectionLoop();
    }
  }, [isCameraActive, isMediaPipeReady, startDetectionLoop, stopDetectionLoop, videoRef, sendFeatures]);

  const error = cameraError || mediaPipeError;
  const isLoading = isCameraLoading || isMediaPipeLoading;
  
  const numHands = results?.landmarks?.length || 0;

  return (
    <Card className="flex flex-col h-full min-h-[400px]">
      <div className="flex justify-between items-center mb-4">
        <h2 className="text-xl font-medium text-text-primary uppercase tracking-wide">Live Camera</h2>
        
        <div className="flex items-center space-x-6 text-sm">
          <div className="flex items-center space-x-2">
            <span className="text-text-secondary">MediaPipe:</span>
            {mediaPipeError ? (
              <span className="text-accent-danger font-medium">Error</span>
            ) : isMediaPipeLoading ? (
              <span className="text-accent-warning font-medium">Initializing...</span>
            ) : isMediaPipeReady ? (
              <span className="text-accent-success font-medium">Ready</span>
            ) : (
              <span className="text-gray-400 font-medium">Wait...</span>
            )}
          </div>
          
          <div className="flex items-center space-x-2">
            <span className="text-text-secondary">Hands:</span>
            <span className="text-text-primary font-mono bg-bg-tertiary px-2 py-0.5 rounded">{numHands}</span>
          </div>

          <div className="flex items-center space-x-2">
            <span className="text-text-secondary">Camera:</span>
            {isCameraActive ? (
              <span className="text-accent-success font-medium flex items-center gap-1.5">
                <span className="w-2 h-2 rounded-full bg-accent-success animate-pulse"></span>
                Active
              </span>
            ) : (
              <span className="text-gray-400 font-medium flex items-center gap-1.5">
                <span className="w-2 h-2 rounded-full bg-gray-500"></span>
                Ready
              </span>
            )}
          </div>
        </div>
      </div>

      <div className="relative w-full aspect-video bg-black rounded-lg overflow-hidden border border-gray-800 flex items-center justify-center">
        {/* The actual video element */}
        <video
          ref={videoRef}
          autoPlay
          playsInline
          muted
          className={`absolute inset-0 w-full h-full object-cover transition-opacity duration-300 ${isCameraActive && !isLoading ? 'opacity-100' : 'opacity-0'} scale-x-[-1]`}
        />
        
        {/* The Hand Skeleton Canvas */}
        {isCameraActive && !isLoading && (
          <HandOverlay 
            results={results} 
            width={videoDimensions.width} 
            height={videoDimensions.height} 
            isMirrored={true} 
          />
        )}

        {/* States overlay */}
        {(!isCameraActive || isLoading || error) && (
          <div className="absolute inset-0 flex flex-col items-center justify-center bg-bg-secondary/90 z-20 p-6 text-center">
            {error ? (
              <>
                <AlertCircle className="w-12 h-12 text-accent-danger mb-4" />
                <p className="text-text-primary font-medium max-w-md">{error}</p>
              </>
            ) : isLoading ? (
              <>
                <Loader2 className="w-12 h-12 text-accent-primary animate-spin mb-4" />
                <p className="text-text-primary font-medium">
                  {isMediaPipeLoading ? 'Initializing hand detection...' : 'Requesting camera permission...'}
                </p>
              </>
            ) : (
              <>
                <Camera className="w-16 h-16 text-gray-600 mb-6" />
                <p className="text-text-secondary max-w-sm">
                  Click "Start Camera" to enable the live video preview and hand tracking.
                </p>
              </>
            )}
          </div>
        )}
      </div>

      <div className="mt-4 pt-4 border-t border-gray-800 flex justify-between items-start">
        <CameraControls 
          isCameraActive={isCameraActive}
          isLoading={isLoading}
          cameraDevices={cameraDevices}
          selectedDeviceId={selectedDeviceId}
          onStart={() => startCamera()}
          onStop={stopCamera}
          onSwitchDevice={switchCamera}
        />
        
        {/* Debug info (Development only) */}
        {isCameraActive && !isLoading && (
          <div className="hidden md:flex flex-col items-end text-xs text-text-secondary bg-bg-tertiary px-3 py-2 rounded-lg border border-gray-700 ml-4">
            <span className="font-medium text-text-primary mb-1">Performance Pipeline</span>
            <span>Camera: {metrics?.resolution || "0x0"}</span>
            <span>Delegate: {metrics?.delegate || "CPU"}</span>
            <span>MediaPipe FPS: {metrics?.fps || 0}</span>
            <span>MediaPipe Inference: {metrics?.inferenceMs || 0} ms</span>
            <span>Normalization: {metrics?.normalizationMs || 0} ms</span>
            <span>WS RTT: {wsLastMessage?.rtt !== undefined ? Math.round(wsLastMessage.rtt) : '--'} ms</span>
            <span>Total Latency: {metrics?.totalLatencyMs || 0} ms</span>
            <span>Hands Detected: {results?.landmarks?.length || 0}/2</span>
          </div>
        )}
      </div>
    </Card>
  );
}
