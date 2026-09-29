import { useState, useEffect, useRef, useCallback } from 'react';
import { getCameraErrorMessage } from '../utils/cameraUtils';

export function useCamera() {
  const [stream, setStream] = useState(null);
  const [isCameraActive, setIsCameraActive] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [cameraDevices, setCameraDevices] = useState([]);
  const [selectedDeviceId, setSelectedDeviceId] = useState('');
  
  const videoRef = useRef(null);

  // Clean up stream tracks
  const stopCamera = useCallback(() => {
    if (stream) {
      stream.getTracks().forEach(track => track.stop());
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
    setStream(null);
    setIsCameraActive(false);
  }, [stream]);

  // Enumerate cameras
  const getCameras = useCallback(async () => {
    try {
      const devices = await navigator.mediaDevices.enumerateDevices();
      const videoDevices = devices.filter(device => device.kind === 'videoinput');
      
      // Assign friendly names if labels are missing (often happens before permission is granted)
      const formattedDevices = videoDevices.map((device, index) => ({
        deviceId: device.deviceId,
        label: device.label || `Camera ${index + 1}`
      }));
      
      setCameraDevices(formattedDevices);
      
      // If we have devices and no selected device, pick the first one
      if (formattedDevices.length > 0 && !selectedDeviceId) {
        setSelectedDeviceId(formattedDevices[0].deviceId);
      }
    } catch (err) {
      console.error("Error enumerating devices:", err);
    }
  }, [selectedDeviceId]);

  // Start camera
  const startCamera = useCallback(async (deviceId = selectedDeviceId) => {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      setError("Your browser does not support camera access.");
      return;
    }

    setIsLoading(true);
    setError(null);
    
    // Stop any existing stream first
    if (stream) {
      stream.getTracks().forEach(track => track.stop());
    }

    try {
      const constraints = {
        video: {
          width: { ideal: 640, max: 640 },
          height: { ideal: 480, max: 480 },
          frameRate: { ideal: 30, max: 30 },
          ...(deviceId ? { deviceId: { exact: deviceId } } : {})
        },
        audio: false
      };

      const newStream = await navigator.mediaDevices.getUserMedia(constraints);
      
      if (videoRef.current) {
        videoRef.current.srcObject = newStream;
      }
      
      setStream(newStream);
      setIsCameraActive(true);
      
      // Refresh devices to get actual labels now that permission is granted
      await getCameras();
      
    } catch (err) {
      setError(getCameraErrorMessage(err));
      setIsCameraActive(false);
    } finally {
      setIsLoading(false);
    }
  }, [selectedDeviceId, stream, getCameras]);

  // Handle switching camera
  const switchCamera = useCallback((deviceId) => {
    setSelectedDeviceId(deviceId);
    if (isCameraActive) {
      startCamera(deviceId);
    }
  }, [isCameraActive, startCamera]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (stream) {
        stream.getTracks().forEach(track => track.stop());
      }
    };
  }, [stream]);

  // Initial device enumeration
  useEffect(() => {
    getCameras();
    
    // Listen for device changes (e.g., plugging in a webcam)
    navigator.mediaDevices?.addEventListener('devicechange', getCameras);
    return () => {
      navigator.mediaDevices?.removeEventListener('devicechange', getCameras);
    };
  }, [getCameras]);

  return {
    videoRef,
    stream,
    isCameraActive,
    isLoading,
    error,
    cameraDevices,
    selectedDeviceId,
    startCamera,
    stopCamera,
    switchCamera
  };
}
