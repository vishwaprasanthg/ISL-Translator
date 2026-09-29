import { useState, useEffect, useRef, useCallback } from 'react';
import { FilesetResolver, HandLandmarker } from '@mediapipe/tasks-vision';
import { processLandmarks } from '../services/landmarkProcessor';

export function useMediaPipe() {
  const [isReady, setIsReady] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [results, setResults] = useState(null);
  const [normalizedFeatures, setNormalizedFeatures] = useState(null);
  
  const [metrics, setMetrics] = useState({
    fps: 0,
    inferenceMs: 0,
    normalizationMs: 0,
    wsSendMs: 0,
    totalLatencyMs: 0,
    delegate: "CPU",
    resolution: "0x0"
  });

  const handLandmarkerRef = useRef(null);
  const requestRef = useRef(null);
  const lastVideoTimeRef = useRef(-1);
  const lastSendTimeRef = useRef(0);
  const lastInferenceTimeRef = useRef(0);
  const fpsFrameCountRef = useRef(0);
  const fpsLastTimeRef = useRef(performance.now());
  const isInferencingRef = useRef(false);

  useEffect(() => {
    let isMounted = true;

    async function initMediaPipe() {
      if (handLandmarkerRef.current) return;
      setIsLoading(true);
      setError(null);

      try {
        const vision = await FilesetResolver.forVisionTasks(
          "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm"
        );
        
        let delegate = "GPU";
        let handLandmarker;
        
        try {
          handLandmarker = await HandLandmarker.createFromOptions(vision, {
            baseOptions: {
              modelAssetPath: "/models/hand_landmarker.task",
              delegate: "GPU"
            },
            runningMode: "VIDEO",
            numHands: 2,
            minHandDetectionConfidence: 0.5,
            minHandPresenceConfidence: 0.5,
            minTrackingConfidence: 0.5,
          });
        } catch (gpuErr) {
          console.warn("GPU delegate failed, falling back to CPU", gpuErr);
          delegate = "CPU";
          handLandmarker = await HandLandmarker.createFromOptions(vision, {
            baseOptions: {
              modelAssetPath: "/models/hand_landmarker.task",
              delegate: "CPU"
            },
            runningMode: "VIDEO",
            numHands: 2,
            minHandDetectionConfidence: 0.5,
            minHandPresenceConfidence: 0.5,
            minTrackingConfidence: 0.5,
          });
        }

        if (isMounted) {
          handLandmarkerRef.current = handLandmarker;
          setMetrics(m => ({ ...m, delegate }));
          setIsReady(true);
          setIsLoading(false);
        }
      } catch (err) {
        console.error("MediaPipe initialization failed:", err);
        if (isMounted) {
          setError("MediaPipe initialization failed. Please check your browser/device and try again.");
          setIsLoading(false);
        }
      }
    }

    initMediaPipe();

    return () => {
      isMounted = false;
      if (requestRef.current) {
        cancelAnimationFrame(requestRef.current);
      }
      if (handLandmarkerRef.current) {
        handLandmarkerRef.current.close();
      }
    };
  }, []);

  const detectFromVideo = useCallback((videoElement, onFeaturesReady) => {
    if (!videoElement || !handLandmarkerRef.current || !isReady) return;

    if (videoElement.readyState >= 2 && videoElement.videoWidth > 0) {
      // Fix IMAGE_DIMENSIONS warning by ensuring element attributes exist
      if (videoElement.width !== videoElement.videoWidth) {
        videoElement.width = videoElement.videoWidth;
        videoElement.height = videoElement.videoHeight;
        setMetrics(m => ({ ...m, resolution: `${videoElement.videoWidth}x${videoElement.videoHeight}` }));
      }

      const startTimeMs = performance.now();
      
      // Throttle Inference to ~25 FPS to save CPU (40ms interval)
      if (startTimeMs - lastInferenceTimeRef.current >= 40 && !isInferencingRef.current) {
        lastInferenceTimeRef.current = startTimeMs;
        isInferencingRef.current = true;
          
        try {
            const pipelineStart = performance.now();

            // 1. MediaPipe Inference
            const inferenceStart = performance.now();
            const detectionResult = handLandmarkerRef.current.detectForVideo(videoElement, startTimeMs);
            const inferenceEnd = performance.now();
            
            setResults(detectionResult);
            
            // 2. Normalization
            const normStart = performance.now();
            const features = processLandmarks(detectionResult);
            const normEnd = performance.now();
            
            setNormalizedFeatures(features);

            // 3. WebSocket Transmission (Throttled to ~20 FPS = 50ms)
            let wsSendEnd = normEnd;
            if (startTimeMs - lastSendTimeRef.current >= 50) { 
              lastSendTimeRef.current = startTimeMs;
              if (onFeaturesReady) {
                onFeaturesReady(features, detectionResult);
              }
              wsSendEnd = performance.now();
            }

            const pipelineEnd = performance.now();

            // Calculate Metrics
            fpsFrameCountRef.current += 1;
            
            const currentInferenceMs = Math.round(inferenceEnd - inferenceStart);
            const currentNormalizationMs = Math.round(normEnd - normStart);
            const currentWsSendMs = Math.round(wsSendEnd - normEnd);
            const currentTotalLatencyMs = Math.round(pipelineEnd - pipelineStart);

            if (startTimeMs - fpsLastTimeRef.current >= 1000) {
              setMetrics(m => ({
                ...m,
                fps: Math.round((fpsFrameCountRef.current * 1000) / (startTimeMs - fpsLastTimeRef.current)),
                inferenceMs: currentInferenceMs,
                normalizationMs: currentNormalizationMs,
                wsSendMs: currentWsSendMs,
                totalLatencyMs: currentTotalLatencyMs
              }));
              fpsFrameCountRef.current = 0;
              fpsLastTimeRef.current = startTimeMs;

              // Diagnostic: Log raw hand results
              if (detectionResult.handednesses && detectionResult.handednesses.length > 0) {
                 console.log(`--- Hand Diagnostic ---`);
                 console.log(`Detected Hands: ${detectionResult.handednesses.length}`);
                 detectionResult.handednesses.forEach((hand, idx) => {
                   console.log(`Hand ${idx}: category: ${hand[0].categoryName}, score: ${hand[0].score.toFixed(2)}, landmarks: ${detectionResult.landmarks[idx].length}`);
                 });
              }
            } else if (metrics.inferenceMs === 0) {
              // Update immediately on the very first frame to avoid "--" showing up for a second
              setMetrics(m => ({
                ...m,
                inferenceMs: currentInferenceMs,
                normalizationMs: currentNormalizationMs,
                wsSendMs: currentWsSendMs,
                totalLatencyMs: currentTotalLatencyMs
              }));
            }
          } finally {
            isInferencingRef.current = false;
          }
      }
    }
  }, [isReady]);
  const startDetectionLoop = useCallback((videoElement, onFeaturesReady) => {
    if (!isReady || !videoElement) return;

    // Prevent duplicate loops
    if (requestRef.current) {
      cancelAnimationFrame(requestRef.current);
    }

    const loop = () => {
      detectFromVideo(videoElement, onFeaturesReady);
      requestRef.current = requestAnimationFrame(loop);
    };

    requestRef.current = requestAnimationFrame(loop);
  }, [isReady, detectFromVideo]);

  const stopDetectionLoop = useCallback(() => {
    if (requestRef.current) {
      cancelAnimationFrame(requestRef.current);
      requestRef.current = null;
    }
    setResults(null);
    setNormalizedFeatures(null);
    lastVideoTimeRef.current = -1;
  }, []);

  return {
    isReady,
    isLoading,
    error,
    results,
    normalizedFeatures,
    metrics,
    startDetectionLoop,
    stopDetectionLoop
  };
}
