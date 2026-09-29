import React, { useEffect, useRef } from 'react';

const HAND_CONNECTIONS = [
  // Thumb
  [0, 1], [1, 2], [2, 3], [3, 4],
  // Index
  [0, 5], [5, 6], [6, 7], [7, 8],
  // Middle
  [5, 9], [9, 10], [10, 11], [11, 12],
  // Ring
  [9, 13], [13, 14], [14, 15], [15, 16],
  // Pinky
  [13, 17], [17, 18], [18, 19], [19, 20],
  [0, 17]
];

export default function HandOverlay({ results, width, height, isMirrored }) {
  const canvasRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    if (!results || !results.landmarks || results.landmarks.length === 0) {
      return;
    }

    results.landmarks.forEach((landmarks, index) => {
      // Determine hand color based on handedness
      // Note: Because video is mirrored, handedness from MediaPipe is usually reversed visually 
      // but for Phase 4 we just use what it gives us. 
      const isRightHand = results.handednesses[index][0].categoryName === 'Right';
      
      const landmarkColor = isRightHand ? '#8b5cf6' : '#10b981'; // Secondary purple : Success green
      const connectionColor = isRightHand ? 'rgba(139, 92, 246, 0.6)' : 'rgba(16, 185, 129, 0.6)';
      
      ctx.lineWidth = 3;
      
      // Draw Connections
      HAND_CONNECTIONS.forEach(([i, j]) => {
        const point1 = landmarks[i];
        const point2 = landmarks[j];

        ctx.beginPath();
        ctx.moveTo(point1.x * canvas.width, point1.y * canvas.height);
        ctx.lineTo(point2.x * canvas.width, point2.y * canvas.height);
        ctx.strokeStyle = connectionColor;
        ctx.stroke();
      });

      // Draw Landmarks
      landmarks.forEach((landmark) => {
        ctx.beginPath();
        ctx.arc(landmark.x * canvas.width, landmark.y * canvas.height, 4, 0, 2 * Math.PI);
        ctx.fillStyle = landmarkColor;
        ctx.fill();
        ctx.strokeStyle = '#ffffff';
        ctx.lineWidth = 1.5;
        ctx.stroke();
      });
    });

  }, [results, width, height, isMirrored]);

  return (
    <canvas
      ref={canvasRef}
      width={width}
      height={height}
      className={`absolute inset-0 w-full h-full pointer-events-none ${isMirrored ? 'scale-x-[-1]' : ''}`}
    />
  );
}
