export const NUM_LANDMARKS_PER_HAND = 21;
export const COORDINATES_PER_LANDMARK = 3;
export const FEATURES_PER_HAND = NUM_LANDMARKS_PER_HAND * COORDINATES_PER_LANDMARK;
export const TOTAL_FEATURES = FEATURES_PER_HAND * 2;
export const EPSILON = 1e-6;

/**
 * Normalizes a single hand's landmarks (wrist-relative translation & scale normalization).
 * 
 * @param {Array<{x,y,z}>} landmarks - Array of 21 landmark objects.
 * @returns {Array<number>} - Array of 63 normalized coordinates.
 */
export function normalizeHandLandmarks(landmarks) {
  if (!landmarks || landmarks.length !== NUM_LANDMARKS_PER_HAND) {
    return new Array(FEATURES_PER_HAND).fill(0);
  }

  const wrist = landmarks[0];
  const middleMCP = landmarks[9];

  // 1. Calculate scale (distance from wrist to middle finger MCP)
  const dx = middleMCP.x - wrist.x;
  const dy = middleMCP.y - wrist.y;
  const dz = middleMCP.z - wrist.z;
  let scale = Math.sqrt(dx * dx + dy * dy + dz * dz);
  
  // Prevent division by zero
  if (scale < EPSILON) {
    scale = 1.0;
  }

  const normalized = new Float32Array(FEATURES_PER_HAND);
  let idx = 0;

  // 2. Translate and scale
  for (let i = 0; i < NUM_LANDMARKS_PER_HAND; i++) {
    const landmark = landmarks[i];
    normalized[idx++] = (landmark.x - wrist.x) / scale;
    normalized[idx++] = (landmark.y - wrist.y) / scale;
    normalized[idx++] = (landmark.z - wrist.z) / scale;
  }

  return Array.from(normalized);
}

/**
 * Processes MediaPipe results into a canonical 126-D feature vector.
 * 
 * @param {Object} results - MediaPipe HandLandmarker results.
 * @returns {Float32Array} - Exactly 126 features.
 */
export function processLandmarks(results) {
  const featureVector = new Float32Array(TOTAL_FEATURES);
  featureVector.fill(0);

  if (!results || !results.landmarks || results.landmarks.length === 0) {
    return featureVector;
  }

  let leftHandLandmarks = null;
  let rightHandLandmarks = null;

  // Extract left and right hands explicitly using handedness
  // Note: Depending on camera mirroring, MediaPipe might label the visual left as 'Right'
  // But we strictly follow its classification here for data consistency.
  for (let i = 0; i < results.landmarks.length; i++) {
    const handedness = results.handednesses[i][0].categoryName;
    if (handedness === 'Left' && !leftHandLandmarks) {
      leftHandLandmarks = results.landmarks[i];
    } else if (handedness === 'Right' && !rightHandLandmarks) {
      rightHandLandmarks = results.landmarks[i];
    }
  }

  // Normalize and insert left hand (indices 0 to 62)
  if (leftHandLandmarks) {
    const leftNormalized = normalizeHandLandmarks(leftHandLandmarks);
    featureVector.set(leftNormalized, 0);
  }

  // Normalize and insert right hand (indices 63 to 125)
  if (rightHandLandmarks) {
    const rightNormalized = normalizeHandLandmarks(rightHandLandmarks);
    featureVector.set(rightNormalized, FEATURES_PER_HAND);
  }

  return featureVector;
}
