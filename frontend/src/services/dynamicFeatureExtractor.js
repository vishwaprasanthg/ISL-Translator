import {
  NUM_LANDMARKS_PER_HAND,
  COORDINATES_PER_LANDMARK,
  FEATURES_PER_HAND,
  TOTAL_FEATURES as STATIC_FEATURE_DIM,
  EPSILON,
  normalizeHandLandmarks
} from './landmarkProcessor';

export const GLOBAL_FEATURE_DIM = 14;
export const DYNAMIC_FEATURE_DIM = STATIC_FEATURE_DIM + GLOBAL_FEATURE_DIM; // 140

/**
 * Extracts the canonical 140-D dynamic feature vector:
 *   [0..125]   : Existing 126-D normalised handshape features
 *   [126..128] : Left wrist camera position (x, y, z)
 *   [129]      : Left hand apparent scale (distance wrist to middle MCP)
 *   [130..132] : Right wrist camera position (x, y, z)
 *   [133]      : Right hand apparent scale (distance wrist to middle MCP)
 *   [134..136] : Inter-wrist displacement vector (right - left)
 *   [137]      : Inter-wrist Euclidean distance
 *   [138..139] : Presence mask [left_present, right_present] in {0.0, 1.0}
 *
 * @param {Object} results - MediaPipe HandLandmarker results
 * @returns {Float32Array} - Exactly 140 features
 */
export function extractDynamicFeatures(results) {
  const vector = new Float32Array(DYNAMIC_FEATURE_DIM);
  vector.fill(0);

  if (!results || !results.landmarks || results.landmarks.length === 0) {
    return vector;
  }

  let leftHandLandmarks = null;
  let rightHandLandmarks = null;

  for (let i = 0; i < results.landmarks.length; i++) {
    const handedness = results.handednesses[i][0].categoryName;
    if (handedness === 'Left' && !leftHandLandmarks) {
      leftHandLandmarks = results.landmarks[i];
    } else if (handedness === 'Right' && !rightHandLandmarks) {
      rightHandLandmarks = results.landmarks[i];
    }
  }

  let leftWrist = null;
  let leftScale = 0;
  if (leftHandLandmarks && leftHandLandmarks.length === NUM_LANDMARKS_PER_HAND) {
    const leftNorm = normalizeHandLandmarks(leftHandLandmarks);
    vector.set(leftNorm, 0);

    const w = leftHandLandmarks[0];
    const m = leftHandLandmarks[9];
    const dx = m.x - w.x;
    const dy = m.y - w.y;
    const dz = m.z - w.z;
    leftScale = Math.sqrt(dx * dx + dy * dy + dz * dz);
    leftWrist = w;

    vector[126] = w.x;
    vector[127] = w.y;
    vector[128] = w.z;
    vector[129] = leftScale;
    vector[138] = 1.0; // left present
  }

  let rightWrist = null;
  let rightScale = 0;
  if (rightHandLandmarks && rightHandLandmarks.length === NUM_LANDMARKS_PER_HAND) {
    const rightNorm = normalizeHandLandmarks(rightHandLandmarks);
    vector.set(rightNorm, FEATURES_PER_HAND); // offset 63

    const w = rightHandLandmarks[0];
    const m = rightHandLandmarks[9];
    const dx = m.x - w.x;
    const dy = m.y - w.y;
    const dz = m.z - w.z;
    rightScale = Math.sqrt(dx * dx + dy * dy + dz * dz);
    rightWrist = w;

    vector[130] = w.x;
    vector[131] = w.y;
    vector[132] = w.z;
    vector[133] = rightScale;
    vector[139] = 1.0; // right present
  }

  // Inter-hand spatial relationships (only if both hands are detected)
  if (leftWrist && rightWrist) {
    const dwx = rightWrist.x - leftWrist.x;
    const dwy = rightWrist.y - leftWrist.y;
    const dwz = rightWrist.z - leftWrist.z;
    const dist = Math.sqrt(dwx * dwx + dwy * dwy + dwz * dwz);

    vector[134] = dwx;
    vector[135] = dwy;
    vector[136] = dwz;
    vector[137] = dist;
  }

  return vector;
}
