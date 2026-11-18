import { describe, it, expect } from 'vitest';
import { extractDynamicFeatures, DYNAMIC_FEATURE_DIM } from '../services/dynamicFeatureExtractor';

function makeDummyResults(leftWrist = null, rightWrist = null) {
  const landmarks = [];
  const handednesses = [];

  if (leftWrist) {
    const leftLms = [];
    for (let i = 0; i < 21; i++) {
      if (i === 0) leftLms.push({ x: leftWrist.x, y: leftWrist.y, z: leftWrist.z || 0 });
      else if (i === 9) leftLms.push({ x: leftWrist.x, y: leftWrist.y - 0.1, z: leftWrist.z || 0 });
      else leftLms.push({ x: leftWrist.x + 0.01 * (i % 5), y: leftWrist.y - 0.02 * Math.floor(i / 5), z: 0 });
    }
    landmarks.push(leftLms);
    handednesses.push([{ categoryName: 'Left' }]);
  }

  if (rightWrist) {
    const rightLms = [];
    for (let i = 0; i < 21; i++) {
      if (i === 0) rightLms.push({ x: rightWrist.x, y: rightWrist.y, z: rightWrist.z || 0 });
      else if (i === 9) rightLms.push({ x: rightWrist.x, y: rightWrist.y - 0.1, z: rightWrist.z || 0 });
      else rightLms.push({ x: rightWrist.x + 0.01 * (i % 5), y: rightWrist.y - 0.02 * Math.floor(i / 5), z: 0 });
    }
    landmarks.push(rightLms);
    handednesses.push([{ categoryName: 'Right' }]);
  }

  return { landmarks, handednesses };
}

describe('dynamicFeatureExtractor (JS)', () => {
  it('returns exactly 140 features', () => {
    const results = makeDummyResults(null, { x: 0.6, y: 0.5 });
    const feat = extractDynamicFeatures(results);
    expect(feat.length).toBe(DYNAMIC_FEATURE_DIM);
    expect(feat.length).toBe(140);
  });

  it('handles empty results with all zeros', () => {
    const feat = extractDynamicFeatures(null);
    expect(feat.length).toBe(140);
    expect(Array.from(feat).every(v => v === 0)).toBe(true);
  });

  it('correctly maps right hand only', () => {
    const results = makeDummyResults(null, { x: 0.7, y: 0.4, z: 0.05 });
    const feat = extractDynamicFeatures(results);

    // Left hand is 0
    for (let i = 0; i < 63; i++) expect(feat[i]).toBe(0);
    expect(feat[138]).toBe(0); // Left present is 0

    // Right hand has landmarks
    expect(feat[130]).toBeCloseTo(0.7);
    expect(feat[131]).toBeCloseTo(0.4);
    expect(feat[132]).toBeCloseTo(0.05);
    expect(feat[133]).toBeCloseTo(0.1); // Scale
    expect(feat[139]).toBe(1.0); // Right present is 1

    // Inter-wrist features are 0 because left hand is absent
    expect(feat[134]).toBe(0);
    expect(feat[137]).toBe(0);
  });

  it('correctly computes inter-wrist distance for two hands', () => {
    const results = makeDummyResults({ x: 0.2, y: 0.5, z: 0 }, { x: 0.8, y: 0.5, z: 0 });
    const feat = extractDynamicFeatures(results);

    expect(feat[138]).toBe(1.0); // Left present
    expect(feat[139]).toBe(1.0); // Right present

    // Inter-wrist vector: right (0.8) - left (0.2) = 0.6
    expect(feat[134]).toBeCloseTo(0.6);
    expect(feat[135]).toBeCloseTo(0.0);
    expect(feat[136]).toBeCloseTo(0.0);
    expect(feat[137]).toBeCloseTo(0.6); // Distance
  });
});
