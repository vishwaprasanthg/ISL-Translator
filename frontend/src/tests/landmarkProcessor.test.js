import { describe, it, expect } from 'vitest';
import { processLandmarks, normalizeHandLandmarks, NUM_LANDMARKS_PER_HAND, TOTAL_FEATURES } from '../services/landmarkProcessor';

function generateDummyLandmarks(offset = 0) {
  const landmarks = [];
  for (let i = 0; i < NUM_LANDMARKS_PER_HAND; i++) {
    landmarks.push({
      x: i * 0.1 + offset,
      y: i * 0.2 + offset,
      z: i * 0.3 + offset
    });
  }
  // Enforce the wrist at specific coordinate to easily test
  landmarks[0] = { x: 1 + offset, y: 1 + offset, z: 1 + offset };
  // Enforce Middle MCP (index 9) to test scale normalization
  landmarks[9] = { x: 2 + offset, y: 2 + offset, z: 2 + offset }; // distance from wrist is sqrt(1^2 + 1^2 + 1^2) = sqrt(3)
  return landmarks;
}

describe('Landmark Processor', () => {
  it('TEST 1: Both hands present -> length === 126', () => {
    const results = {
      landmarks: [generateDummyLandmarks(), generateDummyLandmarks(10)],
      handednesses: [[{ categoryName: 'Left' }], [{ categoryName: 'Right' }]]
    };
    const features = processLandmarks(results);
    expect(features.length).toBe(TOTAL_FEATURES);
  });

  it('TEST 2: Only left hand present -> left normalized, right zeros', () => {
    const results = {
      landmarks: [generateDummyLandmarks()],
      handednesses: [[{ categoryName: 'Left' }]]
    };
    const features = processLandmarks(results);
    
    // Left section should have non-zero values (since it is normalized relative to wrist/scale)
    const leftSection = features.slice(0, 63);
    const rightSection = features.slice(63, 126);
    
    expect(leftSection.some(v => v !== 0)).toBe(true);
    expect(rightSection.every(v => v === 0)).toBe(true);
  });

  it('TEST 3: Only right hand present -> left zeros, right normalized', () => {
    const results = {
      landmarks: [generateDummyLandmarks()],
      handednesses: [[{ categoryName: 'Right' }]]
    };
    const features = processLandmarks(results);
    
    const leftSection = features.slice(0, 63);
    const rightSection = features.slice(63, 126);
    
    expect(leftSection.every(v => v === 0)).toBe(true);
    expect(rightSection.some(v => v !== 0)).toBe(true);
  });

  it('TEST 4: No hands -> 126 zeros', () => {
    const results = { landmarks: [], handednesses: [] };
    const features = processLandmarks(results);
    expect(features.every(v => v === 0)).toBe(true);
  });

  it('TEST 5: Wrist normalization -> normalized landmark 0 = [0, 0, 0]', () => {
    const landmarks = generateDummyLandmarks();
    const normalized = normalizeHandLandmarks(landmarks);
    expect(normalized[0]).toBeCloseTo(0);
    expect(normalized[1]).toBeCloseTo(0);
    expect(normalized[2]).toBeCloseTo(0);
  });

  it('TEST 6: Scale normalization -> distance(wrist, landmark 9) ≈ 1', () => {
    const landmarks = generateDummyLandmarks();
    const normalized = normalizeHandLandmarks(landmarks);
    
    // wrist is at index 0,1,2 (always 0,0,0)
    // landmark 9 is at index 27,28,29
    const dx = normalized[27];
    const dy = normalized[28];
    const dz = normalized[29];
    
    const dist = Math.sqrt(dx*dx + dy*dy + dz*dz);
    expect(dist).toBeCloseTo(1, 5);
  });

  it('TEST 7: No NaN -> every output value is finite', () => {
    const results = {
      landmarks: [generateDummyLandmarks()],
      handednesses: [[{ categoryName: 'Left' }]]
    };
    const features = processLandmarks(results);
    expect(features.every(v => !Number.isNaN(v))).toBe(true);
  });

  it('TEST 8: No Infinity -> every output value is finite', () => {
    // Test with all points same (scale = 0)
    const points = new Array(NUM_LANDMARKS_PER_HAND).fill({x: 0, y: 0, z: 0});
    const normalized = normalizeHandLandmarks(points);
    expect(normalized.every(v => Number.isFinite(v))).toBe(true);
  });

  it('TEST 9: Determinism -> same input = same output', () => {
    const landmarks = generateDummyLandmarks();
    const out1 = normalizeHandLandmarks(landmarks);
    const out2 = normalizeHandLandmarks(landmarks);
    expect(out1).toEqual(out2);
  });

  it('TEST 10: Hand ordering -> regardless of MediaPipe order, Left is first', () => {
    const leftLandmarks = generateDummyLandmarks(10);
    const rightLandmarks = generateDummyLandmarks(20);
    
    const results = {
      landmarks: [rightLandmarks, leftLandmarks], // Right first
      handednesses: [[{ categoryName: 'Right' }], [{ categoryName: 'Left' }]]
    };
    
    const features = processLandmarks(results);
    
    const expectedLeft = normalizeHandLandmarks(leftLandmarks);
    const expectedRight = normalizeHandLandmarks(rightLandmarks);
    
    const actualLeft = Array.from(features.slice(0, 63));
    const actualRight = Array.from(features.slice(63, 126));
    
    expect(actualLeft).toEqual(expectedLeft);
    expect(actualRight).toEqual(expectedRight);
  });
});
