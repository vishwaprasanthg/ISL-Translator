import fs from 'fs';
import { processLandmarks } from '../src/services/landmarkProcessor.js';

const inputPath = process.argv[2];
if (!inputPath) {
  console.error("Please provide input json file");
  process.exit(1);
}

const data = JSON.parse(fs.readFileSync(inputPath, 'utf8'));

// Run processLandmarks
const features = processLandmarks(data);

// Output raw values as JSON list
console.log(JSON.stringify(Array.from(features)));
