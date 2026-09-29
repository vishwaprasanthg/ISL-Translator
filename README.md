# AI-Powered Real-Time Indian Sign Language Translator

A production-grade, full-stack ISL translator that recognizes Indian Sign Language from a webcam in real-time and converts signs into English text and speech.

## Features
- Real-time webcam processing using MediaPipe
- 126-D Canonical Feature Representation
- Static (Alphabet/Numbers) and Dynamic (Vocabulary) recognition
- Strict ISL language support
- Privacy-first: Camera access is completely local. No video is uploaded to the backend, saved, or stored. (Phase 3+).
- Secure Context Requirement: Due to browser security policies, local development works on `localhost`, but production deployments MUST use HTTPS to allow camera access.
## Setup
Refer to `docs/DEPLOYMENT.md` for full setup instructions.
