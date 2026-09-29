export function getCameraErrorMessage(error) {
  if (!error) return null;

  switch (error.name) {
    case 'NotAllowedError':
    case 'SecurityError':
      return 'Camera permission was denied. Please allow camera access in your browser settings and try again.';
    case 'NotFoundError':
      return 'No camera was detected. Please ensure a camera is connected.';
    case 'NotReadableError':
    case 'TrackStartError':
      return 'The camera may already be in use by another application or tab.';
    case 'OverconstrainedError':
      return 'The selected camera does not support the required resolution or framerate.';
    default:
      return `An unexpected camera error occurred: ${error.message || 'Unknown error'}`;
  }
}
