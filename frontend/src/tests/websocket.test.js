import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import wsService from '../services/websocket';

describe('WebSocket Service', () => {
  let mockWs;

  beforeEach(() => {
    // Mock the browser WebSocket
    mockWs = {
      send: vi.fn(),
      close: vi.fn(),
      readyState: 1 // WebSocket.OPEN
    };
    
    // Provide a simple class that can be instantiated with 'new'
    global.WebSocket = class {
      constructor(url) {
        this.url = url;
        this.readyState = mockWs.readyState;
        this.send = mockWs.send;
        this.close = mockWs.close;
        
        // Save instance reference so we can trigger events in tests
        mockWs.instance = this;
      }
    };
    // Assign WebSocket.OPEN
    global.WebSocket.OPEN = 1;

    // Helper functions to trigger events on the active instance
    mockWs.onopen = () => mockWs.instance && mockWs.instance.onopen && mockWs.instance.onopen();
    mockWs.onmessage = (e) => mockWs.instance && mockWs.instance.onmessage && mockWs.instance.onmessage(e);
    mockWs.onclose = () => mockWs.instance && mockWs.instance.onclose && mockWs.instance.onclose();
    mockWs.onerror = (e) => mockWs.instance && mockWs.instance.onerror && mockWs.instance.onerror(e);
    
    wsService.disconnect();
    
    // Reset spy tracking
    vi.spyOn(global, 'WebSocket');
  });

  afterEach(() => {
    vi.clearAllMocks();
    wsService.disconnect();
  });

  it('TEST 1: WebSocket connects', () => {
    wsService.connect(vi.fn(), vi.fn());
    expect(global.WebSocket).toHaveBeenCalledWith('ws://localhost:8000/ws/translate');
    expect(wsService.status).toBe('connecting');
    
    // Simulate connection open
    mockWs.onopen();
    expect(wsService.status).toBe('connected');
  });

  it('TEST 2: WebSocket disconnects', () => {
    wsService.connect(vi.fn(), vi.fn());
    wsService.disconnect();
    expect(mockWs.close).toHaveBeenCalled();
    expect(wsService.status).toBe('disconnected');
    expect(wsService.ws).toBeNull();
  });

  it('TEST 3 & 4: Exactly 126 features are sent in the correct structure', () => {
    wsService.connect(vi.fn(), vi.fn());
    mockWs.onopen();
    
    const features = new Float32Array(126).fill(1.5);
    const success = wsService.sendFeatures(features, 'number');
    
    expect(success).toBe(true);
    expect(mockWs.send).toHaveBeenCalledTimes(1);
    
    const sentData = JSON.parse(mockWs.send.mock.calls[0][0]);
    expect(sentData.type).toBe('landmarks');
    expect(sentData.mode).toBe('number');
    expect(sentData.features.length).toBe(126);
    expect(sentData.features[0]).toBe(1.5);
    expect(sentData.timestamp).toBeDefined();
  });

  it('TEST 5: Connection state updates correctly', () => {
    const statusCallback = vi.fn();
    wsService.connect(vi.fn(), statusCallback);
    
    expect(statusCallback).toHaveBeenCalledWith('connecting');
    
    mockWs.onopen();
    expect(statusCallback).toHaveBeenCalledWith('connected');
    
    mockWs.onerror({});
    expect(statusCallback).toHaveBeenCalledWith('error');
  });
});
