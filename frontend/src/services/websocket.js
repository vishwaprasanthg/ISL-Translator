class WebSocketService {
  constructor() {
    this.ws = null;
    this.url = 'ws://localhost:8000/ws/translate';
    this.reconnectTimeout = null;
    this.reconnectDelay = 1000;
    this.maxReconnectDelay = 10000;
    this.onMessageCallback = null;
    this.onStatusChangeCallback = null;
    this.status = 'disconnected';
    this.shouldReconnect = false;
    this.lastRtt = 0;
  }

  setStatus(newStatus) {
    this.status = newStatus;
    if (this.onStatusChangeCallback) {
      this.onStatusChangeCallback(newStatus);
    }
  }

  connect(onMessage, onStatusChange) {
    if (this.status === 'connected' || this.status === 'connecting') return;
    
    this.shouldReconnect = true;
    this.onMessageCallback = onMessage;
    this.onStatusChangeCallback = onStatusChange;
    this._connect();
  }

  _connect() {
    if (!this.shouldReconnect) return;
    
    this.setStatus('connecting');
    try {
      this.ws = new WebSocket(this.url);

      this.ws.onopen = () => {
        this.setStatus('connected');
        this.reconnectDelay = 1000;
      };

      this.ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          
          if (data.type === 'ack' && data.timestamp) {
            this.lastRtt = performance.now() - data.timestamp;
            data.rtt = this.lastRtt; // Embed it in the payload for the UI
          }
          
          if (this.onMessageCallback) {
            this.onMessageCallback(data);
          }
        } catch (e) {
          console.error("Failed to parse websocket message", e);
        }
      };

      this.ws.onclose = () => {
        this.ws = null;
        if (this.shouldReconnect) {
          this.setStatus('connecting');
          // Exponential backoff
          this.reconnectTimeout = setTimeout(() => {
            this.reconnectDelay = Math.min(this.reconnectDelay * 2, this.maxReconnectDelay);
            this._connect();
          }, this.reconnectDelay);
        } else {
          this.setStatus('disconnected');
        }
      };

      this.ws.onerror = (error) => {
        this.setStatus('error');
        // connection close will fire immediately after error
      };
    } catch (error) {
      this.setStatus('error');
    }
  }

  disconnect() {
    this.shouldReconnect = false;
    if (this.reconnectTimeout) {
      clearTimeout(this.reconnectTimeout);
      this.reconnectTimeout = null;
    }
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
    this.setStatus('disconnected');
  }

  sendFeatures(features, mode = 'alphabet') {
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
      return false;
    }
    
    const payload = {
      type: 'landmarks',
      mode: mode,
      features: Array.from(features),
      timestamp: performance.now()
    };
    
    try {
      this.ws.send(JSON.stringify(payload));
      return true;
    } catch (e) {
      console.error("Error sending websocket message", e);
      return false;
    }
  }
}

// Export as singleton to avoid multiple connections
const wsService = new WebSocketService();
export default wsService;
