import { useState, useEffect, useCallback } from 'react';
import wsService from '../services/websocket';

export function useWebSocket() {
  const [connectionStatus, setConnectionStatus] = useState('disconnected');
  const [lastMessage, setLastMessage] = useState(null);
  const [error, setError] = useState(null);

  const handleMessage = useCallback((data) => {
    if (data.type === 'error') {
      setError(data.message);
    } else {
      setError(null);
      setLastMessage(data);
    }
  }, []);

  const handleStatusChange = useCallback((status) => {
    setConnectionStatus(status);
    if (status === 'connected') {
      setError(null);
    } else if (status === 'error') {
      setError('Connection failed');
    }
  }, []);

  const connect = useCallback(() => {
    wsService.connect(handleMessage, handleStatusChange);
  }, [handleMessage, handleStatusChange]);

  const disconnect = useCallback(() => {
    wsService.disconnect();
    setConnectionStatus('disconnected');
    setLastMessage(null);
  }, []);

  const sendFeatures = useCallback((features, mode) => {
    return wsService.sendFeatures(features, mode);
  }, []);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      wsService.disconnect();
    };
  }, []);

  return {
    connect,
    disconnect,
    sendFeatures,
    connectionStatus,
    lastMessage,
    error
  };
}
