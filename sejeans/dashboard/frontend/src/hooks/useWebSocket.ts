import { useState, useEffect, useRef, useCallback } from 'react';
import { InferenceResult, Alert } from '../types';

type ConnectionStatus = 'disconnected' | 'connecting' | 'connected' | 'error';

interface UseWebSocketReturn {
  latestData: InferenceResult | null;
  alerts: Alert[];
  status: ConnectionStatus;
  fps: number;
  connect: () => void;
  disconnect: () => void;
}

const WS_URL = `ws://${window.location.host}/ws/stream`;
const MAX_RECONNECT_ATTEMPTS = 5;
const RECONNECT_DELAY_MS = 3000;

export function useWebSocket(): UseWebSocketReturn {
  const [latestData, setLatestData] = useState<InferenceResult | null>(null);
  const [alerts, setAlerts] = useState<Alert[]>([]);
  const [status, setStatus] = useState<ConnectionStatus>('disconnected');
  const [fps, setFps] = useState<number>(0);

  const wsRef = useRef<WebSocket | null>(null);
  const reconnectAttemptsRef = useRef(0);
  const reconnectTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const frameCountRef = useRef(0);
  const fpsTimerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const isManualDisconnectRef = useRef(false);

  const clearReconnectTimer = () => {
    if (reconnectTimerRef.current) {
      clearTimeout(reconnectTimerRef.current);
      reconnectTimerRef.current = null;
    }
  };

  const stopFpsCounter = () => {
    if (fpsTimerRef.current) {
      clearInterval(fpsTimerRef.current);
      fpsTimerRef.current = null;
    }
    setFps(0);
  };

  const startFpsCounter = () => {
    fpsTimerRef.current = setInterval(() => {
      setFps(frameCountRef.current);
      frameCountRef.current = 0;
    }, 1000);
  };

  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return;

    isManualDisconnectRef.current = false;
    reconnectAttemptsRef.current = 0;
    clearReconnectTimer();

    const attemptConnect = () => {
      setStatus('connecting');

      try {
        const ws = new WebSocket(WS_URL);
        wsRef.current = ws;

        ws.onopen = () => {
          setStatus('connected');
          reconnectAttemptsRef.current = 0;
          frameCountRef.current = 0;
          startFpsCounter();
        };

        ws.onmessage = (event: MessageEvent) => {
          try {
            const raw = JSON.parse(event.data as string);

            // Backend sends InferenceResponse directly (not wrapped)
            // Detect by checking for 'motor' and 'battery' fields
            const data: InferenceResult = raw.data ?? (raw.motor && raw.battery ? raw : null);
            if (data) {
              setLatestData(data);
              frameCountRef.current += 1;

              // Auto-generate alert if status is warning or caution
              if (data.motor.status === 'warning' || data.battery.status === 'warning') {
                const autoAlert: Alert = {
                  id: `auto-${Date.now()}`,
                  timestamp: data.timestamp,
                  level: 'warning',
                  vehicle_id: data.vehicle_id,
                  message: data.motor.status === 'warning'
                    ? `모터 이상 탐지: MSE ${data.motor.anomaly_score.toFixed(1)} (임계값 초과)`
                    : `배터리 이상 탐지: SOH ${data.battery.soh_proxy.toFixed(1)}`,
                };
                setAlerts((prev) => [autoAlert, ...prev].slice(0, 100));
              }
            }

            if (raw.type === 'alert' && raw.alert) {
              setAlerts((prev) => [raw.alert, ...prev].slice(0, 100));
            }
          } catch {
            // silently ignore malformed messages
          }
        };

        ws.onerror = () => {
          setStatus('error');
        };

        ws.onclose = () => {
          stopFpsCounter();
          wsRef.current = null;

          if (isManualDisconnectRef.current) {
            setStatus('disconnected');
            return;
          }

          if (reconnectAttemptsRef.current < MAX_RECONNECT_ATTEMPTS) {
            reconnectAttemptsRef.current += 1;
            setStatus('connecting');
            reconnectTimerRef.current = setTimeout(attemptConnect, RECONNECT_DELAY_MS);
          } else {
            setStatus('error');
          }
        };
      } catch {
        setStatus('error');
      }
    };

    attemptConnect();
  }, []);

  const disconnect = useCallback(() => {
    isManualDisconnectRef.current = true;
    clearReconnectTimer();
    stopFpsCounter();

    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }

    setStatus('disconnected');
  }, []);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      isManualDisconnectRef.current = true;
      clearReconnectTimer();
      stopFpsCounter();
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, []);

  return { latestData, alerts, status, fps, connect, disconnect };
}
