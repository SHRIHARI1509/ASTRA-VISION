import { useState, useEffect, useCallback } from 'react';
import { apiService } from '../services';
import type { TelemetryState } from '../types';

export function useHealthCheck(pollIntervalMs = 5000) {
  const [telemetry, setTelemetry] = useState<TelemetryState>({
    isOnline: false,
    serviceName: 'offline',
    latencyMs: null,
    lastChecked: null,
  });
  const [loading, setLoading] = useState<boolean>(true);

  const performCheck = useCallback(async () => {
    try {
      const { data, latencyMs } = await apiService.checkHealth();
      setTelemetry({
        isOnline: data.status === 'ok',
        serviceName: data.service,
        latencyMs,
        lastChecked: new Date().toLocaleTimeString(),
        error: undefined,
      });
    } catch (err) {
      setTelemetry((prev) => ({
        ...prev,
        isOnline: false,
        error: err instanceof Error ? err.message : 'Connection failed',
        lastChecked: new Date().toLocaleTimeString(),
      }));
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    performCheck();
    const interval = setInterval(performCheck, pollIntervalMs);
    return () => clearInterval(interval);
  }, [performCheck, pollIntervalMs]);

  return { telemetry, loading, refresh: performCheck };
}
