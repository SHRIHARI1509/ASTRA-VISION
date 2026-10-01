import React from 'react';
import type { TelemetryState } from '../types';

interface TelemetryPanelProps {
  telemetry: TelemetryState;
}

export const TelemetryPanel: React.FC<TelemetryPanelProps> = ({ telemetry }) => {
  return (
    <div className="telemetry-grid">
      <div className="telemetry-card">
        <div className="card-header">
          <span className="card-code">SYS-01</span>
          <span className="card-title">API GATEWAY</span>
        </div>
        <div className="card-metric">
          <span className={`status-pill ${telemetry.isOnline ? 'active' : 'inactive'}`}>
            {telemetry.isOnline ? 'HTTP 200 OK' : 'DISCONNECTED'}
          </span>
        </div>
        <div className="card-desc">
          FastAPI routing service mounted at <code>/api</code>.
        </div>
        <div className="card-footer-meta">
          <span>LATENCY: {telemetry.latencyMs !== null ? `${telemetry.latencyMs}ms` : 'N/A'}</span>
          <span>SERVICE: {telemetry.serviceName}</span>
        </div>
      </div>

      <div className="telemetry-card">
        <div className="card-header">
          <span className="card-code">SYS-02</span>
          <span className="card-title">MODEL SERVICES</span>
        </div>
        <div className="card-metric">
          <span className="status-pill standby">STANDBY (PHASE 2)</span>
        </div>
        <div className="card-desc">
          Strict separation maintained. Zero ML weights loaded in Phase 1.
        </div>
        <div className="card-footer-meta">
          <span>ALLOCATION: 0 MB VRAM</span>
          <span>ENGINES: OFFLINE</span>
        </div>
      </div>

      <div className="telemetry-card">
        <div className="card-header">
          <span className="card-code">SYS-03</span>
          <span className="card-title">SECURITY / AIR-GAP</span>
        </div>
        <div className="card-metric">
          <span className="status-pill secure">ISOLATED</span>
        </div>
        <div className="card-desc">
          No external model downloads or cloud inference calls permitted.
        </div>
        <div className="card-footer-meta">
          <span>CORS: LOCALHOST ONLY</span>
          <span>SECRETS: ZERO COMMITTED</span>
        </div>
      </div>
    </div>
  );
};
