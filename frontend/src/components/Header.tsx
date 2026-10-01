import React from 'react';
import type { TelemetryState } from '../types';

interface HeaderProps {
  telemetry: TelemetryState;
  loading: boolean;
  onRefresh: () => void;
}

export const Header: React.FC<HeaderProps> = ({ telemetry, loading, onRefresh }) => {
  return (
    <header className="tactical-header">
      <div className="header-left">
        <div className="hud-symbol">
          <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <polygon points="12 2 22 8.5 22 15.5 12 22 2 15.5 2 8.5 12 2" />
            <circle cx="12" cy="12" r="3" />
            <line x1="12" y1="2" x2="12" y2="6" />
            <line x1="12" y1="18" x2="12" y2="22" />
            <line x1="2" y1="12" x2="6" y2="12" />
            <line x1="18" y1="12" x2="22" y2="12" />
          </svg>
        </div>
        <div className="title-group">
          <div className="brand-label">DEFENCE AI SYSTEMS // ARCH-01</div>
          <h1 className="brand-title">ASTRA VISION</h1>
        </div>
      </div>

      <div className="header-right">
        <div className="telemetry-pill">
          <span className={`status-indicator ${telemetry.isOnline ? 'online' : 'offline'}`} />
          <div className="telemetry-details">
            <span className="telemetry-label">GATEWAY HEALTH</span>
            <span className="telemetry-val">
              {loading
                ? 'CHECKING...'
                : telemetry.isOnline
                ? `ONLINE (${telemetry.latencyMs}ms)`
                : 'OFFLINE / DISCONNECTED'}
            </span>
          </div>
        </div>

        <button
          type="button"
          onClick={onRefresh}
          className="tactical-refresh-btn"
          title="Query /api/health"
        >
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <polyline points="23 4 23 10 17 10" />
            <polyline points="1 20 1 14 7 14" />
            <path d="M3.51 9a9 9 0 0 1 14.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0 0 20.49 15" />
          </svg>
          <span>PING API</span>
        </button>
      </div>
    </header>
  );
};
