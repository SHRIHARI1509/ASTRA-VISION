import React, { useState } from 'react';
import {
  Header,
  StatusBanner,
  AnalysisWorkspace,
  DetectionWorkspace,
  HistoryGallery,
  TelemetryPanel,
  Footer,
} from '../components';
import { useHealthCheck, useHistory } from '../hooks';
import type { OperationalMode } from '../types';

export const Home: React.FC = () => {
  const [operationalMode, setOperationalMode] = useState<OperationalMode>('classification');
  const { telemetry, loading, refresh } = useHealthCheck();
  const { history, addHistoryItem, addBatchHistoryItems, clearHistory, maxLimit } = useHistory();

  return (
    <div className="tactical-layout">
      <Header
        telemetry={telemetry}
        loading={loading}
        onRefresh={refresh}
      />

      <StatusBanner />

      <main className="main-content">
        <section className="hero-section">
          <div className="badge-technical">TACTICAL RECONNAISSANCE PLATFORM</div>
          <h2 className="hero-title">
            AI-Based Defence Object Recognition System
          </h2>
          <p className="hero-description">
            ASTRA VISION provides high-assurance automated target recognition, visual intelligence
            synthesis, and multi-domain object classification for mission-critical defense operations.
          </p>
        </section>

        {/* Operational Mode Navigation (Classification vs Multi-Object Detection) */}
        <div className="operational-mode-selector" data-testid="operational-mode-selector" role="tablist" aria-label="Operational Capabilities">
          <button
            type="button"
            role="tab"
            aria-selected={operationalMode === 'classification'}
            className={`op-mode-btn ${operationalMode === 'classification' ? 'active' : ''}`}
            onClick={() => setOperationalMode('classification')}
            data-testid="mode-classification-tab"
          >
            <span className="op-mode-indicator" />
            TARGET CLASSIFICATION (SigLIP 2)
          </button>
          <button
            type="button"
            role="tab"
            aria-selected={operationalMode === 'detection'}
            className={`op-mode-btn ${operationalMode === 'detection' ? 'active' : ''}`}
            onClick={() => setOperationalMode('detection')}
            data-testid="mode-detection-tab"
          >
            <span className="op-mode-indicator" />
            OBJECT DETECTION (Grounding DINO)
          </button>
        </div>

        {operationalMode === 'classification' ? (
          <>
            {/* Optical & Batch Image Analysis Workspace */}
            <AnalysisWorkspace
              onSingleSuccess={addHistoryItem}
              onBatchSuccess={addBatchHistoryItems}
            />

            {/* In-Session Tactical History & Gallery */}
            <HistoryGallery
              history={history}
              onClearHistory={clearHistory}
              maxLimit={maxLimit}
            />
          </>
        ) : (
          /* Multi-Object Detection Workspace */
          <DetectionWorkspace />
        )}

        {/* Subsystem Telemetry and Verification Panel */}
        <TelemetryPanel telemetry={telemetry} />
      </main>

      <Footer />
    </div>
  );
};

