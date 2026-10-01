import React from 'react';
import {
  Header,
  StatusBanner,
  AnalysisWorkspace,
  HistoryGallery,
  TelemetryPanel,
  Footer,
} from '../components';
import { useHealthCheck, useHistory } from '../hooks';

export const Home: React.FC = () => {
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

        {/* Subsystem Telemetry and Verification Panel */}
        <TelemetryPanel telemetry={telemetry} />
      </main>

      <Footer />
    </div>
  );
};

