import React, { useState } from 'react';
import type { HistoryItem } from '../types';

interface HistoryGalleryProps {
  history: HistoryItem[];
  onClearHistory: () => void;
  maxLimit?: number;
}

export const HistoryGallery: React.FC<HistoryGalleryProps> = ({
  history,
  onClearHistory,
  maxLimit = 50,
}) => {
  const [selectedItem, setSelectedItem] = useState<HistoryItem | null>(null);
  const [showConfirmClear, setShowConfirmClear] = useState<boolean>(false);

  const handleClear = () => {
    onClearHistory();
    setSelectedItem(null);
    setShowConfirmClear(false);
  };

  const formatTimestamp = (isoStr: string): string => {
    try {
      const d = new Date(isoStr);
      return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
    } catch {
      return isoStr;
    }
  };

  return (
    <section className="history-gallery-card" data-testid="history-gallery-section">
      <div className="card-top-bar">
        <div className="frame-meta">
          <span className="channel-id">HIST-01 [SESSION LOG & GALLERY]</span>
          <span className="hud-coords">
            CAPACITY: {history.length} / {maxLimit} ENTRIES (IN-MEMORY SESSION)
          </span>
        </div>
        <div className="history-actions">
          {history.length > 0 && !showConfirmClear && (
            <button
              type="button"
              className="clear-history-btn"
              onClick={() => setShowConfirmClear(true)}
              data-testid="clear-history-btn"
            >
              CLEAR HISTORY
            </button>
          )}

          {showConfirmClear && (
            <div className="clear-confirm-group">
              <span className="confirm-prompt">PURGE ALL {history.length} ENTRIES?</span>
              <button
                type="button"
                className="confirm-clear-btn"
                onClick={handleClear}
                data-testid="confirm-clear-history-btn"
              >
                YES, PURGE
              </button>
              <button
                type="button"
                className="cancel-clear-btn"
                onClick={() => setShowConfirmClear(false)}
              >
                CANCEL
              </button>
            </div>
          )}
        </div>
      </div>

      <div className="gallery-body">
        {history.length === 0 ? (
          <div className="gallery-empty-state" data-testid="gallery-empty-state">
            <div className="empty-icon-radar">
              <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                <circle cx="12" cy="12" r="10" />
                <path d="M12 2a10 10 0 0 1 10 10" />
                <line x1="12" y1="12" x2="19" y2="5" />
              </svg>
            </div>
            <div className="empty-title">NO HISTORICAL RECONNAISSANCE RECORDS</div>
            <p className="empty-desc">
              All successfully classified single targets and multi-image batch runs in this session will
              be catalogued here in chronological order (newest first).
            </p>
          </div>
        ) : (
          <div className="gallery-grid" data-testid="gallery-items-grid">
            {history.map((item, idx) => (
              <div
                key={item.id}
                className={`gallery-item-card ${selectedItem?.id === item.id ? 'selected' : ''}`}
                onClick={() => setSelectedItem(item)}
                role="button"
                tabIndex={0}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' || e.key === ' ') setSelectedItem(item);
                }}
                data-testid={`gallery-item-${idx}`}
                aria-label={`View history entry for ${item.filename}`}
              >
                <div className="gallery-thumb-container">
                  {item.previewUrl ? (
                    <img src={item.previewUrl} alt={item.filename} className="gallery-thumb" />
                  ) : (
                    <div className="gallery-fallback-thumb">
                      <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                        <circle cx="12" cy="12" r="9" />
                        <line x1="12" y1="3" x2="12" y2="7" />
                        <line x1="12" y1="17" x2="12" y2="21" />
                        <line x1="3" y1="12" x2="7" y2="12" />
                        <line x1="17" y1="12" x2="21" y2="12" />
                      </svg>
                    </div>
                  )}
                  <span className="gallery-timestamp-badge">{formatTimestamp(item.timestamp)}</span>
                </div>

                <div className="gallery-item-details">
                  <div className="gallery-filename" title={item.filename}>
                    {item.filename}
                  </div>
                  <div className="gallery-prediction-row">
                    <span className="gallery-label" data-testid={`gallery-label-${idx}`}>
                      {item.prediction.label}
                    </span>
                    <span className="gallery-score" data-testid={`gallery-score-${idx}`}>
                      {(item.score * 100).toFixed(1)}%
                    </span>
                  </div>

                  {item.uncertainty?.is_uncertain && (
                    <div className="gallery-uncertain-tag" data-testid={`gallery-uncertain-${idx}`}>
                      ⚠️ UNCERTAIN
                    </div>
                  )}

                  <div className="gallery-latency-tag">
                    {item.inference.inference_time_ms.toFixed(0)} ms • {item.inference.device.toUpperCase()}
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}

        {/* Selected History Item Modal / Expanded Inspector */}
        {selectedItem && (
          <div className="history-detail-modal-backdrop" onClick={() => setSelectedItem(null)}>
            <div
              className="history-detail-modal"
              onClick={(e) => e.stopPropagation()}
              data-testid="history-detail-modal"
            >
              <div className="modal-header">
                <div className="modal-title-group">
                  <span className="modal-tag">HISTORICAL RECORD INSPECTOR</span>
                  <h3 className="modal-filename">{selectedItem.filename}</h3>
                </div>
                <button
                  type="button"
                  className="modal-close-btn"
                  onClick={() => setSelectedItem(null)}
                  aria-label="Close inspector"
                  data-testid="close-history-modal-btn"
                >
                  ✕
                </button>
              </div>

              <div className="modal-body">
                {selectedItem.previewUrl && (
                  <div className="modal-image-preview">
                    <img src={selectedItem.previewUrl} alt={selectedItem.filename} />
                  </div>
                )}

                <div className="modal-primary-prediction">
                  <span className="meta-label">PRIMARY CLASSIFICATION</span>
                  <div className="prediction-headline">
                    <span className="headline-label">{selectedItem.prediction.label}</span>
                    <span className="headline-score">{(selectedItem.score * 100).toFixed(2)}%</span>
                  </div>
                </div>

                {selectedItem.uncertainty && (
                  <div className={`modal-uncertainty-box ${selectedItem.uncertainty.is_uncertain ? 'warn' : 'ok'}`}>
                    <span className="meta-label">UNCERTAINTY STATUS</span>
                    <div className="uncertainty-status-text">
                      {selectedItem.uncertainty.is_uncertain
                        ? `EVALUATION: UNCERTAIN (${selectedItem.uncertainty.reason})`
                        : 'EVALUATION: ASSURED (CLEAR MARGIN)'}
                    </div>
                    {selectedItem.uncertainty.score_margin !== undefined && selectedItem.uncertainty.score_margin !== null && (
                      <div className="uncertainty-subtext">
                        Score Margin against Runner-up: {(selectedItem.uncertainty.score_margin * 100).toFixed(2)}%
                      </div>
                    )}
                  </div>
                )}

                {selectedItem.top_3 && selectedItem.top_3.length > 0 && (
                  <div className="modal-candidates-box">
                    <span className="meta-label">TOP-3 CANDIDATE HIERARCHY</span>
                    <div className="modal-candidates-list">
                      {selectedItem.top_3.map((cand, cIdx) => (
                        <div key={`modal-cand-${cand.label}-${cIdx}`} className="modal-cand-row">
                          <span className="cand-rank">RANK #{cIdx + 1}</span>
                          <span className="cand-label">{cand.label}</span>
                          <span className="cand-score">{(cand.score * 100).toFixed(2)}%</span>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                <div className="modal-inference-meta">
                  <span>MODEL: {selectedItem.inference.model}</span>
                  <span>DEVICE: {selectedItem.inference.device.toUpperCase()}</span>
                  <span>LATENCY: {selectedItem.inference.inference_time_ms.toFixed(1)} ms</span>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>

      <div className="card-bottom-banner">
        <span className="phase-scope-tag">PHASE 8D SCOPE:</span>
        <span className="phase-scope-desc">
          Session-scoped reconnaissance gallery — local in-memory retention with automatic object URL lifecycle management.
        </span>
      </div>
    </section>
  );
};
