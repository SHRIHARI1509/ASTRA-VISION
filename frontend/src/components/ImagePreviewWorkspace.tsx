import React from 'react';
import type {
  ClientImageMetadata,
  ImageValidationResult,
  ClassificationResponse,
} from '../types';
import { formatBytes } from '../utils';
import { ClassificationResultPanel } from './ClassificationResultPanel';

interface ImagePreviewWorkspaceProps {
  imageMeta: ClientImageMetadata;
  isValidating: boolean;
  validationResult: ImageValidationResult | null;
  validationError: string | null;
  isAnalyzing: boolean;
  classificationResult: ClassificationResponse | null;
  classificationError: string | null;
  onReset: () => void;
  onRetryValidation: () => void;
  onAnalyze: () => void;
  onRetryAnalysis: () => void;
}

export const ImagePreviewWorkspace: React.FC<ImagePreviewWorkspaceProps> = ({
  imageMeta,
  isValidating,
  validationResult,
  validationError,
  isAnalyzing,
  classificationResult,
  classificationError,
  onReset,
  onRetryValidation,
  onAnalyze,
  onRetryAnalysis,
}) => {
  const isActionDisabled = isValidating || isAnalyzing;
  const isImageValid = Boolean(validationResult?.valid && !validationError);

  return (
    <div className="preview-workspace">
      {/* Viewport Frame with Image */}
      <div className="preview-viewport">
        <div className="reticle-corner top-left" />
        <div className="reticle-corner top-right" />
        <div className="reticle-corner bottom-left" />
        <div className="reticle-corner bottom-right" />

        <div className="image-wrapper">
          <img
            src={imageMeta.previewUrl}
            alt={imageMeta.filename}
            className="preview-img"
          />

          {/* Tactical Crosshair Overlay on Image */}
          <div className="viewport-overlay-hud">
            <div className="hud-corner-tag">
              {isAnalyzing
                ? 'INFERENCE: PROCESSING'
                : classificationResult
                ? 'TARGET IDENTIFIED'
                : 'FRAME LOCK: ACTIVE'}
            </div>
            <div className="hud-center-marker" />
          </div>

          {/* Scanning line animation during validation or analysis */}
          {(isValidating || isAnalyzing) && <div className="scanning-bar" />}
        </div>
      </div>

      {/* Main Metadata and Results Dashboard */}
      <div className="preview-meta-dashboard">
        {/* Validation or Analysis Status Callouts */}
        <div className="validation-status-callout">
          {/* Phase 2: Validating */}
          {isValidating && (
            <div className="status-box validating" role="status" aria-live="polite">
              <span className="spinner-hud" />
              <div className="status-text-group">
                <span className="status-title">BACKEND VALIDATION IN PROGRESS</span>
                <span className="status-desc">
                  Performing server-side header integrity, dimensions, and RGB raster decoding...
                </span>
              </div>
            </div>
          )}

          {/* Phase 5: Analyzing */}
          {isAnalyzing && (
            <div className="status-box analyzing" role="status" aria-live="polite">
              <span className="spinner-hud" />
              <div className="status-text-group">
                <span className="status-title">ANALYZING IMAGE...</span>
                <span className="status-desc">
                  Processing visual data through SigLIP 2 zero-shot vision model...
                </span>
              </div>
            </div>
          )}

          {/* Phase 2: Validation Failure */}
          {!isValidating && validationError && (
            <div className="status-box error" role="alert">
              <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <circle cx="12" cy="12" r="10" />
                <line x1="12" y1="8" x2="12" y2="12" />
                <line x1="12" y1="16" x2="12.01" y2="16" />
              </svg>
              <div className="status-text-group">
                <span className="status-title">VALIDATION FAILED</span>
                <span className="status-desc">{validationError}</span>
              </div>
            </div>
          )}

          {/* Phase 5: Classification Failure */}
          {!isAnalyzing && classificationError && (
            <div className="status-box error" role="alert">
              <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <circle cx="12" cy="12" r="10" />
                <line x1="12" y1="8" x2="12" y2="12" />
                <line x1="12" y1="16" x2="12.01" y2="16" />
              </svg>
              <div className="status-text-group">
                <span className="status-title">CLASSIFICATION FAILED</span>
                <span className="status-desc">{classificationError}</span>
              </div>
            </div>
          )}

          {/* Phase 2: Validated and Ready for Analysis */}
          {!isValidating && !isAnalyzing && !classificationResult && !classificationError && validationResult && (
            <div className="status-box valid">
              <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14" />
                <polyline points="22 4 12 14.01 9 11.01" />
              </svg>
              <div className="status-text-group">
                <span className="status-title">IMAGE VALIDATED — READY FOR ANALYSIS</span>
                <span className="status-desc">
                  Input verified. Click 'ANALYZE IMAGE' to execute SigLIP 2 multi-domain classification.
                </span>
              </div>
            </div>
          )}
        </div>

        {/* Phase 5: Display Classification Result if available */}
        {classificationResult ? (
          <ClassificationResultPanel
            result={classificationResult}
            onReset={onReset}
            onReanalyze={onAnalyze}
            isAnalyzing={isAnalyzing}
          />
        ) : (
          /* Structured Image Metadata Grid */
          <div className="metadata-spec-grid">
            <div className="meta-spec-item">
              <span className="spec-label">FILENAME</span>
              <span className="spec-value" title={imageMeta.filename}>
                {imageMeta.filename}
              </span>
            </div>

            <div className="meta-spec-item">
              <span className="spec-label">FORMAT</span>
              <span className="spec-value highlight">
                {validationResult ? validationResult.format : imageMeta.format}
              </span>
            </div>

            <div className="meta-spec-item">
              <span className="spec-label">DIMENSIONS</span>
              <span className="spec-value">
                {validationResult
                  ? `${validationResult.width} × ${validationResult.height} px`
                  : `${imageMeta.width} × ${imageMeta.height} px`}
              </span>
            </div>

            <div className="meta-spec-item">
              <span className="spec-label">FILE SIZE</span>
              <span className="spec-value">
                {formatBytes(validationResult ? validationResult.size_bytes : imageMeta.sizeBytes)}
                <span className="sub-bytes">
                  ({(validationResult ? validationResult.size_bytes : imageMeta.sizeBytes).toLocaleString()} B)
                </span>
              </span>
            </div>

            <div className="meta-spec-item">
              <span className="spec-label">RGB COMPATIBILITY</span>
              <span className="spec-value">
                {validationResult ? 'VERIFIED (RGB 3-CH)' : isValidating ? 'TESTING...' : 'PENDING'}
              </span>
            </div>

            <div className="meta-spec-item">
              <span className="spec-label">ML INFERENCE STATUS</span>
              <span className={`spec-value ${isAnalyzing ? 'highlight' : ''}`}>
                {isAnalyzing
                  ? 'PROCESSING (INFERENCE ACTIVE)'
                  : isImageValid
                  ? 'STANDBY (READY FOR ANALYSIS)'
                  : 'OFFLINE'}
              </span>
            </div>
          </div>
        )}

        {/* Action Controls (when result panel is not showing) */}
        {!classificationResult && (
          <div className="preview-action-row">
            <button
              type="button"
              className="tactical-btn danger"
              onClick={onReset}
              disabled={isActionDisabled}
              aria-label="Remove and reset image"
            >
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <line x1="18" y1="6" x2="6" y2="18" />
                <line x1="6" y1="6" x2="18" y2="18" />
              </svg>
              <span>REMOVE / RESET</span>
            </button>

            {validationError && (
              <button
                type="button"
                className="tactical-btn secondary"
                onClick={onRetryValidation}
                disabled={isActionDisabled}
                aria-label="Retry validation"
              >
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <polyline points="23 4 23 10 17 10" />
                  <polyline points="1 20 1 14 7 14" />
                  <path d="M3.51 9a9 9 0 0 1 14.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0 0 20.49 15" />
                </svg>
                <span>RETRY VALIDATION</span>
              </button>
            )}

            {classificationError && !validationError && (
              <button
                type="button"
                className="tactical-btn secondary"
                onClick={onRetryAnalysis}
                disabled={isActionDisabled}
                aria-label="Retry classification"
              >
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <polyline points="23 4 23 10 17 10" />
                  <polyline points="1 20 1 14 7 14" />
                  <path d="M3.51 9a9 9 0 0 1 14.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0 0 20.49 15" />
                </svg>
                <span>TRY AGAIN</span>
              </button>
            )}

            {isImageValid && (
              <button
                type="button"
                className={`tactical-btn primary analyze-action-btn ${isAnalyzing ? 'loading' : ''}`}
                onClick={onAnalyze}
                disabled={isActionDisabled}
                data-testid="analyze-button"
                aria-label="Analyze image with vision model"
              >
                {isAnalyzing ? (
                  <>
                    <span className="spinner-hud inline" />
                    <span>ANALYZING...</span>
                  </>
                ) : (
                  <>
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                      <circle cx="12" cy="12" r="10" />
                      <line x1="22" y1="12" x2="18" y2="12" />
                      <line x1="6" y1="12" x2="2" y2="12" />
                      <line x1="12" y1="6" x2="12" y2="2" />
                      <line x1="12" y1="22" x2="12" y2="18" />
                    </svg>
                    <span>ANALYZE IMAGE</span>
                  </>
                )}
              </button>
            )}
          </div>
        )}
      </div>
    </div>
  );
};
