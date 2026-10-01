import React from 'react';
import type { ClassificationResponse } from '../types';

export const CONCISE_MODEL_FIT_JUSTIFICATION =
  'SigLIP 2 fits ASTRA VISION because it directly matches images with text-based class descriptions, enabling zero-shot classification without task-specific training.';

export const DEFAULT_MODEL_FIT_JUSTIFICATION = {
  model_name: 'SigLIP 2 Base',
  model_id: 'google/siglip2-base-patch16-512',
  task: 'Image classification / image-text candidate matching',
  justification: CONCISE_MODEL_FIT_JUSTIFICATION,
  full_justification:
    "ASTRA VISION uses google/siglip2-base-patch16-512 because the operational problem requires classifying reconnaissance images against a predefined taxonomy of defence objects (such as tanks, helicopters, and aircraft). SigLIP 2 provides vision-language zero-shot alignment, allowing the system to measure similarity between visual inputs and text descriptions directly. This capability enables the baseline application to evaluate and rank candidate categories without requiring a dedicated task-specific fine-tuning pipeline or collected training datasets. The model's pairwise similarity outputs directly support ASTRA VISION's multi-candidate workflow, producing both a primary prediction and deterministic Top-3 rankings that integrate with our heuristic uncertainty checks.",
};

interface ClassificationResultPanelProps {
  result: ClassificationResponse;
  onReset: () => void;
  onReanalyze: () => void;
  isAnalyzing?: boolean;
}

export const ClassificationResultPanel: React.FC<ClassificationResultPanelProps> = ({
  result,
  onReset,
  onReanalyze,
  isAnalyzing = false,
}) => {
  const prediction = result?.prediction;
  const inference = result?.inference;

  // Format inference time into seconds and ms safely
  const inferenceTimeMs = inference?.inference_time_ms;
  const timeSeconds = typeof inferenceTimeMs === 'number' ? (inferenceTimeMs / 1000).toFixed(2) : 'N/A';
  const timeMs = typeof inferenceTimeMs === 'number' ? Math.round(inferenceTimeMs) : null;

  // Primary prediction and score
  const label = prediction?.label ?? 'Unknown';
  const score = typeof prediction?.score === 'number' ? prediction.score.toFixed(4) : 'N/A';

  // Shorten model ID for display
  const modelRaw = inference?.model ?? 'SigLIP 2';
  const modelDisplayName = modelRaw.includes('siglip2')
    ? 'SigLIP 2 (base-patch16-512)'
    : modelRaw;
  const deviceName = inference?.device ?? 'N/A';

  // Top-3 candidates safely derived from backend top_3 or candidates slice
  const rawTop3 = result?.top_3 || (Array.isArray(result?.candidates) ? result.candidates.slice(0, 3) : []);
  const top3Candidates = rawTop3.filter(
    (c): c is { label: string; score: number } =>
      Boolean(c && typeof c.label === 'string' && typeof c.score === 'number')
  );

  // Uncertainty details (Phase 7C)
  const uncertainty = result?.uncertainty;
  const isUncertain = Boolean(uncertainty?.is_uncertain);

  // Model-Fit Justification (Phase 7D)
  const modelFit = result?.model_fit || DEFAULT_MODEL_FIT_JUSTIFICATION;

  return (
    <div className="classification-result-panel" role="region" aria-label="Classification Result">
      {/* Classification Header Banner */}
      <div className="result-header-bar">
        <div className="result-title-group">
          <span className="result-badge-tag">ASTRA VISION ANALYSIS</span>
          <span className="result-sub-tag">ZERO-SHOT TACTICAL CLASSIFICATION</span>
        </div>
        <div className="result-status-badge">
          <span className="dot active" />
          <span>STATUS: COMPLETE</span>
        </div>
      </div>

      {/* Primary Prediction Display Box */}
      <div className="primary-prediction-box">
        <div className="prediction-meta-label">PRIMARY PREDICTION</div>
        <h2 className="prediction-label-text" data-testid="primary-prediction-label">
          {label}
        </h2>
      </div>

      {/* Uncertainty Warning Banner (SHOULD-HAVE Feature Phase 7C) */}
      {isUncertain && (
        <div
          className="uncertainty-warning-banner"
          role="alert"
          aria-live="polite"
          data-testid="uncertainty-warning-banner"
        >
          <div className="uncertainty-warning-header">
            <span className="uncertainty-icon" aria-hidden="true">⚠️</span>
            <span className="uncertainty-title">UNCERTAIN PREDICTION</span>
            <span className="uncertainty-reason-badge" data-testid="uncertainty-reason-badge">
              {uncertainty?.reason ? `[${uncertainty.reason.replace(/_/g, ' ')}]` : '[HEURISTIC CHECK]'}
            </span>
          </div>
          <p className="uncertainty-description">
            The vision model's available evidence does not strongly distinguish between the leading candidate categories under the heuristic threshold.
          </p>
          {typeof uncertainty?.score_margin === 'number' && (
            <div className="uncertainty-margin-details">
              <span className="margin-item">
                Score Margin: <strong data-testid="uncertainty-margin-value">{uncertainty.score_margin.toFixed(4)}</strong>
              </span>
              <span className="margin-separator">|</span>
              <span className="margin-item">
                Margin Threshold: <span>{(uncertainty.margin_threshold ?? 0.02).toFixed(4)}</span>
              </span>
            </div>
          )}
        </div>
      )}

      {/* Top 3 Predictions Section (SHOULD-HAVE Feature Phase 7A) */}
      {top3Candidates.length > 0 && (
        <div className="top3-predictions-container" data-testid="top3-predictions-container">
          <div className="top3-section-header">
            <span className="top3-heading">TOP 3 PREDICTIONS</span>
            <span className="top3-score-label">MODEL SCORE (RANKED)</span>
          </div>

          <div className="top3-candidates-list" data-testid="top3-candidates-list">
            {top3Candidates.map((candidate, idx) => (
              <div
                key={`${candidate.label}-${idx}`}
                className={`top3-candidate-card rank-${idx + 1} ${idx === 0 ? 'top-rank' : ''}`}
                data-testid={`top3-candidate-${idx + 1}`}
              >
                <div className="top3-rank-col">
                  <span className="rank-tag">{`#${idx + 1}`}</span>
                </div>
                <div className="top3-label-col">
                  <span className="candidate-label" data-testid={`top3-label-${idx + 1}`}>
                    {candidate.label}
                  </span>
                </div>
                <div className="top3-score-col">
                  <span className="candidate-score" data-testid={`top3-score-${idx + 1}`}>
                    {candidate.score.toFixed(4)}
                  </span>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Structured Metrics Grid */}
      <div className="result-metrics-grid">
        <div className="metric-card">
          <span className="metric-label">MODEL SCORE</span>
          <span className="metric-value highlight" data-testid="model-score-value">
            {score}
          </span>
          <span className="metric-caption">SigLIP 2 similarity score</span>
        </div>

        <div className="metric-card">
          <span className="metric-label">INFERENCE TIME</span>
          <span className="metric-value" data-testid="inference-time-value">
            {`${timeSeconds} s`}
          </span>
          <span className="metric-caption">
            {timeMs !== null ? `(${timeMs} ms latency)` : 'Latency N/A'}
          </span>
        </div>

        <div className="metric-card">
          <span className="metric-label">VISION MODEL</span>
          <span className="metric-value model-name" data-testid="model-name-value" title={modelRaw}>
            {modelDisplayName}
          </span>
          <span className="metric-caption">Hugging Face Transformers</span>
        </div>

        <div className="metric-card">
          <span className="metric-label">EXECUTION DEVICE</span>
          <span className="metric-value uppercase" data-testid="device-value">
            {deviceName}
          </span>
          <span className="metric-caption">Hardware backend</span>
        </div>
      </div>

      {/* Model-Fit Justification Section (SHOULD-HAVE Feature Phase 7D) */}
      <div
        className="model-fit-container"
        data-testid="model-fit-container"
        role="region"
        aria-label="Why this model"
      >
        <div className="model-fit-header">
          <div className="model-fit-title-group">
            <span className="model-fit-icon" aria-hidden="true">🎯</span>
            <span className="model-fit-heading">WHY THIS MODEL?</span>
            <span className="model-fit-badge">BASELINE JUSTIFICATION</span>
          </div>
          <div className="model-fit-meta-pill">
            <span className="meta-pill-label">MODEL:</span>
            <span className="meta-pill-val" data-testid="model-fit-name">{modelFit.model_name}</span>
          </div>
        </div>
        <p
          className="model-fit-text"
          data-testid="model-fit-text"
          title={(modelFit as any).full_justification || modelFit.justification}
          data-full-justification={(modelFit as any).full_justification || modelFit.justification}
        >
          {CONCISE_MODEL_FIT_JUSTIFICATION}
        </p>
        <div className="model-fit-footer-meta">
          <span className="meta-item">
            <strong>Identifier:</strong> <span data-testid="model-fit-id">{modelFit.model_id}</span>
          </span>
          <span className="meta-separator">•</span>
          <span className="meta-item">
            <strong>Architecture:</strong> Zero-Shot Vision Transformer
          </span>
          <span className="meta-separator">•</span>
          <span className="meta-item">
            <strong>Candidate Matching:</strong> Image-Text Similarity
          </span>
        </div>
      </div>

      {/* Result Action Buttons */}
      <div className="result-actions-row">
        <button
          type="button"
          className="tactical-btn secondary"
          onClick={onReset}
          disabled={isAnalyzing}
          aria-label="Select another image"
        >
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <polyline points="15 18 9 12 15 6" />
          </svg>
          <span>SELECT ANOTHER IMAGE</span>
        </button>

        <button
          type="button"
          className="tactical-btn primary"
          onClick={onReanalyze}
          disabled={isAnalyzing}
          aria-label="Re-analyze current image"
        >
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <polyline points="23 4 23 10 17 10" />
            <polyline points="1 20 1 14 7 14" />
            <path d="M3.51 9a9 9 0 0 1 14.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0 0 20.49 15" />
          </svg>
          <span>RE-ANALYZE</span>
        </button>
      </div>
    </div>
  );
};
