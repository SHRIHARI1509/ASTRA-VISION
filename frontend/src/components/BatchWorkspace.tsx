import React, { useState, useRef, useCallback } from 'react';
import { apiService } from '../services';
import type {
  BatchItemResult,
  BatchClassificationResponse,
  ClassificationResponse,
} from '../types';

export const MAX_BATCH_SIZE = 20;

interface BatchWorkspaceProps {
  onBatchSuccess?: (items: Array<{ filename: string; result: ClassificationResponse; previewUrl?: string }>) => void;
}

interface StagedFile {
  file: File;
  previewUrl: string;
  isValid: boolean;
  validationError?: string;
}

export const BatchWorkspace: React.FC<BatchWorkspaceProps> = ({ onBatchSuccess }) => {
  const [stagedFiles, setStagedFiles] = useState<StagedFile[]>([]);
  const [batchError, setBatchError] = useState<string | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState<boolean>(false);
  const [currentProgress, setCurrentProgress] = useState<{ current: number; total: number; filename: string } | null>(null);
  const [batchResults, setBatchResults] = useState<BatchClassificationResponse | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const objectUrlsRef = useRef<Set<string>>(new Set());

  const cleanupObjectUrls = useCallback(() => {
    for (const url of objectUrlsRef.current) {
      URL.revokeObjectURL(url);
    }
    objectUrlsRef.current.clear();
  }, []);

  const handleFilesSelected = useCallback((files: FileList | File[]) => {
    setBatchError(null);
    setBatchResults(null);
    cleanupObjectUrls();

    const fileArray = Array.from(files);

    if (fileArray.length === 0) {
      return;
    }

    if (fileArray.length > MAX_BATCH_SIZE) {
      setBatchError(
        `BATCH LIMIT EXCEEDED: Selected ${fileArray.length} files exceeds maximum batch limit of ${MAX_BATCH_SIZE} images. Please select up to ${MAX_BATCH_SIZE} files.`
      );
      setStagedFiles([]);
      return;
    }

    const staged: StagedFile[] = fileArray.map((file) => {
      const ext = '.' + file.name.split('.').pop()?.toLowerCase();
      const validExts = ['.jpg', '.jpeg', '.png', '.webp'];
      const validMimes = ['image/jpeg', 'image/png', 'image/webp'];
      let isValid = true;
      let validationError: string | undefined;

      if (!validExts.includes(ext) && !validMimes.includes(file.type)) {
        isValid = false;
        validationError = `Unsupported format (${ext || file.type || 'unknown'}). Supported: JPG, JPEG, PNG, WEBP.`;
      } else if (file.size === 0) {
        isValid = false;
        validationError = 'Empty file (0 bytes).';
      } else if (file.size > 10 * 1024 * 1024) {
        isValid = false;
        validationError = `File exceeds 10 MB limit (${(file.size / (1024 * 1024)).toFixed(2)} MB).`;
      }

      let previewUrl = '';
      try {
        previewUrl = URL.createObjectURL(file);
        objectUrlsRef.current.add(previewUrl);
      } catch {
        previewUrl = '';
      }

      return {
        file,
        previewUrl,
        isValid,
        validationError,
      };
    });

    setStagedFiles(staged);
  }, [cleanupObjectUrls]);

  const handleFileInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      handleFilesSelected(e.target.files);
    }
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (isAnalyzing) return;
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      handleFilesSelected(e.dataTransfer.files);
    }
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
  };

  const handleLoadSampleBatch = async () => {
    if (isAnalyzing) return;
    try {
      const samples = [
        { path: '/samples/tank.jpg', name: 'batch_tank.jpg', type: 'image/jpeg' },
        { path: '/samples/ship.png', name: 'batch_ship.png', type: 'image/png' },
        { path: '/samples/drone.webp', name: 'batch_drone.webp', type: 'image/webp' },
      ];

      const files: File[] = [];
      for (const s of samples) {
        const res = await fetch(s.path);
        const blob = await res.blob();
        files.push(new File([blob], s.name, { type: s.type }));
      }
      handleFilesSelected(files);
    } catch (err) {
      console.error('Failed to load sample batch:', err);
    }
  };

  const handleLoadMixedSampleBatch = async () => {
    if (isAnalyzing) return;
    try {
      const samples = [
        { path: '/samples/tank.jpg', name: 'recon_tank.jpg', type: 'image/jpeg' },
        { path: '/samples/invalid_doc.pdf', name: 'corrupt_report.pdf', type: 'application/pdf' },
        { path: '/samples/ship.png', name: 'recon_ship.png', type: 'image/png' },
      ];

      const files: File[] = [];
      for (const s of samples) {
        const res = await fetch(s.path);
        const blob = await res.blob();
        files.push(new File([blob], s.name, { type: s.type }));
      }
      handleFilesSelected(files);
    } catch (err) {
      console.error('Failed to load mixed sample batch:', err);
    }
  };

  const handleReset = useCallback(() => {
    cleanupObjectUrls();
    setStagedFiles([]);
    setBatchError(null);
    setIsAnalyzing(false);
    setCurrentProgress(null);
    setBatchResults(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  }, [cleanupObjectUrls]);

  const handleExecuteBatch = async () => {
    if (stagedFiles.length === 0 || isAnalyzing) return;

    setIsAnalyzing(true);
    setBatchError(null);
    setCurrentProgress({ current: 0, total: stagedFiles.length, filename: stagedFiles[0].file.name });

    try {
      const filesToProcess = stagedFiles.map((s) => s.file);
      const response = await apiService.classifyBatchWithProgress(
        filesToProcess,
        (current, total, filename) => {
          setCurrentProgress({ current, total, filename });
        }
      );

      // Attach thumbnails to results for rendering
      const enrichedResults: BatchItemResult[] = response.results.map((res, idx) => {
        const staged = stagedFiles[idx];
        if (res.status === 'success') {
          return {
            ...res,
            thumbnailUrl: staged?.previewUrl,
          };
        }
        return res;
      });

      const finalResponse: BatchClassificationResponse = {
        ...response,
        results: enrichedResults,
      };

      setBatchResults(finalResponse);

      // Propagate successful items to history
      if (onBatchSuccess) {
        const successfulHistoryItems = enrichedResults
          .filter((r): r is BatchItemResult & { status: 'success'; thumbnailUrl?: string } => r.status === 'success')
          .map((r) => ({
            filename: r.filename,
            result: {
              prediction: r.prediction,
              candidates: r.candidates,
              top_3: r.top_3,
              uncertainty: r.uncertainty,
              model_fit: r.model_fit,
              inference: r.inference,
              status: 'success',
            },
            previewUrl: r.thumbnailUrl,
          }));

        if (successfulHistoryItems.length > 0) {
          onBatchSuccess(successfulHistoryItems);
        }
      }
    } catch (err) {
      const msg = err instanceof Error ? err.message : 'Batch classification failed.';
      setBatchError(msg);
    } finally {
      setIsAnalyzing(false);
      setCurrentProgress(null);
    }
  };

  const getWorkflowState = (): string => {
    if (isAnalyzing) return 'ANALYZING BATCH';
    if (batchResults) return 'BATCH RESULTS';
    if (stagedFiles.length > 0) return 'FILES SELECTED';
    return 'BATCH STANDBY';
  };

  return (
    <div className="batch-workspace-container" data-testid="batch-workspace">
      {/* Batch Header Bar */}
      <div className="batch-header-bar">
        <div className="batch-title-group">
          <span className="batch-tag">CH-02 [MULTI-TARGET INGESTION]</span>
          <span className="batch-state-badge" data-testid="batch-state-badge">
            STATE: {getWorkflowState()}
          </span>
        </div>
        <div className="batch-limits-info">
          <span>MAX BATCH: {MAX_BATCH_SIZE} IMAGES</span>
          <span className="separator">|</span>
          <span>SEQUENTIAL SIGLIP 2 PIPELINE</span>
        </div>
      </div>

      {/* Batch Error Banner */}
      {batchError && (
        <div className="client-error-banner" role="alert" data-testid="batch-error-banner">
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <circle cx="12" cy="12" r="10" />
            <line x1="12" y1="8" x2="12" y2="12" />
            <line x1="12" y1="16" x2="12.01" y2="16" />
          </svg>
          <div className="client-error-text">
            <strong>BATCH INGESTION NOTICE:</strong> {batchError}
          </div>
          <button
            type="button"
            className="dismiss-error-btn"
            onClick={() => setBatchError(null)}
            aria-label="Dismiss error"
          >
            DISMISS
          </button>
        </div>
      )}

      {/* Standby / Multi-File Dropzone */}
      {stagedFiles.length === 0 && !batchResults && (
        <div
          className="tactical-dropzone batch-dropzone"
          onDrop={handleDrop}
          onDragOver={handleDragOver}
          onClick={() => fileInputRef.current?.click()}
          role="button"
          tabIndex={0}
          aria-label="Upload multiple reconnaissance images for batch classification"
          data-testid="batch-dropzone"
        >
          <input
            ref={fileInputRef}
            type="file"
            multiple
            accept=".jpg,.jpeg,.png,.webp,image/jpeg,image/png,image/webp"
            onChange={handleFileInputChange}
            style={{ position: 'absolute', opacity: 0, width: '1px', height: '1px', pointerEvents: 'none' }}
            data-testid="batch-file-input"
            id="batch-recon-file-input"
          />

          <div className="reticle-corner top-left" />
          <div className="reticle-corner top-right" />
          <div className="reticle-corner bottom-left" />
          <div className="reticle-corner bottom-right" />

          <div className="dropzone-content">
            <div className="dropzone-icon-hud">
              <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                <rect x="2" y="7" width="20" height="14" rx="2" ry="2" />
                <path d="M16 21V5a2 2 0 0 0-2-2h-4a2 2 0 0 0-2 2v16" />
              </svg>
              <div className="icon-pulse-ring" />
            </div>

            <div className="dropzone-text-group">
              <h3 className="dropzone-title">SELECT OR DROP MULTIPLE RECONNAISSANCE IMAGES</h3>
              <p className="dropzone-subtitle">
                Batch processing pipeline — evaluate up to 20 images against the defence taxonomy
              </p>
            </div>

            <div className="format-badges-container">
              <span className="format-badge">JPG</span>
              <span className="format-badge">JPEG</span>
              <span className="format-badge">PNG</span>
              <span className="format-badge">WEBP</span>
              <span className="size-limit-badge">MAX 20 FILES</span>
              <span className="size-limit-badge">10 MB / FILE</span>
            </div>

            <div className="dropzone-action-cluster" onClick={(e) => e.stopPropagation()}>
              <button
                type="button"
                className="tactical-browse-btn"
                onClick={() => fileInputRef.current?.click()}
                data-testid="batch-browse-btn"
              >
                SELECT BATCH FILES
              </button>

              <div className="sample-quick-loads">
                <span className="sample-prompt">QUICK TEST:</span>
                <button
                  type="button"
                  className="sample-pill-btn"
                  onClick={handleLoadSampleBatch}
                  data-testid="load-sample-batch-btn"
                >
                  3 SAMPLE TARGETS (VALID)
                </button>
                <button
                  type="button"
                  className="sample-pill-btn danger"
                  onClick={handleLoadMixedSampleBatch}
                  data-testid="load-mixed-batch-btn"
                >
                  MIXED BATCH (VALID + INVALID)
                </button>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Files Selected / Staging Area */}
      {stagedFiles.length > 0 && !batchResults && (
        <div className="batch-staging-container" data-testid="batch-staging-area">
          <div className="staging-header">
            <div className="staging-counts">
              <span className="count-badge total" data-testid="staged-count">
                TOTAL: {stagedFiles.length} / {MAX_BATCH_SIZE}
              </span>
              <span className="count-badge ready">
                READY: {stagedFiles.filter((s) => s.isValid).length}
              </span>
              <span className="count-badge invalid">
                INVALID: {stagedFiles.filter((s) => !s.isValid).length}
              </span>
            </div>

            <div className="staging-actions">
              <button
                type="button"
                className="tactical-btn secondary"
                onClick={handleReset}
                disabled={isAnalyzing}
                data-testid="batch-reset-btn"
              >
                CLEAR SELECTION
              </button>
              <button
                type="button"
                className="tactical-btn primary"
                onClick={handleExecuteBatch}
                disabled={isAnalyzing || stagedFiles.length === 0}
                data-testid="batch-analyze-btn"
              >
                {isAnalyzing ? 'ANALYZING...' : `ANALYZE BATCH (${stagedFiles.length} IMAGES)`}
              </button>
            </div>
          </div>

          {/* Honest Progress Indicator */}
          {isAnalyzing && currentProgress && (
            <div className="batch-progress-card" data-testid="batch-progress-card">
              <div className="progress-label-row">
                <span className="progress-status-text" data-testid="batch-progress-text">
                  ANALYZING {currentProgress.current} / {currentProgress.total}: {currentProgress.filename}
                </span>
                <span className="progress-count-text">
                  {Math.round((currentProgress.current / currentProgress.total) * 100)}%
                </span>
              </div>
              <div className="progress-track">
                <div
                  className="progress-fill"
                  style={{
                    width: `${(currentProgress.current / currentProgress.total) * 100}%`,
                  }}
                />
              </div>
              <div className="progress-subnote">
                Sequential SigLIP 2 execution — one model singleton instance on CPU/device.
              </div>
            </div>
          )}

          {/* Staged Items Grid */}
          <div className="staged-grid">
            {stagedFiles.map((item, idx) => (
              <div
                key={`staged-${item.file.name}-${idx}`}
                className={`staged-item-card ${item.isValid ? 'valid' : 'invalid'}`}
              >
                <div className="staged-thumb-wrapper">
                  {item.previewUrl ? (
                    <img src={item.previewUrl} alt={item.file.name} className="staged-thumb" />
                  ) : (
                    <div className="staged-no-thumb">FILE</div>
                  )}
                  <span className={`staged-status-indicator ${item.isValid ? 'ok' : 'err'}`}>
                    {item.isValid ? 'VALID' : 'INVALID'}
                  </span>
                </div>
                <div className="staged-meta">
                  <span className="staged-name" title={item.file.name}>
                    {item.file.name}
                  </span>
                  <span className="staged-size">{(item.file.size / 1024).toFixed(1)} KB</span>
                  {item.validationError && (
                    <span className="staged-error-text">{item.validationError}</span>
                  )}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Batch Results View */}
      {batchResults && (
        <div className="batch-results-container" data-testid="batch-results-view">
          <div className="batch-results-summary-bar">
            <div className="summary-stat-cluster">
              <div className="stat-pill total">
                <span className="stat-label">TOTAL IMAGES</span>
                <span className="stat-val" data-testid="results-total-count">
                  {batchResults.total}
                </span>
              </div>
              <div className="stat-pill success">
                <span className="stat-label">SUCCESSFUL</span>
                <span className="stat-val" data-testid="results-success-count">
                  {batchResults.successful}
                </span>
              </div>
              <div className="stat-pill failed">
                <span className="stat-label">FAILED / REJECTED</span>
                <span className="stat-val" data-testid="results-failed-count">
                  {batchResults.failed}
                </span>
              </div>
            </div>

            <div className="results-actions">
              <button
                type="button"
                className="tactical-btn primary"
                onClick={handleReset}
                data-testid="batch-results-reset-btn"
              >
                START NEW BATCH
              </button>
            </div>
          </div>

          {/* Results Grid */}
          <div className="batch-results-grid">
            {batchResults.results.map((res, idx) => (
              <div
                key={`batch-res-${res.filename}-${idx}`}
                className={`batch-result-card ${res.status === 'success' ? 'card-success' : 'card-error'}`}
                data-testid={`batch-result-card-${idx}`}
              >
                {res.status === 'success' ? (
                  <>
                    <div className="result-card-header">
                      <span className="result-filename" title={res.filename}>
                        {res.filename}
                      </span>
                      <span className="latency-badge">{res.inference.inference_time_ms.toFixed(0)} ms</span>
                    </div>

                    <div className="result-card-body">
                      {res.thumbnailUrl && (
                        <div className="result-thumb-box">
                          <img src={res.thumbnailUrl} alt={res.filename} className="result-thumb-img" />
                        </div>
                      )}

                      <div className="result-primary-box">
                        <div className="result-target-label" data-testid={`batch-result-label-${idx}`}>
                          {res.prediction.label}
                        </div>
                        <div className="result-score-bar-wrapper">
                          <div
                            className="result-score-fill"
                            style={{ width: `${Math.min(res.prediction.score * 100, 100)}%` }}
                          />
                        </div>
                        <div className="result-score-meta">
                          <span>CONFIDENCE SCORE</span>
                          <span className="score-val">{(res.prediction.score * 100).toFixed(1)}%</span>
                        </div>
                      </div>

                      {/* Uncertainty Badge */}
                      {res.uncertainty?.is_uncertain && (
                        <div className="batch-uncertain-badge" data-testid={`batch-uncertain-${idx}`}>
                          ⚠️ UNCERTAIN: {res.uncertainty.reason || 'LOW CONFIDENCE'}
                        </div>
                      )}

                      {/* Top-3 Candidates */}
                      {res.top_3 && res.top_3.length > 0 && (
                        <div className="result-candidates-list">
                          <div className="candidates-header">TOP-3 CANDIDATES</div>
                          {res.top_3.map((cand, cIdx) => (
                            <div key={`cand-${cand.label}-${cIdx}`} className="candidate-row">
                              <span className="cand-rank">#{cIdx + 1}</span>
                              <span className="cand-label">{cand.label}</span>
                              <span className="cand-score">{(cand.score * 100).toFixed(1)}%</span>
                            </div>
                          ))}
                        </div>
                      )}
                    </div>
                  </>
                ) : (
                  <>
                    <div className="result-card-header error-head">
                      <span className="result-filename error-name" title={res.filename}>
                        {res.filename}
                      </span>
                      <span className="error-status-badge">REJECTED</span>
                    </div>
                    <div className="error-card-body">
                      <div className="error-icon-box">
                        <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                          <circle cx="12" cy="12" r="10" />
                          <line x1="15" y1="9" x2="9" y2="15" />
                          <line x1="9" y1="9" x2="15" y2="15" />
                        </svg>
                      </div>
                      <div className="error-code-badge">{res.error.code}</div>
                      <div className="error-reason-text" data-testid={`batch-error-reason-${idx}`}>
                        {res.error.message}
                      </div>
                    </div>
                  </>
                )}
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};
