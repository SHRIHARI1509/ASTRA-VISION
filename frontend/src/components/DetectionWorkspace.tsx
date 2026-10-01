import React, { useState, useCallback, useRef } from 'react';
import { apiService } from '../services';
import type { DetectionApiResponse, DetectedObjectItem } from '../types';

export interface DetectionWorkspaceProps {
  onDetectionSuccess?: (result: DetectionApiResponse, filename: string) => void;
}

export const DetectionWorkspace: React.FC<DetectionWorkspaceProps> = ({
  onDetectionSuccess,
}) => {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [isDetecting, setIsDetecting] = useState<boolean>(false);
  const [detectionResult, setDetectionResult] = useState<DetectionApiResponse | null>(null);
  const [detectionError, setDetectionError] = useState<string | null>(null);
  const [dragActive, setDragActive] = useState<boolean>(false);
  const [boxThreshold, setBoxThreshold] = useState<number>(0.35);

  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFile = useCallback((file: File) => {
    // Validate MIME type
    const validMimes = ['image/jpeg', 'image/png', 'image/webp'];
    if (!validMimes.includes(file.type)) {
      setDetectionError('Invalid image format. Supported formats: JPEG, PNG, WEBP.');
      return;
    }

    if (file.size > 10 * 1024 * 1024) {
      setDetectionError('Image file exceeds the 10 MB maximum size limit.');
      return;
    }

    // Clean up previous preview URL
    if (previewUrl) {
      URL.revokeObjectURL(previewUrl);
    }

    const objectUrl = URL.createObjectURL(file);
    setSelectedFile(file);
    setPreviewUrl(objectUrl);
    setDetectionResult(null);
    setDetectionError(null);
  }, [previewUrl]);

  const handleFileInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      handleFile(e.target.files[0]);
    }
  };

  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      handleFile(e.dataTransfer.files[0]);
    }
  };

  const handleRunDetection = useCallback(async () => {
    if (!selectedFile || isDetecting) return;

    setIsDetecting(true);
    setDetectionError(null);

    try {
      const response = await apiService.detectObjects(selectedFile, boxThreshold);
      setDetectionResult(response);
      if (onDetectionSuccess) {
        onDetectionSuccess(response, selectedFile.name);
      }
    } catch (err) {
      const msg = err instanceof Error ? err.message : 'Object detection request failed.';
      setDetectionError(msg);
    } finally {
      setIsDetecting(false);
    }
  }, [selectedFile, isDetecting, boxThreshold, onDetectionSuccess]);

  const handleReset = useCallback(() => {
    if (previewUrl) {
      URL.revokeObjectURL(previewUrl);
    }
    setSelectedFile(null);
    setPreviewUrl(null);
    setDetectionResult(null);
    setDetectionError(null);
    setIsDetecting(false);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  }, [previewUrl]);

  // Color mapping by defence class
  const getCategoryColor = (className: string): string => {
    switch (className) {
      case 'Tank':
        return '#00ffcc'; // tactical cyan
      case 'Military Vehicle':
        return '#ffb703'; // amber gold
      case 'Fighter Aircraft':
        return '#00b4d8'; // azure blue
      case 'Helicopter':
        return '#a78bfa'; // violet
      case 'Ship':
        return '#38bdf8'; // sky blue
      case 'Drone':
        return '#f43f5e'; // crimson rose
      default:
        return '#10b981'; // emerald
    }
  };

  return (
    <section className="detection-workspace-card" data-testid="detection-workspace">
      {/* Header Bar */}
      <div className="card-top-bar">
        <div className="frame-meta">
          <span className="channel-id">CH-02 [SPATIAL DETECTOR]</span>
          <span className="hud-coords">MODEL: IDEA-Research/grounding-dino-base</span>
        </div>
        <div className="frame-status">
          <span className={`dot ${isDetecting ? 'pulse' : detectionResult ? 'active' : ''}`} />
          <span>
            {isDetecting
              ? 'RUNNING DETECTION...'
              : detectionResult
              ? `DETECTED: ${detectionResult.count} OBJECT${detectionResult.count === 1 ? '' : 'S'}`
              : selectedFile
              ? 'IMAGE LOADED'
              : 'INPUT STANDBY'}
          </span>
        </div>
      </div>

      <div className="detection-header-content">
        <div className="detection-title-wrap">
          <h2 className="detection-heading" data-testid="detection-heading">OBJECT DETECTION</h2>
          <p className="detection-subtitle">
            Detect multiple supported objects and locate them with bounding boxes.
          </p>
        </div>
        <div className="detection-model-badge">
          <span className="badge-detector">PRETRAINED OPEN-VOCABULARY DETECTOR</span>
          <span className="badge-model-id">IDEA-Research/grounding-dino-base</span>
        </div>
      </div>

      {/* Error Alert */}
      {detectionError && (
        <div className="client-error-banner" role="alert" data-testid="detection-error-banner">
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <circle cx="12" cy="12" r="10" />
            <line x1="12" y1="8" x2="12" y2="12" />
            <line x1="12" y1="16" x2="12.01" y2="16" />
          </svg>
          <div className="client-error-text">
            <strong>DETECTION ERROR:</strong> {detectionError}
          </div>
          <button
            type="button"
            className="dismiss-error-btn"
            onClick={() => setDetectionError(null)}
            aria-label="Dismiss error"
          >
            DISMISS
          </button>
        </div>
      )}

      {/* Workspace Body */}
      <div className="detection-workspace-body">
        {!selectedFile ? (
          <div
            className={`detection-dropzone ${dragActive ? 'drag-active' : ''}`}
            onDragEnter={handleDrag}
            onDragLeave={handleDrag}
            onDragOver={handleDrag}
            onDrop={handleDrop}
            onClick={() => fileInputRef.current?.click()}
            data-testid="detection-dropzone"
          >
            <input
              ref={fileInputRef}
              type="file"
              accept=".jpg,.jpeg,.png,.webp"
              onChange={handleFileInputChange}
              style={{ display: 'none' }}
              data-testid="detection-file-input"
            />
            <div className="dropzone-tactical-icon">
              <svg width="44" height="44" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                <rect x="3" y="3" width="18" height="18" rx="2" strokeDasharray="3 3" />
                <path d="M12 8v8" />
                <path d="M8 12h8" />
              </svg>
            </div>
            <div className="dropzone-text-primary">SELECT OR DROP RECONNAISSANCE IMAGE FOR DETECTION</div>
            <div className="dropzone-text-sub">
              Supported Formats: JPEG, PNG, WEBP • Max Size: 10 MB • Auto RGB Conversion
            </div>
            <button
              type="button"
              className="dropzone-action-btn"
              onClick={(e) => {
                e.stopPropagation();
                fileInputRef.current?.click();
              }}
            >
              BROWSE FILES
            </button>
          </div>
        ) : (
          <div className="detection-viewer-layout">
            {/* Visual Canvas Area with Responsive Overlay */}
            <div className="detection-canvas-container" data-testid="detection-canvas-container">
              <div
                className="detection-image-wrapper"
                style={{
                  position: 'relative',
                  display: 'inline-block',
                  maxWidth: '100%',
                  lineHeight: 0,
                }}
              >
                {previewUrl && (
                  <img
                    src={previewUrl}
                    alt={selectedFile.name}
                    className="detection-rendered-image"
                    data-testid="detection-rendered-image"
                    style={{
                      display: 'block',
                      width: '100%',
                      height: 'auto',
                      maxHeight: '520px',
                      objectFit: 'contain',
                      borderRadius: '4px',
                    }}
                  />
                )}

                {/* SVG Bounding Box Overlay (Mathematically scaled via viewBox) */}
                {detectionResult && (
                  <svg
                    className="detection-svg-overlay"
                    data-testid="detection-svg-overlay"
                    viewBox={`0 0 ${detectionResult.image_width} ${detectionResult.image_height}`}
                    preserveAspectRatio="none"
                    style={{
                      position: 'absolute',
                      top: 0,
                      left: 0,
                      width: '100%',
                      height: '100%',
                      pointerEvents: 'none',
                    }}
                  >
                    {detectionResult.detections.map((det: DetectedObjectItem, idx: number) => {
                      const { x1, y1, x2, y2 } = det.box;
                      const width = Math.max(1, x2 - x1);
                      const height = Math.max(1, y2 - y1);
                      const color = getCategoryColor(det.class_name);
                      const pctScore = Math.round(det.score * 100);
                      const labelText = `${det.class_name.toUpperCase()} ${pctScore}%`;
                      const labelWidth = Math.max(85, labelText.length * 9 + 16);
                      const labelHeight = 22;
                      const labelY = Math.max(0, y1 - labelHeight);

                      return (
                        <g key={`box-${idx}`} className="detection-box-group" data-testid={`detection-box-${idx}`}>
                          {/* Translucent boundary fill */}
                          <rect
                            x={x1}
                            y={y1}
                            width={width}
                            height={height}
                            fill={color}
                            fillOpacity="0.16"
                            stroke={color}
                            strokeWidth="3"
                          />
                          {/* Header label badge */}
                          <rect
                            x={x1}
                            y={labelY}
                            width={labelWidth}
                            height={labelHeight}
                            fill={color}
                            rx="2"
                          />
                          {/* Label text */}
                          <text
                            x={x1 + 6}
                            y={labelY + 15}
                            fill="#0d1117"
                            fontSize="11"
                            fontFamily="monospace, sans-serif"
                            fontWeight="bold"
                            letterSpacing="0.5px"
                          >
                            {labelText}
                          </text>
                        </g>
                      );
                    })}
                  </svg>
                )}
              </div>
            </div>

            {/* Side Controls & Results Panel */}
            <div className="detection-sidebar">
              {/* File Info Card */}
              <div className="detection-file-info">
                <div className="file-info-label">MOUNTED RECON TARGET</div>
                <div className="file-info-name" title={selectedFile.name}>
                  {selectedFile.name}
                </div>
                <div className="file-info-size">
                  {(selectedFile.size / 1024).toFixed(1)} KB • {selectedFile.type.split('/')[1]?.toUpperCase()}
                </div>
              </div>

              {/* Threshold Configuration */}
              <div className="detection-threshold-control">
                <div className="threshold-label-row">
                  <span>DETECTION THRESHOLD:</span>
                  <span className="threshold-val">{boxThreshold.toFixed(2)}</span>
                </div>
                <input
                  type="range"
                  min="0.15"
                  max="0.80"
                  step="0.05"
                  value={boxThreshold}
                  disabled={isDetecting}
                  onChange={(e) => setBoxThreshold(parseFloat(e.target.value))}
                  className="threshold-slider"
                  data-testid="detection-threshold-slider"
                />
                <div className="threshold-hint">
                  Filtering threshold (empirical score, not calibrated probability).
                </div>
              </div>

              {/* Action Buttons */}
              <div className="detection-actions-row">
                <button
                  type="button"
                  className="detection-run-btn"
                  onClick={handleRunDetection}
                  disabled={isDetecting}
                  data-testid="run-detection-btn"
                >
                  {isDetecting ? (
                    <>
                      <span className="spinner-icon" /> RUNNING INFERENCE...
                    </>
                  ) : (
                    'RUN OBJECT DETECTION'
                  )}
                </button>
                <button
                  type="button"
                  className="detection-reset-btn"
                  onClick={handleReset}
                  disabled={isDetecting}
                  data-testid="reset-detection-btn"
                >
                  RESET
                </button>
              </div>

              {/* Detections Summary */}
              {detectionResult && (
                <div className="detection-results-panel" data-testid="detection-results-panel">
                  <div className="detection-count-banner" data-testid="detection-count-banner">
                    <span className="count-label">DETECTED OBJECTS:</span>
                    <span className="count-number" data-testid="detection-count-number">
                      {detectionResult.count}
                    </span>
                  </div>

                  {/* Empty Detection State */}
                  {detectionResult.count === 0 ? (
                    <div className="detection-empty-state" data-testid="detection-empty-state">
                      <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                        <circle cx="12" cy="12" r="10" />
                        <line x1="8" y1="12" x2="16" y2="12" />
                      </svg>
                      <p>No matching objects detected.</p>
                      <span className="empty-sub">
                        No targets exceeded the operating threshold ({boxThreshold.toFixed(2)}) for supported taxonomy classes.
                      </span>
                    </div>
                  ) : (
                    <div className="detection-items-list" data-testid="detection-items-list">
                      {detectionResult.detections.map((det, index) => {
                        const pct = Math.round(det.score * 100);
                        const color = getCategoryColor(det.class_name);
                        return (
                          <div key={index} className="detection-item-card" data-testid={`detection-item-${index}`}>
                            <div className="item-header-row">
                              <span className="item-badge" style={{ borderColor: color, color }}>
                                #{index + 1} {det.class_name}
                              </span>
                              <span className="item-score">{pct}%</span>
                            </div>
                            <div className="item-box-coords">
                              BOX: [{det.box.x1}, {det.box.y1}, {det.box.x2}, {det.box.y2}]
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  )}

                  {/* Performance Metadata */}
                  <div className="detection-latency-footer">
                    <span>
                      LATENCY: {detectionResult.inference_time_ms ? `${Math.round(detectionResult.inference_time_ms)} ms` : 'N/A'} • {detectionResult.device?.toUpperCase()}
                    </span>
                    <span>FRAME: {detectionResult.image_width} × {detectionResult.image_height} PX</span>
                  </div>
                </div>
              )}
            </div>
          </div>
        )}
      </div>

      {/* Model Disclosure Footer */}
      <div className="card-bottom-banner">
        <span className="phase-scope-tag">PHASE 10 SCOPE:</span>
        <span className="phase-scope-desc">
          Zero-Shot Open-Vocabulary Detection via IDEA-Research/grounding-dino-base.
          Separate from SigLIP 2 classification pipeline. No synthetic annotations.
        </span>
      </div>
    </section>
  );
};
