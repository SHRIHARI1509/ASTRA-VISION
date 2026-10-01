import React, { useState, useCallback } from 'react';
import { ImageDropzone } from './ImageDropzone';
import { ImagePreviewWorkspace } from './ImagePreviewWorkspace';
import { BatchWorkspace } from './BatchWorkspace';
import { apiService } from '../services';
import { validateImageClientSide } from '../utils';
import type {
  ClientImageMetadata,
  ImageValidationResult,
  ClassificationResponse,
} from '../types';

export interface AnalysisWorkspaceProps {
  onSingleSuccess?: (filename: string, result: ClassificationResponse, previewUrl?: string) => void;
  onBatchSuccess?: (items: Array<{ filename: string; result: ClassificationResponse; previewUrl?: string }>) => void;
}

export const AnalysisWorkspace: React.FC<AnalysisWorkspaceProps> = ({
  onSingleSuccess,
  onBatchSuccess,
}) => {
  const [mode, setMode] = useState<'single' | 'batch'>('single');
  const [selectedImage, setSelectedImage] = useState<ClientImageMetadata | null>(null);
  const [isValidating, setIsValidating] = useState<boolean>(false);
  const [validationResult, setValidationResult] = useState<ImageValidationResult | null>(null);
  const [validationError, setValidationError] = useState<string | null>(null);
  const [clientError, setClientError] = useState<string | null>(null);

  // Phase 5: Classification Engine State
  const [isAnalyzing, setIsAnalyzing] = useState<boolean>(false);
  const [classificationResult, setClassificationResult] = useState<ClassificationResponse | null>(null);
  const [classificationError, setClassificationError] = useState<string | null>(null);

  const handleImageSelected = useCallback(async (file: File) => {
    // Clear errors and previous validation
    setClientError(null);
    setValidationError(null);
    setValidationResult(null);

    // CRITICAL: Immediately clear stale classification results when selecting a new image
    setClassificationResult(null);
    setClassificationError(null);

    // 1. Client-Side Pre-validation
    let metadata: ClientImageMetadata;
    try {
      metadata = await validateImageClientSide(file);
      setSelectedImage(metadata);
    } catch (err) {
      const msg = err instanceof Error ? err.message : 'Invalid image file.';
      setClientError(msg);
      return;
    }

    // 2. Server-Side Independent Validation
    setIsValidating(true);
    try {
      const result = await apiService.validateImage(file);
      setValidationResult(result);
    } catch (err) {
      const msg = err instanceof Error ? err.message : 'Backend image validation failed.';
      setValidationError(msg);
    } finally {
      setIsValidating(false);
    }
  }, []);

  const handleReset = useCallback(() => {
    if (selectedImage?.previewUrl) {
      URL.revokeObjectURL(selectedImage.previewUrl);
    }
    setSelectedImage(null);
    setValidationResult(null);
    setValidationError(null);
    setClientError(null);
    setIsValidating(false);

    // Reset classification state
    setIsAnalyzing(false);
    setClassificationResult(null);
    setClassificationError(null);
  }, [selectedImage]);

  const handleRetryValidation = useCallback(() => {
    if (selectedImage?.file) {
      handleImageSelected(selectedImage.file);
    }
  }, [selectedImage, handleImageSelected]);

  const handleAnalyze = useCallback(async () => {
    // Guard against duplicate clicks or running while validating/analyzing
    if (!selectedImage?.file || isAnalyzing || isValidating) {
      return;
    }

    // Require valid image
    if (!validationResult?.valid || validationError) {
      return;
    }

    setIsAnalyzing(true);
    setClassificationError(null);

    try {
      const result = await apiService.classifyImage(selectedImage.file);
      setClassificationResult(result);
      if (onSingleSuccess) {
        onSingleSuccess(selectedImage.filename, result, selectedImage.previewUrl);
      }
    } catch (err) {
      const msg = err instanceof Error ? err.message : 'Image classification failed.';
      setClassificationError(msg);
    } finally {
      setIsAnalyzing(false);
    }
  }, [selectedImage, isAnalyzing, isValidating, validationResult, validationError, onSingleSuccess]);

  const handleRetryAnalysis = useCallback(() => {
    handleAnalyze();
  }, [handleAnalyze]);

  const isFrameActive = Boolean(isValidating || isAnalyzing);

  return (
    <section className="analysis-workspace-card">
      <div className="card-top-bar">
        <div className="frame-meta">
          <span className="channel-id">CH-01 [OPTICAL / RECONNAISSANCE]</span>
          <span className="hud-coords">
            {mode === 'single'
              ? 'MODE: PHASE 5 END-TO-END CLASSIFICATION'
              : 'MODE: PHASE 8C MULTI-TARGET BATCH INGESTION'}
          </span>
        </div>
        <div className="frame-status">
          <span className={`dot ${isFrameActive ? 'pulse' : classificationResult ? 'active' : ''}`} />
          <span>
            {mode === 'batch'
              ? 'BATCH PIPELINE ACTIVE'
              : isAnalyzing
              ? 'ANALYZING...'
              : isValidating
              ? 'VALIDATING...'
              : classificationResult
              ? 'ANALYSIS COMPLETE'
              : selectedImage
              ? 'IMAGE MOUNTED'
              : 'INPUT STANDBY'}
          </span>
        </div>
      </div>

      {/* Tactical Mode Switcher */}
      <div className="workspace-mode-selector" data-testid="workspace-mode-selector">
        <button
          type="button"
          className={`mode-selector-btn ${mode === 'single' ? 'active' : ''}`}
          onClick={() => setMode('single')}
          data-testid="mode-single-tab"
        >
          TARGET RECONNAISSANCE (SINGLE)
        </button>
        <button
          type="button"
          className={`mode-selector-btn ${mode === 'batch' ? 'active' : ''}`}
          onClick={() => setMode('batch')}
          data-testid="mode-batch-tab"
        >
          MULTI-TARGET BATCH INGESTION (MAX 20)
        </button>
      </div>

      <div className="workspace-body">
        {mode === 'batch' ? (
          <BatchWorkspace onBatchSuccess={onBatchSuccess} />
        ) : (
          <>
            {/* Client Error Notice */}
            {clientError && (
              <div className="client-error-banner" role="alert">
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <circle cx="12" cy="12" r="10" />
                  <line x1="12" y1="8" x2="12" y2="12" />
                  <line x1="12" y1="16" x2="12.01" y2="16" />
                </svg>
                <div className="client-error-text">
                  <strong>INPUT REJECTED:</strong> {clientError}
                </div>
                <button
                  type="button"
                  className="dismiss-error-btn"
                  onClick={() => setClientError(null)}
                  aria-label="Dismiss error"
                >
                  DISMISS
                </button>
              </div>
            )}

            {/* Workspace Display: Dropzone vs Preview */}
            {!selectedImage ? (
              <ImageDropzone onImageSelected={handleImageSelected} disabled={isValidating} />
            ) : (
              <ImagePreviewWorkspace
                imageMeta={selectedImage}
                isValidating={isValidating}
                validationResult={validationResult}
                validationError={validationError}
                isAnalyzing={isAnalyzing}
                classificationResult={classificationResult}
                classificationError={classificationError}
                onReset={handleReset}
                onRetryValidation={handleRetryValidation}
                onAnalyze={handleAnalyze}
                onRetryAnalysis={handleRetryAnalysis}
              />
            )}
          </>
        )}
      </div>

      <div className="card-bottom-banner">
        <span className="phase-scope-tag">
          {mode === 'single' ? 'PHASE 5 SCOPE:' : 'PHASE 8C SCOPE:'}
        </span>
        <span className="phase-scope-desc">
          {mode === 'single'
            ? 'End-to-End Workflow: Ingestion → Validation → SigLIP 2 Inference → Tactical Result Presentation.'
            : 'Multi-Target Ingestion: Sequential Batch Processing (Max 20) → Partial Failure Resilience → Production SigLIP 2.'}
        </span>
      </div>
    </section>
  );
};

