import { describe, it, expect, vi, beforeEach } from 'vitest';
import { renderToString } from 'react-dom/server';
import { ClassificationResultPanel } from '../ClassificationResultPanel';
import { AnalysisWorkspace } from '../AnalysisWorkspace';
import { ImagePreviewWorkspace } from '../ImagePreviewWorkspace';
import { apiService } from '../../services';
import type { ClassificationResponse, ClientImageMetadata, ImageValidationResult } from '../../types';

const mockMetadata: ClientImageMetadata = {
  file: new File(['fake image data'], 'tank_sample.jpg', { type: 'image/jpeg' }),
  previewUrl: 'blob:http://localhost:5173/mock-preview',
  filename: 'tank_sample.jpg',
  format: 'JPEG',
  sizeBytes: 102400,
  width: 800,
  height: 600,
  aspectRatio: 1.33,
};

const mockValidationSuccess: ImageValidationResult = {
  valid: true,
  filename: 'tank_sample.jpg',
  format: 'JPEG',
  size_bytes: 102400,
  width: 800,
  height: 600,
};

const mockClassificationResponse: ClassificationResponse = {
  prediction: {
    label: 'Main Battle Tank',
    score: 0.3132,
  },
  candidates: [
    { label: 'Main Battle Tank', score: 0.3132 },
    { label: 'Armoured Fighting Vehicle', score: 0.2215 },
  ],
  inference: {
    model: 'google/siglip2-base-patch16-512',
    device: 'cpu',
    inference_time_ms: 2289.87,
  },
  status: 'success',
};

describe('ASTRA VISION Phase 5 — Classification Result UI & Workflows', () => {
  // 1. Empty state renders
  it('1. renders the clean empty upload state by default', () => {
    const html = renderToString(<AnalysisWorkspace />);
    expect(html).toContain('SELECT OR DROP RECONNAISSANCE IMAGE');
    expect(html).toContain('INPUT STANDBY');
    expect(html).toContain('CH-01 [OPTICAL / RECONNAISSANCE]');
    expect(html).toContain('PHASE 5 END-TO-END CLASSIFICATION');
  });

  // 2. Valid image metadata preview
  it('2. displays preview workspace and metadata when image is mounted', () => {
    const html = renderToString(
      <ImagePreviewWorkspace
        imageMeta={mockMetadata}
        isValidating={false}
        validationResult={mockValidationSuccess}
        validationError={null}
        isAnalyzing={false}
        classificationResult={null}
        classificationError={null}
        onReset={vi.fn()}
        onRetryValidation={vi.fn()}
        onAnalyze={vi.fn()}
        onRetryAnalysis={vi.fn()}
      />
    );

    expect(html).toContain('tank_sample.jpg');
    expect(html).toContain('800 × 600 px');
    expect(html).toContain('IMAGE VALIDATED — READY FOR ANALYSIS');
    expect(html).toContain('STANDBY (READY FOR ANALYSIS)');
  });

  // 3. Invalid image error rejection state
  it('3. renders validation error when image validation fails', () => {
    const html = renderToString(
      <ImagePreviewWorkspace
        imageMeta={mockMetadata}
        isValidating={false}
        validationResult={null}
        validationError="Unsupported image format (.txt). Supported formats: JPG, JPEG, PNG, WEBP."
        isAnalyzing={false}
        classificationResult={null}
        classificationError={null}
        onReset={vi.fn()}
        onRetryValidation={vi.fn()}
        onAnalyze={vi.fn()}
        onRetryAnalysis={vi.fn()}
      />
    );

    expect(html).toContain('VALIDATION FAILED');
    expect(html).toContain('Unsupported image format');
    expect(html).toContain('RETRY VALIDATION');
  });

  // 4. Analyze action becomes available for a valid image
  it('4. displays the ANALYZE IMAGE action button when image is valid', () => {
    const html = renderToString(
      <ImagePreviewWorkspace
        imageMeta={mockMetadata}
        isValidating={false}
        validationResult={mockValidationSuccess}
        validationError={null}
        isAnalyzing={false}
        classificationResult={null}
        classificationError={null}
        onReset={vi.fn()}
        onRetryValidation={vi.fn()}
        onAnalyze={vi.fn()}
        onRetryAnalysis={vi.fn()}
      />
    );

    expect(html).toContain('ANALYZE IMAGE');
    expect(html).toContain('analyze-action-btn');
    expect(html).not.toContain('disabled=""');
  });

  // 5. Analyze action enters loading state
  it('5. renders active loading state during analysis', () => {
    const html = renderToString(
      <ImagePreviewWorkspace
        imageMeta={mockMetadata}
        isValidating={false}
        validationResult={mockValidationSuccess}
        validationError={null}
        isAnalyzing={true}
        classificationResult={null}
        classificationError={null}
        onReset={vi.fn()}
        onRetryValidation={vi.fn()}
        onAnalyze={vi.fn()}
        onRetryAnalysis={vi.fn()}
      />
    );

    expect(html).toContain('ANALYZING IMAGE...');
    expect(html).toContain('Processing visual data through SigLIP 2 zero-shot vision model...');
    expect(html).toContain('INFERENCE: PROCESSING');
    expect(html).toContain('scanning-bar');
    expect(html).toContain('ANALYZING...');
  });

  // 6. Successful API response displays primary prediction
  it('6. displays primary prediction clearly', () => {
    const html = renderToString(
      <ClassificationResultPanel
        result={mockClassificationResponse}
        onReset={vi.fn()}
        onReanalyze={vi.fn()}
      />
    );

    expect(html).toContain('PRIMARY PREDICTION');
    expect(html).toContain('Main Battle Tank');
    expect(html).toContain('ASTRA VISION ANALYSIS');
  });

  // 7. Model score is displayed with correct terminology (strictly NOT confidence or probability)
  it('7. displays MODEL SCORE with 4 decimals and forbids arbitrary confidence terminology', () => {
    const html = renderToString(
      <ClassificationResultPanel
        result={mockClassificationResponse}
        onReset={vi.fn()}
        onReanalyze={vi.fn()}
      />
    );

    expect(html).toContain('MODEL SCORE');
    expect(html).toContain('0.3132');
    expect(html).toContain('SigLIP 2 similarity score');
    // Critical scope rules: do not label confidence, accuracy, or probability
    expect(html.toLowerCase()).not.toContain('confidence');
    expect(html.toLowerCase()).not.toContain('accuracy');
    expect(html.toLowerCase()).not.toContain('probability');
  });

  // 8. Inference time is displayed correctly
  it('8. displays inference time in seconds with ms latency caption', () => {
    const html = renderToString(
      <ClassificationResultPanel
        result={mockClassificationResponse}
        onReset={vi.fn()}
        onReanalyze={vi.fn()}
      />
    );

    expect(html).toContain('INFERENCE TIME');
    expect(html).toContain('2.29 s');
    expect(html).toContain('(2290 ms latency)');
    expect(html).toContain('SigLIP 2 (base-patch16-512)');
    expect(html).toContain('cpu');
  });

  // 9. API failure displays a user-friendly error
  it('9. displays user-friendly error when classification fails', () => {
    const html = renderToString(
      <ImagePreviewWorkspace
        imageMeta={mockMetadata}
        isValidating={false}
        validationResult={mockValidationSuccess}
        validationError={null}
        isAnalyzing={false}
        classificationResult={null}
        classificationError="Connection lost to vision engine. Please retry."
        onReset={vi.fn()}
        onRetryValidation={vi.fn()}
        onAnalyze={vi.fn()}
        onRetryAnalysis={vi.fn()}
      />
    );

    expect(html).toContain('CLASSIFICATION FAILED');
    expect(html).toContain('Connection lost to vision engine. Please retry.');
    expect(html).toContain('TRY AGAIN');
    expect(html).not.toContain('Traceback (most recent call last)');
  });

  // 10. User can retry after an error
  it('10. allows user to trigger retry callback', () => {
    const onRetryAnalysis = vi.fn();
    const html = renderToString(
      <ImagePreviewWorkspace
        imageMeta={mockMetadata}
        isValidating={false}
        validationResult={mockValidationSuccess}
        validationError={null}
        isAnalyzing={false}
        classificationResult={null}
        classificationError="Inference timeout."
        onReset={vi.fn()}
        onRetryValidation={vi.fn()}
        onAnalyze={vi.fn()}
        onRetryAnalysis={onRetryAnalysis}
      />
    );

    expect(html).toContain('TRY AGAIN');
  });

  // 11. User can reset/remove the image
  it('11. provides reset action to return to empty state', () => {
    const onReset = vi.fn();
    const html = renderToString(
      <ClassificationResultPanel
        result={mockClassificationResponse}
        onReset={onReset}
        onReanalyze={vi.fn()}
      />
    );

    expect(html).toContain('SELECT ANOTHER IMAGE');
    expect(html).toContain('RE-ANALYZE');
  });

  // 12. Selecting a new image clears stale classification results
  it('12. ensures stale result is not rendered in preview workspace when classificationResult is cleared', () => {
    // When a new image is selected, classificationResult is null in state
    const newImageMeta: ClientImageMetadata = {
      ...mockMetadata,
      filename: 'recon_drone.png',
    };

    const html = renderToString(
      <ImagePreviewWorkspace
        imageMeta={newImageMeta}
        isValidating={false}
        validationResult={mockValidationSuccess}
        validationError={null}
        isAnalyzing={false}
        classificationResult={null}
        classificationError={null}
        onReset={vi.fn()}
        onRetryValidation={vi.fn()}
        onAnalyze={vi.fn()}
        onRetryAnalysis={vi.fn()}
      />
    );

    expect(html).toContain('recon_drone.png');
    // Previous result must not be present
    expect(html).not.toContain('Main Battle Tank');
    expect(html).not.toContain('0.3132');
  });

  // 13. Duplicate Analyze actions are prevented while loading
  it('13. disables action buttons while analysis or validation is running', () => {
    const html = renderToString(
      <ImagePreviewWorkspace
        imageMeta={mockMetadata}
        isValidating={false}
        validationResult={mockValidationSuccess}
        validationError={null}
        isAnalyzing={true}
        classificationResult={null}
        classificationError={null}
        onReset={vi.fn()}
        onRetryValidation={vi.fn()}
        onAnalyze={vi.fn()}
        onRetryAnalysis={vi.fn()}
      />
    );

    // Analyze button is disabled while isAnalyzing is true
    expect(html).toContain('disabled=""');
    expect(html).toContain('loading');
  });
});

describe('ClassificationResultPanel — Defensive Edge Cases', () => {
  it('handles partial or malformed responses gracefully with fallbacks', () => {
    const malformedResult: any = {
      prediction: { label: 'Naval Vessel' }, // score missing
      inference: {}, // inference_time_ms and model missing
    };

    const html = renderToString(
      <ClassificationResultPanel
        result={malformedResult}
        onReset={vi.fn()}
        onReanalyze={vi.fn()}
      />
    );

    expect(html).toContain('Naval Vessel');
    expect(html).toContain('N/A');
    expect(html).toContain('SigLIP 2');
    expect(html).toContain('Latency N/A');
  });
});

describe('apiService.classifyImage — Network & Contract Handling', () => {
  beforeEach(() => {
    vi.restoreAllMocks();
  });

  it('submits multipart form data to POST /api/classify and returns parsed result', async () => {
    const fakeFile = new File(['test image'], 'sample.jpg', { type: 'image/jpeg' });
    
    globalThis.fetch = vi.fn().mockResolvedValueOnce({
      ok: true,
      status: 200,
      json: async () => mockClassificationResponse,
    });

    const result = await apiService.classifyImage(fakeFile);
    expect(result.prediction.label).toBe('Main Battle Tank');
    expect(result.prediction.score).toBe(0.3132);
    expect(result.inference.inference_time_ms).toBe(2289.87);

    expect(globalThis.fetch).toHaveBeenCalledWith('/api/classify', expect.objectContaining({
      method: 'POST',
    }));
  });

  it('rejects on HTTP error response with user-friendly message', async () => {
    const fakeFile = new File(['bad image'], 'corrupt.jpg', { type: 'image/jpeg' });
    
    globalThis.fetch = vi.fn().mockResolvedValueOnce({
      ok: false,
      status: 400,
      json: async () => ({
        error: {
          code: 'IMAGE_CORRUPTED',
          message: 'The uploaded image file is corrupted and cannot be decoded.',
        },
      }),
    });

    await expect(apiService.classifyImage(fakeFile)).rejects.toThrow(
      'The uploaded image file is corrupted and cannot be decoded.'
    );
  });

  it('rejects malformed response payload without prediction label', async () => {
    const fakeFile = new File(['test'], 'test.jpg', { type: 'image/jpeg' });
    
    globalThis.fetch = vi.fn().mockResolvedValueOnce({
      ok: true,
      status: 200,
      json: async () => ({
        prediction: {}, // label missing
      }),
    });

    await expect(apiService.classifyImage(fakeFile)).rejects.toThrow(
      'Malformed classification response from server.'
    );
  });
});

describe('ASTRA VISION Phase 7A — Top-3 Predictions (SHOULD-HAVE)', () => {
  const mockTop3Response: ClassificationResponse = {
    prediction: {
      label: 'Main Battle Tank',
      score: 0.3132,
    },
    candidates: [
      { label: 'Main Battle Tank', score: 0.3132 },
      { label: 'Armoured Fighting Vehicle', score: 0.2215 },
      { label: 'Helicopter', score: 0.0450 },
      { label: 'Drone', score: 0.0120 },
      { label: 'Ship', score: 0.0030 },
      { label: 'Fighter Aircraft', score: 0.0005 },
    ],
    top_3: [
      { label: 'Main Battle Tank', score: 0.3132 },
      { label: 'Armoured Fighting Vehicle', score: 0.2215 },
      { label: 'Helicopter', score: 0.0450 },
    ],
    inference: {
      model: 'google/siglip2-base-patch16-512',
      device: 'cpu',
      inference_time_ms: 2289.87,
    },
    status: 'success',
  };

  // 1. Top-3 section appears after successful classification
  it('1. renders TOP 3 PREDICTIONS section after successful classification', () => {
    const html = renderToString(
      <ClassificationResultPanel
        result={mockTop3Response}
        onReset={vi.fn()}
        onReanalyze={vi.fn()}
      />
    );

    expect(html).toContain('TOP 3 PREDICTIONS');
    expect(html).toContain('MODEL SCORE (RANKED)');
    expect(html).toContain('top3-predictions-container');
  });

  // 2. Correct three labels appear
  it('2. displays the correct three candidate labels in top-3', () => {
    const html = renderToString(
      <ClassificationResultPanel
        result={mockTop3Response}
        onReset={vi.fn()}
        onReanalyze={vi.fn()}
      />
    );

    expect(html).toContain('Main Battle Tank');
    expect(html).toContain('Armoured Fighting Vehicle');
    expect(html).toContain('Helicopter');
    // Rank 4 candidate should NOT be in Top-3
    expect(html).not.toContain('top3-score-4');
  });

  // 3. Correct scores appear
  it('3. displays correct formatted scores for all top-3 candidates', () => {
    const html = renderToString(
      <ClassificationResultPanel
        result={mockTop3Response}
        onReset={vi.fn()}
        onReanalyze={vi.fn()}
      />
    );

    expect(html).toContain('0.3132');
    expect(html).toContain('0.2215');
    expect(html).toContain('0.0450');
  });

  // 4. Rank ordering is correct
  it('4. displays rank tags in strict #1, #2, #3 sequence', () => {
    const html = renderToString(
      <ClassificationResultPanel
        result={mockTop3Response}
        onReset={vi.fn()}
        onReanalyze={vi.fn()}
      />
    );

    expect(html).toContain('#1');
    expect(html).toContain('#2');
    expect(html).toContain('#3');
    expect(html).toContain('rank-1 top-rank');
  });

  // 5. Primary prediction remains visually dominant
  it('5. preserves primary prediction dominance in primary-prediction-box', () => {
    const html = renderToString(
      <ClassificationResultPanel
        result={mockTop3Response}
        onReset={vi.fn()}
        onReanalyze={vi.fn()}
      />
    );

    expect(html).toContain('primary-prediction-box');
    expect(html).toContain('PRIMARY PREDICTION');
    expect(html).toContain('prediction-label-text');
  });

  // 6. Top-3 disappears on reset
  it('6. does not render top-3 when classificationResult is reset to null', () => {
    const html = renderToString(
      <ImagePreviewWorkspace
        imageMeta={mockMetadata}
        isValidating={false}
        validationResult={mockValidationSuccess}
        validationError={null}
        isAnalyzing={false}
        classificationResult={null}
        classificationError={null}
        onReset={vi.fn()}
        onRetryValidation={vi.fn()}
        onAnalyze={vi.fn()}
        onRetryAnalysis={vi.fn()}
      />
    );

    expect(html).not.toContain('TOP 3 PREDICTIONS');
    expect(html).not.toContain('top3-predictions-container');
  });

  // 7. Top-3 updates for a new image
  it('7. updates top-3 entries when a new classification result is received', () => {
    const newResult: ClassificationResponse = {
      ...mockTop3Response,
      prediction: { label: 'Helicopter', score: 0.4389 },
      top_3: [
        { label: 'Helicopter', score: 0.4389 },
        { label: 'Fighter Aircraft', score: 0.0812 },
        { label: 'Drone', score: 0.0210 },
      ],
    };

    const html = renderToString(
      <ClassificationResultPanel
        result={newResult}
        onReset={vi.fn()}
        onReanalyze={vi.fn()}
      />
    );

    expect(html).toContain('Helicopter');
    expect(html).toContain('0.4389');
    expect(html).toContain('Fighter Aircraft');
    expect(html).toContain('0.0812');
    expect(html).toContain('Drone');
    expect(html).toContain('0.0210');
  });

  // 8. Top-3 does not appear during loading
  it('8. does not display top-3 predictions during the analyzing state', () => {
    const html = renderToString(
      <ImagePreviewWorkspace
        imageMeta={mockMetadata}
        isValidating={false}
        validationResult={mockValidationSuccess}
        validationError={null}
        isAnalyzing={true}
        classificationResult={null}
        classificationError={null}
        onReset={vi.fn()}
        onRetryValidation={vi.fn()}
        onAnalyze={vi.fn()}
        onRetryAnalysis={vi.fn()}
      />
    );

    expect(html).not.toContain('TOP 3 PREDICTIONS');
    expect(html).toContain('ANALYZING IMAGE...');
  });

  // 9. Top-3 does not appear after failure
  it('9. does not display top-3 predictions in the error state', () => {
    const html = renderToString(
      <ImagePreviewWorkspace
        imageMeta={mockMetadata}
        isValidating={false}
        validationResult={mockValidationSuccess}
        validationError={null}
        isAnalyzing={false}
        classificationResult={null}
        classificationError="Model execution failed."
        onReset={vi.fn()}
        onRetryValidation={vi.fn()}
        onAnalyze={vi.fn()}
        onRetryAnalysis={vi.fn()}
      />
    );

    expect(html).not.toContain('TOP 3 PREDICTIONS');
    expect(html).toContain('CLASSIFICATION FAILED');
  });

  // 10. Malformed/missing candidates do not crash UI (fewer than 3 handled safely)
  it('10. handles fewer-than-three and malformed candidates safely without crashing', () => {
    const partialResult: any = {
      prediction: { label: 'Naval Vessel', score: 0.9123 },
      candidates: [
        { label: 'Naval Vessel', score: 0.9123 },
        { label: 'Submarine', score: 0.0877 },
        // Only 2 candidates, no 3rd candidate
      ],
      inference: {
        model: 'google/siglip2-base-patch16-512',
        device: 'cpu',
        inference_time_ms: 100.0,
      },
    };

    const html = renderToString(
      <ClassificationResultPanel
        result={partialResult}
        onReset={vi.fn()}
        onReanalyze={vi.fn()}
      />
    );

    // Renders exactly the 2 available candidates without crashing
    expect(html).toContain('TOP 3 PREDICTIONS');
    expect(html).toContain('Naval Vessel');
    expect(html).toContain('Submarine');
    expect(html).toContain('#1');
    expect(html).toContain('#2');
    expect(html).not.toContain('#3');
  });
});

describe('ASTRA VISION Phase 7C — Uncertainty & Low-Confidence Warning (SHOULD-HAVE)', () => {
  const confidentResponse: ClassificationResponse = {
    prediction: { label: 'Main Battle Tank', score: 0.8500 },
    candidates: [
      { label: 'Main Battle Tank', score: 0.8500 },
      { label: 'Armoured Fighting Vehicle', score: 0.1000 },
      { label: 'Helicopter', score: 0.0500 },
    ],
    top_3: [
      { label: 'Main Battle Tank', score: 0.8500 },
      { label: 'Armoured Fighting Vehicle', score: 0.1000 },
      { label: 'Helicopter', score: 0.0500 },
    ],
    uncertainty: {
      is_uncertain: false,
      reason: null,
      method: 'heuristic',
      score_margin: 0.7500,
      primary_score: 0.8500,
      margin_threshold: 0.0200,
      score_threshold: 0.0100,
    },
    inference: {
      model: 'google/siglip2-base-patch16-512',
      device: 'cpu',
      inference_time_ms: 1200.0,
    },
    status: 'success',
  };

  const uncertainResponse: ClassificationResponse = {
    prediction: { label: 'Tank', score: 0.3120 },
    candidates: [
      { label: 'Tank', score: 0.3120 },
      { label: 'Military Vehicle', score: 0.3105 },
      { label: 'Helicopter', score: 0.0200 },
    ],
    top_3: [
      { label: 'Tank', score: 0.3120 },
      { label: 'Military Vehicle', score: 0.3105 },
      { label: 'Helicopter', score: 0.0200 },
    ],
    uncertainty: {
      is_uncertain: true,
      reason: 'LOW_SCORE_MARGIN',
      method: 'heuristic',
      score_margin: 0.0015,
      primary_score: 0.3120,
      margin_threshold: 0.0200,
      score_threshold: 0.0100,
    },
    inference: {
      model: 'google/siglip2-base-patch16-512',
      device: 'cpu',
      inference_time_ms: 1500.0,
    },
    status: 'success',
  };

  // 1. Normal result has no warning
  it('1. does not render uncertainty warning for confident prediction', () => {
    const html = renderToString(
      <ClassificationResultPanel
        result={confidentResponse}
        onReset={vi.fn()}
        onReanalyze={vi.fn()}
      />
    );

    expect(html).not.toContain('UNCERTAIN PREDICTION');
    expect(html).not.toContain('uncertainty-warning-banner');
    // Ensure no redundant or false "HIGH CONFIDENCE" badge
    expect(html.toLowerCase()).not.toContain('high confidence');
  });

  // 2. Uncertain result displays warning
  it('2. displays UNCERTAIN PREDICTION warning banner when is_uncertain is true', () => {
    const html = renderToString(
      <ClassificationResultPanel
        result={uncertainResponse}
        onReset={vi.fn()}
        onReanalyze={vi.fn()}
      />
    );

    expect(html).toContain('UNCERTAIN PREDICTION');
    expect(html).toContain('uncertainty-warning-banner');
    expect(html).toContain('LOW SCORE MARGIN');
    expect(html).toContain('0.0015');
  });

  // 3. Primary prediction remains visible
  it('3. ensures primary prediction remains visible and dominant during uncertain prediction', () => {
    const html = renderToString(
      <ClassificationResultPanel
        result={uncertainResponse}
        onReset={vi.fn()}
        onReanalyze={vi.fn()}
      />
    );

    expect(html).toContain('PRIMARY PREDICTION');
    expect(html).toContain('Tank');
    expect(html).toContain('0.3120');
  });

  // 4. Top-3 remains visible
  it('4. ensures top-3 predictions remain visible when uncertain', () => {
    const html = renderToString(
      <ClassificationResultPanel
        result={uncertainResponse}
        onReset={vi.fn()}
        onReanalyze={vi.fn()}
      />
    );

    expect(html).toContain('TOP 3 PREDICTIONS');
    expect(html).toContain('Military Vehicle');
    expect(html).toContain('0.3105');
  });

  // 5. Warning disappears when a new confident result is loaded
  it('5. renders confident result without warning when updated from uncertain result', () => {
    const html = renderToString(
      <ClassificationResultPanel
        result={confidentResponse}
        onReset={vi.fn()}
        onReanalyze={vi.fn()}
      />
    );

    expect(html).not.toContain('UNCERTAIN PREDICTION');
  });

  // 6. Warning disappears after reset
  it('6. does not render warning when classificationResult is reset to null', () => {
    const html = renderToString(
      <ImagePreviewWorkspace
        imageMeta={mockMetadata}
        isValidating={false}
        validationResult={mockValidationSuccess}
        validationError={null}
        isAnalyzing={false}
        classificationResult={null}
        classificationError={null}
        onReset={vi.fn()}
        onRetryValidation={vi.fn()}
        onAnalyze={vi.fn()}
        onRetryAnalysis={vi.fn()}
      />
    );

    expect(html).not.toContain('UNCERTAIN PREDICTION');
    expect(html).not.toContain('uncertainty-warning-banner');
  });

  // 7. Error state does not leave stale warning
  it('7. does not render uncertainty warning in error state', () => {
    const html = renderToString(
      <ImagePreviewWorkspace
        imageMeta={mockMetadata}
        isValidating={false}
        validationResult={mockValidationSuccess}
        validationError={null}
        isAnalyzing={false}
        classificationResult={null}
        classificationError="Connection timed out."
        onReset={vi.fn()}
        onRetryValidation={vi.fn()}
        onAnalyze={vi.fn()}
        onRetryAnalysis={vi.fn()}
      />
    );

    expect(html).not.toContain('UNCERTAIN PREDICTION');
    expect(html).toContain('CLASSIFICATION FAILED');
  });

  // 8. Malformed uncertainty data is handled safely
  it('8. handles malformed or missing uncertainty data safely without crashing', () => {
    const malformedResult: any = {
      prediction: { label: 'Ship', score: 0.5 },
      candidates: [{ label: 'Ship', score: 0.5 }],
      uncertainty: { is_uncertain: true }, // missing reason, score_margin, etc.
    };

    const html = renderToString(
      <ClassificationResultPanel
        result={malformedResult}
        onReset={vi.fn()}
        onReanalyze={vi.fn()}
      />
    );

    expect(html).toContain('UNCERTAIN PREDICTION');
    expect(html).toContain('HEURISTIC CHECK');
    expect(html).toContain('Ship');
  });

  // 9. Warning text is accessible
  it('9. provides accessible role and aria attributes on the warning banner', () => {
    const html = renderToString(
      <ClassificationResultPanel
        result={uncertainResponse}
        onReset={vi.fn()}
        onReanalyze={vi.fn()}
      />
    );

    expect(html).toContain('role="alert"');
    expect(html).toContain('aria-live="polite"');
    expect(html).toContain('aria-hidden="true"');
  });

  // 10. Warning does not falsely display a confidence percentage
  it('10. strictly forbids false confidence percentage terminology in warning', () => {
    const html = renderToString(
      <ClassificationResultPanel
        result={uncertainResponse}
        onReset={vi.fn()}
        onReanalyze={vi.fn()}
      />
    );

    const lower = html.toLowerCase();
    expect(lower).not.toContain('% confidence');
    expect(lower).not.toContain('percent confidence');
    expect(lower).not.toContain('calibrated probability');
    expect(lower).not.toContain('99%');
  });
});

describe('ASTRA VISION Phase 7D — Model-Fit Justification (SHOULD-HAVE)', () => {
  const mockJustificationResult: ClassificationResponse = {
    prediction: { label: 'Main Battle Tank', score: 0.3132 },
    candidates: [
      { label: 'Main Battle Tank', score: 0.3132 },
      { label: 'Armoured Fighting Vehicle', score: 0.2215 },
      { label: 'Helicopter', score: 0.0450 },
    ],
    top_3: [
      { label: 'Main Battle Tank', score: 0.3132 },
      { label: 'Armoured Fighting Vehicle', score: 0.2215 },
      { label: 'Helicopter', score: 0.0450 },
    ],
    uncertainty: {
      is_uncertain: false,
      reason: null,
      method: 'heuristic',
      score_margin: 0.0917,
      primary_score: 0.3132,
      margin_threshold: 0.02,
      score_threshold: 0.01,
    },
    inference: {
      model: 'google/siglip2-base-patch16-512',
      device: 'cpu',
      inference_time_ms: 1250.0,
    },
    model_fit: {
      model_name: 'SigLIP 2 Base',
      model_id: 'google/siglip2-base-patch16-512',
      task: 'Image classification / image-text candidate matching',
      justification:
        "ASTRA VISION uses google/siglip2-base-patch16-512 because the operational problem requires classifying reconnaissance images against a predefined taxonomy of defence objects (such as tanks, helicopters, and aircraft). SigLIP 2 provides vision-language zero-shot alignment, allowing the system to measure similarity between visual inputs and text descriptions directly. This capability enables the baseline application to evaluate and rank candidate categories without requiring a dedicated task-specific fine-tuning pipeline or collected training datasets. The model's pairwise similarity outputs directly support ASTRA VISION's multi-candidate workflow, producing both a primary prediction and deterministic Top-3 rankings that integrate with our heuristic uncertainty checks.",
    },
    status: 'success',
  };

  // 1. Model-fit justification is present
  it('1. renders model-fit justification section with heading and text', () => {
    const html = renderToString(
      <ClassificationResultPanel
        result={mockJustificationResult}
        onReset={vi.fn()}
        onReanalyze={vi.fn()}
      />
    );

    expect(html).toContain('WHY THIS MODEL?');
    expect(html).toContain('model-fit-container');
    expect(html).toContain('BASELINE JUSTIFICATION');
    expect(html).toContain('ASTRA VISION uses google/siglip2-base-patch16-512');
  });

  // 2. Required model name is correct
  it('2. displays correct model name "SigLIP 2 Base"', () => {
    const html = renderToString(
      <ClassificationResultPanel
        result={mockJustificationResult}
        onReset={vi.fn()}
        onReanalyze={vi.fn()}
      />
    );

    expect(html).toContain('SigLIP 2 Base');
  });

  // 3. Model identifier is correct
  it('3. displays correct model identifier "google/siglip2-base-patch16-512"', () => {
    const html = renderToString(
      <ClassificationResultPanel
        result={mockJustificationResult}
        onReset={vi.fn()}
        onReanalyze={vi.fn()}
      />
    );

    expect(html).toContain('google/siglip2-base-patch16-512');
  });

  // 4. Justification is concise (80-150 words)
  it('4. justification text word count is strictly within 80-150 words target', () => {
    const text = mockJustificationResult.model_fit!.justification;
    const wordCount = text.trim().split(/\s+/).length;
    expect(wordCount).toBeGreaterThanOrEqual(80);
    expect(wordCount).toBeLessThanOrEqual(150);
  });

  // 5. Content validation: strictly forbids prohibited accuracy and marketing claims
  it('5. justification contains no prohibited accuracy, marketing, or superiority claims', () => {
    const text = mockJustificationResult.model_fit!.justification.toLowerCase();
    const prohibitedTerms = [
      '\\bbest\\b',
      '\\bhighest accuracy\\b',
      '\\bguaranteed\\b',
      '\\b100%\\b',
      '\\bmilitary-grade\\b',
      '\\bstate-of-the-art\\b',
      '\\bcertainty\\b',
      '\\bsuperior\\b',
      '\\bthe model knows\\b',
      '\\balways accurate\\b',
      '\\bhighly confident\\b',
    ];

    prohibitedTerms.forEach((term) => {
      const regex = new RegExp(term, 'i');
      expect(regex.test(text)).toBe(false);
    });
  });

  // 6. Justification does not claim calibrated confidence
  it('6. justification strictly forbids calibrated confidence or probability claims', () => {
    const text = mockJustificationResult.model_fit!.justification.toLowerCase();
    expect(text).not.toContain('calibrated confidence');
    expect(text).not.toContain('calibrated probability');
    expect(text).not.toContain('ground-truth probability');
  });

  // 7. Existing primary prediction remains unchanged
  it('7. renders primary prediction intact alongside model-fit justification', () => {
    const html = renderToString(
      <ClassificationResultPanel
        result={mockJustificationResult}
        onReset={vi.fn()}
        onReanalyze={vi.fn()}
      />
    );

    expect(html).toContain('PRIMARY PREDICTION');
    expect(html).toContain('Main Battle Tank');
    expect(html).toContain('0.3132');
  });

  // 8. Existing Top-3 remains unchanged
  it('8. renders Top-3 candidates list intact alongside model-fit justification', () => {
    const html = renderToString(
      <ClassificationResultPanel
        result={mockJustificationResult}
        onReset={vi.fn()}
        onReanalyze={vi.fn()}
      />
    );

    expect(html).toContain('TOP 3 PREDICTIONS');
    expect(html).toContain('Armoured Fighting Vehicle');
    expect(html).toContain('0.2215');
    expect(html).toContain('Helicopter');
    expect(html).toContain('0.0450');
  });

  // 9. Existing uncertainty warning coexists properly when result is uncertain
  it('9. coexists with uncertainty warning banner without conflict', () => {
    const uncertainWithJustification: ClassificationResponse = {
      ...mockJustificationResult,
      uncertainty: {
        is_uncertain: true,
        reason: 'LOW_SCORE_MARGIN',
        method: 'heuristic',
        score_margin: 0.005,
        primary_score: 0.3132,
        margin_threshold: 0.02,
        score_threshold: 0.01,
      },
    };

    const html = renderToString(
      <ClassificationResultPanel
        result={uncertainWithJustification}
        onReset={vi.fn()}
        onReanalyze={vi.fn()}
      />
    );

    expect(html).toContain('UNCERTAIN PREDICTION');
    expect(html).toContain('WHY THIS MODEL?');
    expect(html).toContain('SigLIP 2 Base');
  });

  // 10. Reset behavior: not rendered when result is cleared
  it('10. does not display model-fit justification when classificationResult is reset / null', () => {
    const html = renderToString(
      <ImagePreviewWorkspace
        imageMeta={mockMetadata}
        isValidating={false}
        validationResult={mockValidationSuccess}
        validationError={null}
        isAnalyzing={false}
        classificationResult={null}
        classificationError={null}
        onReset={vi.fn()}
        onRetryValidation={vi.fn()}
        onAnalyze={vi.fn()}
        onRetryAnalysis={vi.fn()}
      />
    );

    expect(html).not.toContain('WHY THIS MODEL?');
    expect(html).not.toContain('model-fit-container');
  });

  // 11. Error behavior: not rendered during error
  it('11. does not display model-fit justification in error state', () => {
    const html = renderToString(
      <ImagePreviewWorkspace
        imageMeta={mockMetadata}
        isValidating={false}
        validationResult={mockValidationSuccess}
        validationError={null}
        isAnalyzing={false}
        classificationResult={null}
        classificationError="Classification failed: Internal server error"
        onReset={vi.fn()}
        onRetryValidation={vi.fn()}
        onAnalyze={vi.fn()}
        onRetryAnalysis={vi.fn()}
      />
    );

    expect(html).not.toContain('WHY THIS MODEL?');
    expect(html).not.toContain('model-fit-container');
  });

  // 12. Missing/static justification data falls back to default safely without crashing
  it('12. safely falls back to default static justification if model_fit is missing from payload', () => {
    const responseWithoutModelFit: any = {
      prediction: { label: 'Fighter Aircraft', score: 0.65 },
      candidates: [{ label: 'Fighter Aircraft', score: 0.65 }],
      inference: { model: 'google/siglip2-base-patch16-512', device: 'cpu', inference_time_ms: 1000 },
      // model_fit is undefined
    };

    const html = renderToString(
      <ClassificationResultPanel
        result={responseWithoutModelFit}
        onReset={vi.fn()}
        onReanalyze={vi.fn()}
      />
    );

    expect(html).toContain('WHY THIS MODEL?');
    expect(html).toContain('SigLIP 2 Base');
    expect(html).toContain('google/siglip2-base-patch16-512');
  });

  // 13. Accessibility: region role and aria-label
  it('13. provides accessible region role and aria attributes', () => {
    const html = renderToString(
      <ClassificationResultPanel
        result={mockJustificationResult}
        onReset={vi.fn()}
        onReanalyze={vi.fn()}
      />
    );

    expect(html).toContain('role="region"');
    expect(html).toContain('aria-label="Why this model"');
  });
});


