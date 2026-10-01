export interface HealthResponse {
  status: string;
  service: string;
}

export interface TelemetryState {
  isOnline: boolean;
  serviceName: string;
  latencyMs: number | null;
  lastChecked: string | null;
  error?: string;
}

export interface ImageValidationResult {
  valid: boolean;
  filename: string;
  format: string;
  width: number;
  height: number;
  size_bytes: number;
}

export interface ImageValidationError {
  code: string;
  message: string;
}

export interface ClientImageMetadata {
  file: File;
  previewUrl: string;
  filename: string;
  format: string;
  sizeBytes: number;
  width: number;
  height: number;
  aspectRatio: number;
}

export interface PrimaryPrediction {
  label: string;
  score: number;
}

export interface CandidateScore {
  label: string;
  score: number;
}

export interface InferenceMetadata {
  model: string;
  device: string;
  inference_time_ms: number;
}

export interface UncertaintyDetails {
  is_uncertain: boolean;
  reason?: 'LOW_PRIMARY_SCORE' | 'LOW_SCORE_MARGIN' | 'BOTH' | string | null;
  method: string;
  score_margin?: number | null;
  primary_score?: number | null;
  margin_threshold: number;
  score_threshold: number;
}

export interface ModelFitJustification {
  model_name: string;
  model_id: string;
  task: string;
  justification: string;
}

export interface ClassificationResponse {
  prediction: PrimaryPrediction;
  candidates: CandidateScore[];
  top_3?: CandidateScore[];
  uncertainty?: UncertaintyDetails;
  model_fit?: ModelFitJustification;
  inference: InferenceMetadata;
  status?: string;
}

export type AnalysisStatus = 'IDLE' | 'ANALYZING' | 'SUCCESS' | 'ERROR';

// Phase 8C: Batch Processing Types
export interface BatchItemSuccessResult {
  filename: string;
  status: 'success';
  prediction: PrimaryPrediction;
  candidates: CandidateScore[];
  top_3?: CandidateScore[];
  uncertainty?: UncertaintyDetails;
  model_fit?: ModelFitJustification;
  inference: InferenceMetadata;
  previewUrl?: string;
  thumbnailUrl?: string;
}

export interface BatchItemErrorResult {
  filename: string;
  status: 'error';
  error: {
    code: string;
    message: string;
  };
}

export type BatchItemResult = BatchItemSuccessResult | BatchItemErrorResult;

export interface BatchClassificationResponse {
  status: string;
  total: number;
  successful: number;
  failed: number;
  results: BatchItemResult[];
}

export type BatchProcessingStatus =
  | 'STANDBY'
  | 'FILES_SELECTED'
  | 'VALIDATING'
  | 'ANALYZING'
  | 'COMPLETED'
  | 'ERROR';

// Phase 8D: History / Gallery Types
export interface HistoryItem {
  id: string;
  filename: string;
  timestamp: string;
  prediction: PrimaryPrediction;
  score: number;
  top_3?: CandidateScore[];
  uncertainty?: UncertaintyDetails;
  model_fit?: ModelFitJustification;
  inference: InferenceMetadata;
  previewUrl?: string;
}

// Phase 10: Multi-Object Detection Types
export interface BoundingBoxCoords {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
}

export interface DetectedObjectItem {
  class_name: string;
  score: number;
  box: BoundingBoxCoords;
}

export interface DetectionApiResponse {
  detections: DetectedObjectItem[];
  image_width: number;
  image_height: number;
  count: number;
  detector?: string;
  device?: string;
  inference_time_ms?: number;
}

export type OperationalMode = 'classification' | 'detection';

