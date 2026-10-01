import type {
  HealthResponse,
  ImageValidationResult,
  ClassificationResponse,
  BatchClassificationResponse,
} from '../types';

export const apiService = {
  /**
   * Performs system health check against backend API.
   */
  async checkHealth(): Promise<{ data: HealthResponse; latencyMs: number }> {
    const startTime = performance.now();
    const response = await fetch('/api/health', {
      headers: {
        'Accept': 'application/json',
      },
    });

    if (!response.ok) {
      throw new Error(`Health check failed with status: ${response.status} ${response.statusText}`);
    }

    const data: HealthResponse = await response.json();
    const latencyMs = Math.round(performance.now() - startTime);

    return { data, latencyMs };
  },

  /**
   * Uploads an image to the backend validation endpoint for independent server-side inspection.
   */
  async validateImage(file: File): Promise<ImageValidationResult> {
    const formData = new FormData();
    formData.append('image', file);

    const response = await fetch('/api/images/validate', {
      method: 'POST',
      body: formData,
    });

    const data = await response.json();

    if (!response.ok || !data.valid) {
      const errorMessage =
        data?.error?.message ||
        `Backend image validation failed (HTTP ${response.status})`;
      throw new Error(errorMessage);
    }

    return data as ImageValidationResult;
  },

  /**
   * Submits a validated image to the core classification engine (POST /api/classify).
   */
  async classifyImage(file: File): Promise<ClassificationResponse> {
    const formData = new FormData();
    formData.append('image', file);

    const response = await fetch('/api/classify', {
      method: 'POST',
      body: formData,
    });

    let data: any;
    try {
      data = await response.json();
    } catch {
      throw new Error(`Unexpected server response (HTTP ${response.status})`);
    }

    if (!response.ok) {
      const errorMessage =
        data?.error?.message ||
        `Classification request failed (HTTP ${response.status})`;
      throw new Error(errorMessage);
    }

    if (!data?.prediction?.label || typeof data?.prediction?.score !== 'number') {
      throw new Error('Malformed classification response from server.');
    }

    return data as ClassificationResponse;
  },

  /**
   * Submits multiple images to the batch classification endpoint (POST /api/classify/batch).
   */
  async classifyBatch(files: File[]): Promise<BatchClassificationResponse> {
    if (!files || files.length === 0) {
      throw new Error('No files provided for batch processing.');
    }
    if (files.length > 20) {
      throw new Error(`Batch size ${files.length} exceeds maximum limit of 20 images.`);
    }

    const formData = new FormData();
    for (const file of files) {
      formData.append('images', file);
    }

    const response = await fetch('/api/classify/batch', {
      method: 'POST',
      body: formData,
    });

    let data: any;
    try {
      data = await response.json();
    } catch {
      throw new Error(`Unexpected batch response (HTTP ${response.status})`);
    }

    if (!response.ok) {
      const errorMessage =
        data?.error?.message ||
        `Batch classification failed (HTTP ${response.status})`;
      throw new Error(errorMessage);
    }

    return data as BatchClassificationResponse;
  },

  /**
   * Sequentially executes batch classification with real-time honest progress tracking.
   */
  async classifyBatchWithProgress(
    files: File[],
    onProgress: (current: number, total: number, filename: string) => void
  ): Promise<BatchClassificationResponse> {
    if (!files || files.length === 0) {
      throw new Error('No files provided for batch processing.');
    }
    if (files.length > 20) {
      throw new Error(`Batch size ${files.length} exceeds maximum limit of 20 images.`);
    }

    const results: any[] = [];
    let successful = 0;
    let failed = 0;

    for (let i = 0; i < files.length; i++) {
      const file = files[i];
      onProgress(i + 1, files.length, file.name);

      try {
        const res = await this.classifyImage(file);
        results.push({
          filename: file.name,
          status: 'success',
          prediction: res.prediction,
          candidates: res.candidates,
          top_3: res.top_3,
          uncertainty: res.uncertainty,
          model_fit: res.model_fit,
          inference: res.inference,
        });
        successful++;
      } catch (err) {
        const errorMsg = err instanceof Error ? err.message : 'Classification failed.';
        results.push({
          filename: file.name,
          status: 'error',
          error: {
            code: 'INSPECTION_REJECTED',
            message: errorMsg,
          },
        });
        failed++;
      }
    }

    return {
      status: 'success',
      total: files.length,
      successful,
      failed,
      results,
    };
  },
};
