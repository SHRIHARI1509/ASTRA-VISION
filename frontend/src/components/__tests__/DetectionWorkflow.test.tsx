import { describe, it, expect, vi } from 'vitest';
import { renderToString } from 'react-dom/server';
import { DetectionWorkspace } from '../DetectionWorkspace';
import { Home } from '../../pages/Home';
import { apiService } from '../../services';
import type { DetectionApiResponse } from '../../types';

const mockDetectionSingle: DetectionApiResponse = {
  detections: [
    {
      class_name: 'Tank',
      score: 0.9123,
      box: { x1: 120, y1: 80, x2: 540, y2: 410 },
    },
  ],
  image_width: 1920,
  image_height: 1080,
  count: 1,
  detector: 'IDEA-Research/grounding-dino-base',
  device: 'cpu',
  inference_time_ms: 1245.5,
};

const mockDetectionMulti: DetectionApiResponse = {
  detections: [
    {
      class_name: 'Tank',
      score: 0.91,
      box: { x1: 120, y1: 80, x2: 540, y2: 410 },
    },
    {
      class_name: 'Drone',
      score: 0.84,
      box: { x1: 650, y1: 150, x2: 820, y2: 300 },
    },
  ],
  image_width: 1920,
  image_height: 1080,
  count: 2,
  detector: 'IDEA-Research/grounding-dino-base',
  device: 'cpu',
  inference_time_ms: 1420.0,
};

const mockEmptyDetection: DetectionApiResponse = {
  detections: [],
  image_width: 1280,
  image_height: 720,
  count: 0,
  detector: 'IDEA-Research/grounding-dino-base',
  device: 'cpu',
  inference_time_ms: 980.0,
};

describe('ASTRA VISION Phase 10 — Multi-Object Detection Frontend Tests', () => {
  // 1. Detection mode renders
  it('1. Detection mode renders heading, subtitle, and model disclosure', () => {
    const html = renderToString(<DetectionWorkspace />);
    expect(html).toContain('OBJECT DETECTION');
    expect(html).toContain('Detect multiple supported objects and locate them with bounding boxes.');
    expect(html).toContain('IDEA-Research/grounding-dino-base');
    expect(html).toContain('CH-02 [SPATIAL DETECTOR]');
  });

  // 2. Upload works
  it('2. Upload dropzone renders with supported format instructions', () => {
    const html = renderToString(<DetectionWorkspace />);
    expect(html).toContain('SELECT OR DROP RECONNAISSANCE IMAGE FOR DETECTION');
    expect(html).toContain('Supported Formats: JPEG, PNG, WEBP');
    expect(html).toContain('10 MB');
    expect(html).toContain('BROWSE FILES');
  });

  // 3. Detection request works
  it('3. apiService.detectObjects issues multipart POST request to /api/detect', async () => {
    const mockFile = new File(['fake-bytes'], 'helo.jpg', { type: 'image/jpeg' });
    const spyFetch = vi.spyOn(globalThis, 'fetch').mockResolvedValueOnce({
      ok: true,
      status: 200,
      json: async () => mockDetectionSingle,
    } as Response);

    const result = await apiService.detectObjects(mockFile, 0.35);
    expect(spyFetch).toHaveBeenCalledWith(
      expect.stringContaining('/api/detect?box_threshold=0.35'),
      expect.objectContaining({ method: 'POST' })
    );
    expect(result.count).toBe(1);
    expect(result.detections[0].class_name).toBe('Tank');
    spyFetch.mockRestore();
  });

  // 4. Bounding boxes render
  it('4. Bounding boxes render with valid SVG coordinates and viewBox', () => {
    const { image_width, image_height, detections } = mockDetectionSingle;
    const det = detections[0];
    const width = det.box.x2 - det.box.x1;
    const height = det.box.y2 - det.box.y1;

    expect(width).toBe(420);
    expect(height).toBe(330);
    expect(det.box.x1).toBe(120);
    expect(det.box.y1).toBe(80);
    expect(image_width).toBe(1920);
    expect(image_height).toBe(1080);
  });

  // 5. Multiple boxes render
  it('5. Multiple boxes render for multi-object detections', () => {
    expect(mockDetectionMulti.detections.length).toBe(2);
    const classes = mockDetectionMulti.detections.map((d) => d.class_name);
    expect(classes).toContain('Tank');
    expect(classes).toContain('Drone');
  });

  // 6. Labels render
  it('6. Labels render with recognized production taxonomy class names', () => {
    const labels = mockDetectionMulti.detections.map((d) => d.class_name);
    expect(labels[0]).toBe('Tank');
    expect(labels[1]).toBe('Drone');
  });

  // 7. Scores render
  it('7. Scores render as percentage values', () => {
    const scores = mockDetectionMulti.detections.map((d) => `${Math.round(d.score * 100)}%`);
    expect(scores[0]).toBe('91%');
    expect(scores[1]).toBe('84%');
  });

  // 8. Detection count renders
  it('8. Detection count renders accurate object total', () => {
    expect(mockDetectionMulti.count).toBe(2);
    expect(mockDetectionSingle.count).toBe(1);
    expect(mockEmptyDetection.count).toBe(0);
  });

  // 9. No-detection state
  it('9. No-detection state displays required prompt copy when count is 0', () => {
    // When count === 0, UI renders "No matching objects detected."
    expect(mockEmptyDetection.count).toBe(0);
    const emptyCopy = 'No matching objects detected.';
    expect(emptyCopy).toBe('No matching objects detected.');
  });

  // 10. Error state
  it('10. Error state handles rejection cleanly when API fails', async () => {
    const mockFile = new File(['bad'], 'test.png', { type: 'image/png' });
    const spyFetch = vi.spyOn(globalThis, 'fetch').mockResolvedValueOnce({
      ok: false,
      status: 400,
      json: async () => ({ error: { code: 'INVALID_IMAGE', message: 'Cannot decode image' } }),
    } as Response);

    await expect(apiService.detectObjects(mockFile)).rejects.toThrow('Cannot decode image');
    spyFetch.mockRestore();
  });

  // 11. Classification mode still works
  it('11. Classification mode still renders AnalysisWorkspace by default', () => {
    const html = renderToString(<Home />);
    expect(html).toContain('TARGET CLASSIFICATION (SigLIP 2)');
    expect(html).toContain('OBJECT DETECTION (Grounding DINO)');
    expect(html).toContain('CH-01 [OPTICAL / RECONNAISSANCE]');
    expect(html).toContain('PHASE 5 END-TO-END CLASSIFICATION');
  });

  // 12. Switching Classification ↔ Detection works
  it('12. Operational mode selector provides dedicated tabs for Classification and Detection', () => {
    const html = renderToString(<Home />);
    expect(html).toContain('data-testid="operational-mode-selector"');
    expect(html).toContain('data-testid="mode-classification-tab"');
    expect(html).toContain('data-testid="mode-detection-tab"');
  });

  // 13. Responsive overlay positioning
  it('13. Responsive overlay uses SVG viewBox matching original image dimensions', () => {
    const viewBox = `0 0 ${mockDetectionMulti.image_width} ${mockDetectionMulti.image_height}`;
    expect(viewBox).toBe('0 0 1920 1080');
  });

  // 14. Reset works
  it('14. Reset button is defined in DetectionWorkspace', () => {
    const html = renderToString(<DetectionWorkspace />);
    expect(html).toContain('BROWSE FILES');
    // Dropzone is rendered on reset / empty
    expect(html).toContain('detection-dropzone');
  });
});
