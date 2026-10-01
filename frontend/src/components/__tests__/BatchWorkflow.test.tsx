import { describe, it, expect } from 'vitest';
import { renderToString } from 'react-dom/server';
import { BatchWorkspace, MAX_BATCH_SIZE } from '../BatchWorkspace';
import { AnalysisWorkspace } from '../AnalysisWorkspace';

describe('Phase 8C — Batch Image Processing Frontend Tests', () => {
  it('1. renders batch standby state with tactical dropzone and max limits info', () => {
    const html = renderToString(<BatchWorkspace />);
    expect(html).toContain('CH-02 [MULTI-TARGET INGESTION]');
    expect(html).toContain('BATCH STANDBY');
    expect(html).toContain('MAX BATCH:');
    expect(html).toContain(String(MAX_BATCH_SIZE));
    expect(html).toContain('SELECT OR DROP MULTIPLE RECONNAISSANCE IMAGES');
    expect(html).toContain('SEQUENTIAL SIGLIP 2 PIPELINE');
  });

  it('2. provides tactical mode switcher in AnalysisWorkspace', () => {
    const html = renderToString(<AnalysisWorkspace />);
    expect(html).toContain('TARGET RECONNAISSANCE (SINGLE)');
    expect(html).toContain('MULTI-TARGET BATCH INGESTION (MAX 20)');
    expect(html).toContain('mode-single-tab');
    expect(html).toContain('mode-batch-tab');
  });

  it('3. renders batch limits info indicating 20 files maximum', () => {
    const html = renderToString(<BatchWorkspace />);
    expect(html).toContain('MAX 20 FILES');
    expect(html).toContain('10 MB / FILE');
  });

  it('4. displays sample quick-load buttons for 3 sample targets and mixed batch', () => {
    const html = renderToString(<BatchWorkspace />);
    expect(html).toContain('3 SAMPLE TARGETS (VALID)');
    expect(html).toContain('MIXED BATCH (VALID + INVALID)');
  });

  it('5. enforces MAX_BATCH_SIZE constant of 20', () => {
    expect(MAX_BATCH_SIZE).toBe(20);
  });
});
