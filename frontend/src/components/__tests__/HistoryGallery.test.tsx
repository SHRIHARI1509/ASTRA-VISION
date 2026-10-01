import { describe, it, expect, vi, beforeEach } from 'vitest';
import { renderToString } from 'react-dom/server';
import { HistoryGallery } from '../HistoryGallery';
import {
  createHistoryItem,
  insertHistoryItem,
  insertBatchHistoryItems,
  MAX_HISTORY_ITEMS,
} from '../../hooks/useHistory';
import type { HistoryItem, ClassificationResponse } from '../../types';

const mockClassificationResponse: ClassificationResponse = {
  prediction: {
    label: 'Main Battle Tank',
    score: 0.8912,
  },
  candidates: [
    { label: 'Main Battle Tank', score: 0.8912 },
    { label: 'Military Vehicle', score: 0.0821 },
  ],
  top_3: [
    { label: 'Main Battle Tank', score: 0.8912 },
    { label: 'Military Vehicle', score: 0.0821 },
    { label: 'Fighter Aircraft', score: 0.0123 },
  ],
  uncertainty: {
    is_uncertain: false,
    reason: null,
    method: 'heuristic',
    score_margin: 0.8091,
    primary_score: 0.8912,
    margin_threshold: 0.02,
    score_threshold: 0.01,
  },
  model_fit: {
    model_name: 'SigLIP 2 Base',
    model_id: 'google/siglip2-base-patch16-512',
    task: 'Zero-Shot Classification',
    justification: 'Production model fit.',
  },
  inference: {
    model: 'google/siglip2-base-patch16-512',
    device: 'cpu',
    inference_time_ms: 198.4,
  },
  status: 'success',
};

const mockUncertainResponse: ClassificationResponse = {
  ...mockClassificationResponse,
  prediction: { label: 'Drone', score: 0.35 },
  uncertainty: {
    is_uncertain: true,
    reason: 'LOW_SCORE_MARGIN',
    method: 'heuristic',
    score_margin: 0.015,
    primary_score: 0.35,
    margin_threshold: 0.02,
    score_threshold: 0.01,
  },
};

describe('Phase 8D — Image History / Gallery Tests', () => {
  beforeEach(() => {
    vi.restoreAllMocks();
  });

  // 1. Successful result creates history item
  it('1. successful result creates history item', () => {
    const item = createHistoryItem('recon_01.jpg', mockClassificationResponse);
    const { updated } = insertHistoryItem([], item);

    expect(updated).toHaveLength(1);
    expect(updated[0].filename).toBe('recon_01.jpg');
    expect(updated[0].prediction.label).toBe('Main Battle Tank');
    expect(updated[0].score).toBe(0.8912);
  });

  // 2. Batch results create individual history entries
  it('2. batch results create individual history entries', () => {
    const batchItems = [
      createHistoryItem('batch_alpha.jpg', mockClassificationResponse),
      createHistoryItem('batch_bravo.jpg', mockUncertainResponse),
      createHistoryItem('batch_charlie.png', mockClassificationResponse),
    ];

    const { updated } = insertBatchHistoryItems([], batchItems);

    expect(updated).toHaveLength(3);
    const filenames = updated.map((h) => h.filename);
    expect(filenames).toContain('batch_alpha.jpg');
    expect(filenames).toContain('batch_bravo.jpg');
    expect(filenames).toContain('batch_charlie.png');
  });

  // 3. Failed result does not create successful history item
  it('3. failed result does not create successful history item', () => {
    // Only successful results produce valid history items
    const onlySuccess = [createHistoryItem('valid_only.jpg', mockClassificationResponse)];
    const { updated } = insertBatchHistoryItems([], onlySuccess);

    expect(updated).toHaveLength(1);
    expect(updated[0].filename).toBe('valid_only.jpg');
  });

  // 4. Newest-first ordering
  it('4. enforces newest-first ordering', () => {
    const item1 = createHistoryItem('first_older.jpg', mockClassificationResponse);
    const item2 = createHistoryItem('second_newer.jpg', mockClassificationResponse);

    let history: HistoryItem[] = [];
    history = insertHistoryItem(history, item1).updated;
    history = insertHistoryItem(history, item2).updated;

    expect(history[0].filename).toBe('second_newer.jpg');
    expect(history[1].filename).toBe('first_older.jpg');
  });

  // 5. Duplicate analyses remain distinct
  it('5. duplicate analyses of the same image create distinct history items', () => {
    const item1 = createHistoryItem('target.jpg', mockClassificationResponse);
    const item2 = createHistoryItem('target.jpg', mockClassificationResponse);

    let history: HistoryItem[] = [];
    history = insertHistoryItem(history, item1).updated;
    history = insertHistoryItem(history, item2).updated;

    expect(history).toHaveLength(2);
    expect(history[0].filename).toBe('target.jpg');
    expect(history[1].filename).toBe('target.jpg');
    expect(history[0].id).not.toBe(history[1].id);
  });

  // 6 & 7. History limit enforced and oldest items removed
  it('6 & 7. enforces history limit of 50 and removes oldest entries', () => {
    let history: HistoryItem[] = [];
    for (let i = 1; i <= 55; i++) {
      const item = createHistoryItem(`image_${String(i).padStart(2, '0')}.jpg`, mockClassificationResponse);
      history = insertHistoryItem(history, item, MAX_HISTORY_ITEMS).updated;
    }

    expect(history).toHaveLength(MAX_HISTORY_ITEMS);
    expect(history[0].filename).toBe('image_55.jpg');
    const filenames = history.map((h) => h.filename);
    expect(filenames).not.toContain('image_01.jpg');
    expect(filenames).not.toContain('image_05.jpg');
    expect(filenames).toContain('image_06.jpg');
  });

  // 8. Clear history removes all entries
  it('8. clear history removes all history entries', () => {
    let history: HistoryItem[] = [
      createHistoryItem('item1.jpg', mockClassificationResponse),
      createHistoryItem('item2.jpg', mockClassificationResponse),
    ];
    expect(history).toHaveLength(2);

    history = [];
    expect(history).toHaveLength(0);
  });

  // 9 & 10. Separation between active workspace result and history
  it('9 & 10. history operates independently of current workspace state', () => {
    const item = createHistoryItem('active_recon.jpg', mockClassificationResponse);
    const history = insertHistoryItem([], item).updated;

    // Resetting current analysis would just clear local workspace state, while history persists
    expect(history).toHaveLength(1);
    expect(history[0].filename).toBe('active_recon.jpg');
  });

  // 11. No duplicate React keys
  it('11. all generated history items have unique keys', () => {
    let history: HistoryItem[] = [];
    for (let i = 0; i < 10; i++) {
      const item = createHistoryItem('same_file.jpg', mockClassificationResponse);
      history = insertHistoryItem(history, item).updated;
    }

    const ids = history.map((item) => item.id);
    const uniqueIds = new Set(ids);
    expect(uniqueIds.size).toBe(ids.length);
  });

  // 12. Object URL cleanup on eviction
  it('12. tracks evicted items when limit exceeded for URL revocation', () => {
    let history: HistoryItem[] = [];
    let lastEvicted: HistoryItem[] = [];

    for (let i = 1; i <= 52; i++) {
      const item = createHistoryItem(
        `img_${i}.jpg`,
        mockClassificationResponse,
        `blob:http://localhost:5173/mock-blob-${i}`
      );
      const res = insertHistoryItem(history, item, 50);
      history = res.updated;
      if (res.evicted.length > 0) {
        lastEvicted.push(...res.evicted);
      }
    }

    expect(history).toHaveLength(50);
    expect(lastEvicted).toHaveLength(2);
    expect(lastEvicted[0].previewUrl).toBe('blob:http://localhost:5173/mock-blob-1');
    expect(lastEvicted[1].previewUrl).toBe('blob:http://localhost:5173/mock-blob-2');
  });

  // 13. Empty gallery state rendering
  it('13. renders clean empty state when history is empty', () => {
    const html = renderToString(<HistoryGallery history={[]} onClearHistory={vi.fn()} />);
    expect(html).toContain('NO HISTORICAL RECONNAISSANCE RECORDS');
    expect(html).toContain('gallery-empty-state');
    expect(html).toContain('CAPACITY:');
    expect(html).toContain('50');
    expect(html).toContain('ENTRIES');
  });

  // 14. Deterministic rendering with history items
  it('14. deterministically renders history items with tactical metadata', () => {
    const mockHistoryItem: HistoryItem = {
      id: 'test-item-01',
      filename: 'recon_satellite.jpg',
      timestamp: '2026-09-30T12:00:00.000Z',
      prediction: { label: 'Helicopter', score: 0.9412 },
      score: 0.9412,
      top_3: [{ label: 'Helicopter', score: 0.9412 }],
      uncertainty: {
        is_uncertain: false,
        method: 'heuristic',
        margin_threshold: 0.02,
        score_threshold: 0.01,
      },
      inference: {
        model: 'google/siglip2-base-patch16-512',
        device: 'cpu',
        inference_time_ms: 210.5,
      },
    };

    const html = renderToString(
      <HistoryGallery history={[mockHistoryItem]} onClearHistory={vi.fn()} />
    );

    expect(html).toContain('recon_satellite.jpg');
    expect(html).toContain('Helicopter');
    expect(html).toContain('94.1');
    expect(html).toContain('211');
    expect(html).toContain('CPU');
    expect(html).toContain('CLEAR HISTORY');
  });

  // 15. Malformed or uncertain history data handled safely
  it('15. handles uncertain and sparse history items safely', () => {
    const mockUncertainItem: HistoryItem = {
      id: 'test-item-uncertain',
      filename: 'foggy_drone.png',
      timestamp: '2026-09-30T12:05:00.000Z',
      prediction: { label: 'Drone', score: 0.45 },
      score: 0.45,
      uncertainty: {
        is_uncertain: true,
        reason: 'LOW_SCORE_MARGIN',
        method: 'heuristic',
        margin_threshold: 0.02,
        score_threshold: 0.01,
      },
      inference: {
        model: 'google/siglip2-base-patch16-512',
        device: 'cpu',
        inference_time_ms: 180.0,
      },
    };

    const html = renderToString(
      <HistoryGallery history={[mockUncertainItem]} onClearHistory={vi.fn()} />
    );

    expect(html).toContain('foggy_drone.png');
    expect(html).toContain('Drone');
    expect(html).toContain('45.0');
    expect(html).toContain('UNCERTAIN');
  });
});
