import { useState, useCallback, useEffect, useRef } from 'react';
import type { HistoryItem, ClassificationResponse } from '../types';

export const MAX_HISTORY_ITEMS = 50;

export function createHistoryItem(
  filename: string,
  result: ClassificationResponse,
  previewUrl?: string,
  customId?: string
): HistoryItem {
  return {
    id: customId || `hist-${Date.now()}-${Math.random().toString(36).substring(2, 9)}`,
    filename,
    timestamp: new Date().toISOString(),
    prediction: result.prediction,
    score: result.prediction.score,
    top_3: result.top_3,
    uncertainty: result.uncertainty,
    model_fit: result.model_fit,
    inference: result.inference,
    previewUrl,
  };
}

export function insertHistoryItem(
  prevHistory: HistoryItem[],
  newItem: HistoryItem,
  maxLimit: number = MAX_HISTORY_ITEMS
): { updated: HistoryItem[]; evicted: HistoryItem[] } {
  const combined = [newItem, ...prevHistory];
  if (combined.length > maxLimit) {
    return {
      updated: combined.slice(0, maxLimit),
      evicted: combined.slice(maxLimit),
    };
  }
  return { updated: combined, evicted: [] };
}

export function insertBatchHistoryItems(
  prevHistory: HistoryItem[],
  newItems: HistoryItem[],
  maxLimit: number = MAX_HISTORY_ITEMS
): { updated: HistoryItem[]; evicted: HistoryItem[] } {
  const combined = [...newItems, ...prevHistory];
  if (combined.length > maxLimit) {
    return {
      updated: combined.slice(0, maxLimit),
      evicted: combined.slice(maxLimit),
    };
  }
  return { updated: combined, evicted: [] };
}

export function useHistory() {
  const [history, setHistory] = useState<HistoryItem[]>([]);
  const objectUrlsRef = useRef<Set<string>>(new Set());

  const addHistoryItem = useCallback((
    filename: string,
    result: ClassificationResponse,
    previewUrl?: string
  ) => {
    if (previewUrl && previewUrl.startsWith('blob:')) {
      objectUrlsRef.current.add(previewUrl);
    }

    const newItem = createHistoryItem(filename, result, previewUrl);

    setHistory((prev) => {
      const { updated, evicted } = insertHistoryItem(prev, newItem, MAX_HISTORY_ITEMS);
      for (const item of evicted) {
        if (item.previewUrl && item.previewUrl.startsWith('blob:')) {
          URL.revokeObjectURL(item.previewUrl);
          objectUrlsRef.current.delete(item.previewUrl);
        }
      }
      return updated;
    });
  }, []);

  const addBatchHistoryItems = useCallback((
    items: Array<{ filename: string; result: ClassificationResponse; previewUrl?: string }>
  ) => {
    if (!items || items.length === 0) return;

    const newItems: HistoryItem[] = items.map((item, idx) => {
      if (item.previewUrl && item.previewUrl.startsWith('blob:')) {
        objectUrlsRef.current.add(item.previewUrl);
      }
      return createHistoryItem(
        item.filename,
        item.result,
        item.previewUrl,
        `hist-${Date.now()}-${idx}-${Math.random().toString(36).substring(2, 9)}`
      );
    });

    setHistory((prev) => {
      const { updated, evicted } = insertBatchHistoryItems(prev, newItems, MAX_HISTORY_ITEMS);
      for (const item of evicted) {
        if (item.previewUrl && item.previewUrl.startsWith('blob:')) {
          URL.revokeObjectURL(item.previewUrl);
          objectUrlsRef.current.delete(item.previewUrl);
        }
      }
      return updated;
    });
  }, []);

  const clearHistory = useCallback(() => {
    for (const url of objectUrlsRef.current) {
      URL.revokeObjectURL(url);
    }
    objectUrlsRef.current.clear();
    setHistory([]);
  }, []);

  useEffect(() => {
    return () => {
      for (const url of objectUrlsRef.current) {
        URL.revokeObjectURL(url);
      }
      objectUrlsRef.current.clear();
    };
  }, []);

  return {
    history,
    addHistoryItem,
    addBatchHistoryItems,
    clearHistory,
    maxLimit: MAX_HISTORY_ITEMS,
  };
}
