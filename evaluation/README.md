# ASTRA VISION — Held-Out Model Performance Evaluation Report

## 1. Overview
- **Evaluated Model**: `google/siglip2-base-patch16-512`
- **Dataset Path**: `C:\FILES\astra-vision\data\held_out_dataset`
- **Timestamp**: `2026-09-30T16:41:54.954195+00:00`
- **Total Evaluated Images**: 30
- **Device**: `auto`

## 2. Key Metrics Summary
- **Top-1 Overall Accuracy**: 100.00%
- **Top-3 Secondary Accuracy**: 100.00%
- **Macro Precision**: 1.0000
- **Macro Recall**: 1.0000
- **Macro F1 Score**: 1.0000

## 3. Per-Class Performance
| Class | Precision | Recall | F1 Score | Support |
|---|---|---|---|---|
| Fighter Aircraft | 1.0000 | 1.0000 | 1.0000 | 5 |
| Helicopter | 1.0000 | 1.0000 | 1.0000 | 5 |
| Tank | 1.0000 | 1.0000 | 1.0000 | 5 |
| Ship | 1.0000 | 1.0000 | 1.0000 | 5 |
| Military Vehicle | 1.0000 | 1.0000 | 1.0000 | 5 |
| Drone | 1.0000 | 1.0000 | 1.0000 | 5 |

## 4. Confusion Matrix (Counts)
```
Ground Truth \ Predicted  | Fighter  | Helicopt | Tank     | Ship     | Military | Drone   
----------------------------------------------------------------------------------------------
Fighter Aircraft          | 5        | 0        | 0        | 0        | 0        | 0       
Helicopter                | 0        | 5        | 0        | 0        | 0        | 0       
Tank                      | 0        | 0        | 5        | 0        | 0        | 0       
Ship                      | 0        | 0        | 0        | 5        | 0        | 0       
Military Vehicle          | 0        | 0        | 0        | 0        | 5        | 0       
Drone                     | 0        | 0        | 0        | 0        | 0        | 5       
```

## 5. Uncertainty Analysis (Descriptive)
- **Uncertain Predictions**: 12 (40.0%)
- **Non-Uncertain Predictions**: 18
- **Accuracy among Uncertain**: 100.00%
- **Accuracy among Non-Uncertain**: 100.00%
- *Note: Descriptive analysis only; operating thresholds were not tuned.*

## 6. Inference Performance
- **Total Time**: 61.72 s
- **Average Latency**: 2052.12 ms/image
- **Throughput**: 0.49 images/second

## 7. Failure Analysis
- Total misclassified samples: 0
See `misclassifications.json` for details on each misclassified instance.
