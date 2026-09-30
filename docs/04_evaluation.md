# Evaluation: separate experiments

## Recorded validation

Epoch 59: mAP50-95 0.3521852248; AP50 0.6454196683. Maximum of available log entries; epoch 21 is missing.

## COCO test-split evaluation artifact

The prior audit independently recovered these values from RT-DETR/rtdetr_pytorch/output/rtdetr_r18vd_rdd_test/eval.pth. It contains precision/recall arrays for 5,758 images and five classes, but no checkpoint hash or original annotation path. Test configuration points to the test split. Do not unconditionally attribute this artifact to epoch 59.

| Metric | Value |
|---|---:|
| mAP50-95 | 0.349558240 |
| AP50 | 0.643720273 |
| AP75 | 0.328032664 |
| AP small | 0.242258321 |
| AP medium | 0.289552379 |
| AP large | 0.413015851 |
| AR100 | 0.612933009 |

| Class | AP50-95 | AP50 | AP75 |
|---|---:|---:|---:|
| D00 | 0.357249 | 0.635500 | 0.349389 |
| D10 | 0.319145 | 0.616985 | 0.290522 |
| D20 | 0.348215 | 0.661173 | 0.321012 |
| D40 | 0.473635 | 0.757173 | 0.502081 |
| Other | 0.249548 | 0.547771 | 0.177160 |

## Separate epoch-71 threshold experiment

[evaluation_summary.json](../results/metrics/evaluation_summary.json) records checkpoint.pth with hash `2bb70bb33b976216f8384afb9cc5cda97599094c8b3856e670c56233cc644254`, matching the epoch-71 checkpoint in the audit. Confidence >= 0.60; matching IoU >= 0.50; class-aware one-to-one matching: TP 4997, FP 1489, FN 4678, precision 77.04%, recall 51.65%. This is not COCO AP or an epoch-59 result. Selected highest-confidence true positives in that metadata are illustrative, not representative, and are not copied here.

Preserved model_test_results.txt and historical results_summary.txt group these figures under an epoch-59 heading. This document corrects that ambiguity without editing evidence. No threshold-optimization conclusion follows from a single operating point.
