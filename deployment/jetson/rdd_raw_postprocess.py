"""NumPy implementation of the repository's RDD RTDETRPostProcessor.

Fixed RDD contract: focal/sigmoid scores, five classes, top 300, no category
remapping. No NMS, clipping, confidence filtering or calibration is done here.
TopK equal-score order is not specified by PyTorch; NumPy uses flattened index
order for deterministic ties. This preserves the same top-k semantics, not an
arbitrary PyTorch CPU/GPU tie ordering.
"""
import numpy as np


def postprocess_raw(pred_logits, pred_boxes, orig_target_sizes, num_top_queries=300):
    logits = np.asarray(pred_logits, dtype=np.float32)
    boxes = np.asarray(pred_boxes, dtype=np.float32)
    sizes = np.asarray(orig_target_sizes, dtype=np.float32)
    if logits.ndim != 3 or logits.shape[-1] != 5:
        raise ValueError('Expected logits [batch, queries, 5]')
    if boxes.shape != logits.shape[:2] + (4,) or sizes.shape != (logits.shape[0], 2):
        raise ValueError('Expected boxes [batch, queries, 4] and sizes [batch, 2] in WIDTH,HEIGHT order')
    if not np.isfinite(logits).all() or not np.isfinite(boxes).all() or not np.isfinite(sizes).all():
        raise ValueError('Non-finite raw model output or target size')
    if np.any(sizes <= 0):
        raise ValueError('Target image dimensions must be positive')
    if not isinstance(num_top_queries, int) or not 1 <= num_top_queries <= logits.shape[1] * 5:
        raise ValueError('Invalid top-k count')

    # Same cxcywh -> xyxy arithmetic and original-size scaling as box_convert.
    half_w = boxes[..., 2] / np.float32(2)
    half_h = boxes[..., 3] / np.float32(2)
    xyxy = np.stack((boxes[..., 0]-half_w, boxes[..., 1]-half_h,
                     boxes[..., 0]+half_w, boxes[..., 1]+half_h), axis=-1)
    xyxy *= np.tile(sizes, (1, 2))[:, None, :]

    # Stable sigmoid, including large negative logits, in float32.
    scores = np.empty_like(logits)
    positive = logits >= 0
    scores[positive] = np.float32(1) / (np.float32(1) + np.exp(-logits[positive]))
    exp_value = np.exp(logits[~positive])
    scores[~positive] = exp_value / (np.float32(1) + exp_value)
    flat = scores.reshape(scores.shape[0], -1)
    index = np.argsort(-flat, axis=1, kind='mergesort')[:, :num_top_queries]
    batch = np.arange(flat.shape[0])[:, None]
    selected_scores = flat[batch, index]
    labels = index % 5
    query_index = index // 5
    selected_boxes = xyxy[batch, query_index]
    return labels.astype(np.int64), selected_boxes, selected_scores
