"""Fixed-threshold RDD test evaluation and illustrative true-positive selection.

This reports precision/recall at one operating point, not COCO AP. Predictions
are matched by descending confidence to unused same-class ground truth at the
specified IoU. The highest-confidence matched detection per class is selected
only for illustration; no thresholds or model parameters are optimized.
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageDraw

from infer import CLASS_NAMES, RDDPredictor, probability


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def box_iou(box, other):
    intersection = max(0., min(box[2], other[2]) - max(box[0], other[0])) * max(
        0., min(box[3], other[3]) - max(box[1], other[1]))
    area = max(0., box[2] - box[0]) * max(0., box[3] - box[1])
    other_area = max(0., other[2] - other[0]) * max(0., other[3] - other[1])
    union = area + other_area - intersection
    return intersection / union if union > 0 else 0.


def match_predictions(labels, boxes, scores, annotations, confidence, iou_threshold):
    """Return thresholded detections with one-to-one TP/FP assignments."""
    used = set()
    matches = []
    for index in np.argsort(-np.asarray(scores), kind='stable'):
        score = float(scores[index])
        if score < confidence:
            continue
        label = int(labels[index])
        box = [float(x) for x in boxes[index]]
        candidates = []
        for gt in annotations:
            if gt['id'] in used or gt['category_id'] != label:
                continue
            x, y, w, h = gt['bbox']
            gt_box = [x, y, x + w, y + h]
            candidates.append((box_iou(box, gt_box), gt['id'], gt_box))
        best = max(candidates, key=lambda item: item[0]) if candidates else None
        tp = best is not None and best[0] >= iou_threshold
        detection = dict(category_id=label, class_name=CLASS_NAMES[label], score=score,
                         bbox_xyxy=box, true_positive=tp)
        if tp:
            used.add(best[1])
            detection.update(annotation_id=best[1], iou=best[0], gt_bbox_xyxy=best[2])
        matches.append(detection)
    return matches


def metrics(tp, fp, gt):
    return dict(tp=tp, fp=fp, fn=gt-tp, ground_truth=gt,
                precision=tp/(tp+fp) if tp+fp else None,
                recall=tp/gt if gt else None)


def main(args):
    if args.batch_size < 1 or args.iou_threshold <= 0:
        raise ValueError('batch size and IoU threshold must be positive')
    dataset = json.loads(args.annotations.read_text(encoding='utf-8'))
    categories = {c['id']: c['name'] for c in dataset['categories']}
    if categories != dict(enumerate(CLASS_NAMES)):
        raise ValueError(f'Unexpected category mapping: {categories}')
    images = sorted(dataset['images'], key=lambda image: image['id'])
    if not images or len({image['id'] for image in images}) != len(images):
        raise ValueError('Image IDs must be nonempty and unique')
    image_ids = {image['id'] for image in images}
    annotations = defaultdict(list)
    gt_counts = [0] * len(CLASS_NAMES)
    annotation_ids = set()
    for gt in dataset['annotations']:
        if gt.get('iscrowd', 0) or gt.get('ignore', 0):
            raise ValueError('This simple evaluator does not implement crowd/ignore matching')
        if gt['id'] in annotation_ids or gt['image_id'] not in image_ids:
            raise ValueError('Duplicate annotation ID or unknown image ID')
        if gt['category_id'] not in categories or gt['bbox'][2] <= 0 or gt['bbox'][3] <= 0:
            raise ValueError('Invalid category or box')
        annotation_ids.add(gt['id'])
        annotations[gt['image_id']].append(gt)
        gt_counts[gt['category_id']] += 1

    checkpoint_hash = sha256(args.resume)
    predictor = RDDPredictor(args.config, args.resume, args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tp, fp = [0] * len(CLASS_NAMES), [0] * len(CLASS_NAMES)
    examples = {}
    started = time.monotonic()
    predictions_path = args.output_dir / 'predictions.jsonl'
    with predictions_path.open('w', encoding='utf-8') as stream:
        for start in range(0, len(images), args.batch_size):
            records = images[start:start+args.batch_size]
            batch = []
            for record in records:
                with Image.open(args.images_dir / record['file_name']) as source:
                    image = source.convert('RGB')
                if image.size != (record['width'], record['height']):
                    raise ValueError(f"Image size differs from annotation: {record['file_name']}")
                batch.append(image)
            for record, prediction in zip(records, predictor.predict(batch)):
                detections = match_predictions(*prediction, annotations[record['id']],
                                               args.conf_threshold, args.iou_threshold)
                stream.write(json.dumps(dict(image_id=record['id'], file_name=record['file_name'],
                                             detections=detections)) + '\n')
                for detection in detections:
                    label = detection['category_id']
                    if detection['true_positive']:
                        tp[label] += 1
                        if label not in examples or detection['score'] > examples[label]['score']:
                            examples[label] = dict(detection, image_id=record['id'], file_name=record['file_name'])
                    else:
                        fp[label] += 1
            done = start + len(records)
            if start == 0 or done % 200 == 0 or done == len(images):
                print(f'{done}/{len(images)} images; {time.monotonic()-started:.1f}s', flush=True)

    for label, example in examples.items():
        with Image.open(args.images_dir / example['file_name']) as source:
            image = source.convert('RGB')
        draw = ImageDraw.Draw(image)
        draw.rectangle(example['gt_bbox_xyxy'], outline='lime', width=3)
        draw.rectangle(example['bbox_xyxy'], outline='red', width=3)
        text = f"{CLASS_NAMES[label]} score={example['score']:.3f} IoU={example['iou']:.3f} | red=prediction green=GT"
        draw.rectangle((0, 0, image.width, 22), fill='black')
        draw.text((4, 4), text, fill='white')
        path = args.output_dir / f'{CLASS_NAMES[label]}_true_positive.jpg'
        image.save(path)
        example['visualization'] = str(path.resolve())

    final_hash = sha256(args.resume)
    if final_hash != checkpoint_hash:
        raise RuntimeError('Checkpoint hash changed during evaluation')
    report = dict(
        checkpoint=str(args.resume.resolve()), checkpoint_sha256=checkpoint_hash,
        checkpoint_unchanged=True, weight_source=predictor.weight_source,
        config=str(args.config.resolve()), annotations=str(args.annotations.resolve()),
        annotations_sha256=sha256(args.annotations), images_dir=str(args.images_dir.resolve()),
        image_count=len(images), confidence_threshold=args.conf_threshold,
        iou_threshold=args.iou_threshold, batch_size=args.batch_size, device=args.device,
        inference='FP32 full-image 640x640; no NMS; no test-set tuning',
        matching='Descending confidence, class-aware, one-to-one IoU matching; not COCO AP',
        example_selection='Highest-confidence matched TP per class; illustrative, not representative',
        overall=metrics(sum(tp), sum(fp), sum(gt_counts)),
        per_class={name: metrics(tp[i], fp[i], gt_counts[i]) for i, name in enumerate(CLASS_NAMES)},
        examples={CLASS_NAMES[i]: example for i, example in sorted(examples.items())},
        missing_classes=[name for i, name in enumerate(CLASS_NAMES) if i not in examples],
        predictions=str(predictions_path.resolve()), elapsed_seconds=time.monotonic()-started)
    path = args.output_dir / 'evaluation_summary.json'
    path.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({key: report[key] for key in ('overall', 'per_class', 'missing_classes')}, indent=2))
    print(f'Saved: {path}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-c', '--config', type=Path, required=True)
    parser.add_argument('-r', '--resume', type=Path, required=True)
    parser.add_argument('--annotations', type=Path, required=True)
    parser.add_argument('--images-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, default=Path('output/rdd_test_examples'))
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--conf-threshold', type=probability, default=0.6)
    parser.add_argument('--iou-threshold', type=probability, default=0.5)
    main(parser.parse_args())
