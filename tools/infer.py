import torch
import torchvision.transforms as T
import numpy as np 
from PIL import Image, ImageDraw, ImageFont
import os 
import sys 
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import argparse
from src.core import YAMLConfig 
from pathlib import Path
import json

CLASS_NAMES = ('D00', 'D10', 'D20', 'D40', 'Other')


def to_numpy(value):
    return value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)


def probability(value):
    value = float(value)
    if not 0 <= value <= 1:
        raise argparse.ArgumentTypeError('must be between 0 and 1')
    return value


class RDDPredictor:
    """Read-only checkpoint loading and full-image FP32 inference."""

    def __init__(self, config, checkpoint, device='cpu'):
        self.device = torch.device(device)
        if self.device.type == 'cuda' and not torch.cuda.is_available():
            raise ValueError('CUDA requested but unavailable')
        cfg = YAMLConfig(str(config))
        if cfg.yaml_cfg.get('num_classes') != len(CLASS_NAMES) or cfg.yaml_cfg.get('remap_mscoco_category', False):
            raise ValueError('RDD requires five classes and remap_mscoco_category=False')
        # The full checkpoint supplies every parameter; avoid a backbone download.
        cfg.yaml_cfg['PResNet']['pretrained'] = False
        state = torch.load(checkpoint, map_location='cpu', weights_only=True)
        self.weight_source = 'ema' if 'ema' in state else 'model'
        weights = state['ema']['module'] if self.weight_source == 'ema' else state['model']
        self.model = cfg.model
        self.model.load_state_dict(weights, strict=True)
        self.model.to(self.device)
        self.model.eval()
        self.postprocessor = cfg.postprocessor.deploy().to(self.device)
        self.transforms = T.Compose([T.Resize((640, 640)), T.ToTensor()])

    @torch.no_grad()
    def predict(self, images):
        tensors = torch.stack([self.transforms(image) for image in images]).to(self.device)
        sizes = torch.tensor([image.size for image in images], device=self.device)
        labels, boxes, scores = self.postprocessor(self.model(tensors), sizes)
        labels, boxes, scores = map(to_numpy, (labels, boxes, scores))
        return list(zip(labels, boxes, scores))

def postprocess(labels, boxes, scores, iou_threshold=0.55):
    def calculate_iou(box1, box2):
        x1, y1, x2, y2 = box1
        x3, y3, x4, y4 = box2
        xi1 = max(x1, x3)
        yi1 = max(y1, y3)
        xi2 = min(x2, x4)
        yi2 = min(y2, y4)
        inter_width = max(0, xi2 - xi1)
        inter_height = max(0, yi2 - yi1)
        inter_area = inter_width * inter_height
        box1_area = (x2 - x1) * (y2 - y1)
        box2_area = (x4 - x3) * (y4 - y3)
        union_area = box1_area + box2_area - inter_area
        iou = inter_area / union_area if union_area != 0 else 0
        return iou
    merged_labels = []
    merged_boxes = []
    merged_scores = []
    used_indices = set()
    for i in range(len(boxes)):
        if i in used_indices:
            continue
        current_box = boxes[i]
        current_label = labels[i]
        current_score = scores[i]
        boxes_to_merge = [current_box]
        scores_to_merge = [current_score]
        used_indices.add(i)
        for j in range(i + 1, len(boxes)):
            if j in used_indices:
                continue
            if labels[j] != current_label:
                continue  
            other_box = boxes[j]
            iou = calculate_iou(current_box, other_box)
            if iou >= iou_threshold:
                boxes_to_merge.append(other_box.tolist())  
                scores_to_merge.append(scores[j])
                used_indices.add(j)
        xs = np.concatenate([[box[0], box[2]] for box in boxes_to_merge])
        ys = np.concatenate([[box[1], box[3]] for box in boxes_to_merge])
        merged_box = [np.min(xs), np.min(ys), np.max(xs), np.max(ys)]
        merged_score = max(scores_to_merge)
        merged_boxes.append(merged_box)
        merged_labels.append(current_label)
        merged_scores.append(merged_score)
    return [np.array(merged_labels)], [np.array(merged_boxes)], [np.array(merged_scores)]
def slice_image(image, slice_height, slice_width, overlap_ratio):
    img_width, img_height = image.size
    
    slices = []
    coordinates = []
    step_x = max(1, int(slice_width * (1 - overlap_ratio)))
    step_y = max(1, int(slice_height * (1 - overlap_ratio)))
    
    for y in range(0, img_height, step_y):
        for x in range(0, img_width, step_x):
            box = (x, y, min(x + slice_width, img_width), min(y + slice_height, img_height))
            slice_img = image.crop(box)
            slices.append(slice_img)
            coordinates.append((x, y))
    return slices, coordinates
def merge_predictions(predictions, slice_coordinates, orig_image_size, slice_width, slice_height, threshold=0.30):
    merged_labels = []
    merged_boxes = []
    merged_scores = []
    orig_height, orig_width = orig_image_size
    for i, (label, boxes, scores) in enumerate(predictions):
        x_shift, y_shift = slice_coordinates[i]
        scores = np.array(scores).reshape(-1)
        valid_indices = scores >= threshold
        valid_labels = np.array(label).reshape(-1)[valid_indices]
        valid_boxes = np.array(boxes).reshape(-1, 4)[valid_indices]
        valid_scores = scores[valid_indices]
        for j, box in enumerate(valid_boxes):
            box[0] = np.clip(box[0] + x_shift, 0, orig_width)  
            box[1] = np.clip(box[1] + y_shift, 0, orig_height)
            box[2] = np.clip(box[2] + x_shift, 0, orig_width)  
            box[3] = np.clip(box[3] + y_shift, 0, orig_height) 
            valid_boxes[j] = box
        merged_labels.extend(valid_labels)
        merged_boxes.extend(valid_boxes)
        merged_scores.extend(valid_scores)
    return np.array(merged_labels), np.array(merged_boxes), np.array(merged_scores)
def draw(images, labels, boxes, scores, thrh = 0.6, path = ""):
    for i, im in enumerate(images):
        draw = ImageDraw.Draw(im)
        scr = to_numpy(scores[i])
        lab_all = to_numpy(labels[i])
        box_all = to_numpy(boxes[i])
        keep = scr >= thrh
        lab = lab_all[keep]
        box = box_all[keep]
        scrs = scr[keep]
        names = CLASS_NAMES
        for j,b in enumerate(box):
            draw.rectangle(list(b), outline='red',)
            draw.text((b[0], b[1]), text=f"{names[int(lab[j])]} {float(scrs[j]):.2f}", font=ImageFont.load_default(), fill='blue')
        if path == "":
            im.save(f'results_{i}.jpg')
        else:
            im.save(path)
            
@torch.no_grad()
def main(args):
    predictor = RDDPredictor(args.config, args.resume, args.device)
    with Image.open(args.im_file) as source:
        im_pil = source.convert('RGB')
    w, h = im_pil.size
    if args.sliced:
        num_boxes = args.numberofboxes
        
        aspect_ratio = w / h
        if num_boxes < 1:
            raise ValueError('--numberofboxes must be positive')
        num_cols = max(1, int(np.sqrt(num_boxes * aspect_ratio)))
        num_rows = max(1, int(num_boxes / num_cols))
        slice_height = max(1, h // num_rows)
        slice_width = max(1, w // num_cols)
        overlap_ratio = 0.2
        slices, coordinates = slice_image(im_pil, slice_height, slice_width, overlap_ratio)
        predictions = []
        for i, slice_img in enumerate(slices):
            predictions.append(predictor.predict([slice_img])[0])
        
        merged_labels, merged_boxes, merged_scores = merge_predictions(predictions, coordinates, (h, w), slice_width, slice_height, threshold=args.conf_threshold)
        labels, boxes, scores = postprocess(merged_labels, merged_boxes, merged_scores)
    else:
        labels, boxes, scores = zip(*predictor.predict([im_pil]))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / (Path(args.im_file).stem + '_predictions.jpg')
    draw([im_pil], labels, boxes, scores, args.conf_threshold, str(output_path))
    detections = [dict(category_id=int(label), class_name=CLASS_NAMES[int(label)],
                       bbox_xyxy=to_numpy(box).tolist(), score=float(score))
                  for label, box, score in zip(labels[0], boxes[0], scores[0])
                  if score >= args.conf_threshold]
    output_path.with_suffix('.json').write_text(json.dumps({
        'image': str(Path(args.im_file).resolve()), 'checkpoint': str(Path(args.resume).resolve()),
        'weight_source': predictor.weight_source, 'confidence_threshold': args.conf_threshold,
        'sliced': args.sliced, 'predictions': detections}, indent=2), encoding='utf-8')
    print(f'Saved {len(detections)} detections: {output_path}')
  
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=True)
    parser.add_argument('-r', '--resume', type=str, required=True)
    parser.add_argument('-f', '--im-file', type=str, required=True)
    parser.add_argument('-s', '--sliced', action='store_true')
    parser.add_argument('-d', '--device', type=str, default='cpu')
    parser.add_argument('-nc', '--numberofboxes', type=int, default=25)
    parser.add_argument('--conf-threshold', type=probability, default=0.6)
    parser.add_argument('--output-dir', type=Path, default=Path('output/rdd_inference'))
    args = parser.parse_args()
    main(args)











