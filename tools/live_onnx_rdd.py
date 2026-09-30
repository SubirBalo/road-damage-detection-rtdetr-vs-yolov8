"""Live RDD2022 detection from a webcam or video using ONNX Runtime.

Press q in the preview window to quit. Requires numpy, Pillow, opencv-python
and ONNX Runtime (onnxruntime-gpu for CUDA); no PyTorch dependency.
"""
import argparse
from pathlib import Path
import time

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = ROOT / 'output/deployment/RDD2022_RTDETR_R18_E59_640.onnx'
CLASS_NAMES = ('D00', 'D10', 'D20', 'D40', 'Other')
COLORS = ((0, 200, 255), (255, 180, 0), (0, 220, 0), (0, 80, 255), (220, 0, 220))
WINDOW = 'RDD2022 | q: quit'


def probability(value):
    number = float(value)
    if not 0 <= number <= 1:
        raise argparse.ArgumentTypeError('confidence must be between 0 and 1')
    return number


def create_session(model_path):
    if not model_path.is_file():
        raise FileNotFoundError(f'ONNX model not found: {model_path}')
    available = ort.get_available_providers()
    providers = ['CPUExecutionProvider']
    if 'CUDAExecutionProvider' in available:
        providers.insert(0, 'CUDAExecutionProvider')
    try:
        session = ort.InferenceSession(str(model_path), providers=providers)
    except Exception as error:
        if 'CUDAExecutionProvider' not in providers:
            raise
        print(f'CUDA session initialization failed: {error}\nFalling back to CPU.', flush=True)
        session = ort.InferenceSession(str(model_path), providers=['CPUExecutionProvider'])
    active = session.get_providers()
    print(f'Active ONNX Runtime execution provider: {active[0]}', flush=True)
    print(f'Session providers: {", ".join(active)}', flush=True)
    inputs = {item.name: item for item in session.get_inputs()}
    if set(inputs) != {'images', 'orig_target_sizes'}:
        raise ValueError(f'Unexpected ONNX inputs: {list(inputs)}')
    if inputs['images'].type != 'tensor(float)' or inputs['orig_target_sizes'].type != 'tensor(int64)':
        raise ValueError('Expected float32 images and int64 orig_target_sizes')
    if not {'labels', 'boxes', 'scores'} <= {item.name for item in session.get_outputs()}:
        raise ValueError('Expected ONNX outputs: labels, boxes, scores')
    return session


def preprocess(frame):
    """Match torchvision's validated PIL Resize + ToTensor pipeline exactly."""
    height, width = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # PIL bilinear matches T.Resize((640, 640)) on the original RGB PIL image.
    # OpenCV resize has different interpolation details, so do not substitute it.
    resized = Image.fromarray(rgb).resize((640, 640), Image.Resampling.BILINEAR)
    images = np.asarray(resized, dtype=np.float32).transpose(2, 0, 1)[None]
    images = np.ascontiguousarray(images / np.float32(255.0))
    # Exported postprocessor expects [width, height], not [height, width].
    sizes = np.array([[width, height]], dtype=np.int64)
    return {'images': images, 'orig_target_sizes': sizes}


def draw_predictions(frame, labels, boxes, scores, threshold):
    height, width = frame.shape[:2]
    for label, box, score in zip(labels.reshape(-1), boxes.reshape(-1, 4), scores.reshape(-1)):
        label = int(label)
        if not np.isfinite(score) or score < threshold or not 0 <= label < len(CLASS_NAMES):
            continue
        if not np.isfinite(box).all():
            continue
        # The ONNX postprocessor already returns original-image pixel xyxy boxes.
        x1, y1, x2, y2 = np.rint(box).astype(int)
        x1, x2 = np.clip([x1, x2], 0, width - 1)
        y1, y2 = np.clip([y1, y2], 0, height - 1)
        if x2 <= x1 or y2 <= y1:
            continue
        color = COLORS[label]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        text = f'{CLASS_NAMES[label]} {float(score):.2f}'
        cv2.putText(frame, text, (x1, max(18, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, text, (x1, max(18, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)


def main(args):
    source = int(args.source) if args.source.isdecimal() else args.source
    if isinstance(source, str):
        if not Path(source).is_file():
            raise FileNotFoundError(f'Video not found: {source}')
        if args.save and Path(source).resolve() == args.output.resolve():
            raise ValueError('Output must differ from the input video')
    if args.save:
        if args.output.suffix.lower() not in {'.mp4', '.avi'}:
            raise ValueError('Recording output must end in .mp4 or .avi')
        if args.output.exists():
            raise FileExistsError(f'Refusing to overwrite existing recording: {args.output}')
    session = create_session(args.model)
    capture = cv2.VideoCapture(source)
    writer = None
    frames = 0
    smoothed_fps = None
    try:
        if not capture.isOpened():
            raise RuntimeError(f'Cannot open source: {args.source}')
        recording_fps = capture.get(cv2.CAP_PROP_FPS)
        if not np.isfinite(recording_fps) or recording_fps <= 0:
            recording_fps = 30.0
            print('Source FPS unavailable; recording at 30 FPS.', flush=True)
        print('Press q in the preview window to quit.', flush=True)
        while True:
            started = time.perf_counter()
            ok, frame = capture.read()
            if not ok:
                if frames == 0:
                    raise RuntimeError('Source opened but no frame could be read')
                print('End of video or capture stream.', flush=True)
                break
            labels, boxes, scores = session.run(['labels', 'boxes', 'scores'], preprocess(frame))
            draw_predictions(frame, labels, boxes, scores, args.conf_threshold)
            fps = 1.0 / max(time.perf_counter() - started, 1e-9)
            smoothed_fps = fps if smoothed_fps is None else 0.9 * smoothed_fps + 0.1 * fps
            # Capture + preprocessing + inference + boxes; excludes display/encode.
            cv2.putText(frame, f'Processing FPS: {smoothed_fps:.1f}', (10, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(frame, f'Processing FPS: {smoothed_fps:.1f}', (10, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            if args.save:
                if writer is None:
                    args.output.parent.mkdir(parents=True, exist_ok=True)
                    codec = 'mp4v' if args.output.suffix.lower() == '.mp4' else 'MJPG'
                    writer = cv2.VideoWriter(str(args.output), cv2.VideoWriter_fourcc(*codec),
                                             recording_fps, (frame.shape[1], frame.shape[0]))
                    if not writer.isOpened():
                        raise RuntimeError(f'Cannot create video recording: {args.output}')
                    print(f'Recording: {args.output} ({recording_fps:.2f} FPS)', flush=True)
                writer.write(frame)
            frames += 1
            cv2.imshow(WINDOW, frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()
        print(f'Processed {frames} frames.', flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, default=DEFAULT_MODEL)
    parser.add_argument('--source', default='0', help='Webcam index (0) or video-file path')
    parser.add_argument('--conf-threshold', type=probability, default=0.60)
    parser.add_argument('--save', action='store_true', help='Record annotated frames')
    parser.add_argument('--output', type=Path, default=ROOT / 'output/deployment/rdd_live.mp4')
    return parser.parse_args()


if __name__ == '__main__':
    main(parse_args())
