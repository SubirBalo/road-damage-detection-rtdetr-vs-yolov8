"""Python 3.6 compatible Nano runtime: TensorRT + CUDA ctypes + OpenCV + NumPy."""
import ctypes
import ctypes.util
from functools import lru_cache
from pathlib import Path
import time

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
NAMES = ('D00', 'D10', 'D20', 'D40', 'Other')


@lru_cache(maxsize=16)
def resize_coefficients(length, target):
    scale = float(length) / target
    support = max(1.0, scale)
    result = []
    for out in range(target):
        center = (out + 0.5) * scale
        start = max(0, int(center - support + 0.5))
        end = min(length, int(center + support + 0.5))
        positions = np.arange(start, end)
        weights = np.maximum(0., 1. - np.abs((positions - center + 0.5) / support))
        weights /= weights.sum()
        result.append((start, end, np.floor(weights * (1 << 22) + 0.5).astype(np.int64)))
    return result


def resize_rgb(rgb):
    """Pillow bilinear uint8 semantics, implemented in NumPy for offline Nano.

    Includes antialiasing for downsampling and per-pass 22-bit rounding. This
    avoids requiring Pillow and avoids substituting OpenCV's different resize.
    """
    h, w = rgb.shape[:2]
    if w == 640:
        horizontal = rgb
    else:
        horizontal = np.empty((h, 640, 3), np.uint8)
        for x, (lo, hi, weights) in enumerate(resize_coefficients(w, 640)):
            values = (rgb[:, lo:hi].astype(np.int64) * weights[None, :, None]).sum(axis=1)
            horizontal[:, x] = np.clip((values + (1 << 21)) >> 22, 0, 255)
    if h == 640:
        return horizontal
    output = np.empty((640, 640, 3), np.uint8)
    for y, (lo, hi, weights) in enumerate(resize_coefficients(h, 640)):
        values = (horizontal[lo:hi].astype(np.int64) * weights[:, None, None]).sum(axis=0)
        output[y] = np.clip((values + (1 << 21)) >> 22, 0, 255)
    return output


def preprocess(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return np.ascontiguousarray(resize_rgb(rgb).transpose(2, 0, 1)[None], dtype=np.float32) / np.float32(255)


class Cuda:
    def __init__(self):
        name = ctypes.util.find_library('cudart') or '/usr/local/cuda/lib64/libcudart.so'
        self.lib = ctypes.CDLL(name)
        self.lib.cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
        self.lib.cudaMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
        self.lib.cudaFree.argtypes = [ctypes.c_void_p]
        self.lib.cudaDeviceSynchronize.argtypes = []
        for method in ('cudaMalloc', 'cudaMemcpy', 'cudaFree', 'cudaDeviceSynchronize'):
            getattr(self.lib, method).restype = ctypes.c_int

    def check(self, code):
        if code != 0:
            raise RuntimeError('CUDA runtime error code {}'.format(code))


class TensorRTEngine:
    def __init__(self, path):
        import tensorrt as trt
        self.trt = trt
        self.cuda = Cuda()
        self.logger = trt.Logger(trt.Logger.WARNING)
        trt.init_libnvinfer_plugins(self.logger, '')
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.runtime.deserialize_cuda_engine(Path(path).read_bytes())
        if self.engine is None:
            raise RuntimeError('Cannot deserialize engine. Build it on this Nano with 02_build_fp16_engine.sh.')
        self.context = self.engine.create_execution_context()
        self.device, self.host, self.indices = [], {}, {}
        types = {trt.float32: np.float32, trt.float16: np.float16, trt.int32: np.int32,
                 trt.int8: np.int8, trt.bool: np.bool_}
        try:
            for i in range(self.engine.num_bindings):
                name = self.engine.get_binding_name(i)
                if self.engine.binding_is_input(i) and -1 in tuple(self.engine.get_binding_shape(i)):
                    shape = (1,3,640,640) if name == 'images' else (1,2)
                    if not self.context.set_binding_shape(i, shape):
                        raise RuntimeError('Cannot set input shape for ' + name)
            for i in range(self.engine.num_bindings):
                name = self.engine.get_binding_name(i)
                shape = tuple(self.context.get_binding_shape(i))
                if any(dim <= 0 for dim in shape):
                    raise ValueError('Unresolved binding shape: {} {}'.format(name, shape))
                array = np.empty(shape, dtype=types[self.engine.get_binding_dtype(i)])
                ptr = ctypes.c_void_p()
                self.cuda.check(self.cuda.lib.cudaMalloc(ctypes.byref(ptr), array.nbytes))
                self.device.append(ptr)
                self.host[name], self.indices[name] = array, i
            if set(self.host) != {'images', 'orig_target_sizes', 'labels', 'boxes', 'scores'}:
                raise ValueError('Unexpected engine bindings: ' + str(list(self.host)))
        except Exception:
            self.close()
            raise
        print('TensorRT {} engine: {}'.format(trt.__version__, path), flush=True)

    def predict(self, frame):
        values = {'images': preprocess(frame), 'orig_target_sizes': [[frame.shape[1], frame.shape[0]]]}
        for name, value in values.items():
            target = self.host[name]
            np.copyto(target, np.asarray(value, dtype=target.dtype))
            self.cuda.check(self.cuda.lib.cudaMemcpy(self.device[self.indices[name]],
                            ctypes.c_void_p(target.ctypes.data), target.nbytes, 1))
        if not self.context.execute_v2([p.value for p in self.device]):
            raise RuntimeError('TensorRT execute_v2 failed')
        self.cuda.check(self.cuda.lib.cudaDeviceSynchronize())
        for name in ('labels', 'boxes', 'scores'):
            target = self.host[name]
            self.cuda.check(self.cuda.lib.cudaMemcpy(ctypes.c_void_p(target.ctypes.data),
                            self.device[self.indices[name]], target.nbytes, 2))
        return [self.host[name].copy() for name in ('labels', 'boxes', 'scores')]

    def close(self):
        for ptr in self.device:
            self.cuda.lib.cudaFree(ptr)
        self.device = []


class HostONNX:
    """Optional Windows validation backend. Not required or installed on Nano."""
    def __init__(self, path):
        import onnxruntime as ort
        options = ort.SessionOptions()
        options.intra_op_num_threads = 4
        options.log_severity_level = 3
        self.session = ort.InferenceSession(str(path), sess_options=options, providers=['CPUExecutionProvider'])
        self.size_type = np.int64 if self.session.get_inputs()[1].type == 'tensor(int64)' else np.int32

    def predict(self, frame):
        return self.session.run(['labels','boxes','scores'], {'images': preprocess(frame),
            'orig_target_sizes': np.array([[frame.shape[1],frame.shape[0]]], dtype=self.size_type)})

    def close(self):
        pass


def open_runner(args):
    return HostONNX(args.model) if args.backend == 'onnxruntime' else TensorRTEngine(args.engine)


def detections(outputs, threshold):
    labels, boxes, scores = outputs
    return [dict(class_id=int(label), class_name=NAMES[int(label)], confidence=float(score),
                 box=[float(x) for x in box])
            for label, box, score in zip(labels.reshape(-1), boxes.reshape(-1,4), scores.reshape(-1))
            if 0 <= int(label) < 5 and np.isfinite(score) and score >= threshold and np.isfinite(box).all()]


def annotate(frame, items, fps=None):
    image = frame.copy()
    h, w = image.shape[:2]
    for item in items:
        x1,y1,x2,y2 = np.rint(item['box']).astype(int)
        x1,x2 = np.clip([x1,x2],0,w-1)
        y1,y2 = np.clip([y1,y2],0,h-1)
        cv2.rectangle(image,(x1,y1),(x2,y2),(0,220,0),2)
        cv2.putText(image, '{} {:.3f}'.format(item['class_name'],item['confidence']),
                    (x1,max(18,y1-6)), cv2.FONT_HERSHEY_SIMPLEX, .6,(0,220,0),2)
    if fps is not None:
        cv2.putText(image,'Processing FPS {:.2f}'.format(fps),(10,28),cv2.FONT_HERSHEY_SIMPLEX,.7,(0,255,255),2)
    return image


def add_arguments(parser):
    parser.add_argument('--engine', type=Path, default=ROOT/'models/nano_fp16.engine')
    parser.add_argument('--conf-threshold', type=float, default=0.60)
    parser.add_argument('--backend', choices=['tensorrt','onnxruntime'], default='tensorrt',
                        help='onnxruntime is an optional host-only validation backend')
    parser.add_argument('--model', type=Path, default=ROOT/'models/nano_compat.onnx')


def validate_args(args):
    if not 0 <= args.conf_threshold <= 1:
        raise ValueError('Confidence threshold must be between zero and one')
