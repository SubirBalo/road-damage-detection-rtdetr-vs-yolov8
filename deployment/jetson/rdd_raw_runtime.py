"""Separate raw-output runtime for TensorRT 8 / Python 3.6; old runtime untouched."""
import ctypes
from pathlib import Path

import numpy as np
from rdd_runtime import Cuda, preprocess
from rdd_raw_postprocess import postprocess_raw

SHAPES = {'images': (1,3,640,640), 'pred_logits': (1,300,5), 'pred_boxes': (1,300,4)}


class RawTensorRTEngine:
    def __init__(self, path):
        import tensorrt as trt
        self.logger = trt.Logger(trt.Logger.WARNING)
        trt.init_libnvinfer_plugins(self.logger, '')
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.runtime.deserialize_cuda_engine(Path(path).read_bytes())
        if self.engine is None:
            raise RuntimeError('Cannot load raw engine. Build nano_raw.onnx on this Nano first.')
        if self.engine.num_bindings != 3:
            raise ValueError('Use a nano_raw engine, not the old postprocessed engine')
        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError('Cannot create TensorRT execution context')
        self.cuda = Cuda()
        self.host, self.indices, self.device = {}, {}, []
        types = {trt.float32: np.float32, trt.float16: np.float16}
        try:
            for i in range(self.engine.num_bindings):
                name = self.engine.get_binding_name(i)
                dims = tuple(self.context.get_binding_shape(i))
                if name not in SHAPES or dims != SHAPES[name]:
                    raise ValueError('Unexpected raw binding: {} {}'.format(name,dims))
                if self.engine.binding_is_input(i) != (name == 'images'):
                    raise ValueError('Wrong binding direction: ' + name)
                dtype = self.engine.get_binding_dtype(i)
                if dtype not in types:
                    raise ValueError('Raw bindings must be floating point')
                array = np.empty(dims, dtype=types[dtype])
                ptr = ctypes.c_void_p()
                self.cuda.check(self.cuda.lib.cudaMalloc(ctypes.byref(ptr), array.nbytes))
                self.device.append(ptr)
                self.host[name], self.indices[name] = array, i
        except Exception:
            self.close()
            raise
        print('TensorRT {} raw model: {}'.format(trt.__version__,path),flush=True)

    def infer_raw(self, images):
        target = self.host['images']
        np.copyto(target, np.asarray(images, dtype=target.dtype))
        self.cuda.check(self.cuda.lib.cudaMemcpy(self.device[self.indices['images']],
                        ctypes.c_void_p(target.ctypes.data),target.nbytes,1))
        if not self.context.execute_v2([ptr.value for ptr in self.device]):
            raise RuntimeError('Raw TensorRT execution failed')
        self.cuda.check(self.cuda.lib.cudaDeviceSynchronize())
        outputs = []
        for name in ('pred_logits','pred_boxes'):
            array = self.host[name]
            self.cuda.check(self.cuda.lib.cudaMemcpy(ctypes.c_void_p(array.ctypes.data),
                            self.device[self.indices[name]],array.nbytes,2))
            outputs.append(array.astype(np.float32,copy=True))
        return outputs

    def predict(self, frame):
        logits, boxes = self.infer_raw(preprocess(frame))
        return postprocess_raw(logits,boxes,[[frame.shape[1],frame.shape[0]]])

    def close(self):
        for ptr in self.device:
            self.cuda.lib.cudaFree(ptr)
        self.device = []


class RawHostONNX:
    """Optional Windows test backend; not a Nano dependency."""
    def __init__(self, path):
        import onnxruntime as ort
        options = ort.SessionOptions()
        options.intra_op_num_threads = 4
        options.log_severity_level = 3
        self.session = ort.InferenceSession(str(path),sess_options=options,providers=['CPUExecutionProvider'])
        if [(x.name,x.shape,x.type) for x in self.session.get_inputs()] != [('images',[1,3,640,640],'tensor(float)')]:
            raise ValueError('Unexpected raw ONNX input contract')
        if [x.name for x in self.session.get_outputs()] != ['pred_logits','pred_boxes']:
            raise ValueError('Unexpected raw ONNX outputs')

    def infer_raw(self, images):
        return self.session.run(['pred_logits','pred_boxes'],{'images':images})

    def predict(self, frame):
        logits, boxes = self.infer_raw(preprocess(frame))
        return postprocess_raw(logits,boxes,[[frame.shape[1],frame.shape[0]]])

    def close(self):
        pass
