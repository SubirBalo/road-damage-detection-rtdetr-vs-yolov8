# Jetson runtime evidence

Five unchanged Python copies from FINAL_PROJECT_EVIDENCE. Camera entry points use RawTensorRTEngine; the older postprocessed engine class in rdd_runtime.py is not the camera path.

External engine path: `models/nano_raw_fp16.engine`, relative to the process working directory. Engine not included; do not rebuild. Recorded SHA256:

`df55eb2824336081453894f5a795fc506b49f1ce9a866f486ea36c70db967a1a`

This is a historical fingerprint, not fresh authentication of the unavailable engine. The ONNX fingerprint is likewise a record without bundled model bytes.

Camera code requests 640 x 480 preview, 5 FPS and USB2 mode. Recording writes `results/oak_live/oak_rtdetr_demo.mp4`, assumes the parent exists, and has no explicit existing-output protection. Any future execution needs a separately prepared environment and a new output location. No scripts were executed or changed here.

See [deployment analysis](../../docs/05_deployment.md) for timing and evidence scope.
