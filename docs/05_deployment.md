# Deployment and measurement scope

## Implemented path

DepthAI color camera -> 640 x 480 frame -> BGR-to-RGB and bilinear-compatible 640 x 640 resize -> float32 NCHW / 255 -> TensorRT raw engine -> NumPy postprocessing -> confidence >= 0.60 -> annotation -> display / MP4.

Expected raw bindings: images [1,3,640,640], pred_logits [1,300,5], pred_boxes [1,300,4]. Runtime reads actual FP32/FP16 binding types; code alone does not prove engine precision. Postprocessing applies sigmoid, flattens 300 x 5 scores, selects top 300, derives class/query indices and converts normalized cxcywh to original-image xyxy. No NMS. Drawing clips displayed boxes; the raw postprocessor does not clip boxes.

Prior audit inspected ONNX metadata: original opset 18/IR10, compatibility and raw graphs opset 13/IR7. Compatibility work decomposes operations and moves final postprocessing outside the graph. Internal query-selection TopK remains part of the network. No export/build was repeated.

## Evidence classification

- Directly verified from executable/code artifact: capture implementation, 640 x 480 preview / 5 FPS request, USB2 mode, threshold, TensorRT/CUDA calls, postprocessing and recording.
- Directly verified from media/artifact: 204 decodable frames, 640 x 480 annotated screen-based camera demonstration, encoded 5 FPS / 40.8 seconds playback.
- Supported by recorded project evidence: Jetson Nano 4GB, TensorRT 8.0.1.6, FP16, OAK-D identity and reported benchmark timing. Raw device inventory is absent.
- User-reported only: earlier JetPack 4.6 / L4T R32.6.1 context.
- Not verifiable: actual missing engine against recorded hash; full-dataset FP16 parity.

## Benchmark scopes

| Record | Value | Qualification |
|---|---|---|
| RTX 5070 ONNX Runtime CUDA | ~8.11 ms / ~123.37 FPS | Summary; inference-only; raw run log absent |
| Jetson file-video | ~2.37 FPS | Summary; precise timing boundaries unavailable |
| Jetson/OAK-D live | ~2.2 FPS | Recorded observation; sampled media overlays near 2.2 corroborate displayed processing rate, not a live end-to-end average |
| Recording | 204 / 107.825331879 s = 1.891948733 FPS | Frame count verified in media; elapsed time from summary, not raw console log |

Recording aggregate timer includes device initialization, acquisition, preprocessing, inference, postprocessing/filtering, annotation, display and frame-writing calls. Engine/writer initialization precedes it; final writer.release() flush and subsequent cleanup occur after elapsed-time capture. Thus it includes broad pipeline overhead, but not every startup/shutdown cost.

The Processing FPS overlay times prediction/filtering only, excluding capture, annotation, display and writing. Encoded playback of 40.8 seconds is not the reported 107.825-second wall time. These scopes prohibit direct inference-only versus recording throughput comparison.

## Recorded single-image agreement

China_Drone_000008.jpg: Windows D10 confidence ~0.783393, box ~[24.879,337.397,378.026,457.633]; recorded Jetson D10 confidence ~0.783326, box [25.5,337.125,377.75,457.375]. Approximate confidence difference 0.000067; maximum coordinate difference 0.621 pixels. Recorded single-image agreement is not full-dataset TensorRT parity. Raw Jetson output is absent.

Windows nano_raw_validation_final.json records epoch-59 EMA comparison and maximum postprocessed box difference about 0.01325 pixels versus PyTorch. Its Nano-pending fields describe the earlier host-validation stage; later summaries/media do not turn it into raw Nano execution evidence.
