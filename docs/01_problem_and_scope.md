# Problem and scope

Localize and classify five road-damage patterns with RT-DETR-R18, then integrate the adapted detector with an edge-camera pipeline. Defensible contribution: adapted and integrated RT-DETR for road-damage detection and edge deployment.

Recoverable lifecycle: existing split -> YOLO-to-COCO conversion -> configuration/training -> recorded validation selection -> test evaluation artifact -> ONNX compatibility work -> raw-output runtime -> recorded camera demonstration. The test artifact does not independently identify epoch 59. Chronology and original engineering decisions are not inferred from file timestamps.

## Requirements history

A historical requirements document, docs/02_System_Requirements.md, was found during prior read-only inspection of the earlier RT-DETR-L / YOLOv8-L repository (SubirBalo/road-damage-detection-rtdetr-vs-yolov8, inspected revision 8f2c8ef646cfd9e79717708ce6ca6ee776f79071). This reconciliation uses that previously inspected evidence and the owner's clarification. The historical document is now preserved in this checkout at [archive/rtdetr-l-vs-yolov8-l/docs/02_System_Requirements.md](../archive/rtdetr-l-vs-yolov8-l/docs/02_System_Requirements.md); its contents were not changed during integration.

It describes embedded image acquisition, inference, damage classification, visualization, evaluation and qualitative real-time operation. It contains no numeric latency or accuracy acceptance threshold. Its chronology relative to experiments is not proven; it is not an authenticated pre-experiment specification for current R18 work.

- Historical requirements (A): qualitative statements from the earlier comparison project.
- Reconstructed current objectives (B): implemented/audited RT-DETR-R18 behavior supported by current artifacts.
- Later deployment goals (C): ONNX/TensorRT conversion and camera integration in current deployment work; this grouping does not prove when the broad historical goals were first formulated.
- Future recommended requirements (D): prospective measurable acceptance criteria and repeatable tests to define before further evaluation.

No model-selection rationale was recovered. Historical large-model comparison metrics remain separate from current R18 results; no controlled architecture-superiority claim is made.

This package uses copies only, excludes model/dataset binaries and makes no production-readiness, real-road-validation or independently authenticated hardware-benchmark claim.
