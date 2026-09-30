# LinkedIn project description

Adapted and integrated RT-DETR-R18 for five-class road-damage detection and edge-camera deployment, building on upstream RT-DETR by lyuwenyu and contributors.

Using an existing 38,385-image RDD2022 split, the work covers YOLO-to-COCO conversion, model configuration, evaluation utilities and TensorRT/DepthAI camera recording. Best recorded validation mAP50-95 reached 35.22%. A separate COCO test-split evaluation artifact reports 34.96% mAP50-95 and 64.37% AP50, with checkpoint linkage retained as an explicit evidence limitation.

A 204-frame annotated camera demo documents integration. Project records describe Jetson Nano FP16/OAK-D operation with approximately 1.892 FPS for recording including acquisition, processing, display and frame writing. The camera views images on a screen, not a real-road field test. Documentation distinguishes verified code/media from recorded hardware and timing claims.
