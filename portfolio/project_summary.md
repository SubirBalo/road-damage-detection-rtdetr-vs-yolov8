# Road Damage Detection: RT-DETR and Edge Integration

Adapted and integrated RT-DETR-R18 / PResNet-18 for five road-damage classes using an existing 38,385-image RDD2022 split. Work includes configuration adaptation, YOLO-to-COCO conversion, evaluation utilities and TensorRT/DepthAI camera integration. RT-DETR remains attributed to lyuwenyu and contributors.

Best recorded validation mAP50-95: 35.22% at epoch 59; one validation entry is missing. A COCO test-split evaluation artifact records 34.96% mAP50-95 and 64.37% AP50 without independent checkpoint-hash linkage. The separate epoch-71 threshold experiment is not conflated with those results.

Evidence includes capture/recording code and a 204-frame annotated 640 x 480 demonstration. Project records identify Jetson Nano 4GB, TensorRT 8.0.1.6 FP16 and OAK-D, reporting approximately 1.892 FPS for broad recording timing. Hardware identity and elapsed time are documented rather than independently authenticated. Screen-based demonstration is not field validation.

The project demonstrates model adaptation, traceable evaluation and deployment integration, without production-readiness or full-dataset TensorRT-parity claims.
