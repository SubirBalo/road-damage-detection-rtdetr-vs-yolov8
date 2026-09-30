# Environment and integration notes

This is a selected-source evidence package, not a newly tested installation recipe. No packages were installed/changed and no model inference was run during packaging.

## Recorded Windows validation

Reports list PyTorch 2.11.0+cu128, ONNX 1.23.0, ONNX Runtime 1.26.0. The Windows report also lists Python 3.11.16, NumPy 2.4.6 and OpenCV 5.0.0. These are recorded validation environments, not proof of the original training environment. Upstream requirements pin older versions and are deliberately not presented as a validated project lockfile.

## Framework dependency

Upstream reference revision: 29320b6fd828f8e0987a71426cf2d961b09dfed7. Training/evaluation tools require upstream `src`, which is not bundled. RDD configs inherit `configs/runtime.yml` and `configs/rtdetr/include/{dataloader,optimizer,rtdetr_r50vd}.yml`. The original working checkout additionally has torchvision compatibility changes in transforms/COCO adapter and a dataloader modification. A pristine upstream checkout is not asserted to reproduce that environment without those changes.

Copied configs and the conversion tool retain Windows-specific original dataset paths. Adapt paths only in a separately prepared working copy, with review before execution. No command is presented as the recovered original training/export command. The public staging layout is not the original executable framework layout.

## Jetson dependency

Records identify Jetson Nano 4GB, TensorRT 8.0.1.6 FP16. Runtime code depends on TensorRT bindings, CUDA runtime through ctypes, NumPy, OpenCV and Python; camera scripts also require DepthAI. Exact installed Python/DepthAI/OpenCV/NumPy versions are not recorded. JetPack 4.6 / L4T R32.6.1 remain earlier user-reported context, not device-log verification.

Preserve the matching Jetson stack and existing engine; do not rebuild or upgrade it. No build/install commands are supplied. Models and datasets must be obtained separately. These unchanged scripts are evidence snapshots, not idempotent setup tools.
