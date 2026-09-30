# Model and training

RT-DETR-R18 / PResNet-18, five classes, three decoder layers, 300 queries. Logged trainable parameters: 20,088,164. Configured 72 epochs, batch 8, four workers, AdamW main LR 1e-4/backbone LR 1e-5, weight decay 1e-4 with norm/bias exclusions, gradient clipping 0.1. MultiStepLR milestone 1000 was not reached.

EMA is enabled (decay 0.9999, warmup 2000). Despite the runtime YAML AMP default, inspected checkpoints contain populated, evolving GradScaler state and the training code uses autocast with that scaler. This supports AMP use without recovering the launch command.

Evaluation/deployment use 640 x 640. Inherited training configuration includes multiscale sizes 480 through 800, with repeated 640 entries. Augmentation includes photometric distortion, zoom-out, IoU crop, horizontal flip and box sanitization.

The copied [log](../results/metrics/log.txt) has 71 entries for epochs 0-71 except 21. All 72 numbered checkpoints existed in the original project; selected metadata were inspected. Epoch 59 has the highest recorded validation mAP50-95, 0.3521852248, and AP50 0.6454196683. It is the **best recorded validation checkpoint**, not a proven maximum over all 72 evaluations.

The audit found checkpoint0059.pth and RDD2022_RTDETR_R18_BEST_E59.pth byte-identical, SHA256 `0526ee09c256f7b168e1d1521325162b66af5c556753a11b326de3c6913e75fe`. Final checkpoint.pth is epoch 71. Checkpoints are excluded. A pretrained file existed, but a recovered command did not establish its use as training initialization.

Source basis: original configs/rtdetr/rtdetr_r18vd_rdd.yml and inherited include/{optimizer,dataloader,rtdetr_r50vd}.yml, src/solver/det_engine.py, src/zoo/rtdetr/rtdetr.py, and checkpoint state inspected during the audit. Only RDD config snapshots and the log are copied here.
