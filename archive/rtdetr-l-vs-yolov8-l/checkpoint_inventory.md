# Legacy checkpoint inventory

Preservation tag: `pre-r18-audited-migration`, commit `8f2c8ef646cfd9e79717708ce6ca6ee776f79071`.

No checkpoint binary is deleted or rewritten in this commit. All eight paths are retained: four under models/ and four moved with experiment results. Git blob IDs below identify exact duplicate bytes; they are Git object IDs, not SHA256 engine fingerprints. Model attribution follows legacy run/configuration paths, not newly executed checkpoint inspection.

| Model | Pre-migration path | Current path | Git blob ID | Preservation status |
|---|---|---|---|---|
| RT-DETR-L | `models/rtdt/best.pt` | `models/rtdt/best.pt` | `16fd83c7c5fd547673b580e2224fcc7bba7126c2` | Retained; also available at preservation tag |
| RT-DETR-L | `models/rtdt/last.pt` | `models/rtdt/last.pt` | `0c009f0854926876527d8792677093142c2cf55b` | Retained; also available at preservation tag |
| YOLOv8-L | `models/yolov8/best.pt` | `models/yolov8/best.pt` | `c28247a4cdcd949f885ad6b9857dab198e15eca3` | Retained; also available at preservation tag |
| YOLOv8-L | `models/yolov8/last.pt` | `models/yolov8/last.pt` | `45a74369e69fc579f10bec6b802ef1d40d4625b6` | Retained; also available at preservation tag |
| RT-DETR-L | `results/Final result for RT-detr vs Yolo/rtdetr_l_50epochs_earlystop/weights/best.pt` | `archive/rtdetr-l-vs-yolov8-l/results/rtdetr_l_50epochs_earlystop/weights/best.pt` | `16fd83c7c5fd547673b580e2224fcc7bba7126c2` | Retained; also available at preservation tag |
| RT-DETR-L | `results/Final result for RT-detr vs Yolo/rtdetr_l_50epochs_earlystop/weights/last.pt` | `archive/rtdetr-l-vs-yolov8-l/results/rtdetr_l_50epochs_earlystop/weights/last.pt` | `0c009f0854926876527d8792677093142c2cf55b` | Retained; also available at preservation tag |
| YOLOv8-L | `results/Final result for RT-detr vs Yolo/yolov8l_50epochs_batch16/weights/best.pt` | `archive/rtdetr-l-vs-yolov8-l/results/yolov8l_50epochs_batch16/weights/best.pt` | `c28247a4cdcd949f885ad6b9857dab198e15eca3` | Retained; also available at preservation tag |
| YOLOv8-L | `results/Final result for RT-detr vs Yolo/yolov8l_50epochs_batch16/weights/last.pt` | `archive/rtdetr-l-vs-yolov8-l/results/yolov8l_50epochs_batch16/weights/last.pt` | `45a74369e69fc579f10bec6b802ef1d40d4625b6` | Retained; also available at preservation tag |

## Exact duplicates

- `0c009f0854926876527d8792677093142c2cf55b`: `models/rtdt/last.pt` and `archive/rtdetr-l-vs-yolov8-l/results/rtdetr_l_50epochs_earlystop/weights/last.pt`
- `16fd83c7c5fd547673b580e2224fcc7bba7126c2`: `models/rtdt/best.pt` and `archive/rtdetr-l-vs-yolov8-l/results/rtdetr_l_50epochs_earlystop/weights/best.pt`
- `45a74369e69fc579f10bec6b802ef1d40d4625b6`: `models/yolov8/last.pt` and `archive/rtdetr-l-vs-yolov8-l/results/yolov8l_50epochs_batch16/weights/last.pt`
- `c28247a4cdcd949f885ad6b9857dab198e15eca3`: `models/yolov8/best.pt` and `archive/rtdetr-l-vs-yolov8-l/results/yolov8l_50epochs_batch16/weights/best.pt`

The eight paths represent four distinct checkpoint blobs. A best checkpoint is not assumed equal to a last checkpoint. The models/ copies remain in their existing locations. No history rewriting, binary deduplication or removal was performed.
