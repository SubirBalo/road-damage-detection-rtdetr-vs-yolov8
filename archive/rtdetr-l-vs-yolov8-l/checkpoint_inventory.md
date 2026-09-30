# Legacy checkpoint inventory

Preservation tag: `pre-r18-audited-migration`, commit `8f2c8ef646cfd9e79717708ce6ca6ee776f79071`.

All eight legacy checkpoint paths were intentionally removed from the current migration tree: four under models/ and four in archived experiment weight directories. These large RT-DETR-L / YOLOv8-L artifacts are unrelated to the audited RT-DETR-R18 deployment. No historical commits or blobs were rewritten. Git blob IDs below identify exact duplicate bytes; they are Git object IDs, not SHA256 engine fingerprints. Model attribution follows legacy run/configuration paths, not newly executed checkpoint inspection.

| Model | Pre-migration path | Removed current-tree path | Git blob ID | Preservation status |
|---|---|---|---|---|
| RT-DETR-L | `models/rtdt/best.pt` | `models/rtdt/best.pt` | `16fd83c7c5fd547673b580e2224fcc7bba7126c2` | Removed from current tree; retained in history/tag |
| RT-DETR-L | `models/rtdt/last.pt` | `models/rtdt/last.pt` | `0c009f0854926876527d8792677093142c2cf55b` | Removed from current tree; retained in history/tag |
| YOLOv8-L | `models/yolov8/best.pt` | `models/yolov8/best.pt` | `c28247a4cdcd949f885ad6b9857dab198e15eca3` | Removed from current tree; retained in history/tag |
| YOLOv8-L | `models/yolov8/last.pt` | `models/yolov8/last.pt` | `45a74369e69fc579f10bec6b802ef1d40d4625b6` | Removed from current tree; retained in history/tag |
| RT-DETR-L | `results/Final result for RT-detr vs Yolo/rtdetr_l_50epochs_earlystop/weights/best.pt` | `archive/rtdetr-l-vs-yolov8-l/results/rtdetr_l_50epochs_earlystop/weights/best.pt` | `16fd83c7c5fd547673b580e2224fcc7bba7126c2` | Removed from current tree; retained in history/tag |
| RT-DETR-L | `results/Final result for RT-detr vs Yolo/rtdetr_l_50epochs_earlystop/weights/last.pt` | `archive/rtdetr-l-vs-yolov8-l/results/rtdetr_l_50epochs_earlystop/weights/last.pt` | `0c009f0854926876527d8792677093142c2cf55b` | Removed from current tree; retained in history/tag |
| YOLOv8-L | `results/Final result for RT-detr vs Yolo/yolov8l_50epochs_batch16/weights/best.pt` | `archive/rtdetr-l-vs-yolov8-l/results/yolov8l_50epochs_batch16/weights/best.pt` | `c28247a4cdcd949f885ad6b9857dab198e15eca3` | Removed from current tree; retained in history/tag |
| YOLOv8-L | `results/Final result for RT-detr vs Yolo/yolov8l_50epochs_batch16/weights/last.pt` | `archive/rtdetr-l-vs-yolov8-l/results/yolov8l_50epochs_batch16/weights/last.pt` | `45a74369e69fc579f10bec6b802ef1d40d4625b6` | Removed from current tree; retained in history/tag |

## Exact duplicates

- `0c009f0854926876527d8792677093142c2cf55b`: `models/rtdt/last.pt` and `archive/rtdetr-l-vs-yolov8-l/results/rtdetr_l_50epochs_earlystop/weights/last.pt`
- `16fd83c7c5fd547673b580e2224fcc7bba7126c2`: `models/rtdt/best.pt` and `archive/rtdetr-l-vs-yolov8-l/results/rtdetr_l_50epochs_earlystop/weights/best.pt`
- `45a74369e69fc579f10bec6b802ef1d40d4625b6`: `models/yolov8/last.pt` and `archive/rtdetr-l-vs-yolov8-l/results/yolov8l_50epochs_batch16/weights/last.pt`
- `c28247a4cdcd949f885ad6b9857dab198e15eca3`: `models/yolov8/best.pt` and `archive/rtdetr-l-vs-yolov8-l/results/yolov8l_50epochs_batch16/weights/best.pt`

The eight removed paths represent four distinct checkpoint blobs. A best checkpoint is not assumed equal to a last checkpoint. No checkpoint remains at these paths in the current tree. The original paths remain available at the preservation tag, and all archived paths remain available at the preceding archive commit `f3c833f48d6e470b4f132ef96361e8a40c8eb626`. CSVs, plots, configurations, notebook and requirements are retained unchanged.

## Historical retrieval

Use `pre-r18-audited-migration:<pre-migration path>` with the exact original path in the table to locate each preserved blob. For example, `git cat-file -s "pre-r18-audited-migration:models/rtdt/best.pt"` inspects its historical size without restoring it. For archived paths, use commit `f3c833f48d6e470b4f132ef96361e8a40c8eb626` instead. Retrieve binaries only to an explicitly chosen location; do not restore or force-add them to the current project.

Removed working-tree file bytes: 615561436 (approximately 587.05 MiB). Git history still retains the objects; this does not promise a reduction in .git storage or historical clone download size.
