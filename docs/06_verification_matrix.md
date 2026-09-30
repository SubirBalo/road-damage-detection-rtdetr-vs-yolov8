# Requirements and verification matrix

Origins: A historical qualitative requirements from the earlier RT-DETR-L/YOLOv8-L project; B reconstructed objectives of current audited RT-DETR-R18 work; C later deployment goals; D future recommended requirements.

The [archived historical requirements](../archive/rtdetr-l-vs-yolov8-l/docs/02_System_Requirements.md) document embedded image acquisition, inference, damage classification, visualization, evaluation and qualitative real-time operation. No numeric latency or accuracy acceptance threshold is documented. Chronology relative to experiments is unproven. See [requirements history](01_problem_and_scope.md). A historical goal is not evidence of current satisfaction.

R01/R05/R07/R08 overlap thematically with those broad historical goals. Their specific five-class/test-split/FP16/capture criteria retain the B/C origins below; the historical file does not establish those details as original requirements.

| ID | Requirement statement | Origin | Verification method | Evidence | Result | Status | Notes |
|---|---|---|---|---|---|---|---|
| R01 | Five-class detection | B | Inspect mappings/configs | Dataset audit, configs, runtime | Five consistent categories | Verified | No invented per-class thresholds |
| R02 | Train/validation/test dataset usage | B | Inspect references and counts | Dataset audit, RDD configs | Three partitions; references resolved | Verified | Original split provenance unknown |
| R03 | Training history/checkpoint retention | B | Inspect original outputs | Copied log, prior checkpoint inventory | 72 numbered checkpoints; 71 log entries | Partially Verified | Epoch-21 validation absent; binaries excluded here |
| R04 | Best recorded validation checkpoint | B | Rank logs; hash selected copies | log.txt, prior E59 hash audit | E59 recorded maximum; copies identical | Verified | Narrow recorded claim, not global maximum over missing epoch |
| R05 | COCO test-split evaluation | B | Recover arrays; inspect config | Prior eval.pth audit, metric summary | Supplied metrics recovered | Partially Verified | Exact checkpoint linkage absent |
| R06 | ONNX conversion agreement | C | Inspect parity and hashes | Windows JSON reports | Close recorded agreement | Verified | Limited to recorded comparison cases |
| R07 | FP16 Jetson execution | C | Inspect code/summaries/media | Runtime, golden text, demo | Consistent deployment evidence | Partially Verified | Engine/device/build logs absent |
| R08 | OAK-D live/recorded capture | C | Inspect DepthAI code; decode media | Camera scripts, MP4 | Capture/recording implementation and 204 frames | Partially Verified | Hardware identity recorded, not device-logged |
| R09 | Numeric latency/accuracy acceptance requirement | A: qualitative real-time goal only; numeric criteria unavailable | Inspect historical requirements | Earlier docs/02_System_Requirements.md, prior inspection | No numeric latency or accuracy acceptance threshold documented | Not Testable From Current Evidence | Chronology unproven; do not invent thresholds or infer real-time acceptance from FPS |
| R10 | Repeatable deployment acceptance testing | D | Review protocol/raw runs | Recommendation only | Protocol not established | Not Verified | Set future criteria before testing |

R04 is Verified only for the expressly narrower best-recorded statement. The broader all-72-epochs claim remains partially verified. R07/R08 remain Partially Verified despite new code/media evidence.
