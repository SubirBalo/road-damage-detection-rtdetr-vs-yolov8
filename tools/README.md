# Selected tools

Unchanged source snapshots; none were executed during packaging.

- convert_rdd_yolo_to_coco.py: converts an existing split; writes annotation JSON when executed. Does not create the original split.
- evaluate_rdd_examples.py: fixed-threshold evaluation and selected true-positive examples; imports infer.py.
- test_rdd_evaluation.py: existing matching/conversion regression tests; not rerun here.
- infer.py: locally modified upstream inference implementation; requires upstream src and a checkpoint.
- live_onnx_rdd.py: host ONNX camera/video inference; requires an external model/runtime.

The framework is excluded. Paths and dependencies retain their original assumptions. See [environment notes](../environments/environment_notes.md). Scripts may write results when run; their inclusion does not constitute execution or a tested installation recipe.
