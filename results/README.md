# Result provenance and corrections

metrics/log.txt is the unchanged training history: 71 entries, missing epoch 21. evaluation_summary.json is the separate epoch-71 threshold experiment; its original absolute paths are historical provenance, not portable links.

model_test_results.txt and docs/recorded_notes/results_summary.txt are unchanged retrospective summaries. Their epoch-59 heading must not be applied to precision 77.04% / recall 51.65%, which belongs to epoch 71. Their implication that the COCO test artifact binds to epoch 59 is not independently established. Use docs/04_evaluation.md as the interpreted statement.

Benchmark TXT files are recorded project summaries, not raw measurements. The Windows JSON files record host conversion validation, not Nano authentication. Older Nano-pending fields describe the earlier host-validation stage. See docs/05_deployment.md for scopes.

Historical system_architecture.txt also uses 640 x 640 training shorthand. The audited training config includes multiscale resizing; see docs/03_model_and_training.md. Original notes are preserved rather than silently corrected.
