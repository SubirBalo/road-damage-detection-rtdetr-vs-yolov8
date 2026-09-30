# Limitations

- Class imbalance: train D00 has 18,201 boxes versus Other 4,628. Counts alone do not establish causality.
- Other test AP50-95 is weaker at 0.249548. Small-object AP 0.242258321 is lower than medium/large AP.
- Existing split provenance/seed are unknown; filename separation does not exclude content or geographic leakage.
- Epoch-21 validation is missing; epoch 59 is best recorded only.
- COCO test artifact does not independently bind itself to a checkpoint hash. Precision 77.04% / recall 51.65% belongs to the separate epoch-71 experiment.
- Hardware identity, TensorRT 8.0.1.6 and FP16 lack raw device/build logs. Recorded hashes do not authenticate absent binaries.
- Screen-based camera demonstration is not real-road field testing. Reflections, perspective and screen scaling affect imagery; samples are not a labeled accuracy evaluation.
- FPS scopes differ: inference-only, processing overlays and broad recording timing. Elapsed time lacks original console output; final encoding flush is excluded from the timer.
- Recorded single-image agreement does not prove full-dataset TensorRT parity.
- Model/dataset binaries and inherited framework dependencies are excluded. This is not a self-contained runnable release or tested environment lockfile.
- Public redistribution rights for displayed third-party imagery remain unestablished; see THIRD_PARTY_NOTICES.md.

- Historical qualitative requirements exist in the earlier comparison repository, but their chronology relative to experiments is unproven. No numeric latency or accuracy acceptance threshold is documented; reported FPS cannot establish satisfaction of an undefined real-time criterion.
