# Future work: recommendations, not completed experiments

1. Preserve the engine and recover original device/version output, hash-command output, golden JSON and benchmark logs without rebuilding.
2. Recover checkpoint-linked test execution records and epoch-21 validation if available; otherwise retain the gaps.
3. Document dataset acquisition/split provenance and investigate content/geographic leakage.
4. Translate historical qualitative goals into prospective measurable acceptance criteria, without retroactively assigning thresholds. Define acceptance testing before new measurements: device identity, power/clocks/thermals, warmup, timing boundaries, repeated runs and variability. No thresholds are invented here.
5. With separate authorization, evaluate full-dataset converted-model accuracy, small objects and class imbalance. Tune thresholds on validation data, not test-split data.
6. Plan labeled real-road and vehicle-mounted testing. Screen demonstration remains an integration milestone.
7. Prepare a reproducible environment and reviewed upstream patches in isolation; review media rights before publication.

No training, inference, export, engine build, installation or publication was performed during packaging.
