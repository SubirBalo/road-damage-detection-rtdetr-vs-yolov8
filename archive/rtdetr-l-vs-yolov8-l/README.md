# Legacy comparison archive

Historical RT-DETR-L / YOLOv8-L comparison work. These results are separate from the current audited RT-DETR-R18 project and do not establish a controlled comparison with it.

This archive is preserved for project history. original_README.md is an exact copy of the root README at the pre-migration commit. Its claims are historical, not current audited conclusions. The root README remains unchanged for this first migration commit; authoritative current documentation will be integrated separately.

## Experimental separation

The legacy configurations specify different training settings: RT-DETR-L used batch 8, workers 8 and patience 10; YOLOv8-L used batch 16, workers 4 and patience 100. Both configured 50 epochs and 640-pixel input. The historical README's identical-conditions wording must not be treated as a verified controlled comparison.

Legacy metrics, plots, notebook outputs and checkpoints must not be mixed with current RT-DETR-R18 results. The notebook and configuration files retain their original contents and historical paths; this archive is not an updated execution recipe.

## Historical requirements

The requirements document describes qualitative embedded acquisition, inference, classification, visualization, evaluation and real-time operation. Its chronology relative to the experiments is not proven. It supplies no numeric latency or accuracy acceptance threshold; none is inferred from these experiments.

## Contents and preservation

- original_README.md: byte-preserved original project description.
- docs/: historical requirements.
- notebooks/: original comparison notebook, including stored outputs.
- results/: both legacy experiment directories, excluding checkpoint binaries removed from the current tree; historical locations are recorded in checkpoint_inventory.md.
- checkpoint_inventory.md: checkpoint locations, duplicate Git blob identities and preservation status.

The initial archive commit retained all checkpoint copies and removed tracked .DS_Store metadata. The subsequent cleanup intentionally removes all eight legacy checkpoint paths from the current tree while preserving them through prior commits and the preservation tag. See checkpoint_inventory.md for exact paths and duplicate identities. Historical personal paths and embedded imagery remain in the legacy evidence and still require review before further publication; they were not silently edited in this archival step.

Preservation reference: pre-r18-audited-migration at 8f2c8ef646cfd9e79717708ce6ca6ee776f79071
