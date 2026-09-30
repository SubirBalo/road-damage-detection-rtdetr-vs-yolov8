# Attribution and licensing scope

## Project-specific original contributions — MIT

The owner has selected [MIT](LICENSE) for their project-specific original contributions, including original code and documentation. This applies only to material the owner has the right to license. It does not relicense Apache-licensed upstream material or third-party assets. Mixed/derived files retain applicable upstream obligations.

## Upstream RT-DETR — Apache License 2.0

RT-DETR: https://github.com/lyuwenyu/RT-DETR, by lyuwenyu and contributors. Audited upstream revision: 29320b6fd828f8e0987a71426cf2d961b09dfed7.

The upstream license is preserved byte-for-byte at [LICENSES/RT-DETR-Apache-2.0.txt](LICENSES/RT-DETR-Apache-2.0.txt). Root MIT does not replace that license or make the repository uniformly MIT. Preserve upstream attribution and applicable notices. No root upstream NOTICE file was found in the earlier inspection.

`tools/infer.py` is a locally modified upstream framework file; upstream-derived material remains Apache-2.0. RDD configuration snapshots adapt/extend upstream configuration conventions and inheritance and are conservatively classified as upstream-derived adaptations. MIT covers separable owner-original contributions, not a substitute license for upstream content. Other project-specific classifications describe role/provenance, not proof of sole authorship.

See [SOURCE_MAP.md](docs/SOURCE_MAP.md) and its manifest. Local changes to src/data/transforms.py, src/data/coco/coco_dataset.py and inherited dataloader configuration remain outside this package; see environments/environment_notes.md.

## Data and media

Neither license grants rights over third-party datasets or imagery. The demo contains third-party images displayed on a screen, including web-page/watermark content.

**NOT FOR PUBLIC REDISTRIBUTION until image/media rights are reviewed.** The MP4 remains in the separate local evidence package, is absent from this checkout, and is excluded by .gitignore; do not upload or force-add it. No publication has occurred. Unrelated Paddle/RT-DETRv2/YOLO framework trees are excluded.

## Legacy repository notice

The earlier repository MIT notice, Copyright (c) 2026 Subir, is preserved here for legacy material; its full original license is retrievable at pre-r18-audited-migration:LICENSE. Current project-original MIT and upstream Apache scopes above do not erase historical notices.
