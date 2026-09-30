# Packaging verification

All 28 copies were checked against SHA256 fingerprints captured before copying. Source hashes remained unchanged, and all copied bytes match. Python files were parsed as text with ast.parse, not imported or executed. JSON files parsed successfully; Markdown file links resolve. No Git initialization, publication, inference, training, export, engine rebuild or package installation occurred.

All writes were confined to this staging folder. No model/checkpoint/dataset/cache/archive binaries were copied. The selected MP4 is intentionally included. Checks establish packaging integrity and syntax, not environment compatibility, runtime correctness, hardware authentication or media rights.

## Newly authored files

- `.gitignore`
- `README.md`
- `THIRD_PARTY_NOTICES.md`
- `assets/demo/README.md`
- `assets/examples/README.md`
- `configs/README.md`
- `deployment/jetson/README.md`
- `docs/01_problem_and_scope.md`
- `docs/02_dataset.md`
- `docs/03_model_and_training.md`
- `docs/04_evaluation.md`
- `docs/05_deployment.md`
- `docs/06_verification_matrix.md`
- `docs/07_limitations.md`
- `docs/08_future_work.md`
- `docs/PACKAGING_CHECKS.md`
- `docs/SOURCE_MAP.md`
- `docs/source_manifest.json`
- `environments/environment_notes.md`
- `portfolio/linkedin.md`
- `portfolio/project_summary.md`
- `portfolio/resume_bullets.md`
- `results/README.md`
- `results/metrics/audited_coco_metrics.json`
- `tools/README.md`

## Pre-publication corrections

Initial checks above describe the original packaging step. Later corrections changed authored documentation and .gitignore only: test-split terminology, resume training count, local-only demo restriction and pending contribution licensing. Copied evidence, code, configs, LICENSE and MP4 remain unchanged. The README no longer links to the excluded MP4. Machine-specific paths remain in preserved configuration/evidence and provenance records; they were reported rather than silently rewritten.

## Requirements and licensing reconciliation

Historical qualitative requirements from the earlier comparison repository are acknowledged. Their chronology is unproven and no numeric latency/accuracy threshold is supplied. R09 remains Not Testable From Current Evidence.

The owner selected MIT for project-specific original contributions. Root LICENSE now contains MIT; its previous Apache-2.0 bytes are preserved unchanged at LICENSES/RT-DETR-Apache-2.0.txt. The manifest points to that location and derived-file classifications are clarified. This supersedes the earlier pending-licensing state; historical experiment evidence is unchanged.

All 28 manifest-listed artifacts were checked locally against stored SHA256 fingerprints. No original-project access, GitHub access, Git initialization, inference, build or publication occurred for this reconciliation. Demo exclusion remains in place. Root LICENSE is newly authored; the separate Apache file is preserved copied material.

## Migration integration (third migration commit)

The prior sections describe the separate staging package and its earlier corrections. This checkout includes the audited R18 package except restricted MP4; the video was never copied or staged. Its provenance remains recorded with excluded_restricted_media status. Included manifest artifacts retain their hashes. The Git repository and first two migration commits pre-exist this integration.

The legacy archive and .gitattributes remain unchanged. Root README now presents the audited R18 project and links to legacy history. Ignore rules were merged; obsolete .gitkeep placeholders were removed. Historical requirements links resolve to the preserved archive without claiming chronology. Legacy license attribution is retained in THIRD_PARTY_NOTICES.md. No datasets, models, checkpoints, ONNX graphs, engines, archives or caches were imported.

Checks cover Python syntax without execution, JSON parsing, Markdown file links where practical, staged-file scope, credential-pattern scanning, archived-file fingerprints and source staging fingerprints. Historical absolute paths and private user-directory paths within the unchanged legacy notebook/configurations remain; this check does not certify their public suitability. No inference, training, model build, network operation or push occurred during integration.

Whitespace review: git diff --cached --check reports pre-existing trailing whitespace in tools/infer.py and terminal blank lines in two dataset YAML files. These copied artifacts are preserved rather than silently reformatted. Manifest SHA256 values describe the source/working-copy bytes; the preserved .gitattributes text normalization can change newline bytes in Git blobs or later checkouts.
