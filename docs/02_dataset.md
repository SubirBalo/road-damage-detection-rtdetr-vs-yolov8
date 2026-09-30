# Dataset evidence

| ID | Class | Meaning |
|---|---|---|
| 0 | D00 | Longitudinal crack |
| 1 | D10 | Transverse crack |
| 2 | D20 | Alligator/fatigue crack |
| 3 | D40 | Pothole |
| 4 | Other | Other road-damage patterns |

| Split | Images / label files | Valid COCO boxes | Images without boxes |
|---|---:|---:|---:|
| Train | 26,869 | 46,295 | 8,097 |
| Validation | 5,758 | 9,741 | 1,837 |
| Test | 5,758 | 9,675 | 1,790 |
| Total | 38,385 | 65,711 | 11,724 |

| Class | Train boxes | Validation boxes | Test boxes |
|---|---:|---:|---:|
| D00 | 18,201 | 3,890 | 3,925 |
| D10 | 8,386 | 1,769 | 1,675 |
| D20 | 7,526 | 1,553 | 1,537 |
| D40 | 7,554 | 1,564 | 1,587 |
| Other | 4,628 | 965 | 951 |

The prior read-only audit reconstructed conversion in memory and matched COCO boxes to current YOLO labels within rounding precision. The converter clips boxes and skips zero-area boxes. Japan_001265.txt line 4 in the training labels has zero width, explaining 46,296 annotation lines versus 46,295 valid boxes. All COCO-referenced images existed; split filenames did not overlap. Image-content/geographic leakage was not tested.

The original ZIP already contained these splits. No split-generation script, seed or original unsplit/XML source was recovered. Do not claim authorship of the original split. No dataset is redistributed here; acquisition and redistribution provenance remain to be documented.
