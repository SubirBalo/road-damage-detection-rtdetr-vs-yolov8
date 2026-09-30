from pathlib import Path
from PIL import Image
import json


DATASET_ROOT = Path(r"D:\Road_Damage_Project\RDD2022\dataset\RDD_SPLIT")

CLASSES = {
    0: "D00",
    1: "D10",
    2: "D20",
    3: "D40",
    4: "Other",
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def convert_split(split):
    images_dir = DATASET_ROOT / split / "images"
    labels_dir = DATASET_ROOT / split / "labels"
    annotations_dir = DATASET_ROOT / "annotations"

    annotations_dir.mkdir(parents=True, exist_ok=True)

    coco = {
        "images": [],
        "annotations": [],
        "categories": [
            {"id": class_id, "name": class_name}
            for class_id, class_name in CLASSES.items()
        ],
    }

    image_files = sorted(
        [
            p
            for p in images_dir.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        ]
    )

    annotation_id = 1

    for image_id, image_path in enumerate(image_files, start=1):
        with Image.open(image_path) as img:
            width, height = img.size

        coco["images"].append(
            {
                "id": image_id,
                "file_name": image_path.name,
                "width": width,
                "height": height,
            }
        )

        label_path = labels_dir / f"{image_path.stem}.txt"

        # Images with no damage annotations are still valid COCO images.
        if not label_path.exists():
            continue

        with open(label_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        for line_number, line in enumerate(lines, start=1):
            line = line.strip()

            if not line:
                continue

            parts = line.split()

            if len(parts) != 5:
                raise ValueError(
                    f"Invalid YOLO annotation in {label_path}, "
                    f"line {line_number}: {line}"
                )

            class_id = int(parts[0])

            if class_id not in CLASSES:
                raise ValueError(
                    f"Unknown class ID {class_id} in {label_path}"
                )

            x_center = float(parts[1])
            y_center = float(parts[2])
            box_width = float(parts[3])
            box_height = float(parts[4])

            # YOLO normalized coordinates -> pixel coordinates
            x_center *= width
            y_center *= height
            box_width *= width
            box_height *= height

            x_min = x_center - box_width / 2
            y_min = y_center - box_height / 2
            x_max = x_center + box_width / 2
            y_max = y_center + box_height / 2

            # Clip boxes to image boundaries
            x_min = max(0.0, min(x_min, width))
            y_min = max(0.0, min(y_min, height))
            x_max = max(0.0, min(x_max, width))
            y_max = max(0.0, min(y_max, height))

            box_width = x_max - x_min
            box_height = y_max - y_min

            # Ignore invalid zero-area boxes
            if box_width <= 0 or box_height <= 0:
                print(
                    f"WARNING: skipped invalid bbox in "
                    f"{label_path}, line {line_number}"
                )
                continue

            coco["annotations"].append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": class_id,
                    "bbox": [
                        round(x_min, 4),
                        round(y_min, 4),
                        round(box_width, 4),
                        round(box_height, 4),
                    ],
                    "area": round(box_width * box_height, 4),
                    "iscrowd": 0,
                }
            )

            annotation_id += 1

    output_file = annotations_dir / f"instances_{split}.json"

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(coco, f)

    print(f"\n{split.upper()}")
    print(f"Images      : {len(coco['images'])}")
    print(f"Annotations : {len(coco['annotations'])}")
    print(f"Saved       : {output_file}")


if __name__ == "__main__":
    for split_name in ["train", "val", "test"]:
        convert_split(split_name)

    print("\nConversion completed successfully.")