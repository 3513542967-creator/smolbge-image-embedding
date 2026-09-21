from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create the deterministic COCO-5K retrieval split.")
    parser.add_argument("--captions", required=True, type=Path)
    parser.add_argument("--images", required=True, type=Path)
    parser.add_argument("--output", default=Path("data/processed"), type=Path)
    parser.add_argument("--split-ids", default=Path("data/splits/coco5k_seed42.json"), type=Path)
    args = parser.parse_args()

    payload = json.loads(args.captions.read_text(encoding="utf-8"))
    metadata = {int(row["id"]): row for row in payload["images"]}
    captions = defaultdict(list)
    for annotation in sorted(payload["annotations"], key=lambda row: int(row["id"])):
        captions[int(annotation["image_id"])].append(annotation["caption"].strip())

    eligible = [
        image_id
        for image_id, values in captions.items()
        if len(values) >= 5 and (args.images / metadata[image_id]["file_name"]).is_file()
    ]
    if args.split_ids.exists():
        split_ids = json.loads(args.split_ids.read_text(encoding="utf-8"))["splits"]
        missing = [image_id for values in split_ids.values() for image_id in values if image_id not in eligible]
        if missing:
            raise FileNotFoundError(f"The exact published split is missing {len(missing)} images; first ID: {missing[0]}")
    else:
        random.Random(42).shuffle(eligible)
        split_ids = {
            "train": eligible[:4000],
            "validation": eligible[4000:4500],
            "test": eligible[4500:5000],
        }

    sets = [set(values) for values in split_ids.values()]
    assert not (sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
    for split, ids in split_ids.items():
        rows = []
        for image_id in ids:
            row = metadata[image_id]
            rows.append(
                {
                    "image_id": image_id,
                    "image_file": str((args.images / row["file_name"]).resolve()),
                    "captions": captions[image_id][:5],
                }
            )
        write_jsonl(args.output / f"{split}.jsonl", rows)
    print(json.dumps({key: len(value) for key, value in split_ids.items()}, indent=2))


if __name__ == "__main__":
    main()

