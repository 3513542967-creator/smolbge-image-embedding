from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Export image IDs from prepared JSONL files.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", default=Path("data/splits/coco5k_seed42.json"), type=Path)
    args = parser.parse_args()
    splits = {}
    for split in ("train", "validation", "test"):
        rows = [json.loads(line) for line in (args.input / f"{split}.jsonl").read_text().splitlines() if line]
        splits[split] = [int(row["image_id"]) for row in rows]
    payload = {
        "dataset": "COCO val2017 captions",
        "seed": 42,
        "captions_per_image": 5,
        "splits": splits,
        "leakage_check": "image IDs are pairwise disjoint",
    }
    values = [set(ids) for ids in splits.values()]
    assert not (values[0] & values[1] or values[0] & values[2] or values[1] & values[2])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print({key: len(value) for key, value in splits.items()})


if __name__ == "__main__":
    main()
