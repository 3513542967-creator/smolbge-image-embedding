from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file

from smolbge_image_embedding import VisionProjector
from smolbge_image_embedding.metrics import retrieval_metrics
from smolbge_image_embedding.modeling import default_device


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", default=Path("artifacts/cache"), type=Path)
    parser.add_argument("--weights", default=Path("weights/projector.safetensors"), type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    with np.load(args.cache / "test.npz", allow_pickle=False) as values:
        vision = values["vision"].astype(np.float32)
        text = values["text"].astype(np.float32)
    device = default_device()
    model = VisionProjector().to(device).eval()
    model.load_state_dict(load_file(str(args.weights)))
    outputs = []
    with torch.inference_mode():
        for start in range(0, len(vision), 512):
            outputs.append(model(torch.from_numpy(vision[start : start + 512]).to(device)).cpu().numpy())
    result = retrieval_metrics(np.concatenate(outputs), text)
    print(json.dumps(result, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

