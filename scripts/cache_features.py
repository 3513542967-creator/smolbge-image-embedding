from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F
from tqdm import tqdm
from transformers import AutoModel, AutoModelForMultimodalLM, AutoProcessor, AutoTokenizer

from smolbge_image_embedding.modeling import default_device, masked_patch_pool, patch_attention_mask


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def encode_texts(records: list[dict], device: torch.device) -> np.ndarray:
    tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-small-en-v1.5")
    model = AutoModel.from_pretrained("BAAI/bge-small-en-v1.5", dtype=torch.float32).to(device).eval()
    captions = [caption for record in records for caption in record["captions"]]
    outputs = []
    for start in tqdm(range(0, len(captions), 128), desc="BGE captions"):
        tokens = tokenizer(captions[start : start + 128], padding=True, truncation=True, return_tensors="pt")
        tokens = {key: value.to(device) for key, value in tokens.items()}
        with torch.inference_mode():
            cls = model(**tokens).last_hidden_state[:, 0]
            outputs.append(F.normalize(cls.float(), dim=-1).cpu())
    return torch.cat(outputs).numpy().reshape(len(records), 5, -1).astype(np.float32)


def encode_images(records: list[dict], device: torch.device) -> np.ndarray:
    model_id = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
    processor = AutoProcessor.from_pretrained(model_id).image_processor
    processor.do_image_splitting = False
    dtype = torch.bfloat16 if device.type in {"cuda", "mps"} else torch.float32
    container = AutoModelForMultimodalLM.from_pretrained(model_id, dtype=dtype).to(device).eval()
    vision = container.model.vision_model
    patch_size = int(container.config.vision_config.patch_size)
    outputs = []
    for start in tqdm(range(0, len(records), 16), desc="SmolVLM2 images"):
        images = [Image.open(row["image_file"]).convert("RGB") for row in records[start : start + 16]]
        try:
            inputs = processor(images=images, do_image_splitting=False, return_tensors="pt")
            pixels = inputs["pixel_values"].view(-1, *inputs["pixel_values"].shape[-3:]).to(device, dtype)
            mask = patch_attention_mask(inputs["pixel_attention_mask"].to(device), patch_size)
            with torch.inference_mode():
                hidden = vision(pixel_values=pixels, patch_attention_mask=mask, return_dict=True).last_hidden_state
                outputs.append(masked_patch_pool(hidden.float(), mask).cpu())
        finally:
            for image in images:
                image.close()
    return torch.cat(outputs).numpy().astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=Path("data/processed"), type=Path)
    parser.add_argument("--output", default=Path("artifacts/cache"), type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    device = default_device()
    for split in ("train", "validation", "test"):
        rows = read_jsonl(args.data / f"{split}.jsonl")
        text = encode_texts(rows, device)
        vision = encode_images(rows, device)
        np.savez(
            args.output / f"{split}.npz",
            image_ids=np.asarray([row["image_id"] for row in rows]),
            captions=np.asarray([row["captions"] for row in rows]),
            image_files=np.asarray([row["image_file"] for row in rows]),
            vision=vision,
            text=text,
        )
        print(split, vision.shape, text.shape)


if __name__ == "__main__":
    main()

