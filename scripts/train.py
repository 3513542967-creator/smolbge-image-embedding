from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file
from sklearn.linear_model import Ridge
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from smolbge_image_embedding import VisionProjector
from smolbge_image_embedding.metrics import retrieval_metrics
from smolbge_image_embedding.modeling import default_device


def load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as values:
        return {key: values[key] for key in values.files}


def centers(text: np.ndarray) -> np.ndarray:
    output = text.mean(axis=1)
    return output / np.maximum(np.linalg.norm(output, axis=-1, keepdims=True), 1e-12)


def validation_score(image: np.ndarray, text: np.ndarray) -> float:
    image = image / np.maximum(np.linalg.norm(image, axis=-1, keepdims=True), 1e-12)
    text = text / np.maximum(np.linalg.norm(text, axis=-1, keepdims=True), 1e-12)
    similarity = image @ text.T
    labels = np.arange(len(image))
    return float(0.5 * ((similarity.argmax(1) == labels).mean() + (similarity.argmax(0) == labels).mean()))


def project(model: VisionProjector, features: np.ndarray, device: torch.device) -> np.ndarray:
    outputs = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(features), 512):
            outputs.append(model(torch.from_numpy(features[start : start + 512]).to(device)).cpu().numpy())
    return np.concatenate(outputs)


def loss_function(image: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    logits = image @ target.T / 0.07
    labels = torch.arange(len(image), device=image.device)
    contrastive = 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))
    distillation = (1.0 - (image * target).sum(dim=-1)).mean()
    raw = image @ target.T
    negative = raw.masked_fill(torch.eye(len(image), dtype=torch.bool, device=image.device), -torch.inf).max(1).values
    margin = F.relu(0.1 - raw.diag() + negative).mean()
    return contrastive + 0.5 * distillation + 0.25 * margin


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the released contrastive projection adapter.")
    parser.add_argument("--cache", default=Path("artifacts/cache"), type=Path)
    parser.add_argument("--output", default=Path("runs/contrastive"), type=Path)
    args = parser.parse_args()
    torch.manual_seed(42)
    np.random.seed(42)
    device = default_device()
    train = load(args.cache / "train.npz")
    validation = load(args.cache / "validation.npz")
    test = load(args.cache / "test.npz")
    train_target = centers(train["text"]).astype(np.float32)
    validation_target = centers(validation["text"]).astype(np.float32)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(train["vision"]), torch.from_numpy(train_target)),
        batch_size=256,
        shuffle=True,
        generator=torch.Generator().manual_seed(42),
    )
    model = VisionProjector().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    best_score = -1.0
    best_state = None
    stale = 0
    history = []
    for epoch in range(1, 101):
        model.train()
        losses = []
        for vision, target in loader:
            vision = vision.to(device)
            target = F.normalize(target.to(device), dim=-1)
            optimizer.zero_grad(set_to_none=True)
            output = model(vision)
            loss = loss_function(output, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        score = validation_score(project(model, validation["vision"], device), validation_target)
        history.append({"epoch": epoch, "loss": float(np.mean(losses)), "validation_mean_r1": score})
        print(history[-1])
        if score > best_score + 1e-8:
            best_score = score
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        if stale >= 15:
            break
    assert best_state is not None
    model.load_state_dict(best_state)
    args.output.mkdir(parents=True, exist_ok=True)
    save_file({key: value.cpu().contiguous() for key, value in model.state_dict().items()}, args.output / "projector.safetensors")
    result = {
        "validation_mean_r1": best_score,
        "test": retrieval_metrics(project(model, test["vision"], device), test["text"]),
        "history": history,
    }
    (args.output / "training.json").write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

