from __future__ import annotations

import numpy as np


def retrieval_metrics(image_embeddings: np.ndarray, caption_embeddings: np.ndarray) -> dict:
    image = image_embeddings / np.maximum(np.linalg.norm(image_embeddings, axis=-1, keepdims=True), 1e-12)
    captions = caption_embeddings.reshape(-1, caption_embeddings.shape[-1])
    captions = captions / np.maximum(np.linalg.norm(captions, axis=-1, keepdims=True), 1e-12)
    captions_per_image = caption_embeddings.shape[1]
    similarities = image @ captions.T
    image_count = image.shape[0]

    image_to_text_ranks = []
    for image_index in range(image_count):
        order = np.argsort(-similarities[image_index])
        relevant = np.arange(image_index * captions_per_image, (image_index + 1) * captions_per_image)
        image_to_text_ranks.append(min(int(np.where(order == target)[0][0]) for target in relevant) + 1)

    text_to_image_ranks = []
    targets = np.repeat(np.arange(image_count), captions_per_image)
    for caption_index, target_image in enumerate(targets):
        order = np.argsort(-similarities[:, caption_index])
        text_to_image_ranks.append(int(np.where(order == target_image)[0][0]) + 1)

    def summarize(ranks: list[int]) -> dict:
        values = np.asarray(ranks)
        return {
            "recall_at_1": float(np.mean(values <= 1)),
            "recall_at_5": float(np.mean(values <= 5)),
            "recall_at_10": float(np.mean(values <= 10)),
            "median_rank": float(np.median(values)),
            "mean_rank": float(np.mean(values)),
        }

    return {
        "image_to_text": summarize(image_to_text_ranks),
        "text_to_image": summarize(text_to_image_ranks),
    }

