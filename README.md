---
license: apache-2.0
pipeline_tag: feature-extraction
library_name: transformers
tags:
  - multimodal
  - image-embedding
  - text-embedding
  - retrieval
  - rag
  - bge
base_model:
  - HuggingFaceTB/SmolVLM2-500M-Video-Instruct
  - BAAI/bge-small-en-v1.5
datasets:
  - detection-datasets/coco
---

# SmolBGE Image Embedding

[Chinese README](README_zh-CN.md) | [Results](results/evaluation.json) | [Model card](MODEL_CARD.md)

SmolBGE Image Embedding maps an image **directly** into the 384-dimensional embedding space of `BAAI/bge-small-en-v1.5`. It does not generate a caption as an intermediate step. The intended use is image/text retrieval and multimodal RAG with a small trainable adapter.

> Repository: `https://github.com/3513542967-creator/smolbge-image-embedding`  
> Hugging Face model: `https://huggingface.co/yifanouyang/smolbge-image-embedding`

## What problem does it solve?

Standard text RAG cannot index an image unless the image is first converted to text. Captioning can omit colors, counts, secondary objects, layout, and other visual details. This project instead aligns a frozen visual encoder directly with a frozen text embedding model:

- image -> 384-D normalized embedding;
- text -> 384-D normalized embedding;
- cosine similarity can be used for text-to-image, image-to-text, and image-to-image retrieval;
- only an 887,424-parameter projection head is trained.

## Architecture

![Architecture](assets/architecture.svg)

The image path is not autoregressive: no caption is generated during indexing or retrieval.

## Model

| Component | Setting |
|---|---|
| Vision backbone | `HuggingFaceTB/SmolVLM2-500M-Video-Instruct` vision tower |
| Vision feature | masked mean over valid visual patch tokens, 768-D |
| Text backbone | `BAAI/bge-small-en-v1.5` |
| Text feature | normalized CLS embedding, 384-D |
| Trainable adapter | LayerNorm -> Linear(768,768) -> GELU -> Dropout -> Linear(768,384) |
| Output | L2-normalized 384-D vector |
| Trainable parameters | 887,424 |
| Selected objective | bidirectional in-batch contrastive loss + cosine distillation + hard-negative margin |

The two backbones remain frozen. The released `weights/projector.safetensors` contains only the trained adapter.

## Dataset and split

The experiment uses 5,000 images from COCO `val2017`, each with five human captions:

| Split | Images | Captions | Usage |
|---|---:|---:|---|
| Train | 4,000 | 20,000 | adapter training |
| Validation | 500 | 2,500 | checkpoint and method selection |
| Test | 500 | 2,500 | final evaluation only |

The split is deterministic with seed 42 and disjoint at image-ID level. All five captions belonging to an image stay in the same split. Exact IDs are published in [`data/splits/coco5k_seed42.json`](data/splits/coco5k_seed42.json).

COCO images are **not** redistributed. Download COCO `val2017` images and `captions_val2017.json` from the official dataset site and follow each image's original license.

## Evaluation protocol

### Image-to-text

Each of the 500 test images searches all 2,500 test captions. A query is counted as correct at K when at least one of its five ground-truth captions occurs in the top K.

### Text-to-image

Each of the 2,500 test captions searches all 500 test images. A query is counted as correct when its source image occurs in the top K.

All rankings use cosine similarity. The selected method is chosen only by validation bidirectional Recall@1; test performance is not used for model selection.

## Results

Held-out COCO test split: 500 images and 2,500 captions.

| Method | Image->Text R@1 | Image->Text R@5 | Text->Image R@1 | Text->Image R@5 |
|---|---:|---:|---:|---:|
| Random projection | 0.00% | 0.60% | 0.04% | 0.60% |
| Ridge linear | 45.40% | 78.20% | 59.84% | 86.64% |
| Cosine MLP | 44.60% | 76.80% | 59.96% | 86.08% |
| **Contrastive MLP (released)** | **68.00%** | **90.60%** | **60.76%** | **88.16%** |
| Multi-positive MLP | 69.00% | 90.00% | 61.08% | 87.44% |

![Retrieval comparison](results/retrieval_comparison.png)

The multi-positive model has slightly higher test R@1, but it was not selected because its validation selection score was lower. We do not select a checkpoint after looking at test results.

Additional checks:

- image-to-text R@10: **95.00%**;
- text-to-image R@10: **95.12%**;
- freshly decoded JPG embeddings vs cached embeddings: mean cosine **0.999988**;
- output vectors have unit L2 norm.

The paired bootstrap improvement of the selected model over ridge regression for image-to-text R@1 is +22.60 percentage points, with a 95% interval of [+18.00, +27.20]. The text-to-image improvement over ridge is small and its interval crosses zero.

## Installation

```bash
git clone https://github.com/3513542967-creator/smolbge-image-embedding.git
cd smolbge-image-embedding
python -m pip install -e .
```

The first run downloads the two base models from Hugging Face. The released adapter does not include those base weights.

## Inference

```python
from smolbge_image_embedding import SmolBGEEmbedder

model = SmolBGEEmbedder.from_pretrained(".")

image_vector = model.encode_image("example.jpg")       # (384,)
text_vector = model.encode_text("a dog playing outside")  # (384,)

score = float(image_vector @ text_vector)
print(score)
```

Batch inference:

```python
images = model.encode_images(["a.jpg", "b.jpg"])
texts = model.encode_texts(["a red car", "a dog in a park"])
similarities = texts @ images.T
```

## Reproduce the experiment

1. Download COCO `val2017.zip` and `annotations_trainval2017.zip`.
2. Export the exact split manifests:

```bash
python scripts/prepare_coco.py \
  --captions /path/to/annotations/captions_val2017.json \
  --images /path/to/val2017 \
  --output data/processed
```

3. Cache frozen backbone features:

```bash
python scripts/cache_features.py --data data/processed --output artifacts/cache
```

4. Train the released contrastive adapter:

```bash
python scripts/train.py --cache artifacts/cache --output runs/contrastive
```

5. Evaluate:

```bash
python scripts/evaluate.py \
  --cache artifacts/cache \
  --weights runs/contrastive/projector.safetensors
```

## Recommended industrial use

This model is suitable as a fast first-stage retriever for:

- text-to-image search;
- image-to-image search;
- image-to-text retrieval;
- multimodal RAG candidate generation.

For production systems, combine it with OCR, metadata filters, region embeddings, and a VLM reranker when queries require small text, exact counting, spatial relations, or fine-grained local objects.

## Limitations

- Evaluation is currently COCO-domain only; no claim is made about medical, remote-sensing, e-commerce, or document-image generalization.
- The pretrained backbones may have prior exposure to COCO-like data.
- Mean pooling can discard small-object and spatial-layout information.
- Exact-image Recall can penalize semantically valid results from a different but visually similar image.
- English captions were used for training and evaluation.
- The base SmolVLM2 model card excludes high-stakes decision making; this adapter does not change that restriction.

## Licensing

Project code and adapter weights are released under Apache-2.0. The required base models retain their own licenses: SmolVLM2 is Apache-2.0 and BGE-small-en-v1.5 is MIT. COCO images retain the licenses recorded for each source image and are not included here. See [`THIRD_PARTY.md`](THIRD_PARTY.md).

## Citation

Until an archival paper is available, cite the repository URL and version or commit hash. Replace the URL placeholders before publishing.

See [`PUBLISHING.md`](PUBLISHING.md) for the GitHub and Hugging Face release checklist.
