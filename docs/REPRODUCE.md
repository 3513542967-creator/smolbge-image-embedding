# Reproduce

Use Python 3.12. Install from the repository root with `python -m pip install -e ".[dev]"`.

Download COCO val2017 images and caption annotations from [COCO](https://cocodataset.org/#download).
The published image IDs define a **custom split of val2017**, not the standard COCO retrieval benchmark.

```bash
python scripts/prepare_coco.py --captions /path/to/captions_val2017.json --images /path/to/val2017
python scripts/cache_features.py
python scripts/train.py
python scripts/evaluate.py --weights runs/contrastive/projector.safetensors
```

To evaluate the published adapter, use `--weights weights/projector.safetensors`.
A Git clone with LFS pointers requires `git lfs pull` first.

Training freezes both backbones and caches their features. Each image's five normalized caption embeddings are averaged and normalized into a training target. AdamW trains only the projector, with learning rate 3e-4, batch size 256, weight decay 0.01, up to 100 epochs and patience 15. Loss is symmetric InfoNCE (temperature 0.07) + 0.5 × cosine distillation + 0.25 × hard-negative margin (0.1). Selection uses validation bidirectional R@1 against caption centroids.

The standalone trainer reproduces the selected contrastive method. Historical five-method results are in [evaluation.json](../results/evaluation.json); this repository does not include a runner for every historical ablation. Hardware and library changes can affect retraining results.

Evaluation uses raw English captions without a query prefix, truncates text at 128 tokens, and ranks by cosine similarity:

- Image → text: 500 queries against 2,500 captions; any of the five paired captions in top K is a hit.
- Text → image: 2,500 queries against 500 images; the paired image in top K is a hit.

The released adapter was re-evaluated after packaging: [verified metrics](../results/reproduced_metrics.json).
The 68.8% / 61.4% numbers from an exploratory four-caption comparison use a different candidate/query set; the published five-caption protocol is **68.00% / 60.76%**.

Only the adapter-training split is held out. Upstream pretraining exposure cannot be excluded. English, in-domain retrieval is evaluated; image-to-image quality, production latency and end-to-end RAG answer quality have not been benchmarked.
