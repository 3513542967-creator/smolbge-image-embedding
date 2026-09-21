# Model Card

## Model summary

SmolBGE Image Embedding is a lightweight alignment adapter that maps pooled visual features from the frozen SmolVLM2-500M vision tower into the normalized 384-dimensional embedding space of frozen BGE-small-en-v1.5 text embeddings.

The released artifact is an adapter, not a standalone foundation model. Runtime use requires both named base models.

## Intended use

- first-stage retrieval for multimodal RAG;
- text-to-image, image-to-text, and image-to-image search;
- research on low-cost alignment between independently pretrained visual and text embedding spaces.

## Out-of-scope use

- biometric identification or unauthorized surveillance;
- high-stakes medical, employment, credit, legal, or safety decisions;
- claims of robust performance outside the tested COCO domain without additional evaluation.

## Training data

5,000 COCO `val2017` images with five captions per image. Split: 4,000 train, 500 validation, 500 test. Seed: 42. Image IDs are disjoint across splits.

No COCO image is redistributed in this repository.

## Training procedure

Both backbones are frozen. A 768 -> 768 -> 384 MLP is trained with AdamW, batch size 256, learning rate 3e-4, weight decay 0.01, and early stopping on validation bidirectional Recall@1. The selected loss combines symmetric in-batch contrastive learning, cosine distillation toward the mean of five caption embeddings, and a hard-negative margin term.

## Evaluation

The final test contains 500 images and 2,500 captions. Image-to-text retrieval searches all captions; text-to-image retrieval searches all images. Metrics are Recall@1/5/10 and rank statistics under cosine similarity.

Selected model results:

- image-to-text R@1/R@5/R@10: 68.00/90.60/95.00%;
- text-to-image R@1/R@5/R@10: 60.76/88.16/95.12%.

See `results/evaluation.json` for all recorded methods and uncertainty estimates.

## Known limitations

The test is in-domain and relatively small. Pretrained backbones may have prior COCO exposure. Global mean pooling can miss local details. English-only captions were used. Exact-image retrieval metrics may label semantically relevant images as negatives.

## Environmental and compute notes

The adapter was trained locally using cached frozen features on Apple MPS. Frozen feature extraction took approximately 252 seconds. The released adapter is approximately 3.4 MB and has 887,424 trainable parameters.

