# Publishing checklist

## Before publishing

1. Confirm that the GitHub and Hugging Face account names in the READMEs are correct.
2. Add the author name and contact information you want to publish.
3. Review the Apache-2.0 license and third-party notices.
4. Confirm that no COCO images, private paths, credentials, or Hugging Face tokens are present.
5. Run the local checks:

```bash
python -m pip install -e ".[dev]"
pytest
python scripts/evaluate.py --cache /path/to/cache --weights weights/projector.safetensors
```

## GitHub

Create an empty public repository named `smolbge-image-embedding`, then run:

```bash
git init -b main
git lfs install
git lfs track "*.safetensors"
git add .
git commit -m "Initial open-source release"
git remote add origin https://github.com/3513542967-creator/smolbge-image-embedding.git
git push -u origin main
```

Suggested first release tag:

```bash
git tag -a v0.1.0 -m "COCO-5K direct image embedding adapter"
git push origin v0.1.0
```

## Hugging Face model repository

Install and authenticate:

```bash
python -m pip install -U huggingface_hub
hf auth login
```

Create and upload the model repository:

```bash
hf repo create yifanouyang/smolbge-image-embedding --type model
hf upload yifanouyang/smolbge-image-embedding . \
  --repo-type model \
  --exclude ".git/*" \
  --exclude "data/processed/*" \
  --exclude "artifacts/cache/*"
```

The YAML header in `README.md` is already compatible with a Hugging Face model card. After upload, verify that the architecture SVG, metrics plot, adapter weight, model card metadata, and usage example render correctly.

## Recommended release description

> SmolBGE Image Embedding is a lightweight adapter that maps SmolVLM2 visual features directly into the BGE-small text embedding space without caption generation. The released 3.4 MB adapter reaches 68.00% image-to-text R@1 and 60.76% text-to-image R@1 on a fixed held-out COCO-5K split.
