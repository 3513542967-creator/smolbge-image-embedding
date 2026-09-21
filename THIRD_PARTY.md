# Third-party models and data

This repository contains only project code, evaluation records, deterministic split IDs, and the trained projection adapter.

| Dependency | Role | License/source |
|---|---|---|
| HuggingFaceTB/SmolVLM2-500M-Video-Instruct | frozen vision backbone | Apache-2.0; https://huggingface.co/HuggingFaceTB/SmolVLM2-500M-Video-Instruct |
| BAAI/bge-small-en-v1.5 | frozen text embedding backbone | MIT; https://huggingface.co/BAAI/bge-small-en-v1.5 |
| COCO val2017 and captions | training/evaluation data | https://cocodataset.org/; source images retain per-image licenses |
| COCO API | optional dataset tooling | Simplified BSD; https://github.com/cocodataset/cocoapi |

Users are responsible for reviewing and complying with the licenses and terms of the base models and source data.

