from __future__ import annotations

import argparse
from pathlib import Path

from smolbge_image_embedding import SmolBGEEmbedder


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path)
    parser.add_argument("text")
    parser.add_argument("--model", default=Path("."), type=Path)
    args = parser.parse_args()
    model = SmolBGEEmbedder.from_pretrained(args.model)
    image = model.encode_image(args.image)
    text = model.encode_text(args.text, is_query=True)
    print({"cosine_similarity": float(image @ text), "embedding_dim": len(image)})


if __name__ == "__main__":
    main()
