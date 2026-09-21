"""Search a local image folder with an English text query."""
import argparse
import json
from pathlib import Path

from .modeling import SmolBGEEmbedder


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", required=True, type=Path, help="Image file or folder")
    parser.add_argument("--query", required=True, help="English search query")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", choices=["cpu", "mps", "cuda"])
    parser.add_argument("--model", default="yifanouyang/smolbge-image-embedding")
    args = parser.parse_args()
    if args.top_k < 1 or args.batch_size < 1:
        parser.error("--top-k and --batch-size must be positive")
    extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    if args.images.is_file():
        paths = [args.images]
    elif args.images.is_dir():
        paths = sorted(p for p in args.images.rglob("*") if p.suffix.lower() in extensions and p.is_file())
    else:
        parser.error(f"Image path does not exist: {args.images}")
    if not paths:
        parser.error("No supported images found")
    model = SmolBGEEmbedder.from_pretrained(args.model, device=args.device)
    images = model.encode_images(paths, batch_size=args.batch_size)
    query = model.encode_text(args.query)
    scores = images @ query
    order = scores.argsort()[::-1][:args.top_k]
    print(json.dumps([
        {"rank": rank, "image": str(paths[index]), "cosine": float(scores[index])}
        for rank, index in enumerate(order, 1)
    ], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
