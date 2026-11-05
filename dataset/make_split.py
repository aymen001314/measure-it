import argparse
import random
import shutil
from pathlib import Path

EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def main():
    parser = argparse.ArgumentParser(description="Create a deterministic 70/20/10 image-only split.")
    parser.add_argument("--images", type=Path, default=Path("dataset/images"))
    parser.add_argument("--out", type=Path, default=Path("dataset/split"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not args.images.is_dir():
        raise SystemExit(f"Image folder not found: {args.images}")
    images = sorted(p for p in args.images.iterdir() if p.is_file() and p.suffix.lower() in EXTENSIONS)
    if len(images) < 10:
        raise SystemExit(f"Found {len(images)} images; add at least 10 before splitting")
    random.Random(args.seed).shuffle(images)
    n = len(images)
    n_train, n_val = int(n * .7), int(n * .2)
    groups = {"train": images[:n_train], "val": images[n_train:n_train+n_val], "test": images[n_train+n_val:]}
    for name, paths in groups.items():
        dest = args.out / name
        dest.mkdir(parents=True, exist_ok=True)
        for p in paths:
            shutil.copy2(p, dest / p.name)
    print("Split complete (image-only; masks/labels are NOT generated):")
    for name, paths in groups.items():
        print(f"  {name}: {len(paths)}")
    print(f"  total: {n}; seed: {args.seed}")
    print(f"Output: {args.out.resolve()}")


if __name__ == "__main__":
    main()
