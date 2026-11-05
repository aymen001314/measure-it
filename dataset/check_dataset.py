import argparse
from pathlib import Path

import cv2

EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def main():
    parser = argparse.ArgumentParser(description="Check dataset image count and readability.")
    parser.add_argument("--images", type=Path, default=Path("dataset/images"))
    args = parser.parse_args()
    if not args.images.is_dir():
        raise SystemExit(f"Folder not found: {args.images}")
    paths = sorted(p for p in args.images.iterdir() if p.is_file() and p.suffix.lower() in EXTENSIONS)
    bad, sizes = [], set()
    for p in paths:
        im = cv2.imread(str(p))
        if im is None:
            bad.append(p.name)
        else:
            sizes.add((im.shape[1], im.shape[0]))
    print(f"Image files: {len(paths)}; readable: {len(paths)-len(bad)}; unreadable: {len(bad)}")
    print(f"Resolutions (width x height): {sorted(sizes)}")
    if bad:
        print("Unreadable files:", ", ".join(bad))
    if len(paths) < 70:
        print(f"Need {70-len(paths)} more images to reach 70.")
    else:
        print("70+ image count reached. Inspect photos manually for blur, complete object visibility, and a visible scale reference.")


if __name__ == "__main__":
    main()
