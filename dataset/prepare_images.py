import argparse
from pathlib import Path

import cv2
import numpy as np

EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def main():
    parser = argparse.ArgumentParser(description="Optionally undistort dataset images using saved camera calibration.")
    parser.add_argument("--raw", type=Path, default=Path("dataset/images"))
    parser.add_argument("--out", type=Path, default=Path("dataset/undistorted"))
    parser.add_argument("--params", type=Path, default=Path("calibration/camera_params.npz"))
    args = parser.parse_args()
    if not args.raw.is_dir():
        raise SystemExit(f"Input folder not found: {args.raw}")
    paths = sorted(p for p in args.raw.iterdir() if p.is_file() and p.suffix.lower() in EXTENSIONS)
    if not paths:
        raise SystemExit(f"No supported images found in {args.raw}")
    if not args.params.is_file():
        raise SystemExit(f"Calibration file not found: {args.params}; run calibration first or check the path")
    with np.load(args.params) as data:
        matrix, dist = data["camera_matrix"], data["dist_coeffs"]
    args.out.mkdir(parents=True, exist_ok=True)
    ok = 0
    for path in paths:
        image = cv2.imread(str(path))
        if image is None:
            print(f"Skip unreadable: {path.name}")
            continue
        h, w = image.shape[:2]
        new_matrix, _ = cv2.getOptimalNewCameraMatrix(matrix, dist, (w, h), 1, (w, h))
        corrected = cv2.undistort(image, matrix, dist, None, new_matrix)
        target = args.out / path.name
        if not cv2.imwrite(str(target), corrected):
            print(f"Failed to write: {target}")
            continue
        ok += 1
    print(f"Undistorted {ok}/{len(paths)} images into {args.out}")


if __name__ == "__main__":
    main()
