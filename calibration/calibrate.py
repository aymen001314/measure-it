import argparse
import json
from pathlib import Path

import cv2
import numpy as np

EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def main():
    parser = argparse.ArgumentParser(description="Calibrate a camera from checkerboard photos.")
    parser.add_argument("--images", type=Path, default=Path("calibration/images"))
    parser.add_argument("--cols", type=int, default=9, help="Inner corners across the board")
    parser.add_argument("--rows", type=int, default=7, help="Inner corners down the board")
    parser.add_argument("--square-mm", type=float, default=18.8)
    args = parser.parse_args()

    if args.cols < 2 or args.rows < 2 or args.square_mm <= 0:
        parser.error("cols/rows must be >= 2 and square-mm must be positive")
    paths = sorted(p for p in args.images.iterdir() if p.is_file() and p.suffix.lower() in EXTENSIONS) if args.images.is_dir() else []
    if not paths:
        raise SystemExit(f"No supported images found in {args.images}")

    pattern = (args.cols, args.rows)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 40, 1e-5)
    template = np.zeros((args.rows * args.cols, 3), np.float32)
    template[:, :2] = np.mgrid[0:args.cols, 0:args.rows].T.reshape(-1, 2)
    template *= args.square_mm
    objpoints, imgpoints, used = [], [], []
    image_size = None
    for path in paths:
        image = cv2.imread(str(path))
        if image is None:
            print(f"Skip unreadable: {path.name}")
            continue
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image_size = (gray.shape[1], gray.shape[0])
        found, corners = cv2.findChessboardCorners(gray, pattern)
        if not found:
            print(f"Skip (board not detected): {path.name}")
            continue
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        objpoints.append(template.copy())
        imgpoints.append(corners)
        used.append(path.name)
    if len(used) < 5:
        raise SystemExit(f"Only {len(used)} usable photos; need at least 5. Check inner-corner rows/cols and image focus.")

    rms, matrix, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, image_size, None, None)
    per_image = []
    for obj, observed, rv, tv in zip(objpoints, imgpoints, rvecs, tvecs):
        projected, _ = cv2.projectPoints(obj, rv, tv, matrix, dist)
        per_image.append(float(cv2.norm(observed, projected, cv2.NORM_L2) / len(projected)))
    report = {"board_inner_corners": [args.cols, args.rows], "square_mm": args.square_mm,
              "images_used": used, "image_count": len(used), "rms_px": float(rms),
              "mean_reprojection_error_px": float(np.mean(per_image)),
              "camera_matrix": matrix.tolist(), "dist_coeffs": dist.tolist()}
    out = args.images.parent
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "camera_params.npz", camera_matrix=matrix, dist_coeffs=dist,
                        image_width=image_size[0], image_height=image_size[1])
    (out / "camera_params_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Calibration succeeded: {len(used)} images; RMS={rms:.4f}px; mean reprojection error={np.mean(per_image):.4f}px")
    print(f"Saved {out / 'camera_params.npz'} and {out / 'camera_params_report.json'}")


if __name__ == "__main__":
    main()
