import argparse
from pathlib import Path

import cv2
import numpy as np


def click_points(event, x, y, flags, state):
    if event == cv2.EVENT_LBUTTONDOWN and len(state["points"]) < 2:
        state["points"].append((x, y))


def main():
    parser = argparse.ArgumentParser(description="Interactive top-view measurement using a ruler/reference visible in the SAME image.")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--ref-mm", type=float, required=True, help="Known distance in mm between the two clicked reference points")
    parser.add_argument("--out", type=Path, default=Path("outputs"))
    parser.add_argument("--params", type=Path, help="Optional camera_params.npz to undistort first")
    args = parser.parse_args()
    if args.ref_mm <= 0:
        raise SystemExit("--ref-mm must be greater than zero")
    image = cv2.imread(str(args.image))
    if image is None:
        raise SystemExit(f"Cannot read image: {args.image}")
    if args.params:
        if not args.params.is_file():
            raise SystemExit(f"Calibration params not found: {args.params}")
        with np.load(args.params) as data:
            mtx, dist = data["camera_matrix"], data["dist_coeffs"]
        h, w = image.shape[:2]
        new_mtx, _ = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
        image = cv2.undistort(image, mtx, dist, None, new_mtx)

    print("STEP 1: Click two points exactly one known distance apart on the ruler/reference, then press ENTER.")
    state = {"points": []}
    shown = image.copy()
    window = "Click reference endpoints (ENTER confirms, ESC cancels)"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    def draw(event, x, y, flags, param):
        click_points(event, x, y, flags, state)
        if event == cv2.EVENT_LBUTTONDOWN:
            cv2.circle(shown, (x, y), 7, (0, 0, 255), -1)
            if len(state["points"]) == 2:
                cv2.line(shown, state["points"][0], state["points"][1], (0, 255, 0), 2)
            cv2.imshow(window, shown)
    cv2.setMouseCallback(window, draw)
    cv2.imshow(window, shown)
    key = cv2.waitKey(0)
    cv2.destroyWindow(window)
    if key == 27 or len(state["points"]) != 2:
        raise SystemExit("Cancelled: select exactly two reference points.")
    pixels = float(np.linalg.norm(np.subtract(state["points"][0], state["points"][1])))
    if pixels < 2:
        raise SystemExit("Reference points are too close; select a longer known ruler distance.")
    px_per_mm = pixels / args.ref_mm

    print("STEP 2: Drag a tight rectangle around ONLY the eraser top face; press ENTER to accept, C to cancel.")
    roi = cv2.selectROI("Select eraser top face", image, showCrosshair=True, fromCenter=False)
    cv2.destroyAllWindows()
    x, y, w, h = map(int, roi)
    if w < 5 or h < 5:
        raise SystemExit("No valid object region selected.")
    # Within a manually selected object-only ROI, Otsu segmentation isolates the light eraser
    # against the dark background in the provided example; user can verify the saved overlay.
    crop = image[y:y+h, x:x+w]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if cv2.contourArea(c) >= max(25, .03 * w * h)]
    if not contours:
        raise SystemExit("Object not segmented from selected ROI. Use a clearer image/background and retry.")
    contour = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(contour)
    side_a, side_b = rect[1]
    if min(side_a, side_b) <= 0:
        raise SystemExit("Could not calculate object dimensions.")
    dims = sorted((side_a / px_per_mm, side_b / px_per_mm), reverse=True)
    overlay = image.copy()
    box = cv2.boxPoints(rect).astype(int)
    box[:, 0] += x
    box[:, 1] += y
    cv2.drawContours(overlay, [box], 0, (0, 255, 0), 3)
    args.out.mkdir(parents=True, exist_ok=True)
    overlay_path = args.out / f"{args.image.stem}_measurement.jpg"
    cv2.imwrite(str(overlay_path), overlay)
    result = {"image": str(args.image), "reference_mm": args.ref_mm, "reference_pixels": pixels,
              "pixels_per_mm": px_per_mm, "top_view_long_side_mm": dims[0], "top_view_short_side_mm": dims[1],
              "note": "Top-view dimensions only; this does not measure thickness/height."}
    import json
    result_path = args.out / f"{args.image.stem}_measurement.json"
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Scale: {px_per_mm:.3f} px/mm")
    print(f"Estimated top-view long side: {dims[0]:.2f} mm")
    print(f"Estimated top-view short side: {dims[1]:.2f} mm")
    print("Height/thickness is NOT measured from a top-view photo; use a side-view photo and repeat.")
    print(f"Saved overlay: {overlay_path}\nSaved result: {result_path}")


if __name__ == "__main__":
    main()
