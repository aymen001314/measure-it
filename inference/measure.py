import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def displayed_image(image, max_width=1100, max_height=760):
    h, w = image.shape[:2]
    scale = min(max_width / w, max_height / h, 1.0)
    shown = cv2.resize(image, (round(w * scale), round(h * scale))) if scale < 1 else image.copy()
    return shown, scale


def collect_points(image, count, title, instruction):
    shown, scale = displayed_image(image)
    state = {"points": []}
    window = title
    cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)

    def on_mouse(event, x, y, flags, userdata):
        if event != cv2.EVENT_LBUTTONDOWN or len(state["points"]) >= count:
            return
        ox = min(image.shape[1] - 1, max(0, round(x / scale)))
        oy = min(image.shape[0] - 1, max(0, round(y / scale)))
        state["points"].append((ox, oy))
        cv2.circle(shown, (x, y), 5, (0, 0, 255), -1)
        if len(state["points"]) > 1:
            cv2.line(shown, (round(state["points"][-2][0] * scale), round(state["points"][-2][1] * scale)),
                     (x, y), (0, 255, 0), 2)
        cv2.imshow(window, shown)

    cv2.setMouseCallback(window, on_mouse)
    cv2.imshow(window, shown)
    print(instruction)
    key = cv2.waitKey(0)
    # The user may close the OpenCV window with its X button. Some OpenCV builds
    # then raise when destroyWindow is called for a window that is already gone.
    try:
        cv2.destroyWindow(window)
    except cv2.error:
        pass
    if key == 27:
        raise SystemExit("Cancelled by user.")
    if len(state["points"]) != count:
        raise SystemExit(f"Expected {count} clicks but got {len(state['points'])}. Rerun and click carefully.")
    return np.asarray(state["points"], dtype=np.float64)


def main():
    parser = argparse.ArgumentParser(description="Interactive manual measurement: click a known ruler span and the four eraser corners.")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--ref-mm", type=float, required=True, help="Known distance between the two ruler points you will click")
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
            matrix, dist = data["camera_matrix"], data["dist_coeffs"]
        h, w = image.shape[:2]
        new_matrix, _ = cv2.getOptimalNewCameraMatrix(matrix, dist, (w, h), 1, (w, h))
        image = cv2.undistort(image, matrix, dist, None, new_matrix)

    ref = collect_points(image, 2, "Reference scale", "Click two clear ruler marks exactly the stated distance apart, then press a key. ESC cancels.")
    ref_pixels = float(np.linalg.norm(ref[1] - ref[0]))
    if ref_pixels < 5:
        raise SystemExit("Reference clicks are too close. Choose a longer ruler span for better accuracy.")
    px_per_mm = ref_pixels / args.ref_mm

    corners = collect_points(image, 4, "Eraser corners", "Click the FOUR outer corners of the visible eraser face in order around its edge, then press a key. ESC cancels.")
    # Average opposite sides to reduce a little click noise. Perspective distortion is not corrected.
    edges = np.linalg.norm(np.roll(corners, -1, axis=0) - corners, axis=1)
    side_a = float((edges[0] + edges[2]) / 2 / px_per_mm)
    side_b = float((edges[1] + edges[3]) / 2 / px_per_mm)
    long_side, short_side = sorted((side_a, side_b), reverse=True)

    overlay = image.copy()
    poly = corners.astype(np.int32).reshape((-1, 1, 2))
    cv2.polylines(overlay, [poly], True, (0, 255, 0), 4)
    label = f"Long: {long_side:.1f} mm  Short: {short_side:.1f} mm"
    cv2.putText(overlay, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2, cv2.LINE_AA)
    args.out.mkdir(parents=True, exist_ok=True)
    overlay_path = args.out / f"{args.image.stem}_measurement.jpg"
    result_path = args.out / f"{args.image.stem}_measurement.json"
    if not cv2.imwrite(str(overlay_path), overlay):
        raise SystemExit(f"Could not write overlay: {overlay_path}")
    result = {
        "image": str(args.image), "reference_distance_mm": args.ref_mm,
        "reference_distance_pixels": ref_pixels, "pixels_per_mm": px_per_mm,
        "visible_long_side_mm": long_side, "visible_short_side_mm": short_side,
        "method": "manual ruler-point and four-corner measurement",
        "limitations": ["manual clicks", "perspective not corrected", "only visible plane measured; thickness requires a side-view photo"],
    }
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Scale: {px_per_mm:.3f} px/mm")
    print(f"Estimated visible long side: {long_side:.2f} mm")
    print(f"Estimated visible short side: {short_side:.2f} mm")
    print("Thickness is not inferred from a top-view image.")
    print(f"Saved overlay: {overlay_path}\nSaved result: {result_path}")


if __name__ == "__main__":
    main()
