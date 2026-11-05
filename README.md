# measure-it

Camera calibration and an interactive classical-computer-vision measurement baseline for an eraser. The measurement tool uses manual clicks on the ruler and object corners. It is not a trained segmentation model; no model metrics or weights are claimed.

## Known dimensions and reference

User-provided eraser dimensions are **32.5 mm × 19 mm × 10 mm**. Use these as comparison values only after confirming them independently with the ruler. Keep the eraser and scale reference in the same flat plane. Use a side-view image for the 10 mm thickness.

The supplied checkerboard sheet says each square is **nominally 20 mm**; verify the printed square with a ruler before using it. Calibration square size and image measurement scale are different things: calibration corrects lens distortion, while an in-frame ruler/verified checker square provides scale.

## Setup (Windows PowerShell)

Run from the repository root:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Fast submission workflow

1. Place the 71 photos in `dataset/images/`. Keep checkerboard photos in `calibration/images/`.
2. Check image count/readability:

```powershell
python dataset/check_dataset.py --images dataset/images
```

3. If camera calibration images are available, use the actual checkerboard **inner corner count** and square size measured from the print. Example only (9 × 7 inner corners, 20 mm squares):

```powershell
python calibration/calibrate.py --images calibration/images --cols 9 --rows 7 --square-mm 20
```

Do not use 20 unless the printed square actually measures 20 mm. Calibration outputs `calibration/camera_params.npz` and `calibration/camera_params_report.json`.

4. Measure one ruler-visible image. Replace the filename and known ruler span below. For ruler marks 0 cm and 3 cm, use 30 mm:

```powershell
python inference/measure.py --image dataset/images/photo.jpg --ref-mm 30
```

Optional lens correction if calibration params exist: add `--params calibration/camera_params.npz`.

In the first window click two ruler marks exactly 30 mm apart and press a key. In the next window click the four outer corners of the eraser's visible face in order around its perimeter, then press a key. Review the saved overlay in `outputs/` and JSON before using the result. If the green outline is misplaced, rerun and click again. The estimate is manual and perspective is not corrected.

5. For an accuracy report, create a CSV based on `docs/accuracy_template.csv`. Enter real measured and predicted values for at least 10 images; run:

```powershell
python evaluation/evaluate_accuracy.py --csv docs/accuracy_template.csv --out outputs/accuracy_report.json
```

Replace the example row with actual results and add the other images. Never submit the template/example values as measured results. The script reports MAE, median absolute error, and worst absolute error for visible long/short sides.

## Optional preprocessing

```powershell
python dataset/prepare_images.py --raw dataset/images --out dataset/undistorted --params calibration/camera_params.npz
python dataset/make_split.py --images dataset/undistorted --out dataset/split
```

The split contains images only: it does not create labels or train a model. Do not claim IoU, mAP, precision/recall, model weights, or trained-model results unless those are actually produced and evaluated.

## Limits to disclose

- A checkerboard calibration estimates lens distortion; scale comes from a known ruler distance or physically verified checkerboard square in the same image.
- Top-view images provide visible face dimensions, not thickness. Thickness needs a side view and its own scale reference.
- Manual clicks and camera perspective introduce measurement error; use a nearly overhead camera and inspect each overlay.
- 71 images meet the image-count target, but they are not automatically labeled. The current implementation is a reproducible interactive baseline, not a fully automated trained model.
- Do not commit private photos or generated calibration files unless intended.
