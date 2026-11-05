# measure-it

Camera-calibrated image workflow and an **interactive classical-CV measurement baseline** for estimating an object's two visible top-view dimensions. This is not a trained segmentation model and does not infer thickness from a top-view photo.

## Project object and reference

- Object: eraser. User-provided reference dimensions: length 32.5 mm, width 19 mm, thickness/height 10 mm. Validate these against the ruler photos and report measured errors; do not present reference values as model output.
- Scale reference: ruler in the same image, or checkerboard squares only if their printed physical size has been verified. The supplied board says 20 mm nominal per square; measure the printed square with a ruler and use the actual value.
- Keep eraser and scale reference on the same flat plane. Use a side-view image for thickness.

## Setup (Windows PowerShell)

From the repository root:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 1. Check the dataset

Put eraser photos in `dataset/images/` and checkerboard calibration photos in `calibration/images/`.

```powershell
python dataset/check_dataset.py --images dataset/images
```

The script checks file count and readability; manually review blur, full object visibility, and whether a ruler/reference is visible in each image used for measurement.

## 2. Camera calibration

Use the actual number of **inner corners** in the checkerboard (not number of squares), and the square size measured with your ruler. If each square measures 20 mm and the board has 9 inner corners across and 7 down:

```powershell
python calibration/calibrate.py --images calibration/images --cols 9 --rows 7 --square-mm 20
```

Change `--cols`, `--rows`, or `--square-mm` to match the physical board. Calibration writes `calibration/camera_params.npz` and `calibration/camera_params_report.json`.

## 3. Optional undistortion and split

```powershell
python dataset/prepare_images.py --raw dataset/images --out dataset/undistorted --params calibration/camera_params.npz
python dataset/make_split.py --images dataset/undistorted --out dataset/split
```

The split script only splits image files. It does not create masks/labels or train a model; do not report model metrics unless a model is actually trained and evaluated.

## 4. Measure a top-view image

Choose an image with the eraser and a ruler/checkerboard scale in the same plane. For example, if you will click ruler marks 0 and 3 cm apart, `--ref-mm` is 30:

```powershell
python inference/measure.py --image dataset/images/photo.jpg --ref-mm 30 --params calibration/camera_params.npz
```

In the OpenCV window, click the two endpoints of the known reference distance and press Enter. Then select a tight rectangle around only the eraser top face and press Enter. Inspect the saved overlay in `outputs/` and the JSON before using the estimate. Omit `--params` if calibration parameters are unavailable.

The output is the visible top-view long and short sides. Thickness/height requires a separate side-view image and is not inferred by this script.

## Submission honesty / limitations

- Checkerboard calibration estimates lens distortion; scale still comes from a known ruler distance or physically verified square in the same image.
- The 71-image count is useful, but a trained model requires accurate labels and evaluation. This repository's current baseline does not provide trained weights, masks, or model metrics.
- For an accuracy report, run measurements on at least 10 ruler-visible images, record each output against the independently measured ground truth, and calculate the errors. Do not invent results.
- Do not commit private photos or generated calibration files unless intended; large/private data can stay local.
