# measure-it

Camera-calibrated image workflow and an **interactive classical-CV measurement baseline** for estimating an object's two visible top-view dimensions. It does not claim a trained segmentation model or infer thickness from a top-view photo.

## Setup (Windows PowerShell)

From the repository root:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 1. Check the dataset

Put eraser photos in `dataset/images/` (checkerboard calibration images remain in `calibration/images/`). Check count/readability:

```powershell
python dataset/check_dataset.py --images dataset/images
```

The measurement photos need a ruler or another known-length reference visible in the same image and on the same flat plane as the eraser. The provided sample with only an eraser on dark fabric cannot yield an absolute millimeter scale. Retake measurement images with a ruler/reference beside the eraser if needed.

## 2. Calibration (already done locally, if `calibration/camera_params.npz` exists)

This board has 9 x 7 inner corners and the measured square size is 18.8 mm:

```powershell
python calibration/calibrate.py --images calibration/images --cols 9 --rows 7 --square-mm 18.8
```

Outputs: `calibration/camera_params.npz` and `calibration/camera_params_report.json`.

## 3. Optional undistortion and split

```powershell
python dataset/prepare_images.py --raw dataset/images --out dataset/undistorted --params calibration/camera_params.npz
python dataset/make_split.py --images dataset/undistorted --out dataset/split
```

The split is image-only. It does not create labels or train a model; do not report it as a trained ML model or model metrics.

## 4. Measure one photo interactively

Use a photo with a ruler/reference visible. Example: if you click endpoints 10 mm apart on the ruler:

```powershell
python inference/measure.py --image dataset/images/photo.jpg --ref-mm 10 --params calibration/camera_params.npz
```

Omit `--params` if you do not have the calibration file locally. In the window, click two endpoints exactly the stated distance apart, press Enter, then select a tight rectangle around only the eraser's top face and press Enter. Inspect the saved overlay in `outputs/` before trusting the number. The script calculates the top-view long and short sides; height/thickness needs a side-view image. This manual workflow is a baseline, not a fully automatic trained model.

## Important limitations

- A checkerboard calibration estimates lens distortion; it does not by itself provide the object's scale in each image. A visible measured ruler/reference is needed.
- The 71-image count is useful, but model training/evaluation requires accurate object masks and a train/validation/test protocol. The scripts here do not fabricate labels or model metrics.
- Do not commit private photos or generated calibration files unless intended; large/private data can stay local.
