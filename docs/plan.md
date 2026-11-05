# Plan: Measure It

## Object and reference
- Object: eraser. User-provided reference dimensions: length 32.5 mm, width 19 mm, thickness/height 10 mm. Check these values against ruler measurements and report deviations honestly.
- Scale reference: ruler or checkerboard in the same image and plane as the eraser. Printed checkerboard squares are 20 mm nominal according to the supplied sheet; verify printed size with ruler before using it.
- Camera: Tecno Spark 40. Keep focus, zoom, resolution, and orientation fixed where possible.

## Work plan
1. Calibration: use checkerboard images; report valid image count and reprojection error.
2. Dataset: 70+ own images in `dataset/images`; check readability and manually verify reference visibility. Split deterministically 70/20/10.
3. Measurement baseline: interactive object ROI and ruler/reference scale; report top-view length and width. Use side views for thickness.
4. Accuracy: compare outputs against independent ruler measurements on 10+ images; save raw values, MAE, median, and worst error.
5. Document completed work and limitations. Do not claim training/model metrics unless a segmentation model is labeled, trained, and evaluated.

## Current implementation decision
- Classical OpenCV interactive baseline is implemented for a fast, reproducible submission.
- A trained U-Net/MobileNetV2 model is not implemented in the current baseline; trained metrics and weights must not be claimed.

## Decision record
- Approach: interactive ruler scale and manually selected eraser ROI to prioritize a runnable baseline with limited time.
- Trade-off: requires user clicks, and is not a fully automatic trained model.
