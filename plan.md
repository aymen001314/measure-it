# Plan: Measure It

## Object and reference
- Object: Matchbox (Super Hockey), approx 50 x 35 mm (width = longer side, height = shorter side). Chosen because: flat, rigid and opaque, and easy to photograph 70 times in different positions.
- Reference: printed ArUco marker (id 0) taped flat on stiff card, black square side to be measured by me with a ruler after printing.
  (Changed from a plain card: an ArUco marker is detected exactly, so 'reference not found' is clean to detect.)
- Camera: Tecno Spark 40, focus, zoom and resolution locked, all photos in landscape

## Order and time (8 hours of work)
1. Calibration (1h): 20+ checkerboard photos, OpenCV, save parameters, report reprojection error
2. Data (1.5h): 70+ photos with reference in same plane, undistort, label masks, fixed 70/20/10 split
3. Model (2h): U-Net with pretrained MobileNetV2 backbone (PyTorch). Training waits do not count
4. Inference command (1.5h): image -> undistort -> mask -> reference -> width/height mm
5. Accuracy (1h): 10+ instances vs ruler, per-instance table, mean/median/worst error
6. Undistortion comparison (0.5h): raw vs undistorted error side by side
7. Docs, README, Loom (0.5h)

## Decided to skip (for now)
- Items 7 to 10 unless time remains

## Model choice
U-Net + MobileNetV2 backbone. Not Ultralytics YOLO, not Roboflow. Small, simple, and I have used MobileNetV2 before.

## Decision record
(Fill in at the end: approach taken, approach rejected, what rejecting it cost.)
