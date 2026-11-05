# Objectives

## Hardware used
| Field | Entry |
|---|---|
| CPU / GPU / RAM / OS | Fill from the computer used to run the project |
| Camera | Tecno Spark 40 (confirm if this was the capture device) |
| Resolution | Record from `python dataset/check_dataset.py --images dataset/images` |

## Measurement objective (MO-1)
- What: Mean absolute error (mm) for eraser length and width, comparing estimates with independent ruler measurements.
- Reference dimensions supplied for the eraser: 32.5 mm x 19 mm top face; thickness 10 mm. Treat these as ground truth only after confirming them independently with the ruler.
- How: Measure at least 10 ruler-visible images; preserve each prediction and its error. The current script measures top-view sides only; evaluate thickness separately with side-view photos and suitable measurement procedure.
- Target: State the chosen threshold before reporting results; do not mark as met until actual results are calculated.
- Conditions: Camera as close to overhead as possible; object and ruler/reference on one flat plane; sharp focus; adequate light.
- Exclusions: Hidden reference, blur, severe perspective tilt, occluded object.

## Model objective (MO-2)
- Current status: no trained segmentation model, weights, IoU, mAP, precision, or recall are provided by the current interactive baseline. Do not claim these results.

## Results
Fill in only after running locally: calibration reprojection error and image count; count/readability of dataset; per-image measurement results; MAE/median/worst error; limitations and unfinished items.
