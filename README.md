Measure It
Segment an object and report its real width and height in mm using a reference object.
Dataset summary
Object: | Images: | Split: 70/20/10 | Labelling tool:
Hardware
CPU / GPU / RAM / OS:
Setup
pip install -r requirements.txt
How to run
Calibration: python calibration/calibrate.py --images calibration/images --cols 9 --rows 7 --square-mm <measured>
Undistort photos: python dataset/prepare_images.py
Label: python dataset/label_grabcut.py
Split (once, commit before training): python dataset/make_split.py --seed 42
Train: python models/train.py
Test metrics: python models/evaluate.py
Measure one image: python inference/measure.py --image photo.jpg --out overlay.jpg
Accuracy vs calliper: python measurement/accuracy.py
Weights
Link: | SHA-256:
