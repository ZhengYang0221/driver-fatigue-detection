# Driver Fatigue Detection using CNN + SVM

## Overview
Real-time driver fatigue detection using webcam input and a hybrid CNN + SVM classifier.
The system detects the driver's face, extracts deep features using a CNN, and classifies
the driver as **Awake** or **Drowsy** using a linear SVM.

## Demo
- Detecting fatigue through yawning: https://youtu.be/n1Zj1Yj_PsY 
- Detecting fatigue through closing eyelids: https://youtu.be/vb8B4BwHq7s  
- Detecting fatigue through head tilt: https://youtu.be/3cA9WDugWOM

Note: Wearing sunglasses only affect the closing eyelids feature during fatigue detection, yawning anad head tilt still works fine. Will add a video on it in the future.

## Methodology

1. Face detection using OpenCV Haar Cascades
2. Stratified train / val / test split (70 / 15 / 15)
3. CNN training with data augmentation and early stopping
4. Feature extraction from the CNN's `flatten` layer (2304-dim spatial features)
5. SVM classification trained on CNN embeddings
6. Evaluation on held-out test set: accuracy, precision, recall, F1, ROC-AUC, confusion matrix
7. Real-time inference via webcam

## Project Structure

```
driver-fatigue-detection/
├── notebooks/
│   └── driver_fatigue_pipeline.ipynb   # full pipeline (train → evaluate → infer)
├── data/
│   ├── awake/                          # awake driver images (add locally)
│   └── drowsy/                         # drowsy driver images (add locally)
├── models/                             # saved model weights (generated on run)
├── assets/                             # evaluation plots (generated on run)
├── requirements.txt
└── README.md
```

## Dataset Setup

Place your images in the `data/` directories before running the notebook:

```
data/
├── awake/    ← JPEG/PNG images of alert drivers
└── drowsy/   ← JPEG/PNG images of drowsy/fatigued drivers
```

The notebook was developed using a custom-collected dataset.
A suitable public alternative is the [MRL Eye Dataset](http://mrl.cs.vsb.cz/eyedataset).

## Results (Test Set)

| Model     | Accuracy | Precision | Recall |  F1  | ROC-AUC |
|-----------|----------|-----------|--------|------|---------|
| CNN alone |   ~98%   |   ~99%    |  ~98%  | ~98% |   ~99%  |
| CNN + SVM |   ~99%   |   ~99%    |  ~99%  | ~99% |   ~99%  |

*Metrics on a custom-collected dataset (~41K images, 70/15/15 stratified split). Full per-class breakdown in `assets/`.*

## Tech Stack

- Python 3.11
- TensorFlow / Keras
- scikit-learn
- OpenCV
- NumPy, Matplotlib, joblib

## How to Run

```bash
pip install -r requirements.txt
jupyter notebook notebooks/driver_fatigue_pipeline.ipynb
```

Run cells top to bottom. Section 6 streams webcam frames inline in the notebook output —
click the **Stop** button (interrupt kernel) to end the session.

## Limitations

- Sensitive to lighting variation
- Reduced accuracy when eyes are occluded (e.g. sunglasses)
- Performance may degrade in real-world conditions not represented in training data
