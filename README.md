# SceneSense

![Python](https://img.shields.io/badge/Python-3.11-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![License](https://img.shields.io/badge/License-MIT-green)

**SceneSense** is an image classification system that classifies natural scene images into 6 categories using Convolutional Neural Networks (CNN) with TensorFlow.

## Dataset

**Intel Image Classification Dataset** ([Kaggle](https://www.kaggle.com/datasets/puneet6060/intel-image-classification))

- ~25,000 images
- 6 classes: buildings, forest, glacier, mountain, sea, street
- Split: Train (70%) | Validation (15%) | Test (15%)

## Requirements

```
tensorflow>=2.12.0
numpy>=1.24.0
matplotlib>=3.7.0
scikit-learn>=1.2.0
seaborn>=0.12.0
Pillow>=9.5.0
tensorflowjs>=4.10.0
jupyter>=1.0.0
```

## Installation

```bash
git clone https://github.com/ImamAriadi2022/SceneSense.git
cd SceneSense
pip install -r requirements.txt
```

## Usage

### Training

**With raw dataset path (auto-splits):**
```bash
python train.py --data_dir /path/to/raw/intel_dataset
```

**With pre-split dataset (dataset/train, dataset/val, dataset/test already exist):**
```bash
python train.py
```

**Override epochs or batch size:**
```bash
python train.py --data_dir /path/to/data --epochs 30 --batch_size 64
```

### Inference

**Using SavedModel:**
```bash
python inference.py --image_path /path/to/image.jpg
```

**Using TFLite model:**
```bash
python inference.py --tflite_path tflite/model.tflite --image_path /path/to/image.jpg
```

### Jupyter Notebook

```bash
jupyter notebook notebook.ipynb
```

## Project Structure

```
SceneSense/
├── README.md
├── requirements.txt
├── notebook.ipynb
├── train.py
├── inference.py
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── dataset.py
│   ├── augmentation.py
│   ├── model.py
│   ├── callbacks.py
│   ├── trainer.py
│   ├── evaluator.py
│   ├── exporter.py
│   └── utils.py
├── dataset/
├── saved_model/
├── tflite/
├── tfjs_model/
└── outputs/
    ├── accuracy.png
    ├── loss.png
    ├── confusion_matrix.png
    └── classification_report.txt
```

## Results

| Metric    | Value |
|-----------|-------|
| Test Accuracy | ≥ 85% |
| Test Loss     | Low   |

## Exported Formats

- **SavedModel** — for TensorFlow serving / Python deployment
- **TensorFlow Lite** — for mobile / edge devices
- **TensorFlow.js** — for browser / Node.js deployment
- **labels.txt** — class labels for inference

## Features

- Sequential CNN with Conv2D, BatchNormalization, MaxPooling, Dropout
- Data Augmentation (RandomFlip, Rotation, Zoom, Contrast, Translation)
- Callbacks: EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TerminateOnNaN, CSVLogger
- Confusion matrix & classification report
- Training history plots (accuracy & loss)
- Production-ready modular architecture
