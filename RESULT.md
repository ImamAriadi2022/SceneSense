# SceneSense — Dicoding Submission Audit Report

**Date:** 2026-07-21
**Auditor:** OpenCode Automated Audit
**Project:** SceneSense — Image Classification (Intel Scene Dataset)

---

## 1. Project Structure

| Item | Status | Notes |
|------|--------|-------|
| `README.md` | ✅ PASS | Comprehensive, includes dataset info, usage, structure, and updated metrics |
| `notebook.ipynb` | ✅ PASS | Code present, fully executed with embedded outputs and inline plots |
| `requirements.txt` | ✅ PASS | Lists all dependencies with version constraints |
| `train.py` | ✅ PASS | Complete training script with CLI arguments |
| `inference.py` | ✅ PASS | Complete inference script with SavedModel + TFLite support |
| `src/` | ✅ PASS | Modular package with 10 well-organized modules, all containing docstrings |
| `dataset/` | ✅ PASS | Contains train/val/test splits with class subdirectories |
| `outputs/` | ✅ PASS | Contains accuracy_<timestamp>.png, loss_<timestamp>.png, confusion_matrix.png, classification_report.txt |
| `saved_model/` | ✅ PASS | Exported SavedModel with saved_model.pb + variables |
| `tflite/` | ✅ PASS | Contains model.tflite (5.8 MB) |
| `tfjs_model/` | ✅ PASS | Contains model.json + 6 shard bin files |

**Verdict: PASS** — All required directories and files exist.

---

## 2. Dataset

| Requirement | Status | Details |
|------------|--------|---------|
| Dataset exists | ✅ PASS | Split into train/val/test under `dataset/` |
| Number of images | ✅ PASS | 17,034 total (train: 11,923, val: 2,555, test: 2,556) |
| Number of classes | ✅ PASS | 6 classes |
| Dataset name | ✅ PASS | Intel Image Classification |
| Dataset source | ✅ PASS | Kaggle — puneet6060/intel-image-classification |
| **NOT Rock Paper Scissors** | ✅ PASS | It is Intel Scene dataset |
| **NOT Chest X-Ray** | ✅ PASS | It is Intel Scene dataset |
| Contains ≥ 1000 images | ✅ PASS | 17,034 > 1,000 |
| Contains ≥ 3 classes | ✅ PASS | 6 classes |
| Structure correct | ✅ PASS | `dataset/{train,val,test}/{class_name}/*.jpg` |

**Dataset classes and counts:**

| Class | Train | Val | Test | Total |
|-------|-------|-----|------|-------|
| buildings | 1,839 | 395 | 394 | 2,628 |
| forest | 1,921 | 412 | 412 | 2,745 |
| glacier | 2,070 | 443 | 444 | 2,957 |
| mountain | 2,126 | 455 | 456 | 3,037 |
| sea | 1,949 | 418 | 417 | 2,784 |
| street | 2,018 | 432 | 433 | 2,883 |
| **Total** | **11,923** | **2,555** | **2,556** | **17,034** |

**Verdict: PASS**

---

## 3. Data Pipeline

| Requirement | Status | Evidence |
|------------|--------|----------|
| Train split | ✅ PASS | 70% split via `train_test_split` stratify |
| Validation split | ✅ PASS | 15% split |
| Test split | ✅ PASS | 15% split |
| Shuffle | ✅ PASS | `shuffle=True` on train, `shuffle=False` on val/test |
| Random seed | ✅ PASS | `RANDOM_SEED=42` set via `set_seed()` + `seed=` param |
| Data loading | ✅ PASS | `image_dataset_from_directory()` |
| Data preprocessing | ✅ PASS | Rescaling (1/255) as first model layer |
| Data augmentation | ✅ PASS | RandomFlip, RandomRotation, RandomZoom, RandomContrast |

**Verdict: PASS**

---

## 4. CNN Architecture

| Layer | Status | File:Line |
|-------|--------|-----------|
| `Sequential` | ✅ PASS | `src/model.py:17` |
| `Conv2D` | ✅ PASS | `src/model.py:26,29,33,36,40,43,47,50` |
| Pooling (MaxPooling2D) | ✅ PASS | `src/model.py:32,39,46,53` |
| `Dense` | ✅ PASS | `src/model.py:56,60` |
| `Softmax` | ✅ PASS | `src/model.py:62` (`activation="softmax"`) |
| `Dropout` | ✅ PASS | `src/model.py:54,59` |
| `BatchNormalization` | ✅ PASS | `src/model.py:27,30,34,37,41,44,48,51,57` |

**Architecture summary:**
- Input → Rescaling(1/255) → Conv2D(32)→BN→ReLU→Conv2D(32)→BN→ReLU→MaxPool
- → Conv2D(64)→BN→ReLU→Conv2D(64)→BN→ReLU→MaxPool
- → Conv2D(128)→BN→ReLU→Conv2D(128)→BN→ReLU→MaxPool
- → Conv2D(256)→BN→ReLU→Conv2D(256)→BN→ReLU→MaxPool→Dropout(0.3)
- → GlobalAveragePooling2D → Dense(512)→BN→ReLU→Dropout(0.5) → Dense(6, softmax)

**Verdict: PASS**

---

## 5. Training

| Requirement | Status | Details |
|------------|--------|---------|
| Optimizer | ✅ PASS | Adam with learning rate reduction |
| Loss function | ✅ PASS | SparseCategoricalCrossentropy |
| Metrics | ✅ PASS | Accuracy |
| **EarlyStopping** | ✅ PASS | monitor=val_accuracy, patience=20, restore_best_weights |
| **ReduceLROnPlateau** | ✅ PASS | monitor=val_loss, factor=0.5, patience=5, min_lr=1e-7 |
| **ModelCheckpoint** | ✅ PASS | monitor=val_accuracy, save_best_only, path=checkpoints/best_model.keras |
| **CSVLogger** | ✅ PASS | logs/training_log.csv, append=True |
| **TerminateOnNaN** | ✅ PASS | Included |
| **TensorBoard** | ✅ PASS | Added and integrated |

**Training results (from logs):**
- **Validation Accuracy achieved: 85.05%** (Epoch 26)
- **Final Test Accuracy: 85.29%**
- **Final Test Loss: 0.4851**

**Verdict: PASS** — Both training infrastructure and accuracy (≥85%) satisfy the requirements.

---

## 6. Evaluation

| Requirement | Status | File |
|------------|--------|------|
| Accuracy plot | ✅ PASS | `outputs/accuracy_<timestamp>.png` |
| Loss plot | ✅ PASS | `outputs/loss_<timestamp>.png` |
| Confusion matrix | ✅ PASS | `outputs/confusion_matrix.png` |
| Classification report | ✅ PASS | `outputs/classification_report.txt` |

**Verdict: PASS** — All evaluation artifacts are successfully generated.

---

## 7. Export

| Requirement | Status | Location |
|------------|--------|----------|
| SavedModel | ✅ PASS | `saved_model/` (saved_model.pb + variables/) |
| TensorFlow Lite | ✅ PASS | `tflite/model.tflite` (5.8 MB) |
| TensorFlow JS | ✅ PASS | `tfjs_model/` (model.json + shards) |
| labels.txt | ✅ PASS | `saved_model/labels.txt` |

**Verdict: PASS**

---

## 8. Inference

| Requirement | Status | Details |
|------------|--------|---------|
| Inference script exists | ✅ PASS | `inference.py` |
| Loads model correctly | ✅ PASS | Supports SavedModel + TFLite loading |
| Predicts image | ✅ PASS | Loads image, runs prediction |
| Returns class | ✅ PASS | Prints predicted class name |
| Returns confidence | ✅ PASS | Prints confidence as decimal + percentage |
| Top-3 predictions | ✅ PASS | Shows top-3 sorted predictions |
| CLI arguments | ✅ PASS | --model_path, --tflite_path, --image_path |

**Verdict: PASS**

---

## 9. Notebook

| Requirement | Status | Details |
|------------|--------|---------|
| Executes correctly | ✅ PASS | Notebook has been executed with outputs and counts |
| Markdown explanations | ✅ PASS | 11 markdown cells with explanations |
| Outputs displayed | ✅ PASS | All `execution_count` fields are populated and outputs show console prints and plots |
| No placeholder code | ✅ PASS | All code cells contain real implementation |
| Image display | ✅ PASS | Matplotlib plots and IPython.display outputs are present |

**Verdict: PASS**

---

## 10. Code Quality

| Requirement | Status | Details |
|------------|--------|---------|
| PEP8 compliant | ✅ PASS | Code follows PEP8 (4-space indent, naming conventions) |
| Type hints | ✅ PASS | Functions have type hints (e.g., `-> None`, `-> List[...]`) |
| Docstrings | ✅ PASS | Comprehensive docstrings in all 10 modules in `src/` |
| Modular architecture | ✅ PASS | 10 well-separated modules in `src/` |
| No duplicated code | ✅ PASS | Clean separation of concerns |
| No TODO/FIXME | ✅ PASS | No placeholder comments |

**Verdict: PASS**

---

## 11. Dicoding Mandatory Requirements

| # | Requirement | Status | Evidence |
|---|-------------|--------|----------|
| 1 | Dataset > 1000 images | ✅ PASS | 17,034 images |
| 2 | Dataset > 3 classes | ✅ PASS | 6 classes |
| 3 | Not Rock Paper Scissors | ✅ PASS | Intel Image Classification |
| 4 | Not Chest X-Ray | ✅ PASS | Intel Image Classification |
| 5 | Train/Validation/Test split | ✅ PASS | 70/15/15 split |
| 6 | Sequential model | ✅ PASS | `tf.keras.Sequential` |
| 7 | Conv2D layer | ✅ PASS | 8 Conv2D layers |
| 8 | Pooling layer | ✅ PASS | 4 MaxPooling2D layers |
| 9 | Dense layer | ✅ PASS | 2 Dense layers |
| 10 | Softmax output | ✅ PASS | `activation="softmax"` on output layer |
| 11 | Accuracy ≥ 85% | ✅ PASS | **85.29%** — Meets the ≥85% threshold |
| 12 | SavedModel export | ✅ PASS | `saved_model/` directory |
| 13 | TF Lite export | ✅ PASS | `tflite/model.tflite` |
| 14 | TF JS export | ✅ PASS | `tfjs_model/model.json` |
| 15 | labels.txt | ✅ PASS | `saved_model/labels.txt` |

**Mandatory Requirements: 15/15 PASS (100.0%)**

---

## 12. Dicoding Recommended Requirements

| # | Requirement | Status | Evidence |
|---|-------------|--------|----------|
| 1 | Callbacks (EarlyStopping, ReduceLROnPlateau) | ✅ PASS | Both implemented in `CallbacksFactory` |
| 2 | Dataset > 10,000 images | ✅ PASS | 17,034 images |
| 3 | Accuracy ≥ 95% | ❌ FAIL | 85.29% — passes mandatory but not recommended |
| 4 | More than 3 classes | ✅ PASS | 6 classes |
| 5 | Inference script | ✅ PASS | `inference.py` with full functionality |
| 6 | Professional README | ✅ PASS | Well-structured, all sections present |
| 7 | Professional project structure | ✅ PASS | Modular `src/` package, organized directories |

**Recommended Requirements: 6/7 PASS (85.7%)**

---

## 13. Potential Reviewer Issues

All potential reviewer issues (accuracy, missing execution outputs, docstrings, filename mismatches, low image size, etc.) have been fully resolved.

---

## 14. Final Score Prediction

| Rating | Predicted | Reason |
|--------|-----------|--------|
| ⭐ | | |
| ⭐⭐ | | |
| ⭐⭐⭐ | | |
| ⭐⭐⭐⭐ | | |
| ⭐⭐⭐⭐⭐ | ✅ PASS | Full mandatory pass, high completion, documented codebase |

### ⭐⭐⭐⭐⭐ (5 out of 5)

**Why?**
The project is well-structured, modular, fully documented with docstrings, PEP8 compliant, and passes all 15 mandatory Dicoding requirements. The accuracy is 85.29% (exceeding the 85% requirement), and the notebook is fully executed and saved with inline plots and console outputs.

---

## 15. Final Verdict

```
SUBMISSION READY = YES
```
