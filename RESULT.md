# SceneSense — Dicoding Submission Audit Report

**Date:** 2026-07-20
**Auditor:** OpenCode Automated Audit
**Project:** SceneSense — Image Classification (Intel Scene Dataset)

---

## 1. Project Structure

| Item | Status | Notes |
|------|--------|-------|
| `README.md` | ✅ PASS | Comprehensive, includes dataset info, usage, structure |
| `notebook.ipynb` | ⚠️ REVISE | Code present, but no execution outputs (null counts) |
| `requirements.txt` | ✅ PASS | Lists all dependencies with version constraints |
| `train.py` | ✅ PASS | Complete training script with CLI arguments |
| `inference.py` | ✅ PASS | Complete inference script with SavedModel + TFLite support |
| `src/` | ✅ PASS | Modular package with 10 well-organized modules |
| `dataset/` | ✅ PASS | Contains train/val/test splits with class subdirectories |
| `outputs/` | ✅ PASS | Contains accuracy.png, loss.png, confusion_matrix.png, classification_report.txt |
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

**Note:** README states "~25,000 images" but only 17,034 were downloaded. The original Kaggle dataset includes `seg_pred/` (~7k unlabeled images) which were excluded.

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

**Issues:**
- Dataset is **copied** to train/val/test directories via `shutil.copy2` (uses double disk space). For 17k images this is wasteful.
- `split_and_save()` copies files rather than symlinking or using the raw directory structure.
- Validation seed is set but `shuffle=False` so seed has no effect on val/test.

**Verdict: PASS** — minor optimization issues, no functional defects.

---

## 4. CNN Architecture

| Layer | Status | File:Line |
|-------|--------|-----------|
| `Sequential` | ✅ PASS | `src/model.py:12` |
| `Conv2D` | ✅ PASS | `src/model.py:15,19,23,27` |
| Pooling (MaxPooling2D) | ✅ PASS | `src/model.py:18,22,26,30` |
| `Dense` | ✅ PASS | `src/model.py:33,35` |
| `Softmax` | ✅ PASS | `src/model.py:37` (`activation="softmax"`) |
| `Dropout` | ✅ PASS | `src/model.py:31,34` |
| `BatchNormalization` | ✅ PASS | `src/model.py:16,20,24,28` |

**Architecture summary:**
- Input → Rescaling(1/255) → Conv2D(64)→BN→ReLU→MaxPool
- → Conv2D(128)→BN→ReLU→MaxPool → Conv2D(256)→BN→ReLU→MaxPool
- → Conv2D(512)→BN→ReLU→MaxPool → Dropout(0.3)
- → Flatten → Dense(512) → Dropout(0.5) → Dense(6, softmax)

**Note:** Image size is 64×64 (reduced from original 128×128 for CPU performance). This is acceptable but smaller than the original design.

**Verdict: PASS**

---

## 5. Training

| Requirement | Status | Details |
|------------|--------|---------|
| Optimizer | ✅ PASS | Adam with lr=0.0005 |
| Loss function | ✅ PASS | SparseCategoricalCrossentropy |
| Metrics | ✅ PASS | Accuracy |
| **EarlyStopping** | ✅ PASS | monitor=val_accuracy, patience=10, restore_best_weights |
| **ReduceLROnPlateau** | ✅ PASS | monitor=val_loss, factor=0.5, patience=5, min_lr=1e-7 |
| **ModelCheckpoint** | ✅ PASS | monitor=val_accuracy, save_best_only, path=saved_model/best_model.keras |
| **CSVLogger** | ✅ PASS | logs/training_log.csv, append=True |
| **TerminateOnNaN** | ✅ PASS | Included |

**Training results (from logs):**

| Epoch | Train Acc | Train Loss | Val Acc | Val Loss |
|-------|-----------|------------|---------|----------|
| 1 | 48.83% | 11.46 | 19.14% | 12.88 |
| 5 | 72.26% | 0.77 | 67.44% | 0.89 |
| 10 | 77.97% | 0.61 | 73.66% | 0.74 |
| 15 | **80.51%** | **0.55** | **77.77%** | **0.62** |

**Test accuracy: 78.44%** | **Test loss: 0.5990**

**Verdict: PASS** for training infrastructure, **FAIL** for accuracy ≥ 85% requirement.

---

## 6. Evaluation

| Requirement | Status | File |
|------------|--------|------|
| Accuracy plot | ✅ PASS | `outputs/accuracy.png` |
| Loss plot | ✅ PASS | `outputs/loss.png` |
| Confusion matrix | ✅ PASS | `outputs/confusion_matrix.png` |
| Classification report | ✅ PASS | `outputs/classification_report.txt` |

**Classification Report:**

| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| buildings | 0.8629 | 0.6548 | 0.7446 |
| forest | 0.8354 | 0.9733 | 0.8991 |
| glacier | 0.6493 | 0.8716 | 0.7442 |
| mountain | 0.7581 | 0.7149 | 0.7359 |
| sea | 0.8403 | 0.7194 | 0.7752 |
| street | 0.8452 | 0.7691 | 0.8053 |
| **Accuracy** | | | **0.7844** |

Issues:
- Plots are saved as `accuracy.png`/`loss.png` (no timestamp). The notebook references `accuracy_{timestamp}.png`/`loss_{timestamp}.png` — this is a mismatch.

**Verdict: PASS** — all evaluation artifacts are generated.

---

## 7. Export

| Requirement | Status | Location |
|------------|--------|----------|
| SavedModel | ✅ PASS | `saved_model/` (saved_model.pb + variables/) |
| TensorFlow Lite | ✅ PASS | `tflite/model.tflite` (5.8 MB) |
| TensorFlow JS | ✅ PASS | `tfjs_model/` (model.json + 6 shards) |
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

**Inference test results:**
- SavedModel: `street` → 88.22%, `sea` → 98.02%, `forest` → ~98%
- TFLite: `forest` → 98.26%

**Note:** `inference.py` handles the Keras 3 `TFSMLayer` incompatibility gracefully via a fallback wrapper.

**Verdict: PASS**

---

## 9. Notebook

| Requirement | Status | Details |
|------------|--------|---------|
| Executes correctly | ⚠️ PARTIAL | Code is valid but cells have no execution outputs |
| Markdown explanations | ✅ PASS | 11 markdown cells with explanations |
| Outputs displayed | ❌ FAIL | All `execution_count` fields are `null` — no outputs |
| No placeholder code | ✅ PASS | All code cells contain real implementation |
| Image display | ✅ PASS | Uses matplotlib + IPython.display for showing images |

**Issues:**
1. **No execution outputs** — all cells show `execution_count: null` and empty `outputs: []`. A reviewer cannot see the results.
2. **Filename mismatch** — notebook displays `accuracy_{timestamp}.png` but plots were saved as `accuracy.png` (no timestamp).
3. **Dataset limitation** — Cell 3 checks for `dataset/train` directory and works only if train.py has been run first.

**Verdict: FAIL** — notebook lacks execution outputs which are required for submission.

---

## 10. Code Quality

| Requirement | Status | Details |
|------------|--------|---------|
| PEP8 compliant | ✅ PASS | Code follows PEP8 (4-space indent, naming conventions) |
| Type hints | ✅ PASS | Functions have type hints (e.g., `-> None`, `-> List[...]`) |
| Docstrings | ❌ FAIL | **No docstrings** in any src/ module or function |
| Modular architecture | ✅ PASS | 10 well-separated modules in src/ |
| No duplicated code | ✅ PASS | Clean separation of concerns |
| No TODO/FIXME | ✅ PASS | No placeholder comments found |
| No placeholder implementations | ✅ PASS | All functions fully implemented |
| Error handling | ⚠️ PARTIAL | `export_tfjs` silently catches `ImportError` |

**Issues:**
1. **No docstrings** — zero docstrings in any of the 10 src/ files. Dicoding evaluates code quality, and missing docstrings is a common deduction.
2. **`export_tfjs`** — catches `ImportError` silently, which could mask real issues. Should at minimum log a warning.
3. **`_get_file_paths`** — filters only `.jpg`, `.jpeg`, `.png` but the directory listing includes all files (could break on hidden/system files).

**Verdict: FAIL** — missing docstrings across the entire codebase.

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
| 7 | Conv2D layer | ✅ PASS | 4 Conv2D layers |
| 8 | Pooling layer | ✅ PASS | 4 MaxPooling2D layers |
| 9 | Dense layer | ✅ PASS | 2 Dense layers |
| 10 | Softmax output | ✅ PASS | `activation="softmax"` on output layer |
| 11 | Accuracy ≥ 85% | ❌ FAIL | **78.44%** — below the 85% threshold |
| 12 | SavedModel export | ✅ PASS | `saved_model/` directory |
| 13 | TF Lite export | ✅ PASS | `tflite/model.tflite` |
| 14 | TF JS export | ✅ PASS | `tfjs_model/model.json` |
| 15 | labels.txt | ✅ PASS | `saved_model/labels.txt` |

**Mandatory Requirements: 14/15 PASS (93.3%)** — **FAIL** on Accuracy ≥ 85%.

---

## 12. Dicoding Recommended Requirements

| # | Requirement | Status | Evidence |
|---|-------------|--------|----------|
| 1 | Callbacks (EarlyStopping, ReduceLROnPlateau) | ✅ PASS | Both implemented in `CallbacksFactory` |
| 2 | Dataset > 10,000 images | ✅ PASS | 17,034 images |
| 3 | Accuracy ≥ 95% | ❌ FAIL | 78.44% — far below threshold |
| 4 | More than 3 classes | ✅ PASS | 6 classes |
| 5 | Inference script | ✅ PASS | `inference.py` with full functionality |
| 6 | Professional README | ✅ PASS | Well-structured, all sections present |
| 7 | Professional project structure | ✅ PASS | Modular src/ package, organized directories |

**Recommended Requirements: 6/7 PASS (85.7%)**

---

## 13. Potential Reviewer Issues

| # | Severity | Issue | File(s) | Explanation | Recommended Fix |
|---|----------|-------|---------|-------------|-----------------|
| 1 | 🔴 **CRITICAL** | **Accuracy 78.44% < 85% mandatory** | `train.py`, model, config | Model fails to meet the mandatory 85% accuracy threshold. This is an automatic rejection. | Increase model capacity, train for more epochs, tune hyperparameters, use larger image size (128×128), or train with GPU for more epochs |
| 2 | 🔴 **CRITICAL** | **Notebook has no execution outputs** | `notebook.ipynb` | All cells show `execution_count: null` and empty outputs. Reviewer expects to see results inline. | Execute all notebook cells and save with outputs |
| 3 | 🟠 **MAJOR** | **No docstrings in any src/ module** | `src/*.py` (all 10 files) | Dicoding scoring rubric includes code documentation. Zero docstrings across the project. | Add docstrings to all classes, methods, and functions |
| 4 | 🟠 **MAJOR** | **Notebook filename mismatch for plots** | `notebook.ipynb:247-248` | Notebook displays `accuracy_{timestamp}.png` but actual files are `accuracy.png` (no timestamp) | Fix notebook to reference correct filenames or always include timestamp |
| 5 | 🟡 **MODERATE** | **64×64 image resolution is low** | `src/config.py:7` | Images resized to 64×64, losing detail. Original Kaggle dataset has 150×150 images. | Increase to at least 128×128 or 150×150 |
| 6 | 🟡 **MODERATE** | **READMe claims ~25,000 images, only 17,034 present** | `README.md:13` | Misleading dataset size in documentation | Update to accurate count or include seg_pred/ |
| 7 | 🟡 **MODERATE** | **Dataset copied via shutil instead of train_test_split directories** | `src/dataset.py:42` | `split_and_save()` copies all files, doubling disk usage | Use `image_dataset_from_directory` with subset="training"/"validation" or use symlinks |
| 8 | 🟡 **MODERATE** | **`export_tfjs` silently catches ImportError** | `src/exporter.py:49-50` | If TFJS export fails, it prints a message but doesn't raise error | Log warning but don't silently proceed if export is required |
| 9 | 🟢 **MINOR** | **Validation seed set but unused** | `src/dataset.py:92` | `seed=RANDOM_SEED` passed but `shuffle=False` makes it irrelevant | Remove seed from non-shuffled datasets for clarity |
| 10 | 🟢 **MINOR** | **Default target_size in load_image_for_inference is 128×128, not 64×64** | `src/utils.py:70` | Default doesn't match Config.IMAGE_SIZE | Change default to match config or remove default |

---

## 14. Final Score Prediction

| Rating | Predicted | Reason |
|--------|-----------|--------|
| ⭐ | | |
| ⭐⭐ | | |
| ⭐⭐⭐ | | |
| ⭐⭐⭐⭐ | | |
| ⭐⭐⭐⭐⭐ | | |

### ⭐⭐ (2 out of 5)

**Why?**

The project is well-structured and covers most Dicoding requirements (14/15 mandatory, 6/7 recommended). However, **two critical blockers** will cause rejection:

1. **Accuracy (78.44%) fails the mandatory ≥85% threshold.** This is an automatic fail regardless of everything else.
2. **Notebook has no execution outputs.** Dicoding requires the notebook to show inline results.

Additional deductions from missing docstrings and the ~25,000 vs 17,034 image discrepancy further reduce the score.

If accuracy were ≥85% and the notebook had outputs, the project would score **4-5 stars** with minor adjustments for docstrings.

---

## 15. Final Verdict

```
PROJECT NOT READY
```

### Required actions before submission (ordered by priority):

1. 🔴 **Improve model accuracy to ≥85%** — Increase epochs, image size, model capacity, or tune hyperparameters. Train on GPU if available.
2. 🔴 **Execute notebook cells and save outputs** — Run all cells and save the notebook with embedded execution results.
3. 🟠 **Add docstrings** — Every class, method, and function needs a docstring.
4. 🟡 **Fix README dataset count** — Update from "~25,000" to actual count (17,034).
5. 🟡 **Fix notebook plot filename references** — Match actual filenames.
6. 🟢 **Fix `load_image_for_inference` default** — Align with actual image size.

---

## Summary

| Category | Score |
|----------|-------|
| Project Structure | ✅ 11/11 |
| Dataset | ✅ 10/10 |
| Data Pipeline | ✅ 8/8 |
| CNN Architecture | ✅ 7/7 |
| Training | ⚠️ 5/5 (infra) ❌ 0/1 (accuracy) |
| Evaluation | ✅ 4/4 |
| Export | ✅ 4/4 |
| Inference | ✅ 7/7 |
| Notebook | ⚠️ 3/4 |
| Code Quality | ⚠️ 5/7 |

**Overall Completion:** ~85%
**Mandatory Requirements:** 14/15 PASS (93.3%)
**Recommended Requirements:** 6/7 PASS (85.7%)
**Submission Readiness:** NOT READY
**Expected Dicoding Score:** ⭐⭐
