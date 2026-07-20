You are continuing the existing SceneSense project.

The project structure, dataset, and source code already exist.

DO NOT recreate the project.

DO NOT regenerate files unless they contain bugs.

=====================================================
OBJECTIVE
=====================================================

Your goal is to review the entire project, automatically fix every issue, install missing dependencies if possible, train the model, evaluate it, export all required models, perform inference, and leave the project in a fully completed state.

Do not stop after reviewing.

Do not stop after fixing.

Continue until the whole pipeline has successfully finished.

=====================================================
STEP 1 — PROJECT AUDIT
=====================================================

Analyze every file.

Review:

- train.py
- inference.py
- download_dataset.py
- notebook.ipynb
- requirements.txt
- README.md

Inside src:

- config.py
- dataset.py
- augmentation.py
- model.py
- callbacks.py
- trainer.py
- evaluator.py
- exporter.py
- utils.py

Check:

✓ syntax
✓ imports
✓ module dependencies
✓ incorrect TensorFlow API
✓ incorrect keras API
✓ missing functions
✓ wrong paths
✓ invalid callbacks
✓ incorrect dataset loading
✓ incorrect dataset split
✓ incorrect augmentation
✓ model architecture
✓ optimizer
✓ loss function
✓ metrics
✓ export logic
✓ inference logic

If anything is wrong:

FIX IT AUTOMATICALLY.

Do not only explain.

=====================================================
STEP 2 — VERIFY DATASET
=====================================================

Verify the Intel Image Classification dataset.

Expected structure:

dataset/
    raw/
        buildings/
        forest/
        glacier/
        mountain/
        sea/
        street/

Verify:

- all folders exist
- images are readable
- labels are correct
- no broken images
- dataset statistics

Fix any issue automatically.

=====================================================
STEP 3 — VERIFY ENVIRONMENT
=====================================================

Check Python environment.

Verify:

TensorFlow

NumPy

Matplotlib

Pillow

Scikit-learn

TensorFlowJS

If packages are missing:

Install them automatically whenever possible.

=====================================================
STEP 4 — RUN TRAINING
=====================================================

Execute the complete training pipeline.

Requirements:

Use GPU if available.

Otherwise continue with CPU.

Train until completion.

Do NOT stop because training takes time.

Use callbacks:

- EarlyStopping
- ReduceLROnPlateau
- ModelCheckpoint
- CSVLogger
- TerminateOnNaN

Save the best model.

=====================================================
STEP 5 — EVALUATION
=====================================================

After training:

Generate

accuracy

loss

confusion matrix

classification report

validation metrics

Save every output inside

outputs/

=====================================================
STEP 6 — EXPORT
=====================================================

Export automatically to

SavedModel

TensorFlow Lite

TensorFlowJS

Verify exported models.

If export fails,

fix the issue,

retry,

continue.

=====================================================
STEP 7 — INFERENCE
=====================================================

Run inference on several sample images.

Print

Predicted class

Confidence

Verify inference works correctly.

=====================================================
STEP 8 — NOTEBOOK
=====================================================

Ensure notebook.ipynb is complete.

Every section must execute correctly.

No missing outputs.

No placeholder code.

=====================================================
STEP 9 — FINAL VALIDATION
=====================================================

Perform a final audit.

Check:

✓ project builds

✓ training works

✓ inference works

✓ export works

✓ notebook works

✓ README is accurate

✓ requirements are complete

✓ no syntax errors

✓ no missing imports

=====================================================
FINAL REPORT
=====================================================

At the end provide only:

PROJECT STATUS

Completed:
✔ ...

Fixed:
✔ ...

Training Accuracy:
...

Validation Accuracy:
...

Test Accuracy:
...

Exported Models:
✔ SavedModel
✔ TFLite
✔ TFJS

Inference:
✔ Success

Remaining Issues:

If none:

Print

PROJECT COMPLETED SUCCESSFULLY

Do not stop until every possible issue has been fixed automatically.