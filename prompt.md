Review RESULT.md and automatically fix every issue that prevents the project from passing Dicoding.

Priority order:

1. CRITICAL
- Improve model accuracy to at least 90% (target >95% if possible).
- Increase image resolution if needed (prefer 128x128 or 150x150).
- Improve CNN architecture.
- Tune optimizer, learning rate, callbacks, batch size, and epochs.
- Add data augmentation only to the training set.
- Retrain until validation and test accuracy exceed the Dicoding minimum requirement.

2. Execute the entire notebook.
- Ensure every cell is executed.
- Save notebook with execution outputs.
- Verify plots and inference outputs appear correctly.

3. Add comprehensive docstrings to all public modules, classes, and functions.

4. Fix documentation inconsistencies.
- Update dataset size.
- Fix plot filename references.
- Ensure README matches the actual project.

5. Run a complete validation after all fixes.

Finally, regenerate RESULT.md and verify:
- All mandatory requirements PASS.
- All recommended requirements PASS whenever possible.
- Submission Ready = YES.
- Expected Reviewer Score = ⭐⭐⭐⭐⭐.

Do not stop after making code changes. Continue until the project satisfies every mandatory requirement.