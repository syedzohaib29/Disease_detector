SymptomChatbot - Change Summary

What I changed

- Restored and hardened main scripts (`create_dataset.py`, `train_model.py`).
- Migrated inference to a bundle-first approach: a single `model_bundle.pkl` contains the
  trained sklearn Pipeline, the symptom vocabulary (column order), and the LabelEncoder.
- Removed legacy multi-file CLI usage in `predict.py`; it now requires `--bundle`.
- Added tests and ensured they pass locally (2 tests).
- Updated `README.md` to document bundle-first usage.

Why

- A single bundle artifact simplifies deployment and avoids mismatches between model, vocab,
  and label encoder files. It also avoids pickling issues that occur when helper classes are
  defined in local scopes.

Next steps (recommended)

1. Apply formatting and linting: run `black` and `ruff/flake8` to standardize style and fix
   any small issues.
2. Add CI: create a GitHub Actions workflow to run `pytest` on push/pull requests.
3. Integrate LabelEncoder into the sklearn Pipeline using a small transformer so the pipeline
   isn't coupled to an external label encoder (optional).
4. Improve dataset & evaluation: expand the dataset and use cross-validation for realistic
   metrics.

How I validated

- Ran the training and prediction flows during development and iterated on errors (stratify
  issues, classification report label mismatch, pickling issues).
- Ran `pytest` locally; final result: 2 passed.

If you'd like, I can now:
- Run `black` + `ruff` and commit the fixes.
- Add a GitHub Actions workflow that runs the test suite on push.
- Implement label-encoder-as-transformer so only a single sklearn Pipeline is required.

Tell me which of the above you'd like me to do next and I'll implement it.
