# SymptomChatbot (mini)

This small project contains scripts to generate a toy symptoms/diseases CSV, train a RandomForest
classifier, and run inference using a single bundle artifact.

Files
- `create_dataset.py` — writes `symptoms_diseases.csv` with sample diseases and symptom flags.
- `train_model.py` — trains a classifier and saves a single bundle file: `model_bundle.pkl` (contains
	the sklearn Pipeline, the symptom vocabulary/column order, and the LabelEncoder). A `pipeline.pkl`
	may also be written for convenience, but the canonical artifact is `model_bundle.pkl`.

Quick start (Windows PowerShell)

```powershell
python .\create_dataset.py
python .\train_model.py --bundle model_bundle.pkl
```

Run tests (requires pytest)

```powershell
pip install -r requirements.txt
pytest -q
```

Predict usage (bundle-first)

```powershell
# Use the single bundle file created by training
python .\predict.py --bundle model_bundle.pkl --symptoms 1,0,0,1
```

Notes
- The project now uses a bundle-first workflow: prefer loading `model_bundle.pkl` which contains
	everything needed for inference. Legacy multi-file CLI usage (separate model/vocab/label-enc)
	has been removed to simplify deployment.

If you'd like CI or automatic formatting (black/ruff) added, tell me and I can add a GitHub Actions
workflow and apply formatting across the repo.
