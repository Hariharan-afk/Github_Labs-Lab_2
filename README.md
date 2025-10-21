# Breast Cancer Classification with FastAPI and GitHub Actions

A production-ready MLOps pipeline demonstrating automated model training, calibration, and deployment for breast cancer classification.

This project implements an end-to-end machine learning workflow featuring:

- **Binary Classification**: Distinguishes between malignant and benign breast tumors
- **Probability Calibration**: Ensures reliable confidence scores for clinical decision-making
- **REST API**: FastAPI service for real-time predictions
- **Continuous Training**: Automated retraining pipeline via GitHub Actions
- **Version Control**: All models and metrics tracked in-repository

## Dataset
- Source: scikit‑learn `load_breast_cancer`
- Samples: 569; Features: 30 numeric; Task: malignant (0) vs benign (1)
- Labels: `target_names = ["malignant", "benign"]`
- Features include mean/SE/worst of radius, texture, perimeter, area, smoothness, etc.

## Model
- Pipeline: `StandardScaler` + `LinearSVC` (fast and deterministic)
- Calibration: `CalibratedClassifierCV(method='sigmoid', cv=5)` to provide reliable probabilities
- Split: `train_test_split(test_size=0.2, random_state=42, stratify=y)`
- Artifacts saved to `models/` with a timestamp:
  - `model_<timestamp>_svc.joblib` (base pipeline)
  - `model_<timestamp>_svc_calibrated.joblib` (calibrated model used by the API)

## Project Structure

```
.
├── .github/
│   └── workflows/
│       └── retrain_and_calibrate_on_push.yml  # CI/CD pipeline
├── src/
│   ├── train_model.py                         # Training script
│   ├── evaluate_model.py                      # Metrics computation
│   ├── api.py                                 # FastAPI service
│   └── send_sample_prediction.py              # Testing utility
├── models/                                     # Versioned model artifacts
├── metrics/                                    # Evaluation results (JSON)
├── requirements.txt                            # Python dependencies
├── .gitignore                                 # Excludes venvs, caches
└── README.md                                  # This file
```

## Local Usage
Prereqs: Python 3.9+, pip, and optionally a virtual environment.

1) Install dependencies
```
pip install -r requirements.txt
```

2) Train a model (also writes metrics automatically)
```
# PowerShell
$ts = Get-Date -Format "yyyyMMddHHmmss"
python .\src\train_model.py --timestamp $ts

# Bash (alternative)
ts=$(date +"%Y%m%d%H%M%S")
python ./src/train_model.py --timestamp "$ts"
```
Artifacts created:
- `models/model_<timestamp>_svc.joblib`
- `models/model_<timestamp>_svc_calibrated.joblib`
- `metrics/<timestamp>_evaluation.json`
- `metrics/<timestamp>_calibration.json`

3) Run the API
```
uvicorn src.api:app --host 127.0.0.1 --port 8001 --reload
```
Endpoints:
- `GET /` — quick info
- `GET /health` — service status
- `GET /feature_names` — expected 30 feature names
- `GET /samples` — returns ready payloads; copy `samples[i].payload` into `/predict`
- `POST /predict` — accepts `{ "features": [...] }` (length 30) or `{ "features": {name: value} }`

Examples:
- Dict payloads (benign): `http://127.0.0.1:8001/samples?format=dict&label=benign&n=1&seed=42`
- List payloads (malignant): `http://127.0.0.1:8001/samples?format=list&label=malignant&n=2&seed=1`

## Metrics Artifacts
Training triggers evaluation and writes two JSON files to `metrics/` per timestamp:
- `<timestamp>_evaluation.json` — accuracy and F1 for base and calibrated models
- `<timestamp>_calibration.json` — Brier score, log loss, and calibration curve points

## GitHub Actions (CI)
Workflow: `.github/workflows/retrain_and_calibrate_on_push.yml`

- Trigger: pushes to `main`
- Steps:
  - Set up Python and install dependencies
  - Generate a timestamp and ensure `models/` and `metrics/` exist
  - Run `src/train_model.py --timestamp $TIMESTAMP` (saves models and writes metrics)
  - Commit and push new `models/` and `metrics/` artifacts back to the repository
- There is a scheduler to retrain the model once everyday (2:00 am default).

Artifacts committed by CI:
- `models/model_<timestamp>_svc.joblib`
- `models/model_<timestamp>_svc_calibrated.joblib`
- `metrics/<timestamp>_evaluation.json`
- `metrics/<timestamp>_calibration.json`

## Notes
- The API loads the latest calibrated model from `models/`.
- Local virtual environments are ignored by `.gitignore` to avoid accidental commits.
- Older workflow examples under `workflows/` are not active; GitHub only runs files under `.github/workflows/`.

