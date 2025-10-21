import os
import json
import argparse
import joblib
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, brier_score_loss, log_loss
from sklearn.calibration import calibration_curve


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--timestamp", type=str, required=True, help="Timestamp from GitHub Actions")
    args = parser.parse_args()

    timestamp = args.timestamp

    # Load models produced by training
    cal_model_path = os.path.join("models", f"model_{timestamp}_svc_calibrated.joblib")
    base_model_path = os.path.join("models", f"model_{timestamp}_svc.joblib")
    if not os.path.exists(cal_model_path):
        raise FileNotFoundError(f"Calibrated model not found: {cal_model_path}")
    cal_model = joblib.load(cal_model_path)
    base_model = joblib.load(base_model_path) if os.path.exists(base_model_path) else None

    # Recreate deterministic test split from the same dataset
    X, y = load_breast_cancer(return_X_y=True)
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Evaluation metrics for base and calibrated
    eval_metrics = {"dataset": "breast_cancer", "n_test": int(len(y_test))}
    # Base predictions
    if base_model is not None:
        y_pred_base = base_model.predict(X_test)
        eval_metrics.update({
            "accuracy_base": float(accuracy_score(y_test, y_pred_base)),
            "f1_base": float(f1_score(y_test, y_pred_base)),
        })
    # Calibrated predictions
    y_pred_cal = cal_model.predict(X_test)
    eval_metrics.update({
        "accuracy_calibrated": float(accuracy_score(y_test, y_pred_cal)),
        "f1_calibrated": float(f1_score(y_test, y_pred_cal)),
    })

    # Calibration metrics for calibrated model (probabilities)
    y_proba_cal = cal_model.predict_proba(X_test)[:, 1]
    prob_true, prob_pred = calibration_curve(y_test, y_proba_cal, n_bins=10, strategy="uniform")
    calibration_metrics = {
        "dataset": "breast_cancer",
        "n_test": int(len(y_test)),
        "brier_score_calibrated": float(brier_score_loss(y_test, y_proba_cal)),
        "log_loss_calibrated": float(log_loss(y_test, np.clip(y_proba_cal, 1e-15, 1-1e-15))),
        "calibration_curve": {
            "fraction_of_positives": [float(v) for v in prob_true.tolist()],
            "mean_predicted_value": [float(v) for v in prob_pred.tolist()],
        },
    }

    # Save separate JSON files
    os.makedirs('metrics', exist_ok=True)
    eval_path = os.path.join('metrics', f'{timestamp}_evaluation.json')
    cal_path = os.path.join('metrics', f'{timestamp}_calibration.json')
    with open(eval_path, 'w') as f:
        json.dump(eval_metrics, f, indent=4)
    with open(cal_path, 'w') as f:
        json.dump(calibration_metrics, f, indent=4)
               
    
