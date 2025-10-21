import os
import argparse
import sys
import subprocess
from joblib import dump
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import f1_score, accuracy_score


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--timestamp", type=str, required=True, help="Timestamp from GitHub Actions")
    args = parser.parse_args()

    timestamp = args.timestamp
    print(f"Timestamp received from GitHub Actions: {timestamp}")

    # Load Breast Cancer dataset
    X, y = load_breast_cancer(return_X_y=True)

    # Train/test split for reproducible evaluation downstream
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Base pipeline: scaling + LinearSVC (different from prior RandomForest)
    base_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LinearSVC(random_state=42))
    ])

    # Fit base pipeline
    base_pipeline.fit(X_train, y_train)

    # Calibrate probabilities using sigmoid (Platt scaling)
    calibrated_clf = CalibratedClassifierCV(
        estimator=base_pipeline, method="sigmoid", cv=5
    )
    calibrated_clf.fit(X_train, y_train)

    # Quick validation on test set (printed only)
    y_pred_base = base_pipeline.predict(X_test)
    y_pred_cal = calibrated_clf.predict(X_test)
    print({
        "accuracy_base": float(accuracy_score(y_test, y_pred_base)),
        "f1_base": float(f1_score(y_test, y_pred_base)),
        "accuracy_calibrated": float(accuracy_score(y_test, y_pred_cal)),
        "f1_calibrated": float(f1_score(y_test, y_pred_cal)),
    })

    # Ensure models directory exists
    os.makedirs("models", exist_ok=True)

    # Save both base and calibrated models with timestamped names
    version_prefix = f"model_{timestamp}"
    base_filename = os.path.join("models", f"{version_prefix}_svc.joblib")
    cal_filename = os.path.join("models", f"{version_prefix}_svc_calibrated.joblib")

    dump(base_pipeline, base_filename)
    dump(calibrated_clf, cal_filename)

    # Trigger evaluation to write metrics artifacts
    eval_script = os.path.join(os.path.dirname(__file__), "evaluate_model.py")
    subprocess.run([sys.executable, eval_script, "--timestamp", timestamp], check=True)
                    

