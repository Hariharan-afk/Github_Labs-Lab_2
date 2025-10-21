import argparse
import json
import requests
from sklearn.datasets import load_breast_cancer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1", help="API host")
    parser.add_argument("--port", default="8001", help="API port")
    args = parser.parse_args()

    X, _ = load_breast_cancer(return_X_y=True)
    payload = {"features": X[0].tolist()}

    url = f"http://{args.host}:{args.port}/predict"
    resp = requests.post(url, json=payload, timeout=10)
    print(f"Status: {resp.status_code}")
    try:
        print(json.dumps(resp.json(), indent=2))
    except Exception:
        print(resp.text)


if __name__ == "__main__":
    main()

