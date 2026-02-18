import json
import time
from pathlib import Path

import requests

URL = "http://127.0.0.1:8000/predict"
PAYLOAD_PATH = Path ("tests/sample_payload.json")

def main(n_ok=100, n_err=20):
    payload = json.loads(PAYLOAD_PATH.read_text())

    # succès
    for _ in range(n_ok):
        r = requests.post(URL, json=payload, timeout=10)
        print ("OK", r.status_code)
        time.sleep(0.02)

    #erreurs
    for _ in range(n_err):
        r = requests.post(URL, json={"features": {"FAKE_FEATURE": 123}}, timeout=10)
        print("Erreur", r.status_code)
        time.sleep(0.02)

if __name__ == "__main__":
    main()
