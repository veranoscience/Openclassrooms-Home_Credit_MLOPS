
import argparse, json, time
from pathlib import Path
import httpx
import numpy as np

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000/predict")
    ap.add_argument("--payload", default="tests/sample_payload.json")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=20)
    args = ap.parse_args()

    payload = json.loads(Path(args.payload).read_text(encoding="utf-8"))

    times = []
    with httpx.Client(timeout=10.0) as client:
        # warmup
        for _ in range(args.warmup):
            r = client.post(args.url, json=payload)
            r.raise_for_status()

        # bench
        for _ in range(args.n):
            t0 = time.perf_counter()
            r = client.post(args.url, json=payload)
            r.raise_for_status()
            times.append((time.perf_counter() - t0) * 1000)

    arr = np.array(times)
    print(f"n={len(arr)}")
    print(f"mean_ms={arr.mean():.2f}")
    print(f"p50_ms={np.percentile(arr, 50):.2f}")
    print(f"p95_ms={np.percentile(arr, 95):.2f}")
    print(f"max_ms={arr.max():.2f}")

    # ── Timings par étape depuis les logs JSONL ──────────────────────────
    import statistics
    log_path = Path("prod_logs/predictions.jsonl")
    lines = log_path.read_text(encoding="utf-8").strip().split("\n")
    
    # On prend les N dernières lignes (celles du benchmark)
    step_keys = ["hash_ms", "preprocessing_ms", "stats_ms", "inference_ms"]
    step_data = {k: [] for k in step_keys}

    for line in lines[-args.n:]:
        try:
            record = json.loads(line)
            timings = record.get("timings", {})
            for k in step_keys:
                if k in timings:
                    step_data[k].append(timings[k])
        except Exception:
            continue

    print("\n── Timings par étape (côté serveur) ──")
    for k, vals in step_data.items():
        if vals:
            s = sorted(vals)
            print(f"  {k:<22} mean={statistics.mean(vals):.3f}ms   p95={s[int(len(s)*0.95)]:.3f}ms")


if __name__ == "__main__":
    main()
