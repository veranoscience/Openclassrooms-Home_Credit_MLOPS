# monitoring/generate_prod_calls.py
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import httpx


def to_jsonable(v: Any):
    # NaN -> None pour JSON
    if pd.isna(v):
        return None
    # numpy scalars -> python scalars
    if isinstance(v, np.generic):
        return v.item()
    return v


def apply_drift(row: pd.Series, rng: np.random.Generator) -> pd.Series:
    """
    Drift léger et réaliste:
    - augmente / diminue un peu certains montants
    - change légèrement les EXT_SOURCE
    - injecte un peu plus de missing sur quelques colonnes 
    """
    r = row.copy()

    # drift sur variables numériques si elles existent
    for col, (mu, sigma) in {
        "AMT_INCOME_TOTAL": (1.05, 0.10),
        "AMT_CREDIT": (1.03, 0.08),
        "AMT_ANNUITY": (1.02, 0.06),
        "INCOME_PER_PERSON": (1.04, 0.10),
    }.items():
        if col in r.index and pd.notna(r[col]):
            factor = rng.normal(mu, sigma)
            r[col] = max(0.0, float(r[col]) * float(factor))

    # drift EXT_SOURCE (clamp 0..1)
    for col in ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]:
        if col in r.index and pd.notna(r[col]):
            r[col] = float(np.clip(float(r[col]) + rng.normal(0.0, 0.03), 0.0, 1.0))

    # exemple: un peu plus de valeurs manquantes (5% des colonnes)
    idx = r.index.to_list()
    if len(idx) > 0:
        n_drop = int(0.05 * len(idx))
        if n_drop > 0:
            drop_cols = rng.choice(idx, size=n_drop, replace=False)
            for c in drop_cols:
                r[c] = np.nan

    return r


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", type=str, default="http://127.0.0.1:8000", help="Base URL de l'API")
    parser.add_argument("--n", type=int, default=200, help="Nombre de requêtes à envoyer")
    parser.add_argument(
        "--baseline",
        type=str,
        default="monitoring/storage/baseline_ref.csv",
        help="CSV baseline_ref (features de référence)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--drift", action="store_true", help="Active la simulation de drift")
    parser.add_argument("--timeout", type=float, default=30.0)
    args = parser.parse_args()

    base_url = args.url.rstrip("/")
    baseline_path = Path(args.baseline)

    if not baseline_path.exists():
        raise FileNotFoundError(f"Baseline introuvable: {baseline_path}")

    df = pd.read_csv(baseline_path)
    if df.shape[0] == 0:
        raise ValueError("baseline_ref.csv est vide")

    rng = np.random.default_rng(args.seed)

    # sample rows avec replacement si n > len(df)
    take = df.sample(n=args.n, replace=(args.n > len(df)), random_state=args.seed)

    ok, ko = 0, 0
    latencies = []

    with httpx.Client(timeout=args.timeout) as client:
        for i, (_, row) in enumerate(take.iterrows(), start=1):
            if args.drift:
                row = apply_drift(row, rng)

            features = {k: to_jsonable(v) for k, v in row.to_dict().items()}
            payload = {"features": features}

            t0 = time.perf_counter()
            try:
                r = client.post(f"{base_url}/predict", json=payload)
                dt_ms = (time.perf_counter() - t0) * 1000.0
                latencies.append(dt_ms)

                if r.status_code == 200:
                    ok += 1
                else:
                    ko += 1
                    # print un exemple d'erreur
                    if ko <= 3:
                        print(f"[{i}] KO {r.status_code}: {r.text[:200]}")
            except Exception as e:
                ko += 1
                if ko <= 3:
                    print(f"[{i}] Exception: {e}")

            if i % 25 == 0:
                print(f"Progress: {i}/{args.n} | ok={ok} ko={ko}")

    if latencies:
        print("\n--- Résumé ---")
        print(f"ok={ok} | ko={ko}")
        print(f"latency_ms p50={np.percentile(latencies, 50):.1f} | p95={np.percentile(latencies, 95):.1f} | mean={np.mean(latencies):.1f}")
        print("\n➡️ Tes logs API doivent maintenant contenir plus de lignes :")
        print("   - prod_logs/predictions.jsonl")
        print("   - prod_logs/errors.jsonl (si erreurs)")
    else:
        print("Aucune latence mesurée (aucune requête passée).")


if __name__ == "__main__":
    main()
