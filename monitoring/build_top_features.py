from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import joblib
import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", default="reports/baseline_ref.csv")
    ap.add_argument("--model", default="src/app/artifacts/model.pkl")
    ap.add_argument("--feature-cols", default="src/app/artifacts/feature_cols.json")
    ap.add_argument("--out", default="src/app/artifacts/top_features.json")
    ap.add_argument("--top-k", type=int, default=30)
    ap.add_argument("--sample", type=int, default=5000)
    args = ap.parse_args()

    baseline_path = Path(args.baseline)
    model_path = Path(args.model)
    feature_cols_path = Path(args.feature_cols)
    out_path = Path(args.out)

    feature_cols = json.loads(feature_cols_path.read_text(encoding="utf-8"))
    df = pd.read_csv(baseline_path)

    # garde uniquement les colonnes attendues
    df = df[[c for c in feature_cols if c in df.columns]].copy()

    # petit sample pour aller vite
    if len(df) > args.sample:
        df = df.sample(args.sample, random_state=42)

    model = joblib.load(model_path)

    # proba du modèle sur baseline
    proba = pd.Series(model.predict_proba(df)[:, 1], name="proba")

    scores = []
    for col in df.columns:
        s = pd.to_numeric(df[col], errors="coerce")  # non-num -> NaN
        valid = s.notna() & proba.notna()
        if valid.sum() < 50:
            continue
        corr = s[valid].corr(proba[valid], method="spearman")
        if pd.isna(corr):
            continue
        scores.append((col, float(abs(corr))))

    scores.sort(key=lambda x: x[1], reverse=True)
    top = [c for c, _ in scores[: args.top_k]]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(top, indent=2, ensure_ascii=False), encoding="utf-8")

    print("resultats:", out_path)
    print("top_k:", len(top))
    print("examples:", top[:10])


if __name__ == "__main__":
    main()
