from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import psycopg
from evidently import Report
from evidently.presets import DataDriftPreset


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def ensure_outdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def to_numeric_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out = out.replace([np.inf, -np.inf], np.nan)
    return out


def psi(expected: np.ndarray, actual: np.ndarray, buckets: int = 10, eps: float = 1e-6) -> float:
    expected = expected[~np.isnan(expected)]
    actual = actual[~np.isnan(actual)]
    if len(expected) == 0 or len(actual) == 0:
        return np.nan

    qs = np.linspace(0, 1, buckets + 1)
    cuts = np.unique(np.quantile(expected, qs))
    if len(cuts) < 3:
        return np.nan

    exp_counts, _ = np.histogram(expected, bins=cuts)
    act_counts, _ = np.histogram(actual, bins=cuts)

    exp_perc = exp_counts / max(exp_counts.sum(), 1)
    act_perc = act_counts / max(act_counts.sum(), 1)

    exp_perc = np.clip(exp_perc, eps, 1)
    act_perc = np.clip(act_perc, eps, 1)

    return float(np.sum((act_perc - exp_perc) * np.log(act_perc / exp_perc)))


def fetch_predictions(conn, limit: int = 2000) -> pd.DataFrame:
    """
    Récupère les prédictions prod depuis Postgres.
    IMPORTANT: on utilise la colonne `top_features` (JSONB) directement.
    """
    q = """
    SELECT
      request_id, ts, client_id, model_name, model_version,
      latency_ms, status_code,
      threshold, fn_cost, fp_cost,
      proba, prediction, decision,
      missing_rate, missing_count, n_features_sent,
      payload_hash,
      top_features
    FROM predictions
    ORDER BY ts DESC
    LIMIT %s;
    """
    rows: List[Dict[str, Any]] = []
    with conn.cursor() as cur:
        cur.execute(q, (limit,))
        for r in cur.fetchall():
            rows.append(
                {
                    "request_id": r[0],
                    "ts": r[1],
                    "client_id": r[2],
                    "model_name": r[3],
                    "model_version": r[4],
                    "latency_ms": r[5],
                    "status_code": r[6],
                    "threshold": r[7],
                    "fn_cost": r[8],
                    "fp_cost": r[9],
                    "proba": r[10],
                    "prediction": r[11],
                    "decision": r[12],
                    "missing_rate": r[13],
                    "missing_count": r[14],
                    "n_features_sent": r[15],
                    "payload_hash": r[16],
                    "top_features": r[17] or None,  # dict ou None
                }
            )
    return pd.DataFrame(rows)


def fetch_errors(conn, limit: int = 2000) -> pd.DataFrame:
    q = """
    SELECT request_id, ts, latency_ms, status_code, error_json
    FROM errors
    ORDER BY ts DESC
    LIMIT %s;
    """
    rows: List[Dict[str, Any]] = []
    with conn.cursor() as cur:
        cur.execute(q, (limit,))
        for r in cur.fetchall():
            rows.append(
                {
                    "request_id": r[0],
                    "ts": r[1],
                    "latency_ms": r[2],
                    "status_code": r[3],
                    "error_json": r[4] or {},
                }
            )
    return pd.DataFrame(rows)


def compute_ops_summary(pred_df: pd.DataFrame, err_df: pd.DataFrame) -> Dict[str, Any]:
    n_ok = int(len(pred_df))
    n_err = int(len(err_df))
    total = n_ok + n_err
    err_rate = float(n_err / total) if total else 0.0

    lat_ok = pred_df["latency_ms"].dropna().astype(float) if n_ok else pd.Series([], dtype=float)
    lat_err = err_df["latency_ms"].dropna().astype(float) if n_err else pd.Series([], dtype=float)
    lat_all = pd.concat([lat_ok, lat_err], ignore_index=True)

    return {
        "n_requests": total,
        "n_ok": n_ok,
        "n_errors": n_err,
        "error_rate": err_rate,
        "latency_ms_p50": float(lat_all.quantile(0.50)) if len(lat_all) else None,
        "latency_ms_p95": float(lat_all.quantile(0.95)) if len(lat_all) else None,
        "latency_ms_max": float(lat_all.max()) if len(lat_all) else None,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dsn", required=True)
    ap.add_argument("--baseline", default="reports/baseline_ref.csv")
    ap.add_argument("--top-features", default="src/app/artifacts/top_features.json")
    ap.add_argument("--out", default="reports/monitoring")
    ap.add_argument("--limit", type=int, default=2000)
    ap.add_argument("--min-non-null", type=int, default=30)
    args = ap.parse_args()

    out_dir = Path(args.out)
    ensure_outdir(out_dir)

    top_features: List[str] = read_json(Path(args.top_features))
    if not top_features:
        raise RuntimeError("top_features.json est vide -> impossible de faire un drift sur top features.")

    # --- Baseline ---
    baseline = pd.read_csv(args.baseline)
    baseline_top = baseline.reindex(columns=top_features)
    baseline_top = to_numeric_df(baseline_top)

    # --- Prod depuis Postgres ---
    with psycopg.connect(args.dsn) as conn:
        pred_df = fetch_predictions(conn, limit=args.limit)
        err_df = fetch_errors(conn, limit=args.limit)

    # --- Ops summary ---
    ops_summary = compute_ops_summary(pred_df, err_df)
    (out_dir / "ops_summary.json").write_text(
        json.dumps(ops_summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # --- Construire prod_top (à partir de predictions.top_features) ---
    pred_with_top = pred_df[pred_df["top_features"].notna()].copy()
    if len(pred_with_top) == 0:
        raise SystemExit(
            "Aucune ligne avec top_features en base. "
            "Relance l’API + génère des appels, puis ré-ingest."
        )

    prod_top = pd.json_normalize(pred_with_top["top_features"].tolist())
    prod_top = prod_top.reindex(columns=top_features)
    prod_top = to_numeric_df(prod_top)

    # --- Choisir les colonnes utilisables (évite l’erreur Evidently 'empty column') ---
    min_non_null = int(args.min_non_null)
    valid_cols: List[str] = []
    dropped: Dict[str, List[str]] = {
        "dropped_current_empty": [],
        "dropped_reference_empty": [],
        "dropped_too_few_values": [],
    }

    for c in top_features:
        ref_n = int(baseline_top[c].notna().sum())
        cur_n = int(prod_top[c].notna().sum())
        if ref_n == 0:
            dropped["dropped_reference_empty"].append(c)
            continue
        if cur_n == 0:
            dropped["dropped_current_empty"].append(c)
            continue
        if ref_n < min_non_null or cur_n < min_non_null:
            dropped["dropped_too_few_values"].append(c)
            continue
        valid_cols.append(c)

    (out_dir / "dropped_columns.json").write_text(
        json.dumps(
            {
                "min_non_null_required": min_non_null,
                "valid_cols": valid_cols,
                **dropped,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    if len(valid_cols) == 0:
        raise SystemExit(
            "Aucune feature utilisable pour Evidently  "
            
        )

    # --- PSI  ---
    psi_rows = []
    for c in valid_cols:
        p = psi(
            baseline_top[c].to_numpy(dtype=float),
            prod_top[c].to_numpy(dtype=float),
        )
        psi_rows.append({"feature": c, "psi": p})

    psi_df = pd.DataFrame(psi_rows).sort_values("psi", ascending=False)
    psi_df.to_csv(out_dir / "drift_psi.csv", index=False)

    # --- Evidently report (HTML + JSON) ---
    report = Report([DataDriftPreset()])

    # Evidently 0.7.x: report.run(...) retourne l’objet "my_report" qui a save_html/save_json
    try:
        my_report = report.run(reference_data=baseline_top[valid_cols], current_data=prod_top[valid_cols])
    except TypeError:
        my_report = report.run(prod_top[valid_cols], baseline_top[valid_cols])

    my_report.save_html(str(out_dir / "evidently_drift_report.html"))
    my_report.save_json(str(out_dir / "evidently_drift_report.json"))

    print("✅ Monitoring outputs écrits dans:", out_dir)
    print(" - ops_summary.json")
    print(" - dropped_columns.json")
    print(" - drift_psi.csv")
    print(" - evidently_drift_report.html / .json")


if __name__ == "__main__":
    main()
