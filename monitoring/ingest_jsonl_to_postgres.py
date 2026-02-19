from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable

import psycopg
from psycopg.types.json import Json


# -----------------------
# IO helpers
# -----------------------
def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)

def parse_ts(s: str) -> datetime:
    """
    Parse ISO timestamps from API logs.
    Supports "...Z" by converting to "+00:00".
    """
    s = s.strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    return datetime.fromisoformat(s)


# -----------------------
# Ingest predictions
# -----------------------
def ingest_predictions(conn: psycopg.Connection, pred_path: Path) -> int:
    sql = """
    INSERT INTO predictions (
        request_id, ts, client_id, model_name, model_version,
        threshold, fn_cost, fp_cost,
        latency_ms, status_code,
        proba, prediction, decision,
        missing_rate, missing_count, n_features_sent, payload_hash,
        top_features, input_json, output_json,
        hash_ms, preprocessing_ms, stats_ms, inference_ms, logging_ms
    ) VALUES (
        %(request_id)s, %(ts)s, %(client_id)s, %(model_name)s, %(model_version)s,
        %(threshold)s, %(fn_cost)s, %(fp_cost)s,
        %(latency_ms)s, %(status_code)s,
        %(proba)s, %(prediction)s, %(decision)s,
        %(missing_rate)s, %(missing_count)s, %(n_features_sent)s, %(payload_hash)s,
        %(top_features)s, %(input_json)s, %(output_json)s,
        %(hash_ms)s, %(preprocessing_ms)s, %(stats_ms)s, %(inference_ms)s, %(logging_ms)s
    )
    ON CONFLICT (request_id) DO NOTHING;
    """

    n = 0
    with conn.cursor() as cur:
        for rec in iter_jsonl(pred_path):
            inp = rec.get("input", {}) or {}
            out = rec.get("output", {}) or {}
            tim = rec.get("timings", {}) or {}

            top = inp.get("top_features")

            top_json = Json(top) if isinstance(top, dict) and len(top) > 0 else None 

            row = {
                "request_id": rec["request_id"],
                "ts": parse_ts(rec["timestamp"]),
                "client_id": rec.get("client_id"),
                "model_name": rec.get("model_name"),
                "model_version": rec.get("model_version"),

                "threshold": rec.get("threshold"),
                "fn_cost": rec.get("fn_cost"),
                "fp_cost": rec.get("fp_cost"),

                "latency_ms": rec.get("latency_ms"),
                "status_code": int(rec.get("status_code", 200) or 200),

                "proba": out.get("probability_default"),
                "prediction": out.get("prediction"),
                "decision": out.get("decision"),

                "missing_rate": inp.get("missing_rate_aligned"),
                "missing_count": inp.get("missing_count_aligned"),
                "n_features_sent": inp.get("n_features_sent"),
                "payload_hash": inp.get("payload_hash"),

                "top_features": top_json,
                "input_json": Json(inp),
                "output_json": Json(out),

                "hash_ms": tim.get("hash_ms"),
                "preprocessing_ms": tim.get("preprocessing_ms"),
                "stats_ms": tim.get("stats_ms"),
                "inference_ms": tim.get("inference_ms"),
                "logging_ms": tim.get("logging_ms"),
            }

            cur.execute(sql, row)
            n += 1

    conn.commit()
    return n


# -----------------------
# Ingest errors
# -----------------------
def ingest_errors(conn: psycopg.Connection, err_path: Path) -> int:
    sql = """
    INSERT INTO errors (
        request_id, ts, client_id, model_name, model_version,
        latency_ms, status_code, payload_hash,
        error_json, input_json
    ) VALUES (
        %(request_id)s, %(ts)s, %(client_id)s, %(model_name)s, %(model_version)s,
        %(latency_ms)s, %(status_code)s, %(payload_hash)s,
        %(error_json)s, %(input_json)s
    )
    ON CONFLICT (request_id) DO NOTHING;
    """

    n = 0
    with conn.cursor() as cur:
        for rec in iter_jsonl(err_path):
            inp = rec.get("input", {}) or {}

            row = {
                "request_id": rec["request_id"],
                "ts": parse_ts(rec["timestamp"]),
                "client_id": rec.get("client_id"),
                "model_name": rec.get("model_name"),
                "model_version": rec.get("model_version"),
                "latency_ms": rec.get("latency_ms"),
                "status_code": int(rec.get("status_code", 500) or 500),
                "payload_hash": inp.get("payload_hash"),
                "error_json": Json(rec.get("error")),
                "input_json": Json(inp),
            }

            cur.execute(sql, row)
            n += 1

    conn.commit()
    return n


# -----------------------
# CLI
# -----------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dsn", required=True, help="ex: postgresql://homecredit:homecredit@localhost:5432/monitoring")
    ap.add_argument("--pred", default="prod_logs/predictions.jsonl")
    ap.add_argument("--err", default="prod_logs/errors.jsonl")
    args = ap.parse_args()

    pred_path = Path(args.pred)
    err_path = Path(args.err)

    with psycopg.connect(args.dsn) as conn:
        n_pred = ingest_predictions(conn, pred_path)
        n_err = ingest_errors(conn, err_path)

    print(f"inserted predictions: {n_pred}")
    print(f"inserted errors     : {n_err}")


if __name__ == "__main__":
    main()
