import cProfile
import pstats
import json
from pathlib import Path

import joblib
import pandas as pd
import numpy as np

ART_DIR = Path("src/app/artifacts")
MODEL = joblib.load(ART_DIR / "model.pkl")
FEATURE_COLS = json.loads((ART_DIR / "feature_cols.json").read_text(encoding="utf-8"))

payload = json.loads(Path("tests/sample_payload.json").read_text(encoding="utf-8"))["features"]

# --- BEFORE: DataFrame depuis dict  ---
def make_df_slow(features: dict) -> pd.DataFrame:
    row = {c: features.get(c, None) for c in FEATURE_COLS}
    return pd.DataFrame([row], columns=FEATURE_COLS)

# --- AFTER: NumPy -> DataFrame ---
feat_idx = {c: i for i, c in enumerate(FEATURE_COLS)}

def make_df_fast(features: dict) -> pd.DataFrame:
    x = np.full((1, len(FEATURE_COLS)), np.nan, dtype=np.float32)
    for k, v in features.items():
        i = feat_idx.get(k)
        if i is None or v is None:
            continue
        try:
            fv = float(v)
        except Exception:
            continue
        if np.isfinite(fv):
            x[0, i] = fv
    return pd.DataFrame(x, columns=FEATURE_COLS)

def run(mode: str, n: int) -> None:
    maker = make_df_slow if mode == "slow" else make_df_fast
    for _ in range(n):
        X = maker(payload)
        _ = MODEL.predict_proba(X)

def profile(mode: str, n: int, out_txt: Path) -> None:
    maker = make_df_slow if mode == "slow" else make_df_fast

    # Warmup (non profilé)
    for _ in range(20):
        X = maker(payload)
        _ = MODEL.predict_proba(X)

    pr = cProfile.Profile()
    pr.enable()

    for _ in range(n):
        X = maker(payload)
        _ = MODEL.predict_proba(X)

    pr.disable()
    stats = pstats.Stats(pr).sort_stats("cumtime")
    out_txt.parent.mkdir(parents=True, exist_ok=True)
    with out_txt.open("w", encoding="utf-8") as f:
        stats.stream = f
        stats.print_stats(40)

if __name__ == "__main__":
    
    profile("slow", 1000, Path("reports/profiling/cprofile_before.txt"))
    profile("fast", 1000, Path("reports/profiling/cprofile_after.txt"))
    print("OK: reports/profiling/cprofile_before.txt et cprofile_after.txt générés")
