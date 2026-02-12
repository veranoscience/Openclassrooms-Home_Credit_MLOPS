import json
from pathlib import Path
import pandas as pd
import numpy as np

n_ref = 50000
seed=42

data_train = Path("data/processed/train_final.csv")
out_dir = Path("reports")

def build_stats (df: pd.DataFrame):
    stats = {
        "n_rows": int(len(df)),
        "n_cols": int(df.shape[1]),
        "cols": {},
    }

    for c in df.columns:
        s = df[c]
        miss = float(s.isna().mean())
        info = {"missing_rate": miss}

        if pd.api.types.is_numeric_dtype(s):
            s2 = s.replace([np.inf, -np.inf], np.nan).dropna()
            info.update({
                "type": "numeric",
                "mean": float(s2.mean()) if len(s2) else None,
                "std": float(s2.std()) if len(s2) else None,
                "q05": float(s2.quantile(0.05)) if len(s2) else None,
                "q50": float(s2.quantile(0.50)) if len(s2) else None,
                "q95": float(s2.quantile(0.95)) if len(s2) else None,
            })
        else:
            vc = s.astype("object").fillna("<<NA>>").value_counts(normalize=True).head(10)
            info.update({
                "type": "categorical",
                "top_values": {str(k): float(v) for k, v in vc.items()}
            })

        stats["cols"][c] = info

    return stats 

def main():
    train = pd.read_csv(data_train)
    X = train.drop(columns=["TARGET", "SK_ID_CURR"], errors="ignore")

    ref = X.sample(n=min(n_ref, len(X)), random_state=seed)
    ref.to_csv(out_dir / "baseline_ref.csv", index=False)

    baseline_stats = build_stats(ref)
    (out_dir / "baseline_stats.json").write_text(json.dumps(baseline_stats, indent=2, ensure_ascii=False))

    print ("Baseline ref est géneré dans reports")

if __name__ == "__main__":
    main()

