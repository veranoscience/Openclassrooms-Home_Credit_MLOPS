from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

DEFAULT_DIR = Path("reports/monitoring")

# Seuils d'alertes 
ALERT_ERROR_RATE = 0.05      
ALERT_LAT_P95_MS = 200.0     
ALERT_PSI_WARN = 0.20
ALERT_PSI_CRIT = 0.30


def read_json(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))


st.set_page_config(page_title="HomeCredit Monitoring", layout="wide")
st.title(" Monitoring — Home Credit Scoring")

# Sidebar
st.sidebar.header("Configuration")
out_dir = Path(st.sidebar.text_input("Dossier monitoring", str(DEFAULT_DIR))).resolve()

ops_path = out_dir / "ops_summary.json"
psi_path = out_dir / "drift_psi.csv"
drop_path = out_dir / "dropped_columns.json"
html_path = out_dir / "evidently_drift_report.html"

if not ops_path.exists():
    st.error(f"Fichier manquant: {ops_path}")
    st.stop()

ops = read_json(ops_path)

# --- KPIs Ops ---
st.subheader(" Santé de l'API (technique)")
c1, c2, c3, c4 = st.columns(4)


n_requests = ops.get("n_requests", 0)
error_rate = ops.get("error_rate", 0.0) or 0.0
lat_p50 = ops.get("latency_ms_p50", None)
lat_p95 = ops.get("latency_ms_p95", None)

errors_est = int(round(error_rate * n_requests)) if n_requests else 0

c1.metric("Requêtes reçues", f"{n_requests}")
c1.caption("Nombre total d’appels API sur la période analysée")

c2.metric("Requêtes en échec", f"{error_rate*100:.2f}%")
c2.caption(f"Part des requêtes qui échouent (≈ {errors_est} erreurs). Seuil: {ALERT_ERROR_RATE*100:.0f}% max.")

c3.metric("Temps de réponse typique (p50)", f"{lat_p50:.1f}" if lat_p50 is not None else "N/A")
c3.caption("Temps ‘normal’ pour répondre (médiane). Plus bas = mieux")

c4.metric("Temps de réponse élevé (p95)", f"{lat_p95:.1f}" if lat_p95 is not None else "N/A")
c4.caption(f"95% des requêtes sont plus rapids). Seuil: {ALERT_LAT_P95_MS:.0f} ms max.")


# Alertes ops
st.markdown("### Alertes (Ops)")
alerts = []
if error_rate >= ALERT_ERROR_RATE:
    alerts.append(f"Taux d'erreur élevé: {error_rate*100:.2f}% (seuil {ALERT_ERROR_RATE*100:.0f}%)")
if lat_p95 is not None and lat_p95 >= ALERT_LAT_P95_MS:
    alerts.append(f"Latence p95 élevée: {lat_p95:.1f} ms (seuil {ALERT_LAT_P95_MS:.0f} ms)")

if alerts:
    for a in alerts:
        st.warning(a)
else:
    st.success(" Aucun signal ops critique (selon les seuils actuels).")


# --- Drift (PSI) ---
st.subheader("Changement des données d’entrée ( data drift)")

with st.expander("C’est quoi le drift (PSI) ?"):
    st.markdown(f"""
Le **PSI** compare la distribution d'une feature entre une période **référence** et la période **actuelle**
- **OK** : PSI < {ALERT_PSI_WARN}
- **A surveiller** : PSI ≥ {ALERT_PSI_WARN}
- **Critique** : PSI ≥ {ALERT_PSI_CRIT} (risque fort sur le modèle)
""")
    
if not psi_path.exists():
    st.warning(f"Fichier drift PSI manquant: {psi_path}")
else:
    psi_df = pd.read_csv(psi_path)

    # Classifier drift
    def level(psi: float) -> str:
        if pd.isna(psi):
            return "NA"
        if psi >= ALERT_PSI_CRIT:
            return "CRITIQUE"
        if psi >= ALERT_PSI_WARN:
            return "WARNING"
        return "OK"

    psi_df["level"] = psi_df["psi"].apply(level)

    # KPIs drift
    n_warn = int((psi_df["level"] == "WARNING").sum())
    n_crit = int((psi_df["level"] == "CRITIQUE").sum())

    d1, d2, d3 = st.columns(3)
    d1.metric("Features analysées (PSI)", f"{len(psi_df)}")
    d2.metric("Drift WARNING (PSI>=0.20)", f"{n_warn}")
    d3.metric("Drift CRITIQUE (PSI>=0.30)", f"{n_crit}")
    
    psi_df["Niveau"] = psi_df["level"].map({
    "OK": "✅ Stable",
    "WARNING": "🟠 À surveiller",
    "CRITIQUE": "🔴 Critique",
    "NA": "—"
    })

    if n_crit > 0:
        st.error("Drift critique détecté sur au moins une feature (PSI>=0.30)")
    elif n_warn > 0:
        st.warning("Drift modéré détecté (PSI>=0.20)")
    else:
        st.success("Pas de drift notable (selon PSI)")

    st.markdown("### Top 15 features par PSI")
    st.dataframe(psi_df.sort_values("psi", ascending=False).head(15), use_container_width=True)


# --- Colonnes exclues ---
st.subheader("3) Features exclues du drift (qualité / manque de données)")
if drop_path.exists():
    drops = read_json(drop_path)
    st.write("Min non-null requis:", drops.get("min_non_null_required"))

    colA, colB, colC = st.columns(3)
    colA.write("**Vides en prod**")
    colA.write(drops.get("dropped_current_empty", []))

    colB.write("**Vides en baseline**")
    colB.write(drops.get("dropped_reference_empty", []))

    colC.write("**Pas assez de valeurs**")
    colC.write(drops.get("dropped_too_few_values", []))
else:
    st.info("dropped_columns.json non trouvé (ok si tu ne le génères pas).")


# --- Evidently report ---
st.subheader("4) Rapport Evidently")
if html_path.exists():
    html = html_path.read_text(encoding="utf-8")
    st.components.v1.html(html, height=900, scrolling=True)
else:
    st.warning(f"Rapport HTML Evidently non trouvé: {html_path}")
    st.write("Lance d'abord: monitoring/run_monitoring_analysis.py")
