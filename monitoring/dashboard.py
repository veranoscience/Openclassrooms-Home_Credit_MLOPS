from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import streamlit as st

DEFAULT_DIR = Path("reports/monitoring")

ALERT_ERROR_RATE = 0.05
ALERT_LAT_P95_MS = 200.0
ALERT_PSI_WARN = 0.20
ALERT_PSI_CRIT = 0.30


def read_json(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))


def psi_level(psi: float) -> str:
    if pd.isna(psi):
        return "NA"
    if psi >= ALERT_PSI_CRIT:
        return "CRITIQUE"
    if psi >= ALERT_PSI_WARN:
        return "WARNING"
    return "OK"


# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="HomeCredit – Surveillance du modèle", layout="wide")

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.header("Configuration")
out_dir = Path(st.sidebar.text_input("Dossier monitoring", str(DEFAULT_DIR))).resolve()

ops_path  = out_dir / "ops_summary.json"
psi_path  = out_dir / "drift_psi.csv"
drop_path = out_dir / "dropped_columns.json"
html_path = out_dir / "evidently_drift_report.html"

if not ops_path.exists():
    st.error(f"Fichier manquant : {ops_path}")
    st.stop()

ops    = read_json(ops_path)
n_req  = ops.get("n_requests", 0)
err_rt = ops.get("error_rate", 0.0) or 0.0
lat_p95 = ops.get("latency_ms_p95", None)

# ── Titre ─────────────────────────────────────────────────────────────────────
st.title(" Rapport de surveillance du modèle de scoring – Home Credit \n"
          )

# ── Bloc d'introduction ────────────────────────────────────
st.subheader ("Contexte\n"
              "**Date de génération:** 19/02/2026  \n"
              "**Modèle surveillé:** XGBoost -  Scoring risque de défaut de paiment \n")
st.info(

    "Home Credit distribue des prêts à la consommation et utilise un modèle de scoring pour estimer la probabilité qu'un client ne rembourse pas son crédit. \n" 
    "Ce rapport vérifie deux choses :\n"
    "- **Les données** : est-ce que les données reçues en production restent proches des données d'entraînement **(data drift)**?  \n"
    "- **L'API** : est-ce que le modèle répond correctement et rapidement **(monitoring opérationnel)** ?  \n"
    "L'analyse est générée automatiquement via un script Python combinant calcul de PSI, distance de Wasserstein, et le framework Evidently"
)

# ── Carte : Contexte de l'analyse ─────────────────────────────────────────────
st.header("1. Jeux de données analysés")

def _fmt_date(iso: str | None) -> str:
    """Convertit une date ISO en format lisible (JJ/MM/AAAA HH:MM)."""
    if not iso:
        return "—"
    try:
        return pd.to_datetime(iso, utc=True).strftime("%d/%m/%Y %H:%M")
    except Exception:
        return iso

baseline_n   = ops.get("baseline_n_rows")
baseline_src = ops.get("baseline_source", "baseline_ref.csv")
prod_n       = ops.get("prod_n_rows_for_drift")
prod_limit   = ops.get("prod_limit_requested")
date_min     = _fmt_date(ops.get("prod_date_min"))
date_max     = _fmt_date(ops.get("prod_date_max"))

# Fallback baseline : lire baseline_stats.json si le champ est absent
if baseline_n is None:
    stats_path = out_dir.parent / "baseline_stats.json"
    if stats_path.exists():
        try:
            baseline_n = read_json(stats_path).get("n_rows")
        except Exception:
            pass

# Fallback période : date de modification d'ops_summary.json
if date_min == "—" and date_max == "—":
    try:
        mtime = os.path.getmtime(ops_path)
        analysis_date = pd.Timestamp(mtime, unit="s", tz="UTC").strftime("%d/%m/%Y à %H:%M")
    except Exception:
        analysis_date = None
else:
    analysis_date = None

ctx_left, ctx_right = st.columns(2)

with ctx_left:
    st.markdown("**Données de référence (baseline)**")
    st.markdown(
        "Ce sont les données sur lesquelles le modèle a été entraîné. "
        "Elles servent de point de comparaison "
    )
    if baseline_n is not None:
        st.metric("Nombre de lignes", f"{baseline_n:,}".replace(",", " "))
    else:
        st.metric("Nombre de lignes", "—")
    st.caption(f"Source : `{Path(baseline_src).name}`")

with ctx_right:
    st.markdown("**Données actuelles (production)**")
    st.markdown(
        "Ce sont les vraies demandes de crédit reçues par le modèle en production "
       
    )
    if prod_n is not None:
        limit_note = f" (max {prod_limit:,} demandées)".replace(",", " ") if prod_limit else ""
        st.metric("Nombre de lignes analysées", f"{prod_n:,}{limit_note}".replace(",", " "))
    else:
        st.metric("Nombre de requêtes reçues", f"{n_req:,}".replace(",", " "))
    if date_min != "—" or date_max != "—":
        st.caption(f"Période couverte : du **{date_min}** au **{date_max}**")
    elif analysis_date:
        st.caption(
            f"Analyse générée le **{analysis_date}**.  \n"
            "Source `run_monitoring_analysis.py`"
        )
    else:
        st.caption("Période non disponible — relancez l'analyse de monitoring")

# ══════════════════════════════════════════════════════════════════════════════
# RÉSUMÉ Exécutif
# ══════════════════════════════════════════════════════════════════════════════
st.header(" 2. Résumé global")

psi_ok = True
n_crit = 0
n_warn = 0

if psi_path.exists():
    psi_df = pd.read_csv(psi_path)
    psi_df["level"] = psi_df["psi"].apply(psi_level)
    n_crit = int((psi_df["level"] == "CRITIQUE").sum())
    n_warn = int((psi_df["level"] == "WARNING").sum())
    psi_ok = n_crit == 0

api_ok = err_rt < ALERT_ERROR_RATE and (lat_p95 is None or lat_p95 < ALERT_LAT_P95_MS)

if api_ok and psi_ok and n_warn == 0:
    st.success("Tout est normal. Le modèle fonctionne bien et les données sont stables.")
elif n_crit > 0:
    st.error(
        f"Alerte : {n_crit} variable(s) ont changé de façon importante. "
        "Le modèle pourrait donner de moins bons résultats"
    )
elif n_warn > 0:
    st.warning(
        f"Attention : {n_warn} variable(s) ont légèrement changé par rapport à la normale. "
        "Pas d'urgence, mais à surveiller."
    )
elif not api_ok:
    st.warning("L'API rencontre des lenteurs ou des erreurs")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — DATA DRIFT
# ══════════════════════════════════════════════════════════════════════════════
st.header("3. Analyse de la derive des données (data drift) ")

with st.expander("Explication de data drift", expanded=True):
    st.markdown(
        """
Un modèle est entraîné sur des données historiques. Si les nouvelles données lui ressemblent,
il fait de bonnes prédictions. Si elles ont trop changé, ses prédictions deviennent moins fiables.

| Niveau | Signification | Action suggérée |
|---|---|---|
| ✅ Stable | La variable n'a pas changé | Rien à faire |
| ⚠️ À surveiller | Léger changement détecté | Observer les prochains jours |
| 🔴 Critique | Changement important | Investiguer et envisager un ré-entraînement |
"""
    )

if not psi_path.exists():
    st.warning("Données de drift non disponibles. Lancez d'abord l'analyse de monitoring")
else:
    # KPIs drift – langage simple
    total = len(psi_df)
    n_ok  = total - n_warn - n_crit

    col1, col2, col3 = st.columns(3)
    col1.metric("Variables stables", f"{n_ok} / {total}", help="Aucun changement significatif détecté")
    col2.metric("À surveiller", f"{n_warn}", help="Changement léger — PSI entre 0.20 et 0.30")
    col3.metric("Changement important", f"{n_crit}", help="Changement fort — PSI > 0.30")

    # Graphique à barres — Top 10 variables les plus dérivées
    st.subheader("3.1 Top 10 variables les plus impactées")
    st.caption(
        "Chaque barre montre l'intensité du changement pour une variable "
        "Plus la barre est longue, plus la variable a évolué par rapport à la normale"
    )

    top10 = psi_df.sort_values("psi", ascending=False).head(10).copy()
    top10["label"] = top10["level"].map(
        {"OK": "✅ Stable", "WARNING": "⚠️ À surveiller", "CRITIQUE": "🔴 Critique", "NA": "—"}
    )

    # Couleur simulée via une colonne numérique de seuil
    chart_df = top10.set_index("feature")[["psi"]].rename(columns={"psi": "Score de changement (PSI)"})
    st.bar_chart(chart_df)

    # Tableau simplifié
    st.subheader("3.2 Détail par variable")
    display_df = top10[["feature", "psi", "label"]].copy()
    display_df.columns = ["Variable", "Score de changement (PSI)", "Niveau"]
    display_df["Score de changement (PSI)"] = display_df["Score de changement (PSI)"].round(3)
    display_df = display_df.reset_index(drop=True)
    st.dataframe(display_df, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — SANTÉ DE L'API
# ══════════════════════════════════════════════════════════════════════════════
st.header("4. Monitoring opérationnel de l'API")
st.caption("Cette section vérifie que le modèle répond correctement aux requêtes")

lat_p50 = ops.get("latency_ms_p50", None)
errors_est = int(round(err_rt * n_req)) if n_req else 0

a1, a2, a3 = st.columns(3)

a1.metric(
    "Requêtes reçues",
    f"{n_req}",
    help="Nombre total de fois que le modèle a été sollicité sur la période",
)

err_delta = None
if err_rt >= ALERT_ERROR_RATE:
    err_delta = f"Seuil dépassé ({ALERT_ERROR_RATE*100:.0f}% max)"
a2.metric(
    "Taux d'erreur",
    f"{err_rt*100:.1f}%",
    delta=err_delta,
    delta_color="inverse",
    help=f"Part des requêtes ayant échoué (≈ {errors_est} erreur(s) sur {n_req})",
)

lat_display = f"{lat_p95:.0f} ms" if lat_p95 is not None else "N/A"
lat_delta = None
if lat_p95 is not None and lat_p95 >= ALERT_LAT_P95_MS:
    lat_delta = f"Seuil dépassé ({ALERT_LAT_P95_MS:.0f} ms max)"
a3.metric(
    "Temps de réponse (95% des requêtes)",
    lat_display,
    delta=lat_delta,
    delta_color="inverse",
    help="95 % des requêtes sont traitées en moins de cette durée. En dessous de 200 ms, c'est bon",
)

if api_ok:
    st.success("L'API répond correctement et dans les temps")
else:
    if err_rt >= ALERT_ERROR_RATE:
        st.warning(f"Le taux d'erreur est élevé : {err_rt*100:.1f}% (seuil : {ALERT_ERROR_RATE*100:.0f}%)")
    if lat_p95 is not None and lat_p95 >= ALERT_LAT_P95_MS:
        st.warning(f"Le temps de réponse est trop lent : {lat_p95:.0f} ms (seuil : {ALERT_LAT_P95_MS:.0f} ms)")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — RAPPORT EXPERT 
# ══════════════════════════════════════════════════════════════════════════════
with st.expander("Rapport détaillé Evidently"):
    if html_path.exists():
        html = html_path.read_text(encoding="utf-8")
        st.components.v1.html(html, height=900, scrolling=True)
    else:
        st.info(
            "Rapport Evidently non généré.  \n"
            "Pour le créer : `python monitoring/run_monitoring_analysis.py`"
        )
