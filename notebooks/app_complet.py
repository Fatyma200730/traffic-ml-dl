# ============================================================
#  APPLICATION STREAMLIT — Prédiction du Trafic Web
#  Version complète : Modèles + Comparaison + Anomalies
#  Projet PFE | Rachid Bijigune | Vala Bleu | 2025-2026
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings("ignore")

# Import XGBoost avec gestion d'erreur si non installé
try:
    from xgboost import XGBRegressor
    XGBOOST_OK = True
except ImportError:
    XGBOOST_OK = False

# ─────────────────────────────────────────────────────────────
# CONFIGURATION DE LA PAGE
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Prédiction Trafic Web — Vala Bleu",
    page_icon="📊",
    layout="wide"
)

# ─────────────────────────────────────────────────────────────
# CSS PERSONNALISÉ
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    html, body, [class*="css"] { font-family: 'Segoe UI', sans-serif; }

    .titre-principal {
        background: linear-gradient(135deg, #1F3864, #2E75B6);
        color: white; padding: 28px 40px;
        border-radius: 12px; text-align: center; margin-bottom: 25px;
    }
    .titre-principal h1 { margin: 0; font-size: 1.9rem; font-weight: 700; }
    .titre-principal p  { margin: 6px 0 0; font-size: 0.9rem; opacity: 0.85; }

    .kpi-card {
        background: white; border-radius: 10px; padding: 18px;
        text-align: center; border-left: 5px solid #2E75B6;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    }
    .kpi-label { font-size: 0.8rem; color: #666; margin-bottom: 5px;
                 text-transform: uppercase; letter-spacing: 0.5px; }
    .kpi-value { font-size: 1.7rem; font-weight: 700; color: #1F3864; }
    .kpi-icon  { font-size: 1.3rem; margin-bottom: 5px; }

    .section-titre {
        font-size: 1.1rem; font-weight: 600; color: #1F3864;
        border-left: 4px solid #2E75B6; padding-left: 12px;
        margin: 22px 0 12px;
    }

    .pred-box {
        border-radius: 10px; padding: 20px 28px; margin-top: 14px;
    }
    .pred-box-rf  { background:#f0f7f0; border: 2px solid #375623; }
    .pred-box-xgb { background:#f0f4ff; border: 2px solid #2E75B6; }
    .pred-box-bl  { background:#f5f5f5; border: 2px solid #888; }

    .pred-box h3 { margin: 0 0 5px; font-size: 1rem; }
    .pred-valeur { font-size: 2.2rem; font-weight: 700; }
    .pred-details { color: #555; font-size: 0.88rem; margin-top: 7px; }

    .anomalie-alerte {
        background: #fff3f3; border: 2px solid #c0392b;
        border-radius: 10px; padding: 16px 22px; margin-top: 12px;
    }
    .anomalie-ok {
        background: #f0f9f0; border: 2px solid #375623;
        border-radius: 10px; padding: 16px 22px; margin-top: 12px;
    }

    div.stButton > button {
        background: #1F3864; color: white; border: none;
        border-radius: 8px; padding: 12px 40px;
        font-size: 1rem; font-weight: 600; width: 100%;
    }
    div.stButton > button:hover { background: #2E75B6; }

    #MainMenu { visibility: hidden; }
    footer     { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# CONSTANTES COULEURS
# ─────────────────────────────────────────────────────────────
C_BASE = "#888888"   # Baseline
C_RF   = "#375623"   # Random Forest
C_XGB  = "#2E75B6"   # XGBoost
C_ANOM = "#c0392b"   # Anomalies


# ─────────────────────────────────────────────────────────────
# CHARGEMENT DES DONNÉES
# ─────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def charger_donnees():
    """Charge df_petit.csv ou génère des données simulées."""
    chemins = ["df_petit.csv", "data/df_petit.csv",
               "../data/df_petit.csv", "notebooks/df_petit.csv"]
    for chemin in chemins:
        try:
            return pd.read_csv(chemin), True
        except FileNotFoundError:
            continue

    # Données simulées (50 pages, 550 jours)
    np.random.seed(42)
    n_pages, n_jours = 50, 550
    dates = pd.date_range("2015-07-01", periods=n_jours, freq="D")
    noms  = [f"Article_Wikipedia_{i:03d}_fr.wikipedia.org" for i in range(n_pages)]
    rows  = {"Page": noms}
    for i, d in enumerate(dates):
        rows[str(d.date())] = [
            max(0, int(
                np.random.exponential(scale=150)
                + 80 * np.sin(i / 7)            # saisonnalité hebdo
                + 40 * np.sin(i / 30)           # saisonnalité mensuelle
                + np.random.normal(0, 20)        # bruit
            )) for _ in range(n_pages)
        ]
    return pd.DataFrame(rows), False


# ─────────────────────────────────────────────────────────────
# ENTRAÎNEMENT DES 3 MODÈLES
# ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def entrainer_modeles(cache_key):
    """
    Entraîne les 3 modèles sur toutes les pages du dataset.
    Retourne: dict de modèles, df_feat, FEATURES, métriques
    """
    df, _ = charger_donnees()
    date_cols = [c for c in df.columns if c != "Page"]

    # Format long
    df_long = df.fillna(0).melt(id_vars="Page", var_name="date", value_name="views")
    df_long["date"] = pd.to_datetime(df_long["date"])
    df_long = df_long.sort_values(["Page", "date"]).reset_index(drop=True)

    # 6 features temporelles
    df_long["lag_1"]       = df_long.groupby("Page")["views"].shift(1)
    df_long["lag_7"]       = df_long.groupby("Page")["views"].shift(7)
    df_long["lag_14"]      = df_long.groupby("Page")["views"].shift(14)
    df_long["mean_7"]      = df_long.groupby("Page")["views"].transform(
                                 lambda x: x.shift(1).rolling(7).mean())
    df_long["day_of_week"] = df_long["date"].dt.dayofweek
    df_long["month"]       = df_long["date"].dt.month
    df_feat = df_long.dropna().reset_index(drop=True)

    # Split chronologique (60 derniers jours = test)
    cutoff  = df_feat.groupby("Page")["date"].transform(
                  lambda x: x.sort_values().iloc[-60])
    train   = df_feat[df_feat["date"] < cutoff]
    test    = df_feat[df_feat["date"] >= cutoff]

    FEATURES = ["lag_1", "lag_7", "lag_14", "mean_7", "day_of_week", "month"]
    X_tr, y_tr = train[FEATURES], np.log1p(train["views"])
    X_te, y_te = test[FEATURES],  test["views"].values

    modeles   = {}
    metriques = {}

    # ── Baseline (lag_1) ──────────────────────────────────────
    y_base = test["lag_1"].values
    modeles["Baseline"] = None   # pas de modèle — juste lag_1
    metriques["Baseline"] = {
        "MAE" : round(mean_absolute_error(y_te, y_base), 2),
        "RMSE": round(np.sqrt(mean_squared_error(y_te, y_base)), 2),
        "R2"  : 0.00,
        "amélioration": 0.0,
    }

    # ── Random Forest ─────────────────────────────────────────
    rf = RandomForestRegressor(n_estimators=100, max_depth=10,
                               random_state=42, n_jobs=-1)
    rf.fit(X_tr, y_tr)
    y_rf = np.expm1(rf.predict(X_te))
    modeles["Random Forest"] = rf
    mae_rf = mean_absolute_error(y_te, y_rf)
    metriques["Random Forest"] = {
        "MAE" : round(mae_rf, 2),
        "RMSE": round(np.sqrt(mean_squared_error(y_te, y_rf)), 2),
        "R2"  : round(r2_score(y_te, y_rf), 4),
        "amélioration": round((metriques["Baseline"]["MAE"] - mae_rf)
                              / metriques["Baseline"]["MAE"] * 100, 1),
    }

    # ── XGBoost ───────────────────────────────────────────────
    if XGBOOST_OK:
        xgb = XGBRegressor(n_estimators=200, learning_rate=0.05,
                           max_depth=6, random_state=42, verbosity=0)
        xgb.fit(X_tr, y_tr)
        y_xgb = np.expm1(xgb.predict(X_te))
        modeles["XGBoost"] = xgb
        mae_xgb = mean_absolute_error(y_te, y_xgb)
        metriques["XGBoost"] = {
            "MAE" : round(mae_xgb, 2),
            "RMSE": round(np.sqrt(mean_squared_error(y_te, y_xgb)), 2),
            "R2"  : round(r2_score(y_te, y_xgb), 4),
            "amélioration": round((metriques["Baseline"]["MAE"] - mae_xgb)
                                  / metriques["Baseline"]["MAE"] * 100, 1),
        }

    return modeles, df_feat, FEATURES, metriques


# ─────────────────────────────────────────────────────────────
# PRÉDICTION D'UN MODÈLE POUR UNE PAGE
# ─────────────────────────────────────────────────────────────
def predire(page, modele_nom, modele, df_feat, features, horizon=14):
    """
    Retourne le vecteur de prédictions pour les 'horizon' prochains jours.
    Pour Baseline : y_pred = dernière valeur connue (lag_1).
    Pour RF/XGB   : transformation log1p → prédiction → expm1.
    """
    page_data = df_feat[df_feat["Page"] == page].sort_values("date")
    if len(page_data) < 14:
        return None

    derniers = page_data.tail(horizon)
    X_pred   = derniers[features]

    if modele_nom == "Baseline":
        return derniers["lag_1"].values.astype(float)
    else:
        return np.expm1(modele.predict(X_pred))


# ─────────────────────────────────────────────────────────────
# DÉTECTION D'ANOMALIES FUTURES (Z-score sur historique)
# ─────────────────────────────────────────────────────────────
def detecter_anomalies_futures(page, y_pred, df, seuil_z=2.0):
    """
    Compare les prédictions futures avec la distribution historique.
    Un jour prédit est "anomalie" si |z| > seuil_z.
    Retourne un tableau détaillé + liste des indices anormaux.
    """
    date_cols  = [c for c in df.columns if c != "Page"]
    historique = df[df["Page"] == page].iloc[0][date_cols].values.astype(float)
    hist_clean = historique[~np.isnan(historique)]

    mu  = hist_clean.mean()
    sig = hist_clean.std()
    if sig == 0:
        sig = 1

    z_scores   = (y_pred - mu) / sig
    est_anomal = np.abs(z_scores) > seuil_z

    types = []
    for z in z_scores:
        if z > seuil_z:
            types.append("🔴 Pic élevé")
        elif z < -seuil_z:
            types.append("🔵 Creux anormal")
        else:
            types.append("✅ Normal")

    return est_anomal, z_scores, types, mu, sig


# ─────────────────────────────────────────────────────────────
# GRAPHIQUE HISTORIQUE
# ─────────────────────────────────────────────────────────────
def fig_historique(page, df):
    """Courbe trafic historique (550 jours)."""
    date_cols = [c for c in df.columns if c != "Page"]
    vals      = df[df["Page"] == page].iloc[0][date_cols].values.astype(float)
    dates     = pd.to_datetime(date_cols)

    fig, ax = plt.subplots(figsize=(12, 3.5))
    fig.patch.set_facecolor("white"); ax.set_facecolor("#FAFAFA")
    ax.plot(dates, vals, color=C_XGB, lw=1.3, alpha=0.9)
    ax.fill_between(dates, vals, alpha=0.12, color=C_XGB)
    ax.set_xlabel("Date", fontsize=10); ax.set_ylabel("Vues", fontsize=10)
    ax.set_title(f"Historique — {page[:65]}", fontsize=11,
                 fontweight="bold", color="#1F3864")
    ax.grid(alpha=0.2, linestyle="--")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    plt.tight_layout(); return fig


# ─────────────────────────────────────────────────────────────
# GRAPHIQUE PRÉDICTION DU MODÈLE CHOISI + ANOMALIES
# ─────────────────────────────────────────────────────────────
def fig_prediction(page, df, y_pred, modele_nom, est_anomal, dates_futures):
    """Courbe historique récente + prédiction + anomalies futures."""
    date_cols = [c for c in df.columns if c != "Page"]
    vals  = df[df["Page"] == page].iloc[0][date_cols].values.astype(float)
    dates = pd.to_datetime(date_cols)

    n_hist    = 30
    hist_vals = vals[-n_hist:]; hist_dates = dates[-n_hist:]

    col = {"Random Forest": C_RF, "XGBoost": C_XGB, "Baseline": C_BASE}[modele_nom]

    fig, ax = plt.subplots(figsize=(12, 4))
    fig.patch.set_facecolor("white"); ax.set_facecolor("#FAFAFA")

    # Historique
    ax.plot(hist_dates, hist_vals, color="#1F3864", lw=1.8,
            marker="o", ms=3, label="Historique (30j)")

    # Prédiction normale
    ax.plot(dates_futures, y_pred, color=col, lw=2.2,
            linestyle="--", marker="s", ms=5, label=f"Prédiction {modele_nom}")

    # Intervalle de confiance
    ax.fill_between(dates_futures, y_pred * 0.85, y_pred * 1.15,
                    alpha=0.15, color=col)

    # Points anomalies en rouge
    idx_anom = np.where(est_anomal)[0]
    if len(idx_anom) > 0:
        ax.scatter([dates_futures[i] for i in idx_anom],
                   y_pred[idx_anom],
                   color=C_ANOM, zorder=6, s=120,
                   marker="^", label=f"Anomalies ({len(idx_anom)})")

    ax.axvline(dates[-1], color="#C55A11", lw=1.5,
               linestyle=":", alpha=0.7, label="Aujourd'hui")

    ax.set_xlabel("Date", fontsize=10); ax.set_ylabel("Vues", fontsize=10)
    ax.set_title(f"Prédiction — {modele_nom}", fontsize=11,
                 fontweight="bold", color="#1F3864")
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(alpha=0.2, linestyle="--")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    plt.tight_layout(); return fig


# ─────────────────────────────────────────────────────────────
# GRAPHIQUE COMPARAISON DES 3 MODÈLES
# ─────────────────────────────────────────────────────────────
def fig_comparaison_courbes(page, df, preds_dict, dates_futures):
    """Superposition des prédictions des 3 modèles."""
    date_cols = [c for c in df.columns if c != "Page"]
    vals  = df[df["Page"] == page].iloc[0][date_cols].values.astype(float)
    dates = pd.to_datetime(date_cols)

    n_hist    = 30
    hist_vals = vals[-n_hist:]; hist_dates = dates[-n_hist:]

    cols  = {"Baseline": C_BASE, "Random Forest": C_RF, "XGBoost": C_XGB}
    marks = {"Baseline": "o",    "Random Forest": "s",  "XGBoost": "^"}

    fig, ax = plt.subplots(figsize=(12, 4))
    fig.patch.set_facecolor("white"); ax.set_facecolor("#FAFAFA")

    ax.plot(hist_dates, hist_vals, color="#1F3864", lw=2,
            marker=".", ms=4, label="Historique réel")

    for nom, y_pred in preds_dict.items():
        if y_pred is not None:
            ax.plot(dates_futures, y_pred,
                    color=cols[nom], lw=2, linestyle="--",
                    marker=marks[nom], ms=5, label=nom)

    ax.axvline(dates[-1], color="#C55A11", lw=1.5,
               linestyle=":", alpha=0.7)
    ax.set_xlabel("Date", fontsize=10); ax.set_ylabel("Vues", fontsize=10)
    ax.set_title("Comparaison des prédictions — 3 modèles", fontsize=11,
                 fontweight="bold", color="#1F3864")
    ax.legend(fontsize=9); ax.grid(alpha=0.2, linestyle="--")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    plt.tight_layout(); return fig


# ─────────────────────────────────────────────────────────────
# GRAPHIQUE COMPARAISON MAE / RMSE
# ─────────────────────────────────────────────────────────────
def fig_metriques(metriques):
    """Barres horizontales MAE et R² pour les 3 modèles."""
    noms  = list(metriques.keys())
    maes  = [metriques[n]["MAE"]  for n in noms]
    r2s   = [metriques[n]["R2"]   for n in noms]
    cols  = [C_BASE, C_RF, C_XGB]

    fig, axes = plt.subplots(1, 2, figsize=(12, 3))
    fig.patch.set_facecolor("white")

    # MAE
    bars = axes[0].barh(noms, maes, color=cols, edgecolor="white", height=0.5)
    axes[0].set_title("MAE — plus bas = meilleur", fontsize=11,
                       fontweight="bold", color="#1F3864")
    axes[0].set_xlabel("MAE (vues/jour)")
    for bar, v in zip(bars, maes):
        axes[0].text(v + 1, bar.get_y() + bar.get_height()/2,
                     f"{v:.2f}", va="center", fontsize=10, fontweight="bold")
    axes[0].spines["top"].set_visible(False)
    axes[0].spines["right"].set_visible(False)

    # R²
    bars2 = axes[1].barh(noms, r2s, color=cols, edgecolor="white", height=0.5)
    axes[1].set_title("R² — plus haut = meilleur", fontsize=11,
                       fontweight="bold", color="#1F3864")
    axes[1].set_xlabel("R²")
    for bar, v in zip(bars2, r2s):
        axes[1].text(v + 0.005, bar.get_y() + bar.get_height()/2,
                     f"{v:.4f}", va="center", fontsize=10, fontweight="bold")
    axes[1].spines["top"].set_visible(False)
    axes[1].spines["right"].set_visible(False)

    plt.tight_layout(); return fig


# ═════════════════════════════════════════════════════════════
#  INTERFACE PRINCIPALE
# ═════════════════════════════════════════════════════════════

# ── 1. TITRE ─────────────────────────────────────────────────
st.markdown("""
<div class="titre-principal">
    <h1>📊 Prédiction et Analyse du Trafic Web</h1>
    <p>Vala Bleu — Solution Hébergement Web | Random Forest • XGBoost • Baseline | PFE 2025-2026</p>
</div>
""", unsafe_allow_html=True)


# ── Chargement données + modèles ─────────────────────────────
with st.spinner("Chargement des données et entraînement des modèles..."):
    df, donnees_reelles             = charger_donnees()
    modeles, df_feat, FEATURES, metriques = entrainer_modeles(str(len(df)))
    date_cols  = [c for c in df.columns if c != "Page"]
    liste_pages = sorted(df["Page"].tolist())

if not donnees_reelles:
    st.info("ℹ️ Fichier df_petit.csv introuvable — données simulées utilisées.")
if not XGBOOST_OK:
    st.warning("⚠️ XGBoost non installé. Lancez : pip install xgboost")


# ── 2. INFORMATIONS GÉNÉRALES ─────────────────────────────────
st.markdown('<p class="section-titre">Informations générales</p>',
            unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)
kpis = [
    (c1, "🤖", "Modèles disponibles",    "3"),
    (c2, "📉", "Meilleur MAE (RF)",       "289.68 v/j"),
    (c3, "🔴", "Anomalies détectées",     "184"),
    (c4, "📄", "Pages analysées",         "1 000"),
]
for col, icon, label, val in kpis:
    with col:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-icon">{icon}</div>
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{val}</div>
        </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── 3 & 4. PAGE + MODÈLE ─────────────────────────────────────
col_page, col_modele, col_horizon = st.columns([3, 2, 1])

with col_page:
    st.markdown('<p class="section-titre">📄 Choisir une page Wikipédia</p>',
                unsafe_allow_html=True)
    page_choisie = st.selectbox("Page", liste_pages,
                                index=0, label_visibility="collapsed")

with col_modele:
    st.markdown('<p class="section-titre">🤖 Choisir un modèle</p>',
                unsafe_allow_html=True)
    options_modeles = (["Baseline", "Random Forest", "XGBoost"]
                       if XGBOOST_OK else ["Baseline", "Random Forest"])
    modele_choisi = st.selectbox("Modèle", options_modeles,
                                 index=1, label_visibility="collapsed")

with col_horizon:
    st.markdown('<p class="section-titre">📅 Horizon</p>',
                unsafe_allow_html=True)
    horizon = st.selectbox("Horizon", [7, 14, 21, 30],
                            index=1, label_visibility="collapsed")


# ── 5. GRAPHIQUE HISTORIQUE ───────────────────────────────────
st.markdown('<p class="section-titre">📈 Historique du trafic (550 jours)</p>',
            unsafe_allow_html=True)
fig_h = fig_historique(page_choisie, df)
st.pyplot(fig_h, use_container_width=True); plt.close()


# ── 6. BOUTON PRÉDIRE ─────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
col_btn, col_txt = st.columns([1, 3])
with col_btn:
    clic = st.button(f"🔮  Prédire avec {modele_choisi}", type="primary")
with col_txt:
    st.markdown(
        f"<small style='color:#888;line-height:3'>Modèle : <b>{modele_choisi}</b> "
        f"| Horizon : <b>{horizon} jours</b> | Page : {page_choisie[:45]}</small>",
        unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════
#  RÉSULTATS (affiché uniquement après clic)
# ═════════════════════════════════════════════════════════════
if clic:
    with st.spinner("Calcul en cours..."):

        # Dates futures
        last_date    = pd.to_datetime(date_cols[-1])
        dates_futures = pd.date_range(
            last_date + pd.Timedelta(days=1), periods=horizon)

        # Prédictions du modèle choisi
        y_pred = predire(page_choisie, modele_choisi,
                         modeles[modele_choisi], df_feat, FEATURES, horizon)

        # Prédictions de TOUS les modèles (pour comparaison)
        preds_tous = {}
        for nom in modeles:
            preds_tous[nom] = predire(
                page_choisie, nom, modeles[nom], df_feat, FEATURES, horizon)

        # Détection anomalies futures
        est_anomal, z_scores, types_anomal, mu_hist, sig_hist = \
            detecter_anomalies_futures(page_choisie, y_pred, df, seuil_z=2.0)

    if y_pred is None:
        st.error("❌ Données insuffisantes pour cette page.")
        st.stop()

    moyenne = float(np.mean(y_pred))
    n_anom_fut = int(est_anomal.sum())

    # ── TABS : 3 onglets ─────────────────────────────────────
    tab1, tab2, tab3 = st.tabs([
        f"🔮 Prédiction — {modele_choisi}",
        "📊 Comparaison des 3 modèles",
        "🚨 Détection des anomalies futures"
    ])

    # ─────────────────────────────────────────────────────────
    # TAB 1 — PRÉDICTION DU MODÈLE CHOISI
    # ─────────────────────────────────────────────────────────
    with tab1:
        # Encadré résultat principal
        col_box, col_metrics = st.columns([2, 1])
        with col_box:
            colors_box = {
                "Random Forest": ("pred-box-rf",  "#375623"),
                "XGBoost"      : ("pred-box-xgb", "#2E75B6"),
                "Baseline"     : ("pred-box-bl",  "#888"),
            }
            cls, col_val = colors_box[modele_choisi]
            icone = {"Random Forest": "🌲", "XGBoost": "⚡", "Baseline": "📏"}[modele_choisi]
            st.markdown(f"""
            <div class="pred-box {cls}">
                <h3>{icone} Prédiction — {modele_choisi}</h3>
                <div class="pred-valeur" style="color:{col_val}">
                    {moyenne:,.0f}
                    <span style="font-size:1rem;font-weight:400"> vues/jour (moyenne)</span>
                </div>
                <div class="pred-details">
                    📉 Min prédit : <b>{y_pred.min():,.0f}</b> v/j &nbsp;|&nbsp;
                    📈 Max prédit : <b>{y_pred.max():,.0f}</b> v/j &nbsp;|&nbsp;
                    🔴 Anomalies futures : <b>{n_anom_fut}</b> jour(s)
                </div>
            </div>""", unsafe_allow_html=True)

        with col_metrics:
            m = metriques[modele_choisi]
            st.metric("MAE (test)",  f"{m['MAE']} vues")
            st.metric("R² Score",    f"{m['R2']}")
            st.metric("vs Baseline", f"+{m['amélioration']}%")

        # Graphique
        st.markdown("####")
        fig_p = fig_prediction(page_choisie, df, y_pred,
                               modele_choisi, est_anomal, dates_futures)
        st.pyplot(fig_p, use_container_width=True); plt.close()

        # Tableau jour par jour
        df_table = pd.DataFrame({
            "Date"               : [d.strftime("%d/%m/%Y") for d in dates_futures],
            "Jour"               : [d.strftime("%A")       for d in dates_futures],
            "Prédiction (vues)"  : [f"{int(v):,}"          for v in y_pred],
            "Borne −15 %"        : [f"{int(v*0.85):,}"     for v in y_pred],
            "Borne +15 %"        : [f"{int(v*1.15):,}"     for v in y_pred],
            "Statut"             : types_anomal,
        })
        st.dataframe(df_table, use_container_width=True, hide_index=True)

    # ─────────────────────────────────────────────────────────
    # TAB 2 — COMPARAISON DES 3 MODÈLES
    # ─────────────────────────────────────────────────────────
    with tab2:

        # Graphique superposition des courbes
        st.markdown('<p class="section-titre">Comparaison des prédictions</p>',
                    unsafe_allow_html=True)
        fig_comp = fig_comparaison_courbes(
            page_choisie, df, preds_tous, dates_futures)
        st.pyplot(fig_comp, use_container_width=True); plt.close()

        # Graphique métriques
        st.markdown('<p class="section-titre">Comparaison des performances (sur le test set)</p>',
                    unsafe_allow_html=True)
        fig_met = fig_metriques(metriques)
        st.pyplot(fig_met, use_container_width=True); plt.close()

        # Tableau récapitulatif
        st.markdown('<p class="section-titre">Tableau récapitulatif</p>',
                    unsafe_allow_html=True)
        lignes_tab = []
        for nom, m in metriques.items():
            meilleur = "✅ Meilleur" if m["MAE"] == min(
                v["MAE"] for v in metriques.values()) else ""
            lignes_tab.append({
                "Modèle"          : nom,
                "MAE (vues/j)"    : m["MAE"],
                "RMSE (vues/j)"   : m["RMSE"],
                "R² Score"        : m["R2"],
                "Amélioration"    : f"+{m['amélioration']}%",
                ""                : meilleur,
            })
        df_comp = pd.DataFrame(lignes_tab)
        st.dataframe(df_comp, use_container_width=True, hide_index=True)

        # Cartes individuelles
        cols_cards = st.columns(3)
        icones = {"Baseline": "📏", "Random Forest": "🌲", "XGBoost": "⚡"}
        couleurs_css = {
            "Baseline"     : ("pred-box-bl",  "#888"),
            "Random Forest": ("pred-box-rf",  "#375623"),
            "XGBoost"      : ("pred-box-xgb", "#2E75B6"),
        }
        for i, (nom, m) in enumerate(metriques.items()):
            cls, col_v = couleurs_css[nom]
            p = preds_tous[nom]
            moy = f"{float(np.mean(p)):,.0f}" if p is not None else "N/A"
            with cols_cards[i]:
                st.markdown(f"""
                <div class="pred-box {cls}" style="margin-top:16px">
                    <h3>{icones[nom]} {nom}</h3>
                    <div class="pred-valeur" style="color:{col_v};font-size:1.6rem">
                        {moy} v/j
                    </div>
                    <div class="pred-details">
                        MAE : <b>{m['MAE']}</b> &nbsp;|&nbsp;
                        R² : <b>{m['R2']}</b><br>
                        Amélioration : <b>+{m['amélioration']}%</b>
                    </div>
                </div>""", unsafe_allow_html=True)

    # ─────────────────────────────────────────────────────────
    # TAB 3 — DÉTECTION DES ANOMALIES FUTURES
    # ─────────────────────────────────────────────────────────
    with tab3:

        st.markdown('<p class="section-titre">Analyse des anomalies sur les prédictions futures</p>',
                    unsafe_allow_html=True)

        # Bilan global
        if n_anom_fut == 0:
            st.markdown("""
            <div class="anomalie-ok">
                <b>✅ Aucune anomalie détectée dans les prédictions futures.</b><br>
                <span style="color:#555">Le trafic prévu reste dans la plage normale
                (± 2 écarts-types par rapport à l'historique).</span>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="anomalie-alerte">
                <b>🚨 {n_anom_fut} jour(s) anormal(aux) détecté(s) sur {horizon} jours prédits</b><br>
                <span style="color:#555">Ces jours dépassent le seuil de ± 2 écarts-types
                par rapport à la moyenne historique de la page
                ({mu_hist:,.0f} ± {sig_hist:,.0f} vues/jour).</span>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Graphique anomalies
        fig_anom, ax = plt.subplots(figsize=(12, 4))
        fig_anom.patch.set_facecolor("white"); ax.set_facecolor("#FAFAFA")

        # Zone normale
        ax.axhspan(mu_hist - 2*sig_hist, mu_hist + 2*sig_hist,
                   alpha=0.08, color="green", label="Zone normale (±2σ)")
        ax.axhline(mu_hist, color="green", lw=1.2,
                   linestyle="--", alpha=0.6, label=f"Moyenne historique ({mu_hist:,.0f})")
        ax.axhline(mu_hist + 2*sig_hist, color="orange", lw=1,
                   linestyle=":", alpha=0.7)
        ax.axhline(mu_hist - 2*sig_hist, color="orange", lw=1,
                   linestyle=":", alpha=0.7, label="Seuil ±2σ")

        # Prédictions colorées selon anomalie
        for i, (d, v) in enumerate(zip(dates_futures, y_pred)):
            col_pt = C_ANOM if est_anomal[i] else C_RF
            mk     = "^" if est_anomal[i] else "o"
            sz     = 120 if est_anomal[i] else 60
            ax.scatter(d, v, color=col_pt, s=sz, marker=mk, zorder=5)

        ax.plot(dates_futures, y_pred, color="#1F3864", lw=1.5,
                linestyle="-", alpha=0.6, label="Prédictions")

        # Légende manuelle
        patch_ok   = mpatches.Patch(color=C_RF,   label="Jour normal")
        patch_anom = mpatches.Patch(color=C_ANOM, label=f"Anomalie ({n_anom_fut})")
        ax.legend(handles=[patch_ok, patch_anom,
                           mpatches.Patch(color="green", alpha=0.3, label="Zone normale")],
                  fontsize=9, loc="upper right")

        ax.set_xlabel("Date", fontsize=10); ax.set_ylabel("Vues", fontsize=10)
        ax.set_title(f"Détection d'anomalies futures — {modele_choisi}",
                     fontsize=11, fontweight="bold", color="#1F3864")
        ax.grid(alpha=0.2, linestyle="--")
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig_anom, use_container_width=True); plt.close()

        # Tableau détaillé des anomalies
        st.markdown('<p class="section-titre">Détail jour par jour</p>',
                    unsafe_allow_html=True)

        df_anom = pd.DataFrame({
            "Date"                : [d.strftime("%d/%m/%Y") for d in dates_futures],
            "Jour"                : [d.strftime("%A")       for d in dates_futures],
            "Vues prédites"       : [f"{int(v):,}"          for v in y_pred],
            "Z-score"             : [f"{z:+.2f}"            for z in z_scores],
            "Statut"              : types_anomal,
            "Interprétation"      : [
                f"Pic : {int(v):,} vues (+{int((v-mu_hist)/sig_hist*100)}% / μ)"
                if z > 2
                else f"Creux : {int(v):,} vues ({int((v-mu_hist)/sig_hist*100)}% / μ)"
                if z < -2
                else f"Normal — {int(v):,} vues (μ = {int(mu_hist):,})"
                for v, z in zip(y_pred, z_scores)
            ],
        })
        st.dataframe(df_anom, use_container_width=True, hide_index=True)

        # Explication méthode
        with st.expander("ℹ️ Méthode de détection utilisée"):
            st.markdown(f"""
            **Méthode : Z-score sur l'historique**

            - **Moyenne historique (μ)** de la page : `{mu_hist:,.0f}` vues/jour
            - **Écart-type (σ)** de la page : `{sig_hist:,.0f}` vues/jour
            - **Seuil** : |z| > 2.0 → anomalie
            - **Formule** : `z = (prédiction − μ) / σ`

            Un z-score > +2 indique un **pic de trafic anormalement élevé**.
            Un z-score < −2 indique un **creux de trafic anormalement bas**.

            > Ces anomalies permettent à Vala Bleu d'anticiper les
            > pics de charge sur les serveurs et de déclencher un scaling préventif.
            """)

    # ── Barre de succès ──────────────────────────────────────
    st.success(
        f"✅ Prédiction calculée — Modèle : **{modele_choisi}** | "
        f"Page : **{page_choisie[:45]}** | "
        f"Horizon : **{horizon} jours** | "
        f"Anomalies futures : **{n_anom_fut}**"
    )


# ── PIED DE PAGE ──────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:#aaa;font-size:0.82rem;'>"
    "Rachid Bijigune &nbsp;|&nbsp; EST Fkih Ben Salah — Bachelor Big Data &nbsp;|&nbsp; "
    "Stage Vala Bleu 2025-2026 &nbsp;|&nbsp; "
    "Random Forest (MAE=289.68) • XGBoost (MAE=313.45) • Baseline (MAE=340.52)"
    "</div>",
    unsafe_allow_html=True
)
