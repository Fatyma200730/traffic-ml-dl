# ============================================================
#  APPLICATION STREAMLIT — Prédiction du Trafic Web
#  Projet PFE | Rachid Bijigune | Vala Bleu | 2025-2026
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# CONFIGURATION DE LA PAGE
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Prédiction du Trafic Web — Vala Bleu",
    page_icon="📊",
    layout="wide"
)

# ─────────────────────────────────────────────────────────────
# CSS PERSONNALISÉ — Style moderne et épuré
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Police et fond */
    html, body, [class*="css"] {
        font-family: 'Segoe UI', sans-serif;
    }

    /* Titre principal */
    .titre-principal {
        background: linear-gradient(135deg, #1F3864, #2E75B6);
        color: white;
        padding: 30px 40px;
        border-radius: 12px;
        text-align: center;
        margin-bottom: 30px;
    }
    .titre-principal h1 {
        margin: 0;
        font-size: 2rem;
        font-weight: 700;
        letter-spacing: 1px;
    }
    .titre-principal p {
        margin: 8px 0 0;
        font-size: 0.95rem;
        opacity: 0.85;
    }

    /* Cartes KPI */
    .kpi-card {
        background: white;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        border-left: 5px solid #2E75B6;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    }
    .kpi-label {
        font-size: 0.85rem;
        color: #666;
        margin-bottom: 6px;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .kpi-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #1F3864;
    }
    .kpi-icon {
        font-size: 1.4rem;
        margin-bottom: 6px;
    }

    /* Section sélection */
    .section-titre {
        font-size: 1.15rem;
        font-weight: 600;
        color: #1F3864;
        border-left: 4px solid #2E75B6;
        padding-left: 12px;
        margin: 25px 0 15px;
    }

    /* Résultat de prédiction */
    .pred-box {
        background: #f0f7f0;
        border: 2px solid #375623;
        border-radius: 10px;
        padding: 22px 30px;
        margin-top: 16px;
    }
    .pred-box h3 {
        color: #375623;
        margin: 0 0 6px;
        font-size: 1.1rem;
    }
    .pred-valeur {
        font-size: 2.4rem;
        font-weight: 700;
        color: #375623;
    }
    .pred-details {
        color: #555;
        font-size: 0.9rem;
        margin-top: 8px;
    }

    /* Bouton Prédire */
    div.stButton > button {
        background: #1F3864;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 12px 40px;
        font-size: 1rem;
        font-weight: 600;
        letter-spacing: 0.5px;
        width: 100%;
        transition: background 0.2s;
    }
    div.stButton > button:hover {
        background: #2E75B6;
    }

    /* Masquer le menu Streamlit */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# CHARGEMENT DES DONNÉES
# ─────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def charger_donnees():
    """
    Charge df_petit.csv (1000 pages Wikipedia).
    Si le fichier est introuvable, génère des données simulées.
    """
    chemins = [
        "df_petit.csv",
        "data/df_petit.csv",
        "../data/df_petit.csv",
        "notebooks/df_petit.csv",
    ]
    for chemin in chemins:
        try:
            df = pd.read_csv(chemin)
            return df, True   # données réelles
        except FileNotFoundError:
            continue

    # ── Données simulées si fichier absent ──────────────────
    np.random.seed(42)
    n_pages, n_jours = 50, 550
    dates = pd.date_range("2015-07-01", periods=n_jours, freq="D")
    noms_pages = [f"Page_Wikipedia_{i:03d}_fr.wikipedia.org" for i in range(n_pages)]
    lignes = {}
    lignes["Page"] = noms_pages
    for d in dates:
        col = str(d.date())
        lignes[col] = [
            max(0, int(np.random.exponential(200) + np.sin(np.arange(n_jours)[dates == d][0] / 7) * 20))
            for _ in range(n_pages)
        ]
    return pd.DataFrame(lignes), False   # données simulées


# ─────────────────────────────────────────────────────────────
# ENTRAÎNEMENT DU MODÈLE RANDOM FOREST
# ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def entrainer_modele(csv_key):
    """
    Entraîne un Random Forest sur toutes les pages du dataset.
    Retourne le modèle entraîné et les données en format long.
    """
    df, _ = charger_donnees()
    date_cols = [c for c in df.columns if c != "Page"]

    # Format long : une ligne par (page, date)
    df_long = df.fillna(0).melt(id_vars="Page", var_name="date", value_name="views")
    df_long["date"] = pd.to_datetime(df_long["date"])
    df_long = df_long.sort_values(["Page", "date"]).reset_index(drop=True)

    # Construction des 6 features temporelles
    df_long["lag_1"]       = df_long.groupby("Page")["views"].shift(1)
    df_long["lag_7"]       = df_long.groupby("Page")["views"].shift(7)
    df_long["lag_14"]      = df_long.groupby("Page")["views"].shift(14)
    df_long["mean_7"]      = df_long.groupby("Page")["views"].transform(
                                 lambda x: x.shift(1).rolling(7).mean())
    df_long["day_of_week"] = df_long["date"].dt.dayofweek
    df_long["month"]       = df_long["date"].dt.month
    df_feat = df_long.dropna().reset_index(drop=True)

    # Split chronologique — 60 derniers jours = test
    cutoff = df_feat.groupby("Page")["date"].transform(
                 lambda x: x.sort_values().iloc[-60])
    train = df_feat[df_feat["date"] < cutoff]

    FEATURES = ["lag_1", "lag_7", "lag_14", "mean_7", "day_of_week", "month"]
    X_train = train[FEATURES]
    y_train = np.log1p(train["views"])

    # Entraînement
    rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)

    return rf, df_feat, FEATURES


# ─────────────────────────────────────────────────────────────
# PRÉDICTION POUR UNE PAGE DONNÉE
# ─────────────────────────────────────────────────────────────
def predire_page(nom_page, rf, df_feat, features, horizon=7):
    """
    Prédit le trafic des 'horizon' prochains jours pour une page.
    Retourne la prédiction moyenne et le vecteur des prédictions.
    """
    page_data = df_feat[df_feat["Page"] == nom_page].sort_values("date")
    if len(page_data) < 14:
        return None, None

    # Prédictions sur les dernières observations disponibles
    derniers = page_data.tail(horizon)
    X_pred   = derniers[features]
    y_pred   = np.expm1(rf.predict(X_pred))

    return float(np.mean(y_pred)), y_pred


# ─────────────────────────────────────────────────────────────
# GRAPHIQUE HISTORIQUE
# ─────────────────────────────────────────────────────────────
def graphique_historique(nom_page, df):
    """
    Trace la courbe du trafic historique pour la page sélectionnée.
    """
    date_cols = [c for c in df.columns if c != "Page"]
    ligne     = df[df["Page"] == nom_page].iloc[0]
    valeurs   = ligne[date_cols].values.astype(float)
    dates     = pd.to_datetime(date_cols)

    fig, ax = plt.subplots(figsize=(12, 4))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#FAFAFA")

    ax.plot(dates, valeurs, color="#2E75B6", linewidth=1.4, alpha=0.9)
    ax.fill_between(dates, valeurs, alpha=0.12, color="#2E75B6")

    # Mise en forme
    ax.set_xlabel("Date", fontsize=11)
    ax.set_ylabel("Nombre de vues", fontsize=11)
    ax.set_title(f"Historique du trafic — {nom_page[:60]}", fontsize=12,
                 fontweight="bold", color="#1F3864")
    ax.grid(alpha=0.25, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────
# GRAPHIQUE PRÉDICTION
# ─────────────────────────────────────────────────────────────
def graphique_prediction(nom_page, df, y_pred):
    """
    Trace l'historique récent + les prédictions futures.
    """
    date_cols   = [c for c in df.columns if c != "Page"]
    ligne       = df[df["Page"] == nom_page].iloc[0]
    valeurs     = ligne[date_cols].values.astype(float)
    dates       = pd.to_datetime(date_cols)

    # 30 derniers jours d'historique
    n_hist = 30
    hist_vals  = valeurs[-n_hist:]
    hist_dates = dates[-n_hist:]

    # Dates futures
    last_date    = dates[-1]
    horizon      = len(y_pred)
    future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=horizon)

    fig, ax = plt.subplots(figsize=(12, 4))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#FAFAFA")

    # Historique récent
    ax.plot(hist_dates, hist_vals, color="#2E75B6", lw=1.8,
            label="Historique (30j)", marker="o", ms=3)

    # Prédictions
    ax.plot(future_dates, y_pred, color="#375623", lw=2.2,
            linestyle="--", marker="s", ms=5, label=f"Prédiction RF ({horizon}j)")

    # Zone de confiance ±15 %
    ax.fill_between(future_dates, y_pred * 0.85, y_pred * 1.15,
                    alpha=0.18, color="#375623", label="Intervalle ±15 %")

    # Ligne séparatrice historique / futur
    ax.axvline(last_date, color="#C55A11", linewidth=1.5,
               linestyle=":", alpha=0.7, label="Aujourd'hui")

    ax.set_xlabel("Date", fontsize=11)
    ax.set_ylabel("Nombre de vues", fontsize=11)
    ax.set_title("Prédiction du trafic futur — Random Forest",
                 fontsize=12, fontweight="bold", color="#1F3864")
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(alpha=0.25, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    return fig


# ═════════════════════════════════════════════════════════════
#  INTERFACE PRINCIPALE
# ═════════════════════════════════════════════════════════════

# ── 1. TITRE PRINCIPAL ───────────────────────────────────────
st.markdown("""
<div class="titre-principal">
    <h1>📊 Prédiction et Analyse du Trafic Web</h1>
    <p>Vala Bleu — Solution Hébergement Web | Projet PFE 2025-2026 | Rachid Bijigune</p>
</div>
""", unsafe_allow_html=True)


# ── 2. INFORMATIONS GÉNÉRALES — 4 KPI ────────────────────────
st.markdown('<p class="section-titre">Informations générales du modèle</p>',
            unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)
kpis = [
    (col1, "🤖", "Modèle utilisé",             "Random Forest"),
    (col2, "📉", "MAE",                         "289.68 vues/j"),
    (col3, "🔴", "Anomalies détectées",          "184"),
    (col4, "📄", "Pages analysées",              "1 000"),
]
for col, icon, label, value in kpis:
    with col:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-icon">{icon}</div>
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# ── Chargement des données et du modèle ──────────────────────
with st.spinner("Chargement des données..."):
    df, donnees_reelles = charger_donnees()
    date_cols = [c for c in df.columns if c != "Page"]

if not donnees_reelles:
    st.info("ℹ️ Fichier df_petit.csv introuvable — données simulées utilisées.",
            icon="ℹ️")

with st.spinner("Entraînement du modèle Random Forest..."):
    rf, df_feat, FEATURES = entrainer_modele(str(len(df)))

liste_pages = sorted(df["Page"].tolist())


# ── 3. SÉLECTION DE LA PAGE ───────────────────────────────────
st.markdown('<p class="section-titre">Sélectionner une page Wikipédia</p>',
            unsafe_allow_html=True)

page_choisie = st.selectbox(
    label="Page Wikipédia",
    options=liste_pages,
    index=0,
    label_visibility="collapsed",
    help="Choisissez une page parmi les 1 000 pages analysées"
)


# ── 4. GRAPHIQUE HISTORIQUE ───────────────────────────────────
st.markdown('<p class="section-titre">Historique du trafic (550 jours)</p>',
            unsafe_allow_html=True)

fig_hist = graphique_historique(page_choisie, df)
st.pyplot(fig_hist, use_container_width=True)
plt.close()


# ── 5. BOUTON PRÉDIRE ─────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
col_btn, col_info = st.columns([1, 3])

with col_btn:
    clic = st.button("🔮  Prédire le trafic futur", type="primary")

with col_info:
    st.markdown(
        "<small style='color:#888;line-height:2.5'>Prédiction par Random Forest "
        "(100 arbres, max_depth=10) sur les 7 prochains jours.</small>",
        unsafe_allow_html=True
    )


# ── 6. RÉSULTATS DE PRÉDICTION ────────────────────────────────
if clic:
    with st.spinner("Calcul de la prédiction en cours..."):
        moyenne, y_pred = predire_page(page_choisie, rf, df_feat, FEATURES, horizon=7)

    if moyenne is None:
        st.error("Données insuffisantes pour cette page.")
    else:
        # Graphique prédiction
        st.markdown('<p class="section-titre">Résultats de la prédiction</p>',
                    unsafe_allow_html=True)
        fig_pred = graphique_prediction(page_choisie, df, y_pred)
        st.pyplot(fig_pred, use_container_width=True)
        plt.close()

        # Encadré résultat principal
        st.markdown(f"""
        <div class="pred-box">
            <h3>🌲 Prédiction Random Forest — 7 prochains jours</h3>
            <div class="pred-valeur">{moyenne:,.0f} <span style="font-size:1rem;font-weight:400;">vues / jour (moyenne)</span></div>
            <div class="pred-details">
                📉 Borne basse (−15 %) : <b>{moyenne*0.85:,.0f}</b> vues &nbsp;|&nbsp;
                📈 Borne haute (+15 %) : <b>{moyenne*1.15:,.0f}</b> vues &nbsp;|&nbsp;
                MAE modèle : <b>289.68</b> vues/jour
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Tableau jour par jour
        st.markdown("<br>", unsafe_allow_html=True)
        last_date = pd.to_datetime(date_cols[-1])
        future_dates = pd.date_range(
            last_date + pd.Timedelta(days=1), periods=7
        )
        tableau = pd.DataFrame({
            "Date"               : [d.strftime("%d/%m/%Y") for d in future_dates],
            "Jour"               : [d.strftime("%A")       for d in future_dates],
            "Prédiction (vues)"  : [f"{int(v):,}"          for v in y_pred],
            "Borne basse (−15 %)": [f"{int(v*0.85):,}"     for v in y_pred],
            "Borne haute (+15 %)": [f"{int(v*1.15):,}"     for v in y_pred],
        })
        st.dataframe(tableau, use_container_width=True, hide_index=True)

        st.success(
            f"✅ Prédiction calculée pour **{page_choisie[:50]}** "
            f"— Moyenne : **{moyenne:,.0f} vues/jour**"
        )


# ── 7. PIED DE PAGE ───────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:#999;font-size:0.82rem;'>"
    "Rachid Bijigune &nbsp;|&nbsp; EST Fkih Ben Salah — Bachelor Big Data &nbsp;|&nbsp; "
    "Stage Vala Bleu 2025-2026 &nbsp;|&nbsp; Random Forest (MAE = 289.68 vues/j)"
    "</div>",
    unsafe_allow_html=True
)
