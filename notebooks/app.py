import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import warnings
warnings.filterwarnings("ignore")

# ── Configuration ──────────────────────────────────────────
st.set_page_config(
    page_title="Détection & Prédiction du Trafic Web",
    page_icon="📈",
    layout="wide"
)

# ── Chargement des données ──────────────────────────────────
@st.cache_data
def load_data():
    return pd.read_csv("notebooks/df_petit.csv")

df = load_data()
date_cols = [c for c in df.columns if c != "Page"]

# ── Sidebar ─────────────────────────────────────────────────
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/8/80/Wikipedia-logo-v2.svg", width=60)
    st.title("⚙️ Paramètres")
    page_selectee = st.selectbox("Page Wikipedia", df["Page"].tolist())
    n_predict = st.slider("Jours à prédire", 7, 30, 14)
    afficher_anomalies = st.checkbox("Afficher les anomalies", value=True)
    st.divider()
    st.caption("Bachelor Big Data — EST Fkih Ben Salah")
    st.caption("Rachid Bijigune — 2025-2026")

# ── Titre ───────────────────────────────────────────────────
st.title("📈 Détection & Prédiction du Trafic Web")
st.caption("Système ML/DL sur le dataset Wikipedia Web Traffic • Vala Orange")

# ── Données de la page ──────────────────────────────────────
row = df[df["Page"] == page_selectee].iloc[0]
serie = pd.Series(
    row[date_cols].values.astype(float),
    index=pd.to_datetime(date_cols)
)

# ── Métriques rapides ────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total vues", f"{int(serie.sum()):,}")
col2.metric("Moyenne /jour", f"{int(serie.mean()):,}")
col3.metric("Pic maximum", f"{int(serie.max()):,}")
col4.metric("Jours d'historique", len(serie))

st.divider()

# ── Tabs ────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "📊 Historique & Prédiction",
    "🚨 Détection d'anomalies",
    "📋 Comparaison des modèles"
])

# ── Feature engineering ──────────────────────────────────────
def make_features(vals):
    df_tmp = pd.DataFrame({"views": vals})
    df_tmp["lag_1"]        = df_tmp["views"].shift(1)
    df_tmp["lag_7"]        = df_tmp["views"].shift(7)
    df_tmp["lag_14"]       = df_tmp["views"].shift(14)
    df_tmp["mean_7"]       = df_tmp["views"].shift(1).rolling(7).mean()
    df_tmp["day_of_week"]  = pd.Series(range(len(vals))) % 7
    df_tmp["month"]        = (pd.Series(range(len(vals))) // 30) % 12 + 1
    return df_tmp.dropna()

# ── Entraînement Random Forest ────────────────────────────────
vals     = np.log1p(serie.values)
feat_df  = make_features(vals)
features = ["lag_1","lag_7","lag_14","mean_7","day_of_week","month"]
X = feat_df[features].values
y = feat_df["views"].values

model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
model.fit(X[:-n_predict], y[:-n_predict])

# ── Prédiction ────────────────────────────────────────────────
preds_log = model.predict(X[-n_predict:])
preds     = np.expm1(preds_log)
true_vals = np.expm1(y[-n_predict:])
mae       = mean_absolute_error(true_vals, preds)

with tab1:
    st.subheader(f"Trafic : {page_selectee[:70]}")

    fig, ax = plt.subplots(figsize=(13, 4))
    dates_hist = serie.index[:-n_predict]
    dates_pred = serie.index[-n_predict:]

    ax.plot(dates_hist, serie.values[:-n_predict],
            color="steelblue", linewidth=1.2, label="Historique")
    ax.plot(dates_pred, serie.values[-n_predict:],
            color="steelblue", linewidth=1.2, linestyle="--")
    ax.plot(dates_pred, preds,
            color="orange", linewidth=2, label=f"Prédiction RF (MAE={mae:.0f})")

    if afficher_anomalies:
        mean_v = serie.mean()
        std_v  = serie.std()
        z      = np.abs((serie.values - mean_v) / std_v)
        idx_a  = np.where(z > 3)[0]
        if len(idx_a) > 0:
            ax.scatter(serie.index[idx_a], serie.values[idx_a],
                      color="red", zorder=5, s=50, label=f"Anomalies ({len(idx_a)})")

    ax.axvline(dates_pred[0], color="gray", linestyle=":", alpha=0.7)
    ax.text(dates_pred[0], ax.get_ylim()[1]*0.95,
            " ← prédiction", color="gray", fontsize=9)
    ax.set_xlabel("Date"); ax.set_ylabel("Vues")
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)

    col1, col2 = st.columns(2)
    col1.metric("MAE sur les prédictions", f"{mae:.0f} vues")
    col2.metric("Horizon de prédiction", f"{n_predict} jours")

with tab2:
    st.subheader("Anomalies détectées — Z-score (seuil : z > 3)")
    mean_v = serie.mean(); std_v = serie.std()
    z = np.abs((serie.values - mean_v) / std_v)
    mask = z > 3
    anomalies_df = pd.DataFrame({
        "Date":    serie.index[mask].strftime("%Y-%m-%d"),
        "Vues":    serie.values[mask].astype(int),
        "Z-score": z[mask].round(2)
    }).sort_values("Z-score", ascending=False)

    c1, c2, c3 = st.columns(3)
    c1.metric("Anomalies détectées", len(anomalies_df))
    c2.metric("Vues moyennes normales", f"{int(mean_v):,}")
    c3.metric("Pic max anormal", f"{int(serie.values[mask].max()):,}" if mask.any() else "—")

    if len(anomalies_df) > 0:
        st.dataframe(anomalies_df.reset_index(drop=True), use_container_width=True)
    else:
        st.success("Aucune anomalie détectée sur cette page.")

with tab3:
    st.subheader("Résultats de l'étude complète — Ensemble de test (60 000 exemples)")
    resultats = pd.DataFrame({
        "Modèle":      ["Baseline","Decision Tree","XGBoost","LSTM","Random Forest","GRU"],
        "Type":        ["Référence","ML","ML","DL","ML","DL"],
        "MAE":         [340.52, 344.81, 313.45, 321.80, 289.69, 283.34],
        "RMSE":        [4170.12, 4547.81, 4312.09, 4479.47, 4002.83, 4362.33],
        "vs Baseline": ["0.0%","-1.3%","+7.9%","+5.5%","+14.9%","+16.8%"],
        "Statut":      ["","","","","","🥇 Meilleur"]
    })
    st.dataframe(
        resultats.style.highlight_min(subset=["MAE","RMSE"], color="#A9D08E"),
        use_container_width=True, hide_index=True
    )
    st.info("Le **GRU** (Deep Learning, séquences 14 jours) obtient la meilleure performance avec MAE=283.34, soit +16.8% d'amélioration par rapport au Baseline naïf.")