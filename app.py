"""
Dashboard Streamlit — HMM Regime Detection.

Interface interactive pour explorer :
    - Les données (prix, volatilité)
    - Les régimes détectés (Markov Switching)
    - Les statistiques par régime
    - Le backtest des stratégies conditionnelles
    - La génération de rapport PDF

Usage :
    streamlit run app.py

Auteur : Statby2Mf
Projet : HMM Regime Detection (M2 Statistique, UGB Saint-Louis)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from src.data_manager import get_all_features, ASSETS, ASSET_CLASSES
from src.hmm_model import (
    fit_regime_model, ASSET_CLASSES_MAP, REGIME_LABELS,
)
from src.strategy_adaptation import (
    run_all_strategies, compare_strategies, StrategyResult,
)
from src.backtesting import (
    rolling_sharpe, performance_by_regime,
    test_sharpe_significance, analyze_drawdowns,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="HMM Regime Detection",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

PROJECT_ROOT = Path(__file__).resolve().parent
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"

# Couleurs
REGIME_COLORS = {
    "Low Vol": "#28a745",
    "Med Vol": "#ffc107",
    "High Vol": "#dc3545",
}


# ---------------------------------------------------------------------------
# CSS personnalisé
# ---------------------------------------------------------------------------

st.markdown("""
<style>
    .main-title {
        font-size: 2.2em;
        font-weight: bold;
        color: #1a3a5c;
        text-align: center;
        margin-bottom: 0.3em;
    }
    .subtitle {
        font-size: 1em;
        color: #666;
        text-align: center;
        margin-bottom: 1.5em;
    }
    .winner {
        background-color: #d4edda;
        padding: 1em;
        border-radius: 0.5em;
        border-left: 4px solid #28a745;
        margin: 0.5em 0;
    }
    .info-box {
        background-color: #e7f3ff;
        padding: 1em;
        border-radius: 0.5em;
        border-left: 4px solid #1a3a5c;
        margin: 0.5em 0;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False, ttl=3600)
def load_data():
    """Charge les features (cache)."""
    returns, vol = get_all_features(vol_window=20)
    return returns, vol


@st.cache_data(show_spinner=False, ttl=3600)
def fit_all_models():
    """Fit tous les modèles Markov Switching (cache)."""
    returns, vol = load_data()
    results = {}
    for asset_class in ASSET_CLASSES_MAP.keys():
        results[asset_class] = fit_regime_model(
            returns, vol, asset_class,
            n_states=3, maxiter=1000, feature="vol",
        )
    return results


@st.cache_data(show_spinner=False, ttl=3600)
def backtest_all(fit_cache_key: str):
    """Backtest toutes les stratégies (cache)."""
    results = fit_all_models()
    strategies = {}
    for asset_class, fit in results.items():
        strategies[asset_class] = run_all_strategies(fit)
    return strategies


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.markdown('<div class="main-title">📊 HMM Regime Detection</div>',
            unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Détection de régimes de marché par '
    'Markov Switching — Crypto • US • BRVM</div>',
    unsafe_allow_html=True
)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("### ⚙️ Configuration")

    asset_class = st.selectbox(
        "🏷️ Classe d'actifs",
        options=list(ASSET_CLASSES_MAP.keys()),
        format_func=lambda x: x.upper(),
        index=0,
    )

    st.markdown("---")
    st.markdown("### 📄 Rapport PDF")

    if st.button("📄 Générer le rapport PDF", use_container_width=True):
        with st.spinner("Génération du PDF (2-3 min)…"):
            try:
                from src.report_generator import generate_report
                pdf_path = generate_report(verbose=False)
                st.session_state["pdf_path"] = str(pdf_path)
                st.success("✅ Rapport prêt !")
            except Exception as e:
                st.error(f"❌ Erreur : {e}")

    if "pdf_path" in st.session_state:
        path_obj = Path(st.session_state["pdf_path"])
        if path_obj.exists():
            with open(path_obj, "rb") as f:
                st.download_button(
                    label="⬇️ Télécharger",
                    data=f.read(),
                    file_name=path_obj.name,
                    mime="application/pdf",
                    use_container_width=True,
                )
            st.caption(f"📁 {path_obj.name}")

    st.markdown("---")
    st.markdown(
        "**📚 Références**\n"
        "- Hamilton (1989)\n"
        "- Ang & Bekaert (2002)\n"
        "- Guidolin & Timmermann (2007)"
    )
    st.markdown("---")
    st.caption("Statby2Mf — M2 Statistique, UGB Saint-Louis")


# ---------------------------------------------------------------------------
# Chargement des données
# ---------------------------------------------------------------------------

with st.spinner("Chargement des données et fit des modèles…"):
    returns, vol = load_data()
    all_fits = fit_all_models()
    all_strategies = backtest_all("v1")

fit = all_fits[asset_class]


# ---------------------------------------------------------------------------
# Onglets
# ---------------------------------------------------------------------------

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Données",
    "🎯 Régimes détectés",
    "📊 Analyse approfondie",
    "🏆 Stratégies",
    "ℹ️ À propos",
])


# ===========================================================================
# ONGLET 1 — Données
# ===========================================================================

with tab1:
    st.header(f"📈 Données — {asset_class.upper()}")

    tickers = ASSET_CLASSES_MAP[asset_class]

    # KPIs
    comp_ret = fit.composite_returns
    comp_vol = fit.composite_vol

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Observations", f"{len(comp_ret):,}")
    col2.metric("Période",
                f"{(comp_ret.index[-1] - comp_ret.index[0]).days} jours")
    col3.metric("Rendement moyen",
                f"{comp_ret.mean():+.4f}% / jour")
    col4.metric("Volatilité moyenne",
                f"{comp_vol.mean():.3f}% / jour")

    st.info(f"**Actifs inclus** : {', '.join(tickers)}")

    # Prix composite
    st.subheader("Prix composite (base 100)")
    price = 100 * np.exp(np.cumsum(comp_ret / 100))
    fig_price = go.Figure()
    fig_price.add_trace(go.Scatter(
        x=price.index, y=price.values,
        name="Composite", line=dict(color="#1a3a5c", width=1.5),
    ))
    fig_price.update_layout(
        height=400,
        xaxis_title="Date", yaxis_title="Indice base 100",
        margin=dict(t=20, b=20),
    )
    st.plotly_chart(fig_price, use_container_width=True)

    # Volatilité composite
    st.subheader("Volatilité composite 20 jours")
    fig_vol = go.Figure()
    fig_vol.add_trace(go.Scatter(
        x=comp_vol.index, y=comp_vol.values,
        name="Volatilité 20j", line=dict(color="#d62728", width=1.2),
        fill="tozeroy", fillcolor="rgba(214,39,40,0.1)",
    ))
    fig_vol.update_layout(
        height=350,
        xaxis_title="Date", yaxis_title="Volatilité (%)",
        margin=dict(t=20, b=20),
    )
    st.plotly_chart(fig_vol, use_container_width=True)


# ===========================================================================
# ONGLET 2 — Régimes détectés
# ===========================================================================

with tab2:
    st.header(f"🎯 Régimes détectés — {asset_class.upper()}")

    # KPIs
    stats = fit.regime_stats
    col1, col2, col3 = st.columns(3)
    for col, regime in zip([col1, col2, col3], REGIME_LABELS):
        if regime in stats.index:
            row = stats.loc[regime]
            col.markdown(
                f"""
                <div class="info-box">
                    <b style="color:{REGIME_COLORS[regime]}">{regime}</b><br/>
                    Rendement : <b>{row['mean_return']:+.3f}%</b><br/>
                    Volatilité : <b>{row['mean_vol']:.3f}%</b><br/>
                    Fréquence : <b>{row['frequency']:.1%}</b><br/>
                    Durée : <b>{row['avg_duration']:.1f}j</b>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # Prix + régimes colorés
    st.subheader("Prix composite avec régimes")
    fig_regime = go.Figure()

    # Prix
    fig_regime.add_trace(go.Scatter(
        x=price.index, y=price.values,
        name="Prix composite",
        line=dict(color="#1a3a5c", width=1.5),
    ))

    # Bandes colorées par régime
    labels = fit.regime_labels
    prev_regime = None
    start_date = None

    for date, regime in labels.items():
        if regime != prev_regime:
            if prev_regime is not None and start_date is not None:
                fig_regime.add_vrect(
                    x0=start_date, x1=date,
                    fillcolor=REGIME_COLORS.get(prev_regime, "gray"),
                    opacity=0.15, layer="below", line_width=0,
                )
            start_date = date
            prev_regime = regime

    if prev_regime is not None and start_date is not None:
        fig_regime.add_vrect(
            x0=start_date, x1=labels.index[-1],
            fillcolor=REGIME_COLORS.get(prev_regime, "gray"),
            opacity=0.15, layer="below", line_width=0,
        )

    fig_regime.update_layout(
        height=450,
        xaxis_title="Date", yaxis_title="Indice base 100",
        margin=dict(t=20, b=20),
    )
    st.plotly_chart(fig_regime, use_container_width=True)

    st.caption("Bandes colorées : 🟢 Low Vol, 🟡 Med Vol, 🔴 High Vol")

    # Régime actuel
    st.subheader("Régime actuel")
    current_regime = fit.regime_labels.iloc[-1]
    current_proba = fit.state_proba.iloc[-1].max()
    current_date = fit.regime_labels.index[-1]

    st.markdown(
        f"""
        <div class="winner">
            <h3>Régime au {current_date.date()}</h3>
            <p style="font-size:1.5em; color:{REGIME_COLORS[current_regime]}">
                <b>{current_regime}</b>
            </p>
            <p>Probabilité : <b>{current_proba:.2%}</b></p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ===========================================================================
# ONGLET 3 — Analyse approfondie
# ===========================================================================

with tab3:
    st.header(f"📊 Analyse approfondie — {asset_class.upper()}")

    # Matrice de transition
    st.subheader("Matrice de transition")
    tm = fit.transition_matrix

    fig_tm = go.Figure(data=go.Heatmap(
        z=tm.values, x=tm.columns, y=tm.index,
        colorscale="RdYlGn_r", zmin=0, zmax=1,
        text=[[f"{v:.1%}" for v in row] for row in tm.values],
        texttemplate="%{text}", textfont=dict(size=14),
    ))
    fig_tm.update_layout(
        height=350,
        xaxis_title="Régime à t",
        yaxis_title="Régime à t-1",
        margin=dict(t=20, b=20),
    )
    st.plotly_chart(fig_tm, use_container_width=True)

    # Distribution des durées
    st.subheader("Distribution des durées par régime")
    from src.regime_analysis import regime_durations
    durations = regime_durations(fit.regime_labels)

    fig_dur = make_subplots(rows=1, cols=3,
                             subplot_titles=REGIME_LABELS)

    for i, regime in enumerate(REGIME_LABELS, start=1):
        d = durations.get(regime, [])
        if d:
            fig_dur.add_trace(
                go.Histogram(
                    x=d, name=regime,
                    marker_color=REGIME_COLORS[regime],
                    opacity=0.7, nbinsx=15,
                    showlegend=False,
                ),
                row=1, col=i,
            )

    fig_dur.update_layout(
        height=350,
        margin=dict(t=40, b=20),
    )
    st.plotly_chart(fig_dur, use_container_width=True)

    # Test de séparation
    st.subheader("Tests de séparation des régimes")
    from src.regime_analysis import test_regime_separation
    sep = test_regime_separation(fit)
    if len(sep) > 0:
        sep_display = sep.copy()
        sep_display["mean_a"] = sep_display["mean_a"].apply(
            lambda x: f"{x:.3f}%")
        sep_display["mean_b"] = sep_display["mean_b"].apply(
            lambda x: f"{x:.3f}%")
        sep_display["t_stat"] = sep_display["t_stat"].apply(
            lambda x: f"{x:+.2f}")
        sep_display["p_value"] = sep_display["p_value"].apply(
            lambda x: f"{x:.2e}")
        st.dataframe(sep_display, use_container_width=True, hide_index=True)

    # Probabilités de transition futures
    st.subheader("Probabilités de transition (5 prochains jours)")
    from src.regime_analysis import current_regime_proba
    probs = current_regime_proba(fit, horizon=5)
    st.dataframe(probs.round(4), use_container_width=True)


# ===========================================================================
# ONGLET 4 — Stratégies
# ===========================================================================

with tab4:
    st.header(f"🏆 Stratégies — {asset_class.upper()}")

    strategies_results = all_strategies[asset_class]

    # Tableau comparatif
    st.subheader("Comparaison des stratégies")
    table = compare_strategies(strategies_results)

    display = table.copy()
    display["annual_return"] = display["annual_return"].apply(
        lambda x: f"{x:+.2%}")
    display["annual_vol"] = display["annual_vol"].apply(
        lambda x: f"{x:.2%}")
    display["sharpe"] = display["sharpe"].apply(lambda x: f"{x:.3f}")
    display["sortino"] = display["sortino"].apply(lambda x: f"{x:.3f}")
    display["max_drawdown"] = display["max_drawdown"].apply(
        lambda x: f"{x:.2%}")
    display["calmar"] = display["calmar"].apply(lambda x: f"{x:.3f}")
    display["avg_exposure"] = display["avg_exposure"].apply(
        lambda x: f"{x:.1%}")
    display["turnover"] = display["turnover"].apply(lambda x: f"{x:.2f}")

    st.dataframe(display, use_container_width=True, hide_index=True)

    # Graphique Sharpe
    st.subheader("Sharpe ratio par stratégie")
    fig_sharpe = go.Figure()
    colors_bar = ["#28a745" if i == 0 else "#4c72b0"
                   for i in range(len(table))]
    fig_sharpe.add_trace(go.Bar(
        x=table["sharpe"],
        y=table["strategy"],
        orientation="h",
        marker=dict(color=colors_bar),
        text=table["sharpe"].round(3),
        textposition="outside",
    ))
    fig_sharpe.update_layout(
        height=350,
        xaxis_title="Sharpe ratio", yaxis_title="",
        margin=dict(t=20, b=20),
    )
    st.plotly_chart(fig_sharpe, use_container_width=True)

    # Courbes cumulées
    st.subheader("Rendement cumulé")
    fig_cum = go.Figure()
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

    for (name, res), color in zip(strategies_results.items(), colors):
        fig_cum.add_trace(go.Scatter(
            x=res.cumulative_returns.index,
            y=res.cumulative_returns.values,
            name=name, line=dict(color=color, width=1.5),
        ))

    fig_cum.update_layout(
        height=450,
        xaxis_title="Date", yaxis_title="Indice base 100",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(t=60, b=20),
    )
    st.plotly_chart(fig_cum, use_container_width=True)

    # Rolling Sharpe
    st.subheader("Sharpe glissant (252 jours)")
    fig_rs = go.Figure()
    for (name, res), color in zip(strategies_results.items(), colors):
        rs = rolling_sharpe(res.portfolio_returns, window=252)
        fig_rs.add_trace(go.Scatter(
            x=rs.index, y=rs.values,
            name=name, line=dict(color=color, width=1.2),
        ))

    fig_rs.add_hline(y=0, line_dash="dash", line_color="gray")
    fig_rs.update_layout(
        height=400,
        xaxis_title="Date", yaxis_title="Sharpe annualisé",
        margin=dict(t=20, b=20),
    )
    st.plotly_chart(fig_rs, use_container_width=True)

    # Meilleure stratégie
    best = table.iloc[0]
    st.markdown(
        f"""
        <div class="winner">
            <h3>🥇 Meilleure stratégie : {best['strategy']}</h3>
            <p><b>Sharpe</b> : {best['sharpe']:.3f}</p>
            <p><b>Rendement annualisé</b> : {best['annual_return']:+.2%}</p>
            <p><b>Max drawdown</b> : {best['max_drawdown']:.2%}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ===========================================================================
# ONGLET 5 — À propos
# ===========================================================================

with tab5:
    st.header("ℹ️ À propos")

    st.markdown("""
    ## 🎯 Objectif du projet

    Ce projet implémente un **détecteur de régimes de marché** via
    **Markov Switching** (Hamilton, 1989), appliqué à 3 classes d'actifs :
    cryptomonnaies, actions US et actions BRVM.

    ## 🧠 Méthodologie

    1. **Fit** d'un modèle Markov Switching à 3 régimes sur la volatilité
    2. **Identification** des régimes par classification de leur volatilité
    3. **Backtesting** de 5 stratégies d'allocation dynamique
    4. **Tests statistiques** de significativité (t-tests)

    ## 📊 Découvertes clés

    - **BRVM** : décorrélée des marchés mondiaux (< 0.1)
    - **BRVM** : régime High Vol positivement rémunéré (spécificité frontière)
    - **US** : stratégie agressive significative (p = 0.04, drawdown -42%)
    - **Crypto** : aucune stratégie significative sur la période

    ## 📚 Références

    - Hamilton, J. D. (1989). *A New Approach to the Economic Analysis
      of Nonstationary Time Series and the Business Cycle.*
    - Ang, A., & Bekaert, G. (2002). *Regime Switches in Interest Rates.*
    - Guidolin, M., & Timmermann, A. (2007). *Asset allocation under
      multivariate regime switching.*

    ## 👤 Auteur

    **Statby2Mf** — Master 2 Statistique, Université Gaston Berger
    de Saint-Louis
    """)


# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------

st.markdown("---")
st.caption(
    "📊 **HMM Regime Detection** — Statby2Mf — "
    "M2 Statistique, UGB Saint-Louis — "
    "Références : Hamilton (1989), Ang & Bekaert (2002)"
)
