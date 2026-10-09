"""
Backtesting avancé des stratégies conditionnelles aux régimes.

Fournit :
    - Rolling Sharpe ratio (fenêtre glissante)
    - Performance conditionnelle par régime
    - Tests statistiques de significativité
    - Analyse fine des drawdowns
    - Comparaisons multi-classes

Auteur : Statby2Mf
Projet : HMM Regime Detection (M2 Statistique, UGB Saint-Louis)
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ttest_1samp

from src.strategy_adaptation import (
    StrategyResult,
    TRADING_DAYS,
    RISK_FREE_RATE,
)
from src.hmm_model import RegimeFit, REGIME_LABELS


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR = PROJECT_ROOT / "data"


# ---------------------------------------------------------------------------
# Rolling Sharpe
# ---------------------------------------------------------------------------

def rolling_sharpe(
    returns: pd.Series,
    window: int = TRADING_DAYS,
    risk_free: float = RISK_FREE_RATE,
) -> pd.Series:
    """
    Sharpe ratio sur fenêtre glissante.

    Parameters
    ----------
    returns : pd.Series
        Rendements (%) du portefeuille.
    window : int
        Taille de la fenêtre. Défaut : 252 jours (1 an).
    risk_free : float
        Taux sans risque annualisé.

    Returns
    -------
    pd.Series
        Sharpe glissant annualisé.
    """
    r = returns / 100.0
    mean = r.rolling(window).mean() * TRADING_DAYS
    vol = r.rolling(window).std() * np.sqrt(TRADING_DAYS)
    sharpe = (mean - risk_free) / vol
    return sharpe


# ---------------------------------------------------------------------------
# Performance conditionnelle par régime
# ---------------------------------------------------------------------------

def performance_by_regime(
    strategy_results: dict[str, StrategyResult],
    regime_labels: pd.Series,
) -> pd.DataFrame:
    """
    Calcule le rendement moyen et la volatilité de chaque stratégie
    DANS chaque régime.

    Parameters
    ----------
    strategy_results : dict {strategy_name: StrategyResult}
    regime_labels : pd.Series
        Régimes détectés.

    Returns
    -------
    pd.DataFrame
        Colonnes : strategy, regime, mean_return, mean_vol, sharpe, n_days
    """
    rows = []
    for strat_name, res in strategy_results.items():
        # Alignement
        df = pd.DataFrame({
            "returns": res.portfolio_returns,
            "regime": regime_labels,
        }).dropna()

        for regime in REGIME_LABELS:
            mask = df["regime"] == regime
            n = mask.sum()
            if n < 5:
                continue

            r = df.loc[mask, "returns"] / 100.0
            mean_daily = r.mean()
            vol_daily = r.std()

            # Sharpe annualisé (approximatif pour un régime)
            sharpe = (mean_daily * TRADING_DAYS - RISK_FREE_RATE) / \
                     (vol_daily * np.sqrt(TRADING_DAYS)) if vol_daily > 0 else np.nan

            rows.append({
                "strategy": strat_name,
                "regime": regime,
                "mean_return": float(mean_daily * 100),
                "mean_vol": float(vol_daily * 100),
                "sharpe": float(sharpe),
                "n_days": int(n),
            })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Test de significativité du Sharpe
# ---------------------------------------------------------------------------

def test_sharpe_significance(
    returns: pd.Series,
    risk_free: float = RISK_FREE_RATE,
) -> dict:
    """
    Test t sur la moyenne des rendements excédentaires.

    H0 : Sharpe = 0 (le rendement moyen est égal au taux sans risque)
    H1 : Sharpe > 0

    Approche : t-test unilatéral sur r - rf/252.

    Returns
    -------
    dict
        {'sharpe': ..., 't_stat': ..., 'p_value': ..., 'significant': ...}
    """
    r = returns / 100.0
    daily_rf = risk_free / TRADING_DAYS
    excess = r - daily_rf
    excess = excess.dropna()

    if len(excess) < 30:
        return {"sharpe": np.nan, "t_stat": np.nan,
                "p_value": np.nan, "significant": False}

    # Sharpe annualisé
    sharpe = (excess.mean() * TRADING_DAYS) / \
             (excess.std() * np.sqrt(TRADING_DAYS)) if excess.std() > 0 else 0

    # t-test : H0 mean(excess) = 0
    t, p_two_sided = ttest_1samp(excess, 0.0)
    p_one_sided = p_two_sided / 2 if t > 0 else 1 - p_two_sided / 2

    return {
        "sharpe": float(sharpe),
        "t_stat": float(t),
        "p_value": float(p_one_sided),
        "significant": bool(p_one_sided < 0.05 and t > 0),
    }


# ---------------------------------------------------------------------------
# Analyse des drawdowns
# ---------------------------------------------------------------------------

def analyze_drawdowns(cumulative_returns: pd.Series) -> dict:
    """
    Analyse fine des drawdowns.

    Returns
    -------
    dict
        max_dd, max_dd_duration, current_dd, recovery_date
    """
    cum = cumulative_returns
    peak = cum.cummax()
    dd = (cum - peak) / peak

    max_dd = float(dd.min())
    max_dd_date = dd.idxmin()

    # Durée du drawdown max
    if max_dd < 0:
        # Trouve le pic précédent
        peak_before = peak.loc[:max_dd_date].idxmax() if len(peak.loc[:max_dd_date]) > 0 else None
        if peak_before is not None:
            # Trouve la récupération (retour au peak)
            after = cum.loc[max_dd_date:]
            recovered = after[after >= peak.loc[peak_before]]
            if len(recovered) > 0:
                recovery_date = recovered.index[0]
                duration = (recovery_date - peak_before).days
            else:
                recovery_date = None
                duration = None
        else:
            recovery_date = None
            duration = None
    else:
        recovery_date = None
        duration = None

    current_dd = float(dd.iloc[-1])

    return {
        "max_dd": max_dd,
        "max_dd_date": max_dd_date,
        "max_dd_duration_days": duration,
        "current_dd": current_dd,
        "recovery_date": recovery_date,
    }


# ---------------------------------------------------------------------------
# Visualisations
# ---------------------------------------------------------------------------

def plot_rolling_sharpe(
    strategies_results: dict[str, StrategyResult],
    asset_class: str,
    window: int = TRADING_DAYS,
    save_path: Path | None = None,
) -> plt.Figure:
    """Graphique : rolling Sharpe pour chaque stratégie."""
    fig, ax = plt.subplots(figsize=(13, 5))

    colors = plt.cm.tab10(np.linspace(0, 1, len(strategies_results)))

    for (name, res), color in zip(strategies_results.items(), colors):
        rs = rolling_sharpe(res.portfolio_returns, window=window)
        ax.plot(rs.index, rs.values, label=name, color=color, linewidth=1.3)

    ax.axhline(0, color="black", linewidth=0.5)
    ax.axhline(1, color="green", linewidth=0.5, linestyle="--", alpha=0.5)
    ax.axhline(-1, color="red", linewidth=0.5, linestyle="--", alpha=0.5)

    ax.set_title(f"{asset_class.upper()} — Sharpe glissant ({window}j)",
                 fontsize=12, fontweight="bold", color="#1a3a5c")
    ax.set_xlabel("Date")
    ax.set_ylabel("Sharpe annualisé")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    return fig


def plot_performance_heatmap(
    perf_df: pd.DataFrame,
    asset_class: str,
    metric: str = "sharpe",
    save_path: Path | None = None,
) -> plt.Figure:
    """Heatmap : performance par stratégie × régime."""
    pivot = perf_df.pivot_table(
        index="strategy", columns="regime",
        values=metric, aggfunc="mean",
    )
    # Réordonne les colonnes
    pivot = pivot.reindex(columns=[r for r in REGIME_LABELS if r in pivot.columns])

    fig, ax = plt.subplots(figsize=(8, 5))

    im = ax.imshow(pivot.values, cmap="RdYlGn", aspect="auto",
                   vmin=-1.5, vmax=1.5)

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)

    # Annotations
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.values[i, j]
            if np.isnan(val):
                text = "—"
                color = "black"
            else:
                text = f"{val:.2f}"
                color = "white" if abs(val) > 1.0 else "black"
            ax.text(j, i, text, ha="center", va="center",
                    color=color, fontsize=10, fontweight="bold")

    ax.set_title(f"{asset_class.upper()} — {metric.capitalize()} par régime",
                 fontsize=12, fontweight="bold", color="#1a3a5c")
    ax.set_xlabel("Régime")
    ax.set_ylabel("Stratégie")

    fig.colorbar(im, ax=ax, label=metric.capitalize())
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    return fig


def plot_cumulative_comparison(
    strategies_results: dict[str, StrategyResult],
    asset_class: str,
    save_path: Path | None = None,
) -> plt.Figure:
    """Graphique : courbes cumulées de toutes les stratégies."""
    fig, ax = plt.subplots(figsize=(13, 5))

    colors = plt.cm.tab10(np.linspace(0, 1, len(strategies_results)))

    for (name, res), color in zip(strategies_results.items(), colors):
        ax.plot(res.cumulative_returns.index,
                res.cumulative_returns.values,
                label=name, color=color, linewidth=1.5)

    ax.set_title(f"{asset_class.upper()} — Rendement cumulé (base 100)",
                 fontsize=12, fontweight="bold", color="#1a3a5c")
    ax.set_xlabel("Date")
    ax.set_ylabel("Indice base 100")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    return fig


# ---------------------------------------------------------------------------
# Rapport complet par classe
# ---------------------------------------------------------------------------

def full_backtest_report(
    strategy_results: dict[str, StrategyResult],
    fit: RegimeFit,
    asset_class: str,
) -> dict:
    """
    Rapport complet de backtest pour une classe.

    Returns
    -------
    dict
        - perf_by_regime : DataFrame
        - significance : DataFrame
        - drawdowns : DataFrame
    """
    perf = performance_by_regime(strategy_results, fit.regime_labels)

    sig_rows = []
    for name, res in strategy_results.items():
        sig = test_sharpe_significance(res.portfolio_returns)
        dd = analyze_drawdowns(res.cumulative_returns)
        sig_rows.append({
            "strategy": name,
            "sharpe": sig["sharpe"],
            "t_stat": sig["t_stat"],
            "p_value": sig["p_value"],
            "significant": sig["significant"],
            "max_dd": dd["max_dd"],
            "max_dd_days": dd["max_dd_duration_days"],
        })
    sig_df = pd.DataFrame(sig_rows).sort_values("sharpe", ascending=False)

    return {
        "perf_by_regime": perf,
        "significance": sig_df,
        "asset_class": asset_class,
    }


# ---------------------------------------------------------------------------
# Test rapide
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from src.data_manager import get_all_features
    from src.hmm_model import fit_regime_model, ASSET_CLASSES_MAP
    from src.strategy_adaptation import run_all_strategies

    print("=" * 70)
    print("🔍 Backtesting avancé")
    print("=" * 70)

    # 1. Features
    returns, vol = get_all_features(vol_window=20)

    # 2. Fit + backtest pour chaque classe
    print("\n🔄 Fit des modèles + backtest…")

    for asset_class in ASSET_CLASSES_MAP.keys():
        print(f"\n{'═' * 70}")
        print(f"🪙  {asset_class.upper()}")
        print('═' * 70)

        # Fit HMM
        fit = fit_regime_model(
            returns, vol, asset_class,
            n_states=3, maxiter=1000, feature="vol",
        )

        # Backtest
        strategies_results = run_all_strategies(fit)

        # Rapport complet
        report = full_backtest_report(strategies_results, fit, asset_class)

        # Significativité
        print("\n📊 Significativité des Sharpe (t-test unilatéral)")
        sig = report["significance"].copy()
        sig["sharpe"] = sig["sharpe"].apply(lambda x: f"{x:.3f}")
        sig["t_stat"] = sig["t_stat"].apply(lambda x: f"{x:+.2f}")
        sig["p_value"] = sig["p_value"].apply(lambda x: f"{x:.4f}")
        sig["max_dd"] = sig["max_dd"].apply(lambda x: f"{x:.2%}")
        print(sig.to_string(index=False))

        # Performance par régime
        print("\n📊 Performance par régime (Sharpe)")
        perf = report["perf_by_regime"]
        pivot = perf.pivot_table(
            index="strategy", columns="regime",
            values="sharpe", aggfunc="mean",
        ).round(3)
        print(pivot.to_string())

        # Graphiques
        plot_rolling_sharpe(
            strategies_results, asset_class,
            save_path=FIGURES_DIR / f"{asset_class}_rolling_sharpe.png",
        )
        plot_performance_heatmap(
            report["perf_by_regime"], asset_class,
            save_path=FIGURES_DIR / f"{asset_class}_perf_heatmap.png",
        )
        plot_cumulative_comparison(
            strategies_results, asset_class,
            save_path=FIGURES_DIR / f"{asset_class}_cumulative.png",
        )

    # 3. Sauvegarde globale
    print("\n" + "=" * 70)
    print("💾 Sauvegarde des rapports")
    print("=" * 70)

    all_reports = {}
    for asset_class in ASSET_CLASSES_MAP.keys():
        fit = fit_regime_model(returns, vol, asset_class,
                               n_states=3, maxiter=1000, feature="vol")
        strategies_results = run_all_strategies(fit)
        report = full_backtest_report(strategies_results, fit, asset_class)
        all_reports[asset_class] = report

    # Concatène les significativités
    sigs = []
    for cls, rep in all_reports.items():
        df = rep["significance"].copy()
        df.insert(0, "asset_class", cls)
        sigs.append(df)
    df_sig = pd.concat(sigs, ignore_index=True)
    df_sig.to_csv(DATA_DIR / "backtest_significance.csv", index=False)
    print(f"💾 Significativité : {DATA_DIR / 'backtest_significance.csv'}")

    # Concatène les performances par régime
    perfs = []
    for cls, rep in all_reports.items():
        df = rep["perf_by_regime"].copy()
        df.insert(0, "asset_class", cls)
        perfs.append(df)
    df_perf = pd.concat(perfs, ignore_index=True)
    df_perf.to_csv(DATA_DIR / "backtest_perf_by_regime.csv", index=False)
    print(f"💾 Perf par régime : {DATA_DIR / 'backtest_perf_by_regime.csv'}")

    print("\n" + "=" * 70)
    print("✅ Backtesting avancé terminé")
    print("=" * 70)
