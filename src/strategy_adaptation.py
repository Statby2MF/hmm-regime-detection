"""
Stratégies d'investissement conditionnelles aux régimes.

Ce module fournit :
    - 5 stratégies d'allocation dynamique basées sur les régimes
    - Backtesting complet avec métriques (Sharpe, drawdown, turnover)
    - Comparaison avec le Buy & Hold
    - Éviter le look-ahead bias (décision sur régime[t-1])

Auteur : Statby2Mf
Projet : HMM Regime Detection (M2 Statistique, UGB Saint-Louis)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from src.hmm_model import RegimeFit, REGIME_LABELS


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

#: Rendement sans risque annualisé (approximation 2021-2026)
RISK_FREE_RATE = 0.02   # 2% par an

#: Jours de trading par an
TRADING_DAYS = 252


# ---------------------------------------------------------------------------
# Structures
# ---------------------------------------------------------------------------

@dataclass
class StrategyResult:
    """
    Résultat d'une stratégie après backtest.

    Attributes
    ----------
    name : str
        Nom de la stratégie.
    weights : pd.Series
        Poids investi (0 à 1) à chaque date.
    portfolio_returns : pd.Series
        Rendements du portefeuille après application des poids.
    cumulative_returns : pd.Series
        Rendements cumulés (base 100).
    drawdown : pd.Series
        Série de drawdown (négatif).
    metrics : dict
        Métriques de performance.
    """
    name: str
    weights: pd.Series
    portfolio_returns: pd.Series
    cumulative_returns: pd.Series
    drawdown: pd.Series
    metrics: dict

    def summary(self) -> str:
        m = self.metrics
        return (
            f"═══ {self.name} ═══\n"
            f"  Rendement annualisé : {m['annual_return']:+.2%}\n"
            f"  Volatilité annuelle : {m['annual_vol']:.2%}\n"
            f"  Sharpe ratio        : {m['sharpe']:.3f}\n"
            f"  Sortino ratio       : {m['sortino']:.3f}\n"
            f"  Max drawdown        : {m['max_drawdown']:.2%}\n"
            f"  Calmar ratio        : {m['calmar']:.3f}\n"
            f"  Turnover annuel     : {m['turnover']:.2f}\n"
            f"  Exposition moyenne  : {m['avg_exposure']:.1%}\n"
            f"  Rendement cumulé    : {m['total_return']:+.2%}"
        )


# ---------------------------------------------------------------------------
# Fonctions de calcul des poids
# ---------------------------------------------------------------------------

def weights_buy_hold(
    regime_labels: pd.Series,
) -> pd.Series:
    """Stratégie 1 : 100% investi en permanence."""
    return pd.Series(1.0, index=regime_labels.index, name="Buy & Hold")


def weights_regime_switch_simple(
    regime_labels: pd.Series,
) -> pd.Series:
    """
    Stratégie 2 : Régime-Switch simple.
        Low Vol  : 100%
        Med Vol  : 75%
        High Vol : 25%
    """
    mapping = {"Low Vol": 1.00, "Med Vol": 0.75, "High Vol": 0.25}
    return regime_labels.map(mapping).rename("Regime-Switch Simple")


def weights_regime_switch_aggressive(
    regime_labels: pd.Series,
) -> pd.Series:
    """
    Stratégie 3 : Régime-Switch agressif.
        Low Vol  : 100%
        Med Vol  : 100%
        High Vol : 0%
    """
    mapping = {"Low Vol": 1.00, "Med Vol": 1.00, "High Vol": 0.00}
    return regime_labels.map(mapping).rename("Regime-Switch Agressif")


def weights_regime_cash(
    regime_labels: pd.Series,
) -> pd.Series:
    """
    Stratégie 4 : Régime-Cash (très prudent).
        Low Vol  : 100%
        Med Vol  : 50%
        High Vol : 0%
    """
    mapping = {"Low Vol": 1.00, "Med Vol": 0.50, "High Vol": 0.00}
    return regime_labels.map(mapping).rename("Regime-Cash")


def weights_inverted_regime(
    regime_labels: pd.Series,
) -> pd.Series:
    """
    Stratégie 5 : Régime inversé (adapté à la BRVM).
        Low Vol  : 30%
        Med Vol  : 70%
        High Vol : 100%

    ⚠️ Basée sur la découverte : sur BRVM, High Vol = meilleur rendement.
    Attention : ne PAS utiliser sur crypto/US !
    """
    mapping = {"Low Vol": 0.30, "Med Vol": 0.70, "High Vol": 1.00}
    return regime_labels.map(mapping).rename("Régime Inversé (BRVM)")


def weights_vol_target(
    composite_vol: pd.Series,
    target_vol: float = 1.0,
    max_leverage: float = 1.0,
) -> pd.Series:
    """
    Stratégie 6 : Vol Target.

    Poids = target_vol / vol_actuelle, plafonné à max_leverage.

    Parameters
    ----------
    composite_vol : pd.Series
        Volatilité roulante (%).
    target_vol : float
        Volatilité cible (% journalier). Typiquement 1%.
    max_leverage : float
        Poids maximum. Défaut : 1.0 (pas de levier).
    """
    weights = (target_vol / composite_vol).clip(upper=max_leverage)
    weights = weights.fillna(0.0)
    return weights.rename("Vol Target")


# ---------------------------------------------------------------------------
# Backtesting
# ---------------------------------------------------------------------------

def compute_max_drawdown(cumulative: pd.Series) -> float:
    """Calcule le maximum drawdown d'une série cumulée."""
    peak = cumulative.cummax()
    drawdown = (cumulative - peak) / peak
    return float(drawdown.min())


def compute_drawdown_series(cumulative: pd.Series) -> pd.Series:
    """Série complète de drawdown."""
    peak = cumulative.cummax()
    return (cumulative - peak) / peak


def compute_turnover(weights: pd.Series) -> float:
    """Turnover annuel = somme des variations absolues annualisée."""
    changes = weights.diff().abs().sum()
    n_years = len(weights) / TRADING_DAYS
    if n_years == 0:
        return 0.0
    return float(changes / n_years)


def compute_metrics(
    portfolio_returns: pd.Series,
    weights: pd.Series,
    risk_free: float = RISK_FREE_RATE,
) -> dict:
    """
    Calcule toutes les métriques de performance.

    Parameters
    ----------
    portfolio_returns : pd.Series
        Rendements (%) du portefeuille.
    weights : pd.Series
        Poids utilisés.
    risk_free : float
        Taux sans risque annualisé.

    Returns
    -------
    dict
        Métriques de performance.
    """
    # Rendements en décimal
    r = portfolio_returns / 100.0

    # Annualisation
    total_return = (1 + r).prod() - 1
    n_years = len(r) / TRADING_DAYS
    annual_return = (1 + total_return) ** (1 / max(n_years, 0.01)) - 1
    annual_vol = r.std() * np.sqrt(TRADING_DAYS)

    # Sharpe
    excess = annual_return - risk_free
    sharpe = excess / annual_vol if annual_vol > 0 else np.nan

    # Sortino (downside deviation)
    downside = r[r < 0].std() * np.sqrt(TRADING_DAYS)
    sortino = excess / downside if downside > 0 else np.nan

    # Drawdown
    cum = (1 + r).cumprod()
    max_dd = compute_max_drawdown(cum)
    calmar = annual_return / abs(max_dd) if max_dd < 0 else np.nan

    return {
        "total_return": float(total_return),
        "annual_return": float(annual_return),
        "annual_vol": float(annual_vol),
        "sharpe": float(sharpe),
        "sortino": float(sortino),
        "max_drawdown": float(max_dd),
        "calmar": float(calmar),
        "turnover": compute_turnover(weights),
        "avg_exposure": float(weights.mean()),
        "n_days": len(r),
    }


def backtest_strategy(
    name: str,
    returns: pd.Series,
    weights: pd.Series,
) -> StrategyResult:
    """
    Backtest une stratégie avec poids donnés.

    ⚠️ Anti-look-ahead : on utilise weights[t-1] pour le rendement r[t].

    Parameters
    ----------
    name : str
    returns : pd.Series
        Rendements (%) de l'actif ou composite.
    weights : pd.Series
        Poids à chaque date.

    Returns
    -------
    StrategyResult
    """
    # Alignement
    df = pd.DataFrame({
        "returns": returns,
        "weights": weights,
    }).dropna()

    if len(df) < 10:
        raise ValueError(f"Pas assez d'observations ({len(df)})")

    # ⚠️ Anti-look-ahead : le poids appliqué au jour t est celui de t-1
    df["weights_applied"] = df["weights"].shift(1).fillna(df["weights"].iloc[0])

    # Rendement du portefeuille
    df["portfolio_returns"] = df["weights_applied"] * df["returns"]

    # Cumulé
    cum = (1 + df["portfolio_returns"] / 100).cumprod() * 100
    drawdown = compute_drawdown_series(cum)

    metrics = compute_metrics(df["portfolio_returns"], df["weights_applied"])

    return StrategyResult(
        name=name,
        weights=df["weights_applied"],
        portfolio_returns=df["portfolio_returns"],
        cumulative_returns=cum,
        drawdown=drawdown,
        metrics=metrics,
    )


# ---------------------------------------------------------------------------
# Pipeline complet
# ---------------------------------------------------------------------------

def run_all_strategies(
    fit: RegimeFit,
    strategies: list[str] | None = None,
    target_vol: float = 1.0,
) -> dict[str, StrategyResult]:
    """
    Lance toutes les stratégies sur un fit de régimes.

    Parameters
    ----------
    fit : RegimeFit
    strategies : list of str, optional
        Si None, exécute toutes. Sinon : 'buy_hold', 'simple', 'aggressive',
        'regime_cash', 'inverted', 'vol_target'.
    target_vol : float
        Volatilité cible pour Vol Target.

    Returns
    -------
    dict {strategy_name: StrategyResult}
    """
    if strategies is None:
        strategies = ["buy_hold", "simple", "aggressive", "regime_cash",
                      "vol_target"]
        # 'inverted' seulement si BRVM
        if fit.asset_class == "brvm":
            strategies.append("inverted")

    returns = fit.composite_returns
    regime_labels = fit.regime_labels

    results = {}

    for strat in strategies:
        if strat == "buy_hold":
            w = weights_buy_hold(regime_labels)
        elif strat == "simple":
            w = weights_regime_switch_simple(regime_labels)
        elif strat == "aggressive":
            w = weights_regime_switch_aggressive(regime_labels)
        elif strat == "regime_cash":
            w = weights_regime_cash(regime_labels)
        elif strat == "inverted":
            w = weights_inverted_regime(regime_labels)
        elif strat == "vol_target":
            w = weights_vol_target(fit.composite_vol, target_vol=target_vol)
        else:
            continue

        # Aligne les poids sur les rendements (par prudence)
        w = w.reindex(returns.index).fillna(0.0)

        try:
            results[strat] = backtest_strategy(
                name=w.name or strat, returns=returns, weights=w,
            )
        except Exception as e:
            print(f"   ⚠️ Stratégie {strat} échouée : {e}")

    return results


def compare_strategies(
    results: dict[str, StrategyResult],
) -> pd.DataFrame:
    """
    Tableau comparatif de toutes les stratégies.
    """
    rows = []
    for name, res in results.items():
        m = res.metrics
        rows.append({
            "strategy": name,
            "annual_return": m["annual_return"],
            "annual_vol": m["annual_vol"],
            "sharpe": m["sharpe"],
            "sortino": m["sortino"],
            "max_drawdown": m["max_drawdown"],
            "calmar": m["calmar"],
            "avg_exposure": m["avg_exposure"],
            "turnover": m["turnover"],
        })
    df = pd.DataFrame(rows).sort_values("sharpe", ascending=False)
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Test rapide
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from src.data_manager import get_all_features
    from src.hmm_model import fit_regime_model, ASSET_CLASSES_MAP

    print("=" * 70)
    print("🔍 Test du module Strategy Adaptation")
    print("=" * 70)

    # 1. Features
    returns, vol = get_all_features(vol_window=20)

    # 2. Fit régimes
    print("\n🔄 Fit des régimes…")
    all_results = {}
    for asset_class in ASSET_CLASSES_MAP.keys():
        print(f"   → {asset_class}…")
        fit = fit_regime_model(
            returns, vol, asset_class,
            n_states=3, maxiter=1000, feature="vol",
        )
        all_results[asset_class] = fit

    # 3. Backtest pour chaque classe
    print("\n" + "=" * 70)
    print("📊 BACKTESTING DES STRATÉGIES")
    print("=" * 70)

    for asset_class, fit in all_results.items():
        print(f"\n{'═' * 70}")
        print(f"🪙  {asset_class.upper()}")
        print('═' * 70)

        strategies_results = run_all_strategies(fit)
        table = compare_strategies(strategies_results)

        # Affichage compact
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

        print()
        print(display.to_string(index=False))

        # Meilleur Sharpe
        best = table.iloc[0]
        bh = table[table["strategy"].str.contains("Buy", case=False)]
        if len(bh) > 0:
            bh_sharpe = bh.iloc[0]["sharpe"]
            gain = (best["sharpe"] - bh_sharpe) / abs(bh_sharpe) * 100
            print(f"\n   🏆 Meilleur Sharpe : {best['strategy']} "
                  f"({best['sharpe']:.3f})")
            print(f"   📈 Gain vs Buy & Hold : {gain:+.1f}%")

    print("\n" + "=" * 70)
    print("✅ Backtest terminé")
    print("=" * 70)
