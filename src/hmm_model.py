"""
Détection de régimes de marché via Markov Switching Regression.

Ce module utilise `statsmodels.tsa.regime_switching.markov_regression`
(implémentation de référence de Hamilton 1989) pour détecter des régimes
cachés dans les séries de rendements financiers.

Approche :
    1. Pour chaque classe d'actifs (crypto, US, BRVM), on construit un
       indice composite (rendement moyen + vol moyenne).
    2. On fit un MarkovRegression avec k régimes.
    3. On identifie les régimes en classant par rendement moyen.

Références :
    - Hamilton, J. D. (1989). A New Approach to the Economic Analysis of
      Nonstationary Time Series and the Business Cycle.
    - Ang, A., & Bekaert, G. (2002). Regime Switches in Interest Rates.
    - Guidolin, M., & Timmermann, A. (2007). Asset allocation under
      multivariate regime switching.

Auteur : Statby2Mf
Projet : HMM Regime Detection (M2 Statistique, UGB Saint-Louis)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

#: Mapping classe → liste de tickers
ASSET_CLASSES_MAP: dict[str, list[str]] = {
    "crypto": ["BTC-USD", "ETH-USD"],
    "us":     ["US_SPY"],
    "brvm":   ["BRVM_SNTS", "BRVM_BOAB", "BRVM_ECOC"],
}

#: Noms de régimes (ordre d'affichage après identification)
REGIME_LABELS = ["Bear", "Sideways", "Bull"]


# ---------------------------------------------------------------------------
# Structures
# ---------------------------------------------------------------------------

@dataclass
class RegimeFit:
    """
    Résultat d'un fit Markov Switching sur une classe d'actifs.

    Attributes
    ----------
    asset_class : str
        'crypto', 'us', ou 'brvm'.
    tickers : list[str]
        Tickers inclus dans la classe.
    model : object
        Le modèle fitté (statsmodels).
    states : pd.Series
        Série des états inférés (0..K-1) indexée par date.
    state_proba : pd.DataFrame
        Probabilités a posteriori de chaque état.
    regime_labels : pd.Series
        États renommés en {Bear, Sideways, Bull}.
    regime_stats : pd.DataFrame
        Stats par régime (rendement moyen, vol, fréquence, durée).
    transition_matrix : pd.DataFrame
        Matrice de transition entre régimes nommés.
    composite_returns : pd.Series
        Rendement composite utilisé comme feature.
    composite_vol : pd.Series
        Volatilité composite utilisée comme feature.
    n_states : int
        Nombre d'états.
    log_likelihood : float
        Log-vraisemblance au point optimal.
    aic : float
        Akaike Information Criterion.
    bic : float
        Bayesian Information Criterion.
    """

    asset_class: str
    tickers: list[str]
    model: object = field(repr=False)
    states: pd.Series
    state_proba: pd.DataFrame
    regime_labels: pd.Series
    regime_stats: pd.DataFrame
    transition_matrix: pd.DataFrame
    composite_returns: pd.Series
    composite_vol: pd.Series
    n_states: int
    log_likelihood: float
    aic: float
    bic: float

    def summary(self) -> str:
        """Résumé texte lisible."""
        lines = [
            f"═══ Markov Switching — {self.asset_class.upper()} ═══",
            f"  Actifs        : {', '.join(self.tickers)}",
            f"  États         : {self.n_states}",
            f"  Observations  : {len(self.states)}",
            f"  Log-vraisemb. : {self.log_likelihood:.2f}",
            f"  AIC / BIC     : {self.aic:.2f} / {self.bic:.2f}",
            "",
            "  📊 Statistiques par régime :",
        ]
        for regime in REGIME_LABELS:
            if regime in self.regime_stats.index:
                row = self.regime_stats.loc[regime]
                lines.append(
                    f"    {regime:10s} : "
                    f"ret={row['mean_return']:+.3f}%, "
                    f"vol={row['mean_vol']:.3f}%, "
                    f"fréq={row['frequency']:.1%}, "
                    f"durée={row['avg_duration']:.1f}j"
                )
        lines.append("")
        lines.append("  🔄 Matrice de transition (en %) :")
        tm = (self.transition_matrix * 100).round(1)
        lines.append("    " + " " * 10 + "  ".join(f"{c:>7s}"
                                                    for c in tm.columns))
        for idx, row in tm.iterrows():
            lines.append(f"    {idx:10s}  "
                         + "  ".join(f"{v:>7.1f}" for v in row.values))
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Construction de l'indice composite
# ---------------------------------------------------------------------------

def build_composite_index(
    returns: pd.DataFrame,
    vol: pd.DataFrame,
    asset_class: str,
) -> tuple[pd.Series, pd.Series]:
    """
    Construit un indice composite pour une classe d'actifs.

    - Rendement composite : moyenne équipondérée des rendements
    - Volatilité composite : moyenne équipondérée des vols 20j
    """
    if asset_class not in ASSET_CLASSES_MAP:
        raise ValueError(f"Classe inconnue : {asset_class}")

    tickers = ASSET_CLASSES_MAP[asset_class]
    available = [t for t in tickers if t in returns.columns]
    if not available:
        raise ValueError(f"Aucun ticker {tickers} dans les données")

    comp_ret = returns[available].mean(axis=1)
    comp_vol = vol[available].mean(axis=1)

    comp_ret.name = f"{asset_class}_return"
    comp_vol.name = f"{asset_class}_vol"
    return comp_ret, comp_vol


# ---------------------------------------------------------------------------
# Fit Markov Switching
# ---------------------------------------------------------------------------

def fit_markov_switching(
    returns: pd.Series,
    n_states: int = 3,
    switching_variance: bool = True,
    maxiter: int = 500,
) -> MarkovRegression:
    """
    Fit un modèle Markov Switching sur une série de rendements.

    Parameters
    ----------
    returns : pd.Series
        Série de rendements (%). Doit être stationnaire.
    n_states : int
        Nombre de régimes. Défaut : 3.
    switching_variance : bool
        Si True, la variance change entre régimes (recommandé pour finance).
        Si False, seule la moyenne change.
    maxiter : int
        Nombre max d'itérations EM. Défaut : 500.

    Returns
    -------
    MarkovRegression
        Modèle fitté.
    """
    model = MarkovRegression(
        returns,
        k_regimes=n_states,
        trend="c",                    # constante par régime
        switching_variance=switching_variance,
    )
    result = model.fit(maxiter=maxiter, disp=False)
    return result


# ---------------------------------------------------------------------------
# Extraction des états et probabilités
# ---------------------------------------------------------------------------

def extract_states(result, index: pd.DatetimeIndex) -> tuple[pd.Series, pd.DataFrame]:
    """
    Extrait la séquence d'états et les probabilités a posteriori.

    Parameters
    ----------
    result : MarkovRegressionResults
        Modèle fitté.
    index : pd.DatetimeIndex
        Index des dates (aligné avec les observations).

    Returns
    -------
    (states, probabilities) : tuple of pd.Series, pd.DataFrame
    """
    # Probabilités lissées (smoothed)
    proba = result.smoothed_marginal_probabilities
    proba.index = index[:len(proba)]
    proba.columns = [f"state_{k}" for k in range(proba.shape[1])]

    # État = argmax des probabilités
    states = proba.idxmax(axis=1).str.replace("state_", "").astype(int)
    states.name = "state"

    return states, proba


# ---------------------------------------------------------------------------
# Identification des régimes
# ---------------------------------------------------------------------------

def identify_regimes(
    states: pd.Series,
    returns: pd.Series,
    n_states: int,
) -> dict[int, str]:
    """
    Identifie les régimes en classant par rendement moyen.

    Règle :
        - Rendement le plus faible → 'Bear'
        - Rendement intermédiaire → 'Sideways'
        - Rendement le plus élevé → 'Bull'
    """
    state_returns = {}
    for k in range(n_states):
        mask = states == k
        if mask.sum() > 0:
            state_returns[k] = float(returns[mask].mean())
        else:
            state_returns[k] = np.nan

    sorted_states = sorted(state_returns.items(), key=lambda x: x[1])

    mapping = {}
    if n_states == 2:
        # Bear / Bull seulement
        mapping[sorted_states[0][0]] = "Bear"
        mapping[sorted_states[-1][0]] = "Bull"
    else:
        # Bear / Sideways / Bull
        mapping[sorted_states[0][0]] = "Bear"
        if len(sorted_states) >= 3:
            mapping[sorted_states[1][0]] = "Sideways"
        mapping[sorted_states[-1][0]] = "Bull"
    return mapping


def compute_regime_stats(
    states_labeled: pd.Series,
    composite_returns: pd.Series,
    composite_vol: pd.Series,
) -> pd.DataFrame:
    """Calcule les stats descriptives par régime."""
    rows = []
    for regime in REGIME_LABELS:
        mask = states_labeled == regime
        n = mask.sum()
        if n == 0:
            rows.append({
                "regime": regime,
                "mean_return": np.nan,
                "mean_vol": np.nan,
                "frequency": 0.0,
                "avg_duration": np.nan,
            })
            continue

        # Durée moyenne des blocs consécutifs
        blocks = (states_labeled != states_labeled.shift()).cumsum()
        regime_blocks = blocks[mask].value_counts()
        avg_dur = float(regime_blocks.mean()) if len(regime_blocks) > 0 else np.nan

        rows.append({
            "regime": regime,
            "mean_return": float(composite_returns[mask].mean()),
            "mean_vol": float(composite_vol[mask].mean()),
            "frequency": float(n / len(states_labeled)),
            "avg_duration": avg_dur,
        })

    return pd.DataFrame(rows).set_index("regime")


def compute_transition_matrix(states_labeled: pd.Series) -> pd.DataFrame:
    """Calcule la matrice de transition empirique entre régimes nommés."""
    tm = pd.crosstab(
        states_labeled.shift(1),
        states_labeled,
        normalize="index",
    )
    tm = tm.reindex(index=REGIME_LABELS, columns=REGIME_LABELS, fill_value=0.0)
    return tm


# ---------------------------------------------------------------------------
# Pipeline complet
# ---------------------------------------------------------------------------

def fit_regime_model(
    returns: pd.DataFrame,
    vol: pd.DataFrame,
    asset_class: str,
    n_states: int = 3,
    switching_variance: bool = True,
    maxiter: int = 500,
) -> RegimeFit:
    """
    Pipeline complet : composite → Markov Switching → états → régimes identifiés.
    """
    # 1. Composite
    comp_ret, comp_vol = build_composite_index(returns, vol, asset_class)

    # 2. Aligne sur les mêmes dates
    features_df = pd.DataFrame({
        "return": comp_ret,
        "vol": comp_vol,
    }).dropna()

    returns_series = features_df["return"]
    vol_series = features_df["vol"]
    dates = features_df.index

    # 3. Fit Markov Switching
    result = fit_markov_switching(
        returns_series, n_states=n_states,
        switching_variance=switching_variance, maxiter=maxiter,
    )

    # 4. États et probabilités
    states, proba = extract_states(result, dates)

    # 5. Identification des régimes
    state_to_regime = identify_regimes(states, returns_series, n_states)
    regime_labels = states.map(state_to_regime).rename("regime")

    # 6. Stats par régime
    regime_stats = compute_regime_stats(
        regime_labels, returns_series, vol_series
    )

    # 7. Matrice de transition
    tm = compute_transition_matrix(regime_labels)

    # 8. AIC / BIC
    log_lik = float(result.llf)
    aic = float(result.aic)
    bic = float(result.bic)

    return RegimeFit(
        asset_class=asset_class,
        tickers=ASSET_CLASSES_MAP[asset_class],
        model=result,
        states=states,
        state_proba=proba,
        regime_labels=regime_labels,
        regime_stats=regime_stats,
        transition_matrix=tm,
        composite_returns=returns_series,
        composite_vol=vol_series,
        n_states=n_states,
        log_likelihood=log_lik,
        aic=aic,
        bic=bic,
    )


# ---------------------------------------------------------------------------
# Comparaison multi-classes
# ---------------------------------------------------------------------------

def fit_all_classes(
    returns: pd.DataFrame,
    vol: pd.DataFrame,
    n_states: int = 3,
    switching_variance: bool = True,
    maxiter: int = 500,
) -> dict[str, RegimeFit]:
    """Fit un modèle Markov Switching pour chaque classe d'actifs."""
    results = {}
    for asset_class in ASSET_CLASSES_MAP.keys():
        print(f"\n🔄 Fit Markov Switching — {asset_class.upper()}…")
        try:
            results[asset_class] = fit_regime_model(
                returns, vol, asset_class,
                n_states=n_states,
                switching_variance=switching_variance,
                maxiter=maxiter,
            )
            print(f"   ✅ OK (log-lik = {results[asset_class].log_likelihood:.2f})")
        except Exception as e:
            print(f"   ❌ Échec : {e}")
    return results


# ---------------------------------------------------------------------------
# Test rapide
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from src.data_manager import get_all_features

    print("=" * 70)
    print("🔍 Test du module Regime Detection (Markov Switching)")
    print("=" * 70)

    print("\n📊 Chargement des features…")
    returns, vol = get_all_features(vol_window=20)
    print(f"   Rendements : {returns.shape}")
    print(f"   Vol 20j    : {vol.shape}")
    print(f"   Période    : {returns.index[0].date()} → {returns.index[-1].date()}")

    print("\n" + "=" * 70)
    print("🔄 Fit des modèles Markov Switching (3 classes)")
    print("=" * 70)
    results = fit_all_classes(returns, vol, n_states=3)

    for asset_class, fit in results.items():
        print("\n" + "=" * 70)
        print(fit.summary())
        print("=" * 70)

    print("\n" + "=" * 70)
    print("📊 COMPARAISON INTER-CLASSES")
    print("=" * 70)
    print()
    print(f"{'Classe':10s} {'Régime':10s} {'Ret moy':>10s} {'Vol moy':>10s} "
          f"{'Fréq':>8s} {'Durée':>8s}")
    print("-" * 70)
    for asset_class, fit in results.items():
        for regime in REGIME_LABELS:
            if regime in fit.regime_stats.index:
                row = fit.regime_stats.loc[regime]
                print(f"{asset_class:10s} {regime:10s} "
                      f"{row['mean_return']:>+9.3f}% "
                      f"{row['mean_vol']:>9.3f}% "
                      f"{row['frequency']:>7.1%} "
                      f"{row['avg_duration']:>7.1f}j")

    print("\n" + "=" * 70)
    print(f"🎯 RÉGIMES ACTUELS (au {returns.index[-1].date()})")
    print("=" * 70)
    for asset_class, fit in results.items():
        current = fit.regime_labels.iloc[-1]
        proba = fit.state_proba.iloc[-1]
        print(f"   {asset_class.upper():8s} : {current:10s} "
              f"(p={proba.max():.2%})")
