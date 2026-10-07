"""
Modèle HMM (Hidden Markov Model) pour la détection de régimes de marché.

Ce module fournit :
    - Un fit HMM gaussien sur un indice composite (rendements + vol)
    - L'inférence des états cachés via l'algorithme de Viterbi
    - L'identification automatique des régimes (bull / sideways / bear)
    - La prédiction du régime courant et des probabilités de transition

Approche :
    1. Pour chaque classe d'actifs (crypto, US, BRVM), on construit un
       indice composite (rendement moyen + vol moyenne).
    2. On fit un GaussianHMM sur ces features 2D.
    3. On identifie les régimes en classant par rendement moyen.

Références :
    - Hamilton, J. D. (1989). A New Approach to the Economic Analysis
      of Nonstationary Time Series and the Business Cycle.
    - Ang, A., & Bekaert, G. (2002). Regime Switches in Interest Rates.
    - Guidolin, M., & Timmermann, A. (2007). Asset allocation under
      multivariate regime switching.

Auteur : Statby2Mf
Projet : HMM Regime Detection (M2 Statistique, UGB Saint-Louis)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd

# Import robuste : sklearn.hmm (sklearn >= 1.5) OU hmmlearn (fallback)
try:
    from sklearn.hmm import GaussianHMM
    _HMM_SOURCE = "sklearn"
except ImportError:
    try:
        from hmmlearn.hmm import GaussianHMM
        _HMM_SOURCE = "hmmlearn"
    except ImportError:
        raise ImportError(
            "Impossible d'importer GaussianHMM.\n"
            "Installe scikit-learn >= 1.5 ou hmmlearn :\n"
            "  pip install --upgrade scikit-learn\n"
            "  OU\n"
            "  pip install --only-binary :all: hmmlearn"
        )

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
    Résultat d'un fit HMM sur une classe d'actifs.

    Attributes
    ----------
    asset_class : str
        'crypto', 'us', ou 'brvm'.
    tickers : list[str]
        Tickers inclus dans la classe.
    model : GaussianHMM
        Le modèle fitté (objet sklearn).
    states : pd.Series
        Série des états inférés (0, 1, 2) indexée par date.
    state_proba : pd.DataFrame
        Probabilités a posteriori de chaque état (shape n×K).
    regime_labels : pd.Series
        États renommés en {Bear, Sideways, Bull} (mappage automatique).
    regime_stats : pd.DataFrame
        Stats par régime (rendement moyen, vol moyenne, fréquence, durée).
    transition_matrix : pd.DataFrame
        Matrice de transition entre régimes nommés.
    composite_returns : pd.Series
        Rendement composite utilisé comme feature.
    composite_vol : pd.Series
        Volatilité composite utilisée comme feature.
    n_states : int
        Nombre d'états (3).
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
            f"═══ HMM Régimes — {self.asset_class.upper()} ═══",
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

    Parameters
    ----------
    returns : pd.DataFrame
        Log-rendements (%) — colonnes = tickers.
    vol : pd.DataFrame
        Volatilité roulante 20j.
    asset_class : str
        'crypto', 'us', 'brvm'.

    Returns
    -------
    (composite_returns, composite_vol) : tuple of pd.Series
        Moyenne équipondérée par date.
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
# Fit HMM
# ---------------------------------------------------------------------------

def fit_hmm(
    features: np.ndarray,
    n_states: int = 3,
    n_iter: int = 500,
    random_state: int = 42,
    covariance_type: str = "full",
) -> GaussianHMM:
    """
    Fit un GaussianHMM sur les features 2D.

    Parameters
    ----------
    features : np.ndarray
        Matrice (n_obs, n_features). Typiquement (n, 2) = (ret, vol).
    n_states : int
        Nombre d'états cachés. Défaut : 3.
    n_iter : int
        Nombre max d'itérations Baum-Welch. Défaut : 500.
    random_state : int
        Seed pour reproductibilité.
    covariance_type : str
        'full', 'diag', 'tied', 'spherical'. Défaut : 'full'.

    Returns
    -------
    GaussianHMM
        Modèle fitté.
    """
    model = GaussianHMM(
        n_components=n_states,
        covariance_type=covariance_type,
        n_iter=n_iter,
        random_state=random_state,
        tol=1e-4,
    )
    model.fit(features)
    return model


def compute_aic_bic(model: GaussianHMM, n_obs: int, n_features: int = 2) -> tuple[float, float]:
    """
    Calcule AIC et BIC pour un HMM.

    Nombre de paramètres :
        K - 1              (transitions, chaque ligne somme à 1)
        + K*(K-1)          (transitions complètes)
        + K * n_features   (moyennes)
        + K * n_features^2 (covariances full)
    """
    K = model.n_components
    n_params = K * (K - 1) + K * n_features + K * n_features * n_features
    log_lik = model.score(features=None) if hasattr(model, "score") else model.score_.x  # fallback

    # ⚠️ sklearn ne renvoie pas directement la log-vraisemblance totale après fit
    # → on la récupère via model.score(samples) qui est la log-vraisemblance MOYENNE
    # On multiplie par n_obs pour retrouver la log-vraisemblance totale
    try:
        avg_log_lik = model.score(None)  # marche pas toujours
    except Exception:
        avg_log_lik = 0.0

    # Fallback : utiliser l'attribut interne
    if hasattr(model, "monitor_") and hasattr(model.monitor_, "history"):
        log_lik = model.monitor_.history[-1]
    else:
        log_lik = avg_log_lik

    aic = -2 * log_lik + 2 * n_params
    bic = -2 * log_lik + n_params * np.log(n_obs)
    return float(aic), float(bic)


# ---------------------------------------------------------------------------
# Identification des régimes
# ---------------------------------------------------------------------------

def identify_regimes(
    model: GaussianHMM,
    states: np.ndarray,
    composite_returns: pd.Series,
    composite_vol: pd.Series,
) -> dict[int, str]:
    """
    Identifie les régimes en les classant par rendement moyen.

    Règle :
        - Rendement le plus faible → 'Bear'
        - Rendement intermédiaire → 'Sideways'
        - Rendement le plus élevé → 'Bull'

    Parameters
    ----------
    model : GaussianHMM
    states : np.ndarray
        Séquence d'états (0..K-1) indexée par date.
    composite_returns : pd.Series
    composite_vol : pd.Series

    Returns
    -------
    dict
        {state_id: regime_label}
    """
    # Calcule le rendement moyen de chaque état
    state_returns = {}
    for k in range(model.n_components):
        mask = states == k
        if mask.sum() > 0:
            state_returns[k] = composite_returns.iloc[mask].mean() if hasattr(composite_returns, 'iloc') else composite_returns[mask].mean()
        else:
            state_returns[k] = np.nan

    # Trie par rendement croissant
    sorted_states = sorted(state_returns.items(), key=lambda x: x[1])

    # Mappe : plus faible rendement → Bear, plus haut → Bull
    mapping = {}
    if len(sorted_states) >= 1:
        mapping[sorted_states[0][0]] = "Bear"
    if len(sorted_states) >= 3:
        mapping[sorted_states[1][0]] = "Sideways"
    if len(sorted_states) >= 2:
        mapping[sorted_states[-1][0]] = "Bull"
    return mapping


def compute_regime_stats(
    states_labeled: pd.Series,
    composite_returns: pd.Series,
    composite_vol: pd.Series,
) -> pd.DataFrame:
    """
    Calcule les statistiques descriptives par régime.

    Returns
    -------
    pd.DataFrame
        Index = régimes, colonnes = [mean_return, mean_vol, frequency, avg_duration]
    """
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

        # Durée moyenne = longueur moyenne des blocs consécutifs
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


def compute_transition_matrix(
    states_labeled: pd.Series,
) -> pd.DataFrame:
    """
    Calcule la matrice de transition empirique entre régimes nommés.

    Returns
    -------
    pd.DataFrame
        Matrice 3×3 (Bear, Sideways, Bull), valeurs = probabilités.
    """
    # Compte les transitions
    tm = pd.crosstab(
        states_labeled.shift(1),
        states_labeled,
        normalize="index",
    )
    # Réindexe dans l'ordre canonique
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
    n_iter: int = 500,
    random_state: int = 42,
) -> RegimeFit:
    """
    Pipeline complet : composite → HMM → états → régimes identifiés.

    Parameters
    ----------
    returns : pd.DataFrame
        Log-rendements (%) de tous les actifs.
    vol : pd.DataFrame
        Volatilité roulante 20j.
    asset_class : str
    n_states : int
    n_iter : int
    random_state : int

    Returns
    -------
    RegimeFit
    """
    # 1. Composite
    comp_ret, comp_vol = build_composite_index(returns, vol, asset_class)

    # 2. Features 2D (avec dropna pour aligner)
    features_df = pd.DataFrame({
        "return": comp_ret,
        "vol": comp_vol,
    }).dropna()

    features = features_df.values
    dates = features_df.index

    # 3. Fit HMM
    model = fit_hmm(features, n_states=n_states,
                    n_iter=n_iter, random_state=random_state)

    # 4. Inférence des états (Viterbi)
    states_arr = model.predict(features)
    states = pd.Series(states_arr, index=dates, name="state")

    # 5. Probabilités a posteriori
    proba_arr = model.predict_proba(features)
    state_proba = pd.DataFrame(
        proba_arr,
        index=dates,
        columns=[f"state_{k}" for k in range(n_states)],
    )

    # 6. Identification des régimes
    state_to_regime = identify_regimes(
        model, states_arr,
        features_df["return"].reset_index(drop=True),
        features_df["vol"].reset_index(drop=True),
    )
    regime_labels = states.map(state_to_regime).rename("regime")

    # 7. Stats par régime
    comp_ret_aligned = comp_ret.loc[dates]
    comp_vol_aligned = comp_vol.loc[dates]
    regime_stats = compute_regime_stats(regime_labels, comp_ret_aligned, comp_vol_aligned)

    # 8. Matrice de transition
    tm = compute_transition_matrix(regime_labels)

    # 9. AIC / BIC
    log_lik = float(model.monitor_.history[-1]) if hasattr(model, "monitor_") else float(model.score(features) * len(features))
    n_params = n_states * (n_states - 1) + n_states * 2 + n_states * 2 * 2
    aic = -2 * log_lik + 2 * n_params
    bic = -2 * log_lik + n_params * np.log(len(features))

    return RegimeFit(
        asset_class=asset_class,
        tickers=ASSET_CLASSES_MAP[asset_class],
        model=model,
        states=states,
        state_proba=state_proba,
        regime_labels=regime_labels,
        regime_stats=regime_stats,
        transition_matrix=tm,
        composite_returns=comp_ret_aligned,
        composite_vol=comp_vol_aligned,
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
    n_iter: int = 500,
) -> dict[str, RegimeFit]:
    """
    Fit un HMM pour chaque classe d'actifs.

    Returns
    -------
    dict of {asset_class: RegimeFit}
    """
    results = {}
    for asset_class in ASSET_CLASSES_MAP.keys():
        print(f"\n🔄 Fit HMM — {asset_class.upper()}…")
        try:
            results[asset_class] = fit_regime_model(
                returns, vol, asset_class,
                n_states=n_states, n_iter=n_iter,
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
    print("🔍 Test du module HMM Regime Detection")
    print("=" * 70)

    # 1. Charger les features
    print("\n📊 Chargement des features…")
    returns, vol = get_all_features(vol_window=20)
    print(f"   Rendements : {returns.shape}")
    print(f"   Vol 20j    : {vol.shape}")
    print(f"   Période    : {returns.index[0].date()} → {returns.index[-1].date()}")

    # 2. Fit HMM pour chaque classe
    print("\n" + "=" * 70)
    print("🔄 Fit des HMM (3 classes)")
    print("=" * 70)
    results = fit_all_classes(returns, vol, n_states=3, n_iter=500)

    # 3. Résumé par classe
    for asset_class, fit in results.items():
        print("\n" + "=" * 70)
        print(fit.summary())
        print("=" * 70)

    # 4. Comparaison inter-classes
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

    # 5. Régimes actuels
    print("\n" + "=" * 70)
    print(f"🎯 RÉGIMES ACTUELS (au {returns.index[-1].date()})")
    print("=" * 70)
    for asset_class, fit in results.items():
        current = fit.regime_labels.iloc[-1]
        proba = fit.state_proba.iloc[-1]
        print(f"   {asset_class.upper():8s} : {current:10s} "
              f"(p={proba.max():.2%})")
