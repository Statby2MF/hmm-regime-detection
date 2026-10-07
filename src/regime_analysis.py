"""
Analyse approfondie des régimes détectés.

Ce module fournit :
    - Visualisations des régimes (prix, vol, transitions)
    - Statistiques avancées par régime
    - Analyse des durées (distribution, histogramme)
    - Corrélations inter-classes
    - Détection et analyse des transitions

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

from src.hmm_model import RegimeFit, REGIME_LABELS, ASSET_CLASSES_MAP


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

#: Couleurs par régime
REGIME_COLORS = {
    "Low Vol": "#28a745",     # vert
    "Med Vol": "#ffc107",     # jaune
    "High Vol": "#dc3545",    # rouge
}


# ---------------------------------------------------------------------------
# Statistiques avancées
# ---------------------------------------------------------------------------

def regime_durations(regime_labels: pd.Series) -> dict[str, list[int]]:
    """
    Calcule la liste des durées de chaque régime (en jours).

    Returns
    -------
    dict {regime: [durées]}
    """
    durations = {r: [] for r in REGIME_LABELS}
    if len(regime_labels) == 0:
        return durations

    # Détecte les blocs consécutifs
    changes = (regime_labels != regime_labels.shift()).cumsum()
    for _, group in regime_labels.groupby(changes):
        regime = group.iloc[0]
        if regime in durations:
            durations[regime].append(len(group))
    return durations


def compute_transition_events(regime_labels: pd.Series) -> pd.DataFrame:
    """
    Détecte les changements de régime et renvoie leurs dates.

    Returns
    -------
    pd.DataFrame
        Colonnes : date, from_regime, to_regime
    """
    prev = regime_labels.shift(1)
    changes = regime_labels[regime_labels != prev].index

    events = []
    for date in changes:
        events.append({
            "date": date,
            "from_regime": prev.loc[date],
            "to_regime": regime_labels.loc[date],
        })
    return pd.DataFrame(events)


def test_regime_separation(fit: RegimeFit) -> pd.DataFrame:
    """
    Test simple de séparation : compare les moyennes de vol entre régimes.

    Utilise un t-test entre chaque paire de régimes.

    Returns
    -------
    pd.DataFrame
        Colonnes : regime_a, regime_b, t_stat, p_value, significant
    """
    from scipy.stats import ttest_ind

    vol = fit.composite_vol
    labels = fit.regime_labels
    rows = []

    regimes = [r for r in REGIME_LABELS if r in labels.unique()]
    for i, r_a in enumerate(regimes):
        for r_b in regimes[i + 1:]:
            a = vol[labels == r_a].dropna()
            b = vol[labels == r_b].dropna()
            if len(a) < 5 or len(b) < 5:
                continue
            t, p = ttest_ind(a, b, equal_var=False)
            rows.append({
                "regime_a": r_a,
                "regime_b": r_b,
                "mean_a": float(a.mean()),
                "mean_b": float(b.mean()),
                "t_stat": float(t),
                "p_value": float(p),
                "significant": p < 0.05,
            })
    return pd.DataFrame(rows)


def current_regime_proba(fit: RegimeFit, horizon: int = 5) -> pd.DataFrame:
    """
    Renvoie les probabilités de transition depuis le régime courant
    pour les `horizon` prochains jours.

    Utilise la matrice de transition empirique.
    """
    current = fit.regime_labels.iloc[-1]
    tm = fit.transition_matrix

    # Distribution de probabilités sur les prochains jours
    probs = pd.DataFrame(0.0, index=range(1, horizon + 1),
                          columns=REGIME_LABELS)

    # Distribution initiale (one-hot sur le régime courant)
    dist = pd.Series(0.0, index=REGIME_LABELS)
    if current in dist.index:
        dist[current] = 1.0

    for h in range(1, horizon + 1):
        dist = dist @ tm  # multiplication matricielle
        probs.loc[h] = dist

    return probs


# ---------------------------------------------------------------------------
# Visualisations
# ---------------------------------------------------------------------------

def plot_regimes_on_prices(
    fit: RegimeFit,
    asset_class: str,
    save_path: Path | None = None,
) -> plt.Figure:
    """
    Graphique : prix composite + bandes colorées par régime.
    """
    fig, ax = plt.subplots(figsize=(13, 5))

    # Prix reconstruits (normalisés à 100)
    ret = fit.composite_returns
    price = (1 + ret / 100).cumprod() * 100

    ax.plot(price.index, price.values, color="#1a3a5c", linewidth=1.2,
            label=f"{asset_class} composite (base 100)")

    # Bandes colorées par régime
    labels = fit.regime_labels
    prev_regime = None
    start_idx = None

    for i, (date, regime) in enumerate(labels.items()):
        if regime != prev_regime:
            if prev_regime is not None and start_idx is not None:
                ax.axvspan(start_idx, date,
                           color=REGIME_COLORS.get(prev_regime, "#cccccc"),
                           alpha=0.20)
            start_idx = date
            prev_regime = regime

    # Dernière bande
    if prev_regime is not None and start_idx is not None:
        ax.axvspan(start_idx, labels.index[-1],
                   color=REGIME_COLORS.get(prev_regime, "#cccccc"),
                   alpha=0.20)

    # Légende custom
    from matplotlib.patches import Patch
    legend_elements = [plt.Line2D([0], [0], color="#1a3a5c", lw=1.5,
                                   label=f"{asset_class} composite")]
    for regime in REGIME_LABELS:
        legend_elements.append(Patch(facecolor=REGIME_COLORS[regime],
                                       alpha=0.4, label=regime))
    ax.legend(handles=legend_elements, loc="upper left", fontsize=9)

    ax.set_title(f"{asset_class.upper()} — Régimes détectés sur le prix composite",
                 fontsize=12, fontweight="bold", color="#1a3a5c")
    ax.set_ylabel("Indice base 100")
    ax.set_xlabel("Date")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    return fig


def plot_regimes_on_vol(
    fit: RegimeFit,
    asset_class: str,
    save_path: Path | None = None,
) -> plt.Figure:
    """
    Graphique : volatilité + bandes colorées.
    """
    fig, ax = plt.subplots(figsize=(13, 5))

    vol = fit.composite_vol
    ax.plot(vol.index, vol.values, color="#1a3a5c", linewidth=1.2,
            label="Volatilité 20j")

    # Bandes colorées
    labels = fit.regime_labels
    prev_regime = None
    start_idx = None

    for date, regime in labels.items():
        if regime != prev_regime:
            if prev_regime is not None and start_idx is not None:
                ax.axvspan(start_idx, date,
                           color=REGIME_COLORS.get(prev_regime, "#cccccc"),
                           alpha=0.25)
            start_idx = date
            prev_regime = regime

    if prev_regime is not None and start_idx is not None:
        ax.axvspan(start_idx, labels.index[-1],
                   color=REGIME_COLORS.get(prev_regime, "#cccccc"),
                   alpha=0.25)

    # Lignes horizontales aux moyennes
    for regime in REGIME_LABELS:
        if regime in fit.regime_stats.index:
            mean_vol = fit.regime_stats.loc[regime, "mean_vol"]
            ax.axhline(y=mean_vol, color=REGIME_COLORS[regime],
                       linestyle="--", linewidth=0.8, alpha=0.6)

    ax.set_title(f"{asset_class.upper()} — Volatilité et régimes",
                 fontsize=12, fontweight="bold", color="#1a3a5c")
    ax.set_ylabel("Volatilité (%)")
    ax.set_xlabel("Date")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    return fig


def plot_transition_heatmap(
    fit: RegimeFit,
    asset_class: str,
    save_path: Path | None = None,
) -> plt.Figure:
    """
    Heatmap de la matrice de transition.
    """
    fig, ax = plt.subplots(figsize=(6, 5))

    tm = fit.transition_matrix
    im = ax.imshow(tm.values, cmap="YlOrRd", aspect="auto", vmin=0, vmax=1)

    ax.set_xticks(range(len(tm.columns)))
    ax.set_xticklabels(tm.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(tm.index)))
    ax.set_yticklabels(tm.index)

    # Annotations
    for i in range(tm.shape[0]):
        for j in range(tm.shape[1]):
            value = tm.values[i, j]
            color = "white" if value > 0.5 else "black"
            ax.text(j, i, f"{value:.1%}",
                    ha="center", va="center", color=color, fontsize=10)

    ax.set_title(f"{asset_class.upper()} — Matrice de transition",
                 fontsize=12, fontweight="bold", color="#1a3a5c")
    ax.set_xlabel("Régime à t")
    ax.set_ylabel("Régime à t-1")

    fig.colorbar(im, ax=ax, label="Probabilité")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    return fig


def plot_duration_distribution(
    fit: RegimeFit,
    asset_class: str,
    save_path: Path | None = None,
) -> plt.Figure:
    """
    Histogramme des durées par régime.
    """
    durations = regime_durations(fit.regime_labels)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)

    for ax, regime in zip(axes, REGIME_LABELS):
        d = durations.get(regime, [])
        if not d:
            ax.text(0.5, 0.5, "Pas de données",
                    ha="center", va="center", transform=ax.transAxes)
        else:
            ax.hist(d, bins=min(15, len(d)), color=REGIME_COLORS[regime],
                    alpha=0.7, edgecolor="black")
            ax.axvline(np.mean(d), color="black", linestyle="--",
                       linewidth=1.5, label=f"Moy: {np.mean(d):.1f}j")

        ax.set_title(f"{regime}  (n={len(d)})", fontsize=11,
                     fontweight="bold", color=REGIME_COLORS[regime])
        ax.set_xlabel("Durée (jours)")
        if ax == axes[0]:
            ax.set_ylabel("Fréquence")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"{asset_class.upper()} — Distribution des durées par régime",
                 fontsize=12, fontweight="bold", color="#1a3a5c", y=1.02)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    return fig


# ---------------------------------------------------------------------------
# Analyse comparative multi-classes
# ---------------------------------------------------------------------------

def compare_regimes_across_classes(
    results: dict[str, RegimeFit],
) -> pd.DataFrame:
    """
    Construit un tableau comparatif des régimes entre classes.

    Returns
    -------
    pd.DataFrame
        Colonnes : asset_class, regime, mean_return, mean_vol,
                   frequency, avg_duration, persistence
    """
    rows = []
    for asset_class, fit in results.items():
        for regime in REGIME_LABELS:
            if regime in fit.regime_stats.index:
                row = fit.regime_stats.loc[regime]
                # Persistance = diagonale de la matrice de transition
                pers = (fit.transition_matrix.loc[regime, regime]
                        if regime in fit.transition_matrix.index
                        else np.nan)
                rows.append({
                    "asset_class": asset_class,
                    "regime": regime,
                    "mean_return": float(row["mean_return"]),
                    "mean_vol": float(row["mean_vol"]),
                    "frequency": float(row["frequency"]),
                    "avg_duration": float(row["avg_duration"]),
                    "persistence": float(pers),
                })
    return pd.DataFrame(rows)


def compute_cross_class_correlations(
    results: dict[str, RegimeFit],
) -> pd.DataFrame:
    """
    Calcule les corrélations entre rendements composites par classe.
    """
    series = {}
    for asset_class, fit in results.items():
        series[asset_class] = fit.composite_returns

    df = pd.DataFrame(series).dropna()
    return df.corr()


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def export_regimes(
    results: dict[str, RegimeFit],
    output_dir: Path = None,
) -> None:
    """
    Exporte les régimes et stats en CSV.
    """
    if output_dir is None:
        output_dir = PROJECT_ROOT / "data"

    # Régimes (labels par date et par classe)
    df_regimes = pd.DataFrame({
        f"{cls}_regime": fit.regime_labels
        for cls, fit in results.items()
    })
    df_regimes.index.name = "date"
    df_regimes.to_csv(output_dir / "regimes.csv")
    print(f"💾 Régimes : {output_dir / 'regimes.csv'}")

    # Stats par régime
    df_stats = compare_regimes_across_classes(results)
    df_stats.to_csv(output_dir / "regime_stats.csv", index=False)
    print(f"💾 Stats : {output_dir / 'regime_stats.csv'}")


# ---------------------------------------------------------------------------
# Test rapide
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from src.data_manager import get_all_features
    from src.hmm_model import fit_all_classes

    print("=" * 70)
    print("🔍 Analyse approfondie des régimes")
    print("=" * 70)

    # 1. Charger les features
    returns, vol = get_all_features(vol_window=20)

    # 2. Fit les modèles
    print("\n🔄 Fit des modèles…")
    results = {}
    for asset_class in ASSET_CLASSES_MAP.keys():
        print(f"   → {asset_class}…")
        from src.hmm_model import fit_regime_model
        results[asset_class] = fit_regime_model(
            returns, vol, asset_class,
            n_states=3, maxiter=1000, feature="vol",
        )

    # 3. Comparaison inter-classes
    print("\n" + "=" * 70)
    print("📊 TABLEAU COMPARATIF COMPLET")
    print("=" * 70)
    df = compare_regimes_across_classes(results)
    print(df.to_string(index=False))

    # 4. Régimes actuels
    print("\n" + "=" * 70)
    print("🎯 RÉGIMES ACTUELS")
    print("=" * 70)
    for asset_class, fit in results.items():
        current = fit.regime_labels.iloc[-1]
        proba = fit.state_proba.iloc[-1]
        print(f"   {asset_class.upper():8s} : {current:10s} (p={proba.max():.2%})")

    # 5. Probabilités de transition depuis régime courant
    print("\n" + "=" * 70)
    print("🔮 PROBABILITÉS DE TRANSITION (5 prochains jours)")
    print("=" * 70)
    for asset_class, fit in results.items():
        print(f"\n   {asset_class.upper()} :")
        probs = current_regime_proba(fit, horizon=5)
        print(probs.round(4).to_string())

    # 6. Tests de séparation
    print("\n" + "=" * 70)
    print("🧪 TESTS DE SÉPARATION DES RÉGIMES")
    print("=" * 70)
    for asset_class, fit in results.items():
        print(f"\n   {asset_class.upper()} :")
        sep = test_regime_separation(fit)
        if len(sep) > 0:
            print(sep[["regime_a", "regime_b", "mean_a", "mean_b",
                       "t_stat", "p_value", "significant"]].to_string(index=False))
        else:
            print("   Pas assez de données")

    # 7. Durées
    print("\n" + "=" * 70)
    print("⏱️  DURÉES DE RÉGIME")
    print("=" * 70)
    for asset_class, fit in results.items():
        print(f"\n   {asset_class.upper()} :")
        durations = regime_durations(fit.regime_labels)
        for regime, d in durations.items():
            if d:
                print(f"      {regime:10s} : n={len(d):2d} blocs, "
                      f"moy={np.mean(d):5.1f}j, "
                      f"médiane={np.median(d):5.1f}j, "
                      f"max={max(d):3d}j")

    # 8. Corrélations inter-classes
    print("\n" + "=" * 70)
    print("🔗 CORRÉLATIONS INTER-CLASSES (rendements)")
    print("=" * 70)
    corr = compute_cross_class_correlations(results)
    print(corr.round(3).to_string())

    # 9. Transitions majeures
    print("\n" + "=" * 70)
    print("🔄 TRANSITIONS MAJEURES (5 dernières par classe)")
    print("=" * 70)
    for asset_class, fit in results.items():
        print(f"\n   {asset_class.upper()} :")
        events = compute_transition_events(fit.regime_labels)
        if len(events) > 0:
            print(events.tail(5).to_string(index=False))

    # 10. Génération des graphiques
    print("\n" + "=" * 70)
    print("📈 Génération des graphiques")
    print("=" * 70)
    for asset_class, fit in results.items():
        plot_regimes_on_prices(
            fit, asset_class,
            save_path=FIGURES_DIR / f"{asset_class}_regimes_prices.png",
        )
        plot_regimes_on_vol(
            fit, asset_class,
            save_path=FIGURES_DIR / f"{asset_class}_regimes_vol.png",
        )
        plot_transition_heatmap(
            fit, asset_class,
            save_path=FIGURES_DIR / f"{asset_class}_transition_matrix.png",
        )
        plot_duration_distribution(
            fit, asset_class,
            save_path=FIGURES_DIR / f"{asset_class}_durations.png",
        )
    print(f"   ✅ Graphiques sauvegardés dans {FIGURES_DIR}")

    # 11. Export CSV
    print("\n" + "=" * 70)
    print("💾 Export des données")
    print("=" * 70)
    export_regimes(results)

    print("\n" + "=" * 70)
    print("✅ Analyse terminée")
    print("=" * 70)
  
