"""
Data manager du projet HMM Regime Detection.

Fournit une API haut-niveau pour :
    - Charger les prix depuis data/prices.csv
    - Nettoyer les données
    - Calculer les log-rendements (en %)
    - Calculer les features pour le HMM (rendements + vol 20j)
    - Fournir un rapport de qualité

Univers analysé :
    - Crypto      : BTC-USD, ETH-USD
    - Actions US  : SP500 (US_SPY), Nasdaq (US_QQQ)
    - Actions BRVM: Sonatel, BOAB, Ecobank

Auteur : Statby2Mf
Projet : HMM Regime Detection (M2 Statistique, UGB Saint-Louis)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PRICES_FILE = DATA_DIR / "prices.csv"

#: Univers final — classés par classe d'actifs
ASSETS: dict[str, str] = {
    # Crypto
    "BTC-USD": "Bitcoin",
    "ETH-USD": "Ethereum",
    # Actions US
    "US_SPY": "S&P 500 ETF",
    "US_QQQ": "Nasdaq 100 ETF",
    # Actions BRVM
    "BRVM_SNTS": "Sonatel",
    "BRVM_BOAB": "Bank of Africa",
    "BRVM_ECOC": "Ecobank",
}

#: Classes d'actifs pour regroupements futurs
ASSET_CLASSES: dict[str, str] = {
    "BTC-USD": "crypto",
    "ETH-USD": "crypto",
    "US_SPY": "us_equity",
    "US_QQQ": "us_equity",
    "BRVM_SNTS": "brvm",
    "BRVM_BOAB": "brvm",
    "BRVM_ECOC": "brvm",
}


# ---------------------------------------------------------------------------
# Rapport de qualité
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DataQualityReport:
    """Rapport synthétique de qualité des données."""
    n_obs: int
    n_assets: int
    start: pd.Timestamp
    end: pd.Timestamp
    missing_before_ffill: int
    missing_after_ffill: int
    missing_by_asset: dict

    def __str__(self) -> str:
        lines = [
            "DataQualityReport(",
            f"  Période      : {self.start.date()} → {self.end.date()}",
            f"  Observations : {self.n_obs} jours × {self.n_assets} actifs",
            f"  NaN avant    : {self.missing_before_ffill}",
            f"  NaN après    : {self.missing_after_ffill}",
            "  NaN par actif:",
        ]
        for asset, n in self.missing_by_asset.items():
            lines.append(f"    - {asset:15s} : {n}")
        lines.append(")")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Chargement
# ---------------------------------------------------------------------------

def load_prices() -> pd.DataFrame:
    """
    Charge les prix depuis data/prices.csv.

    Returns
    -------
    pd.DataFrame
        Prix nettoyés, index = dates, colonnes = tickers.

    Raises
    ------
    FileNotFoundError
        Si data/prices.csv n'existe pas.
        → Lance d'abord : python download_data.py
    """
    if not PRICES_FILE.exists():
        raise FileNotFoundError(
            f"Fichier introuvable : {PRICES_FILE}\n"
            f"→ Lance d'abord : python download_data.py"
        )
    prices = pd.read_csv(PRICES_FILE, index_col=0, parse_dates=True)
    return prices


# ---------------------------------------------------------------------------
# Nettoyage
# ---------------------------------------------------------------------------

def clean_prices(prices: pd.DataFrame) -> tuple[pd.DataFrame, DataQualityReport]:
    """
    Nettoie les prix et renvoie un rapport de qualité.

    Stratégie :
        1. Suppression des jours où TOUT est NaN
        2. Forward-fill limité (max 3 jours = weekend BRVM)
        3. Garde uniquement les dates où TOUS les actifs existent
           (pour garantir un dataset rectangulaire sans NaN)
    """
    n_before = int(prices.isna().sum().sum())

    # 1. Supprime les lignes 100% vides
    prices = prices.dropna(how="all")

    # 2. Forward-fill limité (les BRVM ne cotent pas les weekends mais peuvent
    #    avoir des jours fériés → 3 jours max)
    prices = prices.ffill(limit=3)

    # 3. Garde les dates où tous les actifs ont un prix
    prices = prices.dropna()

    n_after = int(prices.isna().sum().sum())

    report = DataQualityReport(
        n_obs=len(prices),
        n_assets=prices.shape[1],
        start=prices.index[0],
        end=prices.index[-1],
        missing_before_ffill=n_before,
        missing_after_ffill=n_after,
        missing_by_asset={col: int(prices[col].isna().sum())
                          for col in prices.columns},
    )
    return prices, report


# ---------------------------------------------------------------------------
# Rendements et features
# ---------------------------------------------------------------------------

def compute_log_returns(prices: pd.DataFrame, scale: float = 100.0) -> pd.DataFrame:
    """
    Calcule les log-rendements.

    Parameters
    ----------
    prices : pd.DataFrame
    scale : float
        Facteur multiplicatif. 100 → rendements en %.
        Recommandé pour la stabilité numérique du HMM.

    Returns
    -------
    pd.DataFrame
        Log-rendements.
    """
    prices_clean, _ = clean_prices(prices)
    returns = scale * np.log(prices_clean / prices_clean.shift(1))
    return returns.dropna()


def compute_rolling_vol(returns: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """
    Calcule la volatilité roulante (écart-type des rendements sur N jours).

    Parameters
    ----------
    returns : pd.DataFrame
    window : int
        Fenêtre en jours. Défaut : 20 (~1 mois de trading).

    Returns
    -------
    pd.DataFrame
        Volatilité roulante (même shape que returns).
    """
    return returns.rolling(window=window).std()


def compute_features(
    prices: pd.DataFrame,
    vol_window: int = 20,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Calcule les features pour le HMM : rendements + volatilité roulante.

    Returns
    -------
    (returns, volatility) : tuple of pd.DataFrame
        Les deux DataFrames alignés sur les mêmes dates.
    """
    returns = compute_log_returns(prices)
    vol = compute_rolling_vol(returns, window=vol_window)

    # Alignement : on perd les `vol_window` premières lignes
    common_idx = returns.index.intersection(vol.dropna().index)
    return returns.loc[common_idx], vol.loc[common_idx]


# ---------------------------------------------------------------------------
# API haut-niveau
# ---------------------------------------------------------------------------

def get_returns(ticker: str) -> pd.Series:
    """Renvoie les log-rendements (%) d'un actif."""
    prices = load_prices()
    if ticker not in prices.columns:
        raise ValueError(
            f"Ticker '{ticker}' introuvable. "
            f"Disponibles : {list(prices.columns)}"
        )
    returns = compute_log_returns(prices[[ticker]])[ticker]
    return returns.rename(ticker)


def get_all_returns() -> pd.DataFrame:
    """Renvoie les log-rendements (%) de tous les actifs."""
    prices = load_prices()
    return compute_log_returns(prices)


def get_all_features(vol_window: int = 20) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Renvoie les features (returns, vol) pour tous les actifs."""
    prices = load_prices()
    return compute_features(prices, vol_window=vol_window)


def get_quality_report() -> DataQualityReport:
    """Renvoie le rapport de qualité."""
    prices = load_prices()
    _, report = clean_prices(prices)
    return report


# ---------------------------------------------------------------------------
# Test rapide
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("🔍 Test du data_manager (HMM Regime Detection)\n")

    report = get_quality_report()
    print(report)
    print()

    print("📊 Aperçu des rendements (%) :")
    returns = get_all_returns()
    print(f"Shape : {returns.shape}")
    print(f"Période : {returns.index[0].date()} → {returns.index[-1].date()}")
    print()
    print(returns.describe().round(3).to_string())
    print()

    print("📊 Features (rendements + vol 20j) :")
    r, v = get_all_features(vol_window=20)
    print(f"Rendements : {r.shape}, Vol : {v.shape}")
    print(f"\nVolatilité moyenne par actif :")
    print(v.mean().round(4).to_string())
