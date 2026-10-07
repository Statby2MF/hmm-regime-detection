"""
Script de préparation des données pour HMM Regime Detection.

Stratégie :
    - Crypto (BTC, ETH) : téléchargés via yfinance
    - Actions US (SPY, QQQ) : lus depuis data/brvm/*.csv (CSV existants)
    - Actions BRVM (SNTS, BOAB, ECOC) : lus depuis data/brvm/*.csv

Usage :
    python download_data.py
    python download_data.py --force

Auteur : Statby2Mf
Projet : HMM Regime Detection (M2 Statistique, UGB Saint-Louis)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yfinance as yf
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

BRVM_DIR = DATA_DIR / "brvm"
BRVM_DIR.mkdir(exist_ok=True)

PRICES_FILE = DATA_DIR / "prices.csv"

#: Crypto à télécharger via yfinance
CRYPTO_TICKERS = {
    "BTC-USD": "Bitcoin",
    "ETH-USD": "Ethereum",
}

#: Fichiers CSV locaux à charger (US + BRVM)
#: Nom final → nom du fichier dans data/brvm/
#: Fichiers CSV locaux à charger (US + BRVM)
#: Note : US_QQQ exclu (historique trop court, démarre en 2024-09)
LOCAL_FILES = {
    # US
    "US_SPY": "US_SPY.csv",
    # BRVM (les 3 avec le plus d'historique)
    "BRVM_SNTS": "BRVM_SNTS.csv",
    "BRVM_BOAB": "BRVM_BOAB.csv",
    "BRVM_ECOC": "BRVM_ECOC.csv",
}

START_DATE = "2018-01-01"


# ---------------------------------------------------------------------------
# Téléchargement crypto (yfinance)
# ---------------------------------------------------------------------------

def download_crypto(ticker: str, start: str = START_DATE) -> pd.Series:
    """Télécharge une crypto via yfinance."""
    print(f"   → {ticker}… ", end="", flush=True)
    df = yf.download(ticker, start=start, auto_adjust=True, progress=False)

    if df.empty:
        raise RuntimeError(f"Aucune donnée pour {ticker}")

    if isinstance(df.columns, pd.MultiIndex):
        prices = df["Close"][ticker]
    else:
        prices = df["Close"]

    prices = prices.dropna().sort_index()
    prices.name = ticker
    print(f"{len(prices)} obs ({prices.index[0].date()} → {prices.index[-1].date()})")
    return prices


def download_all_crypto(start: str = START_DATE) -> pd.DataFrame:
    """Télécharge toutes les cryptos."""
    print(f"📥 Téléchargement crypto ({len(CRYPTO_TICKERS)} actifs)…")
    series_list = [download_crypto(t, start) for t in CRYPTO_TICKERS.keys()]
    return pd.concat(series_list, axis=1, join="outer", sort=False).sort_index()


# ---------------------------------------------------------------------------
# Chargement CSV locaux (US + BRVM)
# ---------------------------------------------------------------------------

def load_csv_file(filename: str, col_name: str) -> pd.Series | None:
    """Charge un CSV avec format Date/Close."""
    path = BRVM_DIR / filename
    if not path.exists():
        print(f"   ⚠️ {filename} introuvable → ignoré")
        return None

    df = pd.read_csv(path)

    # Détection colonne date
    date_col = next(
        (c for c in ["Date", "date", "DATE"] if c in df.columns),
        df.columns[0],
    )

    # Détection colonne prix
    price_col = next(
        (c for c in ["Close", "close", "CLOSE"] if c in df.columns),
        df.columns[1],
    )

    df[date_col] = pd.to_datetime(df[date_col], utc=True).dt.tz_localize(None)
    df = df.set_index(date_col)[price_col]
    df = df.sort_index()
    df.name = col_name
    df = df[~df.index.duplicated(keep="last")]  # enlève les doublons
    return df


def load_all_local() -> pd.DataFrame:
    """Charge tous les CSV locaux."""
    print(f"\n📥 Chargement CSV locaux ({len(LOCAL_FILES)} actifs)…")
    series_list = []
    for col_name, filename in LOCAL_FILES.items():
        print(f"   → {col_name}… ", end="", flush=True)
        s = load_csv_file(filename, col_name)
        if s is not None:
            print(f"{len(s)} obs ({s.index[0].date()} → {s.index[-1].date()})")
            series_list.append(s)
    if not series_list:
        return pd.DataFrame()
    return pd.concat(series_list, axis=1, join="outer", sort=False).sort_index()


# ---------------------------------------------------------------------------
# Fusion
# ---------------------------------------------------------------------------

def merge_all_sources(crypto_df: pd.DataFrame,
                       local_df: pd.DataFrame) -> pd.DataFrame:
    """Fusionne crypto + local."""
    print("\n🔗 Fusion des sources…")
    parts = [df for df in [crypto_df, local_df] if not df.empty]
    return pd.concat(parts, axis=1, join="outer", sort=False).sort_index()


# ---------------------------------------------------------------------------
# Point d'entrée
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true",
                        help="Retélécharge même si le CSV existe")
    args = parser.parse_args()

    if PRICES_FILE.exists() and not args.force:
        print(f"ℹ️  {PRICES_FILE} existe déjà. Utilise --force.")
        return

    # 1. Crypto
    crypto_df = download_all_crypto()

    # 2. CSV locaux
    local_df = load_all_local()

    # 3. Fusion
    prices = merge_all_sources(crypto_df, local_df)

    # 4. Sauvegarde
    prices.to_csv(PRICES_FILE)
    print(f"\n💾 Sauvegardé : {PRICES_FILE}")
    print(f"   Shape : {prices.shape}")
    print(f"   Colonnes : {list(prices.columns)}")

    # 5. Rapport qualité
    print("\n" + "=" * 70)
    print("📊 RAPPORT DE QUALITÉ")
    print("=" * 70)
    print(f"Période : {prices.index[0].date()} → {prices.index[-1].date()}")
    print(f"Observations : {len(prices)}")
    print(f"Actifs : {prices.shape[1]}")
    print("\nNaN par colonne :")
    print(prices.isna().sum().to_string())
    print("\nDernière date par actif (avant alignment) :")
    for col in prices.columns:
        s = prices[col].dropna()
        if len(s) > 0:
            print(f"   {col:15s} : {s.index[-1].date()}")
    print("=" * 70)


if __name__ == "__main__":
    main()
