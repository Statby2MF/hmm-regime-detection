"""
Script de téléchargement des données de marché pour HMM Regime Detection.

Ce script télécharge les données des classes d'actifs suivantes :
    - Crypto     : BTC-USD, ETH-USD         (via yfinance)
    - Actions US : ^GSPC (S&P 500)          (via yfinance)
    - Actions BRVM : ABJC, BICC, BOAB       (via CSV locaux — voir data/brvm/)

Usage :
    python download_data.py
    python download_data.py --force     # force le retéléchargement

Auteur : Statby2Mf
Projet : HMM Regime Detection (M2 Statistique, UGB Saint-Louis)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yfinance as yf

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

BRVM_DIR = DATA_DIR / "brvm"
BRVM_DIR.mkdir(exist_ok=True)

PRICES_FILE = DATA_DIR / "prices.csv"

#: Tickers à télécharger via yfinance.
TICKERS_YF = {
    "BTC-USD": "Bitcoin",
    "ETH-USD": "Ethereum",
    "^GSPC": "SP500",
}

#: Fichiers BRVM attendus dans data/brvm/.
#: Format : nom_fichier → nom_colonne_final
BRVM_FILES = {
    "brvm_abjc.csv": "BRVM_ABJC",
    "brvm_bicc.csv": "BRVM_BICC",
    "brvm_boab.csv": "BRVM_BOAB",
}

START_DATE = "2018-01-01"


# ---------------------------------------------------------------------------
# Téléchargement yfinance (individuel, comme dans GARCH-EVT)
# ---------------------------------------------------------------------------

def download_single_ticker(ticker: str, start: str = START_DATE) -> pd.Series:
    """Télécharge un ticker et renvoie sa série de prix."""
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


def download_yf_data(start: str = START_DATE) -> pd.DataFrame:
    """Télécharge tous les tickers yfinance."""
    print(f"📥 Téléchargement yfinance ({len(TICKERS_YF)} actifs)…")
    series_list = [download_single_ticker(t, start) for t in TICKERS_YF.keys()]
    df = pd.concat(series_list, axis=1, join="outer").sort_index()
    return df


# ---------------------------------------------------------------------------
# Chargement BRVM
# ---------------------------------------------------------------------------

def load_brvm_data() -> pd.DataFrame:
    """
    Charge les CSV BRVM depuis data/brvm/.

    ⚠️ À adapter selon le format EXACT de tes fichiers.
    Hypothèse : chaque CSV a une colonne 'Date' et une colonne de prix
                (ex : 'Close', 'close', 'Cours', ...).
    """
    print(f"📥 Chargement BRVM ({len(BRVM_FILES)} actifs)…")

    series_list = []
    for filename, col_name in BRVM_FILES.items():
        path = BRVM_DIR / filename
        if not path.exists():
            print(f"   ⚠️ {filename} introuvable → ignoré")
            continue

        df = pd.read_csv(path)

        # --- Détection automatique des colonnes ---
        # Cherche une colonne de date
        date_col = None
        for candidate in ["Date", "date", "DATE", "datetime", "Datetime"]:
            if candidate in df.columns:
                date_col = candidate
                break
        if date_col is None:
            # Sinon, prend la première colonne comme date
            date_col = df.columns[0]

        # Cherche une colonne de prix
        price_col = None
        for candidate in ["Close", "close", "CLOSE", "Cours", "cours",
                          "Prix", "prix", "Adj Close"]:
            if candidate in df.columns:
                price_col = candidate
                break
        if price_col is None:
            # Sinon, prend la 2e colonne
            price_col = df.columns[1]

        df[date_col] = pd.to_datetime(df[date_col])
        df = df.set_index(date_col)[price_col]
        df = df.sort_index()
        df.name = col_name
        series_list.append(df)
        print(f"   → {col_name} : {len(df)} obs "
              f"({df.index[0].date()} → {df.index[-1].date()})")

    if not series_list:
        print("   ⚠️ Aucune donnée BRVM chargée")
        return pd.DataFrame()

    return pd.concat(series_list, axis=1, join="outer").sort_index()


# ---------------------------------------------------------------------------
# Fusion
# ---------------------------------------------------------------------------

def merge_all_sources(yf_df: pd.DataFrame, brvm_df: pd.DataFrame) -> pd.DataFrame:
    """Fusionne yfinance + BRVM sur l'index de dates."""
    print("\n🔗 Fusion des sources…")
    parts = [df for df in [yf_df, brvm_df] if not df.empty]
    merged = pd.concat(parts, axis=1, join="outer").sort_index()
    return merged


# ---------------------------------------------------------------------------
# Point d'entrée
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true",
                        help="Retélécharge même si le CSV existe")
    args = parser.parse_args()

    if PRICES_FILE.exists() and not args.force:
        print(f"ℹ️  {PRICES_FILE} existe déjà. Utilise --force pour retélécharger.")
        return

    # 1. Téléchargement yfinance
    yf_df = download_yf_data()

    # 2. Chargement BRVM
    brvm_df = load_brvm_data()

    # 3. Fusion
    prices = merge_all_sources(yf_df, brvm_df)

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
    print("=" * 70)


if __name__ == "__main__":
    main()
