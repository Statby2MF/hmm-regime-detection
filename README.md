# 📊 HMM Regime Detection

> **Détection automatique de régimes de marché par Markov Switching** — Application aux cryptomonnaies, actions US et actions BRVM, avec backtesting de stratégies d'allocation dynamique.

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-red.svg)](https://streamlit.io/)
[![statsmodels](https://img.shields.io/badge/statsmodels-0.14+-green.svg)](https://www.statsmodels.org/)

---

## 🎯 Objectifs

Ce projet implémente un **détecteur de régimes de marché** via **Markov Switching** (Hamilton, 1989), appliqué à **3 classes d'actifs** :

- **Cryptomonnaies** (BTC, ETH)
- **Actions US** (S&P 500)
- **Actions BRVM** (Sonatel, BOA, Ecobank)

Pour chaque classe, le projet :
1. **Détecte 3 régimes** de volatilité : Low Vol / Med Vol / High Vol
2. **Modélise les transitions** entre régimes (matrice de Markov)
3. **Backteste 5 stratégies** d'allocation dynamique
4. **Teste la significativité** statistique (t-tests)
5. **Génère un rapport PDF** automatique

---

## 📸 Aperçu

### Dashboard interactif

![Dashboard Régimes](docs/screenshots/dashboard_regimes.png)

### Backtesting des stratégies

![Dashboard Stratégies](docs/screenshots/dashboard_strategies.png)

### Rapport PDF généré automatiquement

![Résumé exécutif](docs/screenshots/report_summary.png)

![Régimes détectés](docs/screenshots/report_regimes.png)

---

## 🏆 Résultats clés

### Découvertes empiriques majeures

| Découverte | Valeur | Interprétation |
|------------|--------|----------------|
| **Persistance des régimes** | > 93% | Régimes stables et exploitables |
| **Corrélation BRVM–US** | **-0.04** | Quasi-décorrélée → excellent diversificateur |
| **Corrélation BRVM–Crypto** | **0.03** | Indépendance quasi-totale |
| **BRVM High Vol Sharpe** | **+1.24** | Contre-intuitif : la volatilité rémunère |
| **US stratégie Agressive** | **p = 0.04** | Significatif, drawdown -42% |
| **Crypto** | Aucune strat. signif. | Marché imprévisible sur 5 ans |

### Tableau récapitulatif

| Classe | Régime optimal | Sharpe | Max Drawdown | Significatif ? |
|--------|---------------|--------|--------------|----------------|
| **Crypto** | Regime-Cash | 0.13 | -48.5% | ❌ (p = 0.35) |
| **US** | Régime-Switch Agressive | **0.60** | **-14.2%** | ✅ (p = 0.04) |
| **BRVM** | Vol Target | **1.19** | -13.3% | ✅ (p = 0.0003) |

---

## 🏗️ Architecture
