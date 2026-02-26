# Asymptotic and Finite-Sample Analysis of Monte Carlo VaR Estimators

This project studies statistical properties of Monte Carlo Value-at-Risk (VaR) and Expected Shortfall (ES) estimators under:

- Heavy-tailed returns (Student-t)
- Finite-sample concentration bounds (DKW)
- Large-sample asymptotics
- Variance reduction techniques
- Importance sampling
- Model misspecification
- Time-varying volatility (GARCH)
- ES backtesting and proper scoring rules

The goal is to bridge theoretical probability with practical risk modeling.

---

## Main Findings

- Asymptotic quantile theory works well at large sample sizes but is unstable in extreme tails.
- DKW bounds guarantee coverage but are conservative.
- Antithetic variates provide little benefit for quantile estimators.
- Importance sampling significantly reduces variance when tuned properly.
- Static Gaussian VaR fails backtests under volatility clustering.
- GARCH(1,1) with Student-t innovations produces well-calibrated VaR and ES forecasts.

---

## Repository Structure

- `src/` – Core implementation (Monte Carlo, heavy tails, backtests, GARCH, etc.)
- `experiments/` – YAML configurations for reproducible experiments
- `scripts/` – Data download and experiment runners
- `notebooks/` – Visualization and analysis notebook
- `paper/` – Generated tables and figures

---

## Reproducing Results

### 1️⃣ Install dependencies

