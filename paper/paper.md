# Asymptotic and Finite‑Sample Analysis of Monte Carlo Value‑at‑Risk and Expected Shortfall Estimators under Heavy‑Tailed Returns

---

## Abstract

This project studies Monte Carlo estimators of extreme‑tail risk measures—**Value‑at‑Risk (VaR)** and **Expected Shortfall (ES)**—under realistic features of financial returns, including **heavy tails**, **finite‑sample uncertainty**, **model misspecification**, and **time‑varying volatility**. The project integrates:

1. **Large‑sample theory** for quantile estimators (asymptotic normality / weak convergence),
2. **Finite‑sample concentration bounds** via the Dvoretzky–Kiefer–Wolfowitz (DKW) inequality,
3. **Variance‑reduced Monte Carlo** (antithetic variates; importance sampling by mean‑shift),
4. **Backtesting** using Kupiec unconditional coverage and Christoffersen independence tests,
5. **Proper scoring rules** for joint VaR–ES evaluation (Fissler–Ziegel FZ0),
6. **Dynamic volatility models** (GARCH(1,1)) with Gaussian and Student‑t innovations.

Empirically, static Gaussian and historical methods tend to miscalibrate 99% VaR/ES on real returns; GARCH‑Normal improves independence but often underestimates tail risk; **GARCH‑Student‑t** provides substantially improved calibration and diagnostics.

---

## 1. Risk measures and notation

Let portfolio return be \(R\) and **loss** be \(L=-R\). Fix a confidence level \(\alpha\in(0,1)\), e.g. \(\alpha=0.99\).

### 1.1 Value‑at‑Risk (VaR)

\[
\mathrm{VaR}_\alpha(L)=\inf\{\ell:\Pr(L\le \ell)\ge \alpha\}=q_\alpha(L)
\]

### 1.2 Expected Shortfall (ES)

\[
\mathrm{ES}_\alpha(L)=\mathbb{E}[L\mid L\ge \mathrm{VaR}_\alpha(L)]
\]

Empirical estimators compute the sample quantile and the conditional tail average.

---

## 2. Scenario generation (Monte Carlo)

### 2.1 Gaussian parametric Monte Carlo (exp01)

Assume a multivariate normal model for asset returns:
\[
X\sim\mathcal N(\mu,\Sigma)
\]
Estimate \(\mu\) and \(\Sigma\) from historical returns and simulate scenarios \(X^{(i)}\). Portfolio return is \(R^{(i)}=w^\top X^{(i)}\); losses are \(L^{(i)}=-R^{(i)}\). VaR/ES are computed from \(\{L^{(i)}\}\).

**Figure:** exp01 VaR/ES comparison  
![](paper/figures/exp01_var_cvar.png)

### 2.2 Historical bootstrap (exp01)

Resample historical return vectors with replacement to obtain scenarios:
\[
X^{(i)} \sim \hat F_{\mathrm{emp}}
\]
This preserves empirical dependence/tail behavior (to the extent present in sample).

---

## 3. Heavy tails via multivariate Student‑t (exp02)

Using the scale‑mixture representation to model heavy tails:

- \(Z \sim \mathcal N(0,\Sigma)\)
- \(U \sim \chi^2_\nu\) independent
- \[
X = \mu + \frac{Z}{\sqrt{U/\nu}}
\]

This yields \(X \sim t_\nu(\mu,\Sigma)\). As \(\nu\to\infty\), the distribution approaches Gaussian; for small \(\nu\), tails are much heavier.

**Figure:** exp02 normal vs Student‑t VaR/ES  
![](paper/figures/exp02_var_cvar.png)

---

## 4. Large‑sample theory: quantile asymptotics (exp03)

Let \(F\) be the CDF of losses \(L\), and \(q_\alpha=F^{-1}(\alpha)\) the true quantile. The sample quantile \(\hat q_\alpha\) satisfies (under regularity, \(f(q_\alpha)>0\)):

\[
\sqrt{n}(\hat q_\alpha-q_\alpha) \Rightarrow \mathcal N\left(0,\ \frac{\alpha(1-\alpha)}{f(q_\alpha)^2}\right)
\]

A plug‑in standard error uses an estimate \(\hat f(\hat q_\alpha)\) (KDE). The resulting approximate CI is:

\[
\hat q_\alpha \pm z_{1-\delta/2}\sqrt{\frac{\alpha(1-\alpha)}{n\,\hat f(\hat q_\alpha)^2}}
\]

**Key diagnostic:** when \(\hat f(\hat q_\alpha)\) is near zero (common at small \(n\) in extreme tails), CI width can inflate dramatically.

**Figures:** RMSE, coverage, CI width, and density diagnostics  
![](paper/figures/exp03_rmse_loglog.png)  
![](paper/figures/exp03_coverage.png)  
![](paper/figures/exp03_ci_width.png)  
![](paper/figures/exp03_fhat_diagnostics.png)

---

## 5. Finite‑sample bounds: DKW quantile brackets (exp04)

The DKW inequality provides a uniform bound on the empirical CDF:

\[
\Pr\left(\sup_x |F_n(x)-F(x)|>\varepsilon\right) \le 2e^{-2n\varepsilon^2}
\]

Given confidence level \(\gamma\), choose \(\varepsilon=\sqrt{\frac{1}{2n}\log\frac{2}{1-\gamma}}\). Then with probability at least \(\gamma\):

\[
F_n(x)-\varepsilon \le F(x) \le F_n(x)+\varepsilon\quad \forall x
\]

This can be translated into a quantile bracket:
\[
q_{\alpha-\varepsilon}(F_n) \le q_\alpha(F) \le q_{\alpha+\varepsilon}(F_n)
\]
(clipping \(\alpha\pm\varepsilon\) into \([0,1]\)).

**Figures:** coverage and width comparison vs asymptotic CIs  
![](paper/figures/exp04_coverage_compare.png)  
![](paper/figures/exp04_width_compare.png)

---

## 6. Variance reduction (exp05–exp06)

### 6.1 Antithetic variates (exp05)

For \(Z\sim\mathcal N(0,I)\), pair scenarios \(Z\) and \(-Z\) to induce negative correlation. This often reduces variance for smooth functionals; however **quantiles are non‑smooth**, so gains may be limited.

**Figures:** VaR/ES RMSE and std comparison  
![](paper/figures/step1_variance_reduction_std_var.png)  
![](paper/figures/step1_variance_reduction_rmse_var.png)

### 6.2 Importance sampling via mean‑shift (exp06)

To estimate rare tail events more efficiently, shift the sampling distribution toward losses. Under a Gaussian model, a mean‑shift proposal \(g\) (vs original \(f\)) yields an unbiased estimator via likelihood ratios:

\[
\mathbb E_f[h(X)] = \mathbb E_g\left[h(X)\frac{f(X)}{g(X)}\right]
\]

We implement a tunable **shift scale** and evaluate variance/RMSE improvements.

**Figures:** shift sweep diagnostics  
![](paper/figures/exp06_shift_std_var.png)  
![](paper/figures/exp06_shift_rmse_var.png)

---

## 7. Model misspecification (exp07)

We generate a synthetic “true” path from multivariate Student‑t with degrees of freedom \(\nu_{\text{true}}\), then forecast VaR with:

- a **Gaussian** model (misspecified),
- a Student‑t model with \(\nu_{\text{model}}\) (less misspecified / correct).

Backtest via exception rate and Kupiec UC statistic.

**Figures:** exception rates and Kupiec LR across settings  
![](paper/figures/exp07_exceptions.png)  
![](paper/figures/exp07_kupiec_lr.png)

---

## 8. VaR/ES backtesting + proper scoring (exp08–exp10)

### 8.1 Backtesting tests

Define exceptions:
\[
I_t = \mathbf{1}\{L_t > \widehat{\mathrm{VaR}}_{\alpha,t}\}
\]
Nominal exception probability is \(p = 1-\alpha\).

**Kupiec UC test** (unconditional coverage) compares observed exceptions \(k\) among \(n\) forecasts to the binomial rate \(p\). The LR statistic is asymptotically \(\chi^2(1)\).

**Christoffersen IND test** checks whether exceptions cluster (Markov vs iid Bernoulli), also asymptotically \(\chi^2(1)\).

### 8.2 Proper scoring: Fissler–Ziegel FZ0

We use an FZ0‑type strictly consistent score for joint VaR–ES forecasts (loss version):
\[
S_t = (\mathbf{1}\{L_t > V_t\}-p)\frac{V_t}{E_t} + \frac{L_t}{E_t} + \log(E_t),
\quad E_t>0
\]
Lower is better. Scores complement UC/IND tests (a good score does not guarantee nominal coverage).

**Figure:** model ranking by mean FZ0 score  
![](paper/figures/fz0_model_ranking.png)

---

## 9. Conditional volatility: GARCH(1,1) (exp09–exp10)

### 9.1 GARCH(1,1) dynamics

Model portfolio returns:
\[
r_t = \mu + \sigma_t z_t,\quad
\sigma_t^2 = \omega + \alpha\,\varepsilon_{t-1}^2 + \beta\,\sigma_{t-1}^2
\]
with innovations:
- exp09: \(z_t\sim\mathcal N(0,1)\)
- exp10: \(z_t\sim t_\nu\) (Student‑t), estimating \(\nu\)

This captures volatility clustering and tail thickness simultaneously.

**Figures:** GARCH‑t estimated \(\sigma_t\) and \(\nu_t\) series  
![](paper/figures/exp10_sigma_series.png)  
![](paper/figures/exp10_nu_series.png)

---

## 10. Empirical summary (key table)

| model          |   alpha |   window |   exception_rate |   kupiec_p_uc |   christoffersen_p_ind |   es_ratio |   fz0_mean |    n |   n_exceed |
|:---------------|--------:|---------:|-----------------:|--------------:|-----------------------:|-----------:|-----------:|-----:|-----------:|
| normal         |    0.99 |      500 |       0.021978   |   4.34995e-05 |              0.0417073 |   1.12225  |   -3.20957 | 1547 |         34 |
| historical     |    0.99 |      500 |       0.0168067  |   0.0142198   |              0.0083077 |   0.870115 |   -2.94279 | 1547 |         26 |
| garch11_normal |    0.99 |      500 |       0.0226244  |   1.84661e-05 |              0.240866  |   0.925417 |   -3.39014 | 1547 |         35 |
| garch11_t      |    0.99 |      500 |       0.00904977 |   0.702639    |              0.612966  |   0.75895  |   -3.07765 | 1547 |         14 |

Interpretation highlights:
- Static Gaussian / historical methods tend to **reject** UC/IND at \(\alpha=0.99\).
- GARCH‑Normal often improves independence but may still **reject UC**.
- **GARCH‑Student‑t** shows calibrated exceptions (high UC/IND p‑values) and provides interpretable tail thickness via \(\nu_t\).

---

## 11. Conclusion

Extreme-tail inference is inherently data-hungry and sensitive to model assumptions.

Asymptotic CLTs may break down when density near the quantile is small.\ DKW bounds provide finite-sample guarantees at the cost of conservatism.\ Variance reduction improves efficiency but cannot fix model misspecification.\ Dynamic volatility combined with heavy-tailed innovations provides the most reliable 99% VaR/ES calibration in our experiments.

---
## 12. Reproducibility

- Experiments are configured as YAML in `experiments/`.
- Runner: `python -m scripts.run_all_experiments`
- Each experiment writes CSV tables to `paper/tables/` and figures to `paper/figures/`.
- The notebook `notebooks/02_experiments_overview.ipynb` loads those artifacts and produces visualizations.

---

## Appendix: Where the code lives

- Quantile theory and inverse CDF helpers: `src/theory.py`
- DKW bounds: `src/bounds.py`
- Heavy tails: `src/heavy_tail.py`
- Importance sampling: `src/importance_sampling.py`
- Backtests (Kupiec/Christoffersen): `src/backtests.py`
- GARCH wrappers: `src/garch_model.py`
- Proper scoring: `src/scoring.py`
- Experiments entry points: `src/experiments.py`, YAML runner: `src/run_from_yaml.py`

---
