import numpy as np
import pandas as pd
from arch import arch_model
from scipy.stats import norm, t

def _normalize_weights(weights):
    w = np.asarray(weights, dtype=float)
    return w / w.sum()

def garch_rolling_var(
    returns_df,
    weights,
    window=500,
    alpha=0.99,
    dist="t",
    refit_every=20,
    scale=100.0,
):
    """
    Rolling 1-day VaR using a GARCH(1,1) on PORTFOLIO returns.

    dist: "normal" or "t"
    window: rolling estimation window size
    refit_every: refit model every N days (speed)
    scale: arch fits better if returns are scaled (e.g., percent). We'll scale and then scale back.

    Returns a DataFrame with columns:
      VaR, RealizedLoss, Exception
    """
    w = _normalize_weights(weights)
    port_ret = (returns_df.values @ w)
    idx = returns_df.index

    # scale returns for numerical stability
    r = pd.Series(port_ret * scale, index=idx)

    var_list, loss_list, exc_list = [], [], []

    fit_res = None
    last_fit_t = None

    for t_i in range(window, len(r)):
        train = r.iloc[t_i - window:t_i]

        need_refit = (fit_res is None) or (last_fit_t is None) or ((t_i - last_fit_t) >= refit_every)
        if need_refit:
            am = arch_model(
                train,
                mean="Constant",
                vol="GARCH",
                p=1,
                q=1,
                dist=("t" if dist == "t" else "normal"),
            )
            fit_res = am.fit(disp="off")
            last_fit_t = t_i

        # 1-step ahead forecast
        fcst = fit_res.forecast(horizon=1, reindex=False)
        mu_hat = float(fit_res.params.get("mu", 0.0))
        # variance forecast is last row, horizon=1 column
        sigma_hat = float(np.sqrt(fcst.variance.values[-1, 0]))

        # VaR in RETURN space (scaled)
        # VaR(loss) = -(mu - z_alpha * sigma)  (z_alpha > 0)
        if dist == "normal":
            z = float(norm.ppf(alpha))
        else:
            # Student-t quantile with df = nu
            nu = float(fit_res.params.get("nu"))
            z = float(t.ppf(alpha, df=nu))

        var_scaled = -(mu_hat - z * sigma_hat)  # positive loss (scaled units)

        # realized loss (scaled)
        realized_loss_scaled = float(-(r.iloc[t_i]))

        exc = int(realized_loss_scaled > var_scaled)

        # scale back to original units (log-return units)
        var_list.append(var_scaled / scale)
        loss_list.append(realized_loss_scaled / scale)
        exc_list.append(exc)

    out = pd.DataFrame(
        {"VaR": var_list, "RealizedLoss": loss_list, "Exception": exc_list},
        index=idx[window:]
    )
    return out