from __future__ import annotations

import argparse
from pathlib import Path
import yaml  

from src.experiments import (
    exp01_classical_var,
    exp02_normal_vs_t,
    exp03_asymptotics,
    exp04_dkw_bounds,
    exp05_variance_reduction,
    exp06_importance_sampling,
    exp07_misspecification,
    exp08_es_backtesting,
    exp09_garch_var_es,
    exp10_garch_t_var_es,
    _parse_weights,  # uses weights or weights-file
)

def main():
    ap = argparse.ArgumentParser(description="Run an experiment from a YAML config.")
    ap.add_argument("config", help="Path to experiments/*.yml")
    args = ap.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")

    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))

    exp = cfg["exp"]
    returns = cfg["returns"]
    weights = _parse_weights(cfg.get("weights"), cfg.get("weights_file"))

    if exp == "exp01":
        df = exp01_classical_var(
            returns_path=returns,
            weights=weights,
            n_sims=int(cfg.get("n_sims", 20000)),
            alpha=float(cfg.get("alpha", 0.99)),
            seed=int(cfg.get("seed", 0)),
            out_csv=str(cfg.get("out", "paper/tables/exp01_classical_var.csv")),
        )

    elif exp == "exp02":
        df = exp02_normal_vs_t(
            returns_path=returns,
            weights=weights,
            n_sims=int(cfg.get("n_sims", 20000)),
            alpha=float(cfg.get("alpha", 0.99)),
            seed=int(cfg.get("seed", 0)),
            nu=cfg.get("nu", None),
            out_csv=str(cfg.get("out", "paper/tables/exp02_normal_vs_t.csv")),
        )

    elif exp == "exp03":
        n_grid = cfg.get("n_grid", [250, 500, 1000, 2000])
        df = exp03_asymptotics(
            returns_path=returns,
            weights=weights,
            alpha=float(cfg.get("alpha", 0.99)),
            conf=float(cfg.get("conf", 0.95)),
            seed=int(cfg.get("seed", 0)),
            n_grid=[int(x) for x in n_grid],
            reps=int(cfg.get("reps", 200)),
            out_summary=str(cfg.get("out", "paper/tables/exp03_asymptotics_summary.csv")),
        )

    elif exp == "exp04":
        n_grid = cfg.get("n_grid", [250, 500, 1000, 2000])
        df = exp04_dkw_bounds(
            returns_path=returns,
            weights=weights,
            alpha=float(cfg.get("alpha", 0.99)),
            conf=float(cfg.get("conf", 0.95)),
            delta=float(cfg.get("delta", 0.05)),
            seed=int(cfg.get("seed", 0)),
            n_grid=[int(x) for x in n_grid],
            reps=int(cfg.get("reps", 200)),
            out_csv=str(cfg.get("out", "paper/tables/exp04_dkw_bounds.csv")),
        )

    elif exp == "exp05":
        df = exp05_variance_reduction(
            returns_path=returns,
            weights=weights,
            alpha=float(cfg.get("alpha", 0.99)),
            seed=int(cfg.get("seed", 0)),
            n_sims=int(cfg.get("n_sims", 20000)),
            reps=int(cfg.get("reps", 200)),
            out_csv=str(cfg.get("out", "paper/tables/exp05_variance_reduction.csv")),
        )

    elif exp == "exp06":
        df = exp06_importance_sampling(
            returns_path=returns,
            weights=weights,
            alpha=float(cfg.get("alpha", 0.99)),
            seed=int(cfg.get("seed", 0)),
            n_sims=int(cfg.get("n_sims", 20000)),
            reps=int(cfg.get("reps", 200)),
            shift_scale=float(cfg.get("shift_scale", 1.0)),
            out_csv=str(cfg.get("out", "paper/tables/exp06_importance_sampling.csv")),
        )

    elif exp == "exp07":
        df = exp07_misspecification(
            returns_path=returns,
            weights=weights,
            alpha=float(cfg.get("alpha", 0.99)),
            seed=int(cfg.get("seed", 0)),
            T=int(cfg.get("T", 2500)),
            window=int(cfg.get("window", 500)),
            nu_true=float(cfg.get("nu_true", 5.0)),
            nu_model=float(cfg.get("nu_model", 5.0)),
            out_csv=str(cfg.get("out", "paper/tables/exp07_misspecification.csv")),
        )

    elif exp == "exp08":
        df = exp08_es_backtesting(
            returns_path=returns,
            weights=weights,
            alpha=float(cfg.get("alpha", 0.99)),
            window=int(cfg.get("window", 500)),
            seed=int(cfg.get("seed", 0)),
            out_csv=str(cfg.get("out", "paper/tables/exp08_es_backtesting.csv")),
        )
    
    elif exp == "exp09":
        df = exp09_garch_var_es(
            returns_path=returns,
            weights=weights,
            alpha=float(cfg.get("alpha", 0.99)),
            window=int(cfg.get("window", 500)),
            seed=int(cfg.get("seed", 0)),
            out_csv=str(cfg.get("out", "paper/tables/exp09_garch_var_es.csv")),
        )

    elif exp == "exp10":
        df = exp10_garch_t_var_es(
            returns_path=returns,
            weights=weights,
            alpha=float(cfg.get("alpha", 0.99)),
            window=int(cfg.get("window", 500)),
            seed=int(cfg.get("seed", 0)),
            out_csv=str(cfg.get("out", "paper/tables/exp10_garch_t_var_es.csv")),
        )

    else:
        raise ValueError(f"Unknown exp: {exp}")

    print(df.to_string(index=False))

if __name__ == "__main__":
    main()