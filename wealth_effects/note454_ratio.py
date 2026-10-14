"""Auxiliary income-normalized wealth relation, distinct from note 454's ECM.

Beach, Gamber and Moran's cited ratio specification motivates this benchmark.
Its transfer MPC of one is assumed. Public note 454 does not disclose whether
its long-run equation uses this normalization. No reported MPC is imposed.
"""
import numpy as np
import pandas as pd
from scipy.stats import t
import statsmodels.api as sm


REQUIRED_LEVELS = ("C_real_quarter_bn", "Y_real_quarter_bn",
                   "Transfers_real_quarter_bn", "W80_real_stock_bn", "W20_real_stock_bn")
WEALTH_COLUMNS = ("wealth_bottom80", "wealth_top20")


def fit_ratio_dols(panel, start="1990Q1", end="2019Q4", leads_lags=1, hac_lags=4):
    """Fit ratio OLS (order zero) or DOLS, using only the specified training data.

    Positive orders augment both wealth ratios with first differences at all
    leads, contemporaneous differences and lags up to that order. Differences
    are formed AFTER trimming to the training window, then unavailable boundary
    rows are explicitly removed. Consequently, no post-training observation or
    presample difference enters this auxiliary regression.

    Consumption, income and transfer flows are multiplied by four; wealth
    stocks are not. The wealth slopes therefore measure annual consumption
    dollars per dollar of wealth, under the maintained transfer-MPC assumption.
    Returns coefficients/fitted/diagnostics and the full params/covariance/X/y
    objects for further research. HAC intervals are conditional associations.
    """
    for name, value in (("leads_lags", leads_lags), ("hac_lags", hac_lags)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(name + " must be a nonnegative integer")
    if not isinstance(panel.index, pd.PeriodIndex) or panel.index.freqstr != "Q-DEC":
        raise ValueError("A quarterly Q-DEC PeriodIndex is required")
    if panel.index.duplicated().any() or not panel.index.is_monotonic_increasing:
        raise ValueError("Quarterly dates must be unique and sorted")
    if not panel.index.equals(pd.period_range(panel.index.min(), panel.index.max(), freq="Q")):
        raise ValueError("Missing quarter; no interpolation")
    first, last = pd.Period(start, freq="Q"), pd.Period(end, freq="Q")
    if first > last:
        raise ValueError("Training end precedes start")
    dates = pd.period_range(first, last, freq="Q")
    if not dates.isin(panel.index).all():
        raise ValueError("Requested training quarter absent")
    if not set(REQUIRED_LEVELS).issubset(panel.columns):
        raise ValueError("Missing canonical ratio fields: " + str(sorted(set(REQUIRED_LEVELS) - set(panel.columns))))
    levels = panel.reindex(dates)[list(REQUIRED_LEVELS)].copy()
    if not np.isfinite(levels.to_numpy(dtype=float)).all():
        raise ValueError("Nonfinite observed ratio input; no filling")
    if (levels.C_real_quarter_bn <= 0).any() or (levels.Y_real_quarter_bn <= 0).any() or (levels.Transfers_real_quarter_bn < 0).any():
        raise ValueError("Positive consumption/income and nonnegative transfer flows required")
    C = 4. * levels.C_real_quarter_bn
    Y = 4. * levels.Y_real_quarter_bn
    T = 4. * levels.Transfers_real_quarter_bn
    denominator = Y - T
    if (denominator <= 0).any():
        raise ValueError("Annualized income excluding transfers must be positive")
    y = (C - T) / denominator
    X = pd.DataFrame({"const": 1.,
        "wealth_bottom80": levels.W80_real_stock_bn / denominator,
        "wealth_top20": levels.W20_real_stock_bn / denominator}, index=dates)
    if leads_lags:
        differences = X[list(WEALTH_COLUMNS)].diff()
        for name in WEALTH_COLUMNS:
            for offset in range(-leads_lags, leads_lags + 1):
                label = f"lead{-offset}" if offset < 0 else f"lag{offset}"
                X[f"d_{name}_{label}"] = differences[name].shift(offset)
    valid = X.notna().all(axis=1)
    removed = [str(date) for date in X.index[~valid]]
    X, y = X.loc[valid], y.loc[valid]
    if len(X) <= X.shape[1] + 10:
        raise ValueError("Too few observations for the specified ratio/DOLS fit")
    if hac_lags >= len(X):
        raise ValueError("HAC bandwidth must be shorter than the effective sample")
    scales = X.std().replace(0., 1.)
    scales["const"] = 1.
    standardized = X / scales
    if np.linalg.matrix_rank(standardized.to_numpy()) != X.shape[1]:
        raise ValueError("Rank-deficient ratio/DOLS design; no inferred zero or pseudoinverse slopes")
    result = sm.OLS(y, standardized, missing="raise").fit(
        cov_type="HAC", cov_kwds={"maxlags": hac_lags, "use_correction": True}, use_t=True)
    params = result.params / scales
    covariance = result.cov_params().div(scales, axis=0).div(scales, axis=1)
    model = "ratio_ols" if leads_lags == 0 else "ratio_dols"
    critical = t.ppf(.975, result.df_resid)
    error = np.sqrt(np.maximum(np.diag(covariance), 0.))
    rows = []
    for name, value, se in zip(params.index, params.to_numpy(), error):
        wealth_slope = name in WEALTH_COLUMNS
        rows.append({"model": model, "regressor": name, "estimate": value,
            "hac_standard_error": se, "ci_low": value - critical * se,
            "ci_high": value + critical * se,
            "annual_consumption_mpc": value if wealth_slope else np.nan,
            "quarterly_consumption_mpc": value / 4. if wealth_slope else np.nan,
            "quarterly_mpc_ci_low": (value - critical * se) / 4. if wealth_slope else np.nan,
            "quarterly_mpc_ci_high": (value + critical * se) / 4. if wealth_slope else np.nan,
            "role": "long_run_wealth_slope" if wealth_slope else "nontransfer_income_slope" if name == "const" else "DOLS_nuisance_difference_control",
            "units": "annual real consumption dollars per real wealth dollar" if wealth_slope else "dimensionless ratio coefficient",
            "status": "Estimated auxiliary conditional relation; transfer MPC one assumed; not note454 author coefficients"})
    long_prediction = X[["const", *WEALTH_COLUMNS]] @ params[["const", *WEALTH_COLUMNS]]
    full_prediction = X @ params
    fitted = pd.DataFrame({"quarter": X.index.astype(str), "model": model,
        "observed_consumption_income_ratio": y.to_numpy(),
        "fitted_long_run_ratio": long_prediction.to_numpy(),
        "fitted_ratio_including_dols_terms": full_prediction.to_numpy(),
        "residual_ratio": (y - full_prediction).to_numpy(),
        "annual_real_nontransfer_income_bn": denominator.reindex(X.index).to_numpy(),
        "observed_real_consumption_saar_bn": C.reindex(X.index).to_numpy(),
        "implied_long_run_real_consumption_saar_bn": (T + denominator * long_prediction).reindex(X.index).to_numpy()})
    diagnostics = {"model": model, "requested_training_start": str(first),
        "requested_training_end": str(last), "effective_sample_start": str(X.index[0]),
        "effective_sample_end": str(X.index[-1]), "observations": len(X),
        "requested_training_observations": len(dates), "removed_boundary_quarters": removed,
        "parameters": X.shape[1], "degrees_of_freedom": int(result.df_resid),
        "leads_lags": leads_lags, "hac_lags": hac_lags,
        "standardized_condition_number": float(np.linalg.cond(standardized)),
        "r_squared": float(result.rsquared), "residual_ratio_rmse": float(np.sqrt(np.mean((y - full_prediction) ** 2))),
        "flow_units": "C, Y and transfers: actual quarterly real flows multiplied by four to form annualized flow levels",
        "stock_units": "Real W80/W20 stocks are not annualized",
        "transfer_mpc": 1., "transfer_mpc_status": "Maintained assumption, not an estimated restriction tested here",
        "training_isolation": "Ratios, differences and all leads/lags constructed inside the stated training window; no holdout leads",
        "inference": "Unrestricted statsmodels OLS; Bartlett HAC n/(n-k) correction and pointwise t intervals",
        "replication_status": "Auxiliary adaptation motivated by Beach/Gamber/Moran; not the undisclosed Banque de France equation",
        "identification": "Conditional level relation; no causal identification or cointegration validation established by this fit",
        "source_group_definition": panel.attrs.get("wealth_group_definition", "Canonical W80/W20 income-group wealth; see input provenance"),
    }
    return {"coefficients": pd.DataFrame(rows), "fitted": fitted, "diagnostics": diagnostics,
            "params": params, "covariance": covariance, "X": X, "y": y,
            "index": X.index.copy()}
