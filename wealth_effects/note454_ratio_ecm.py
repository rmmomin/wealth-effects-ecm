"""Auxiliary Beach-compatible ratio/DOLS long-run normalization for note 454.

This is a disclosed functional-form sensitivity, not the unpublished Banque
de France equation. A transfer MPC of one is assumed by the referenced ratio
normalization. DOLS annual-dollar MPCs and log-model elasticities differ.
The same short-run log-growth inputs and anchored accounting are retained.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd
from statsmodels.stats.diagnostic import acorr_breusch_godfrey

from .note454_ratio import fit_ratio_dols
from .note454_ecm import (
    SHORT_COLUMNS, _check_panel, _circular_blocks, _level_radius, _ols, _spec,
    annualization_factor, short_design,
)


@dataclass
class RatioECMFit:
    long_run_params: pd.Series
    short_run_params: pd.Series
    short_run_covariance: pd.DataFrame
    long_X: pd.DataFrame
    short_X: pd.DataFrame
    y: pd.Series
    residuals: pd.Series
    cointegration_residual: pd.Series
    index: pd.PeriodIndex
    spec: dict
    diagnostics: dict
    df_resid: int
    ratio_fit: dict


def _levels(panel):
    names = ("C_real_quarter_bn", "Y_real_quarter_bn", "Transfers_real_quarter_bn",
             "W80_real_stock_bn", "W20_real_stock_bn")
    if not set(names).issubset(panel.columns) or not np.isfinite(panel[list(names)].to_numpy()).all():
        raise ValueError("Finite observed consumption/income/transfer/wealth levels required")
    if (panel.Y_real_quarter_bn - panel.Transfers_real_quarter_bn <= 0).any():
        raise ValueError("Income excluding transfers must be positive")


def _target(params, Y_quarter, T_quarter, W80_stock, W20_stock):
    """Annual-flow ratio prediction converted to an actual quarterly flow."""
    annual = (4. * T_quarter + params["const"] * (4. * Y_quarter - 4. * T_quarter)
              + params["wealth_bottom80"] * W80_stock + params["wealth_top20"] * W20_stock)
    if not np.isfinite(np.asarray(annual)).all() or (np.asarray(annual) <= 0).any():
        raise ValueError("Nonpositive ratio/DOLS target consumption; log ECM unsupported for these inputs")
    return annual / 4.


def ratio_log_target(params, panel):
    _levels(panel)
    return np.log(_target(params, panel.Y_real_quarter_bn, panel.Transfers_real_quarter_bn,
                          panel.W80_real_stock_bn, panel.W20_real_stock_bn))


def shapley_target_changes(params, panel, base, dates):
    """Exact two-group log-target allocation: income/transfers versus wealth.

    Both orders of changing the two bundles are averaged. All four corners
    must imply positive consumption; no clipping manufactures a log target.
    """
    _levels(panel)
    base = pd.Period(base, freq="Q")
    dates = pd.PeriodIndex(dates, freq="Q")
    if base not in panel.index or not dates.isin(panel.index).all():
        raise ValueError("Shapley allocation dates absent")
    b = panel.loc[base]
    current = panel.reindex(dates)
    f00 = np.log(_target(params, b.Y_real_quarter_bn, b.Transfers_real_quarter_bn,
                         b.W80_real_stock_bn, b.W20_real_stock_bn))
    f10 = np.log(_target(params, current.Y_real_quarter_bn, current.Transfers_real_quarter_bn,
                         b.W80_real_stock_bn, b.W20_real_stock_bn))
    f01 = np.log(_target(params, b.Y_real_quarter_bn, b.Transfers_real_quarter_bn,
                         current.W80_real_stock_bn, current.W20_real_stock_bn))
    f11 = ratio_log_target(params, current)
    income = .5 * ((f10 - f00) + (f11 - f01))
    wealth = .5 * ((f01 - f00) + (f11 - f10))
    np.testing.assert_allclose(income + wealth, f11 - f00, atol=1e-12)
    return pd.DataFrame({"income_target_log_change": income, "wealth_target_log_change": wealth,
                         "total_target_log_change": f11 - f00}, index=dates)


def fit_ratio_ecm(panel, spec=None):
    """DOLS ratio stage followed by the identical short-run log-growth OLS."""
    _check_panel(panel)
    _levels(panel)
    spec = _spec(spec)
    order = spec.get("ratio_leads_lags", 1)
    ratio = fit_ratio_dols(panel, start=spec["start"], end=spec["end"],
                           leads_lags=order, hac_lags=spec["hac_lags"])
    params = ratio["params"]
    target = ratio_log_target(params, panel)
    residual = panel.logC - target
    dates = pd.period_range(spec["start"], spec["end"], freq="Q")
    X = short_design(panel, residual, spec).reindex(dates)
    y = panel.dlogC.reindex(dates)
    short, covariance, result, condition = _ols(y, X, spec["hac_lags"])
    errors = y - X @ short
    radius = _level_radius(short, spec["consumption_lags"])
    bg = acorr_breusch_godfrey(result, nlags=4)
    diagnostics = {
        "sample_start": str(dates[0]), "sample_end": str(dates[-1]),
        "observations": len(dates), "short_run_parameters": len(short),
        "short_run_df_resid": int(result.df_resid), "short_run_condition_number": condition,
        "short_run_r_squared": float(result.rsquared),
        "short_run_rmse_log_growth": float(np.sqrt(np.mean(errors ** 2))),
        "adjustment_coefficient": float(short["ECT_L1"]),
        "level_recursion_spectral_radius": radius, "stable_level_recursion": radius < 1.,
        "breusch_godfrey_4lag_pvalue": float(bg[1]), "ratio_long_run": ratio["diagnostics"],
        "functional_form_status": "Auxiliary Beach-compatible ratio/DOLS target, not an authors' Banque de France ECM equation",
        "transfer_assumption": "Long-run MPC out of transfers fixed at one by the referenced ratio normalization",
        "wealth_coefficient_units": "Annual consumption dollars per wealth-stock dollar; wealth is not annualized",
        "target_error_definition": "log observed quarterly C minus log positive ratio-predicted annual C/4",
        "cointegration_test_status": "No ordinary residual ADF/MacKinnon Engle-Granger test claimed for this nonlinear DOLS target",
        "short_run_inference": "Pointwise HAC conditional on fitted ratio/DOLS target; full two-stage bootstrap reported separately",
        "identification": "Conditional functional-form sensitivity, not causal wealth effects",
    }
    return RatioECMFit(params, short, covariance, ratio["X"], X, y, errors,
                       residual, dates, spec, diagnostics, int(result.df_resid), ratio)


def decompose_ratio_ecm(fit, panel, start="1990Q1", end="2025Q4"):
    """Anchored nonlinear target allocation with exact linear dynamic feedback."""
    _check_panel(panel)
    _levels(panel)
    dates = pd.period_range(start, end, freq="Q")
    base = dates[0] - 1
    if base not in panel.index or not dates.isin(panel.index).all():
        raise ValueError("Ratio decomposition needs base and all requested quarters")
    target = ratio_log_target(fit.long_run_params, panel)
    residual = panel.logC - target
    X = short_design(panel, residual, fit.spec).reindex(dates)
    if not np.isfinite(X.to_numpy()).all():
        raise ValueError("Missing ratio decomposition presample input")
    changes = shapley_target_changes(fit.long_run_params, panel, base, dates - 1)
    changes.index = dates
    channels = ["income", "wealth", "credit", "unemployment", "deterministic", "initial_state", "unexplained"]
    growth = pd.DataFrame(0., index=dates, columns=channels)
    levels = growth.copy()
    lam = fit.short_run_params["ECT_L1"]
    mapping = {"dlogY_ex_transfers": "income", "dlogTransfers": "income",
               "dlogCredit": "credit", "dlogSP500": "wealth", "dUnemployment": "unemployment"}
    for date in dates:
        innovation = pd.Series(0., index=channels)
        innovation["income"] = -lam * changes.loc[date, "income_target_log_change"]
        innovation["wealth"] = -lam * changes.loc[date, "wealth_target_log_change"]
        innovation["deterministic"] = fit.short_run_params["const"]
        innovation["initial_state"] = lam * residual.loc[base]
        for name, channel in mapping.items():
            for lag in range(fit.spec["short_lags"] + 1):
                term = fit.short_run_params[f"{name}_L{lag}"] * panel.loc[date - lag, name]
                innovation[channel if date - lag >= dates[0] else "initial_state"] += term
        innovation["unexplained"] = panel.loc[date, "dlogC"] - X.loc[date] @ fit.short_run_params
        growth.loc[date] = innovation
        if date > dates[0]:
            growth.loc[date] += lam * levels.loc[date - 1]
        for lag in range(1, fit.spec["consumption_lags"] + 1):
            phi = fit.short_run_params[f"consumption_L{lag}"]
            if date - lag >= dates[0]:
                growth.loc[date] += phi * growth.loc[date - lag]
            else:
                growth.loc[date, "initial_state"] += phi * panel.loc[date - lag, "dlogC"]
        levels.loc[date] = growth.loc[date] + (levels.loc[date - 1] if date > dates[0] else 0.)
    np.testing.assert_allclose(growth.sum(axis=1), panel.dlogC.reindex(dates), atol=1e-10, rtol=1e-9)
    np.testing.assert_allclose(levels.sum(axis=1), panel.logC.reindex(dates) - panel.loc[base, "logC"], atol=1e-10, rtol=1e-9)
    annualized = growth.mul(annualization_factor(panel.dlogC.reindex(dates)), axis=0)
    observed = 100. * np.expm1(4. * panel.dlogC.reindex(dates))
    np.testing.assert_allclose(annualized.sum(axis=1), observed, atol=1e-8, rtol=1e-9)
    rows = []
    for channel in channels:
        for date in dates:
            rows.append({"quarter": str(date), "channel": channel,
                "log_growth_contribution": float(growth.loc[date, channel]),
                "log_level_contribution": float(levels.loc[date, channel]),
                "annualized_consumption_growth_pp": float(annualized.loc[date, channel]),
                "observed_annualized_growth_pct": float(observed.loc[date]),
                "base_quarter": str(base), "status": "Auxiliary ratio/DOLS nonlinear-target allocation; conditional, not causal"})
    return pd.DataFrame(rows)


def _simulate_ratio_consumption(fit, panel, errors):
    out = panel.copy()
    target = ratio_log_target(fit.long_run_params, panel)
    params = fit.short_run_params
    for j, date in enumerate(fit.index):
        value = params["const"] + params["ECT_L1"] * (out.loc[date - 1, "logC"] - target.loc[date - 1]) + errors[j]
        for lag in range(1, fit.spec["consumption_lags"] + 1):
            value += params[f"consumption_L{lag}"] * out.loc[date - lag, "dlogC"]
        for name in SHORT_COLUMNS:
            for lag in range(fit.spec["short_lags"] + 1):
                value += params[f"{name}_L{lag}"] * panel.loc[date - lag, name]
        out.loc[date, "dlogC"] = value
        out.loc[date, "logC"] = out.loc[date - 1, "logC"] + value
    out.loc[out.index[1:], "dlogC"] = out.logC.diff().iloc[1:]
    out["C_real_quarter_bn"] = np.exp(out.logC)
    return out


def bootstrap_ratio_ecm(fit, panel, decomposition_start="1990Q1", decomposition_end="2025Q4",
                        draws=399, block_length=4, seed=2454):
    """Conditional circular-block residual bootstrap of BOTH ratio-ECM stages.

    Consumption is recursively simulated while all other observed input paths
    remain fixed. DOLS and the short-run equation are then reestimated; each
    draw's historical allocation uses the original observed panel. Inference
    is model-conditional, not unconditional or identification-robust, and does
    not validate cointegration for the nonlinear generated target.
    """
    from .note454_ecm import annual_decomposition

    if not isinstance(draws, int) or draws < 2 or not isinstance(block_length, int) or not 1 <= block_length <= len(fit.index):
        raise ValueError("Bootstrap needs at least two draws and a valid positive block length")
    rng = np.random.default_rng(seed)
    residuals = fit.residuals.to_numpy() - fit.residuals.mean()
    parameter_rows, annual_rows, failures = [], [], []
    instability, nonnegative_loading, nonpositive_target = 0, 0, 0
    for draw in range(draws):
        try:
            simulated = _simulate_ratio_consumption(fit, panel, _circular_blocks(residuals, block_length, rng))
            replicate = fit_ratio_ecm(simulated, fit.spec)
            annual = annual_decomposition(decompose_ratio_ecm(replicate, panel, decomposition_start, decomposition_end))
            for stage, params in (("long_run", replicate.long_run_params), ("short_run", replicate.short_run_params)):
                for name, value in params.items():
                    parameter_rows.append({"draw": draw, "stage": stage, "regressor": name, "estimate": float(value)})
            annual_rows.append(annual.assign(draw=draw))
            instability += not replicate.diagnostics["stable_level_recursion"]
            nonnegative_loading += replicate.short_run_params["ECT_L1"] >= 0.
        except (ValueError, AssertionError, np.linalg.LinAlgError, FloatingPointError) as error:
            message = str(error)
            nonpositive_target += "Nonpositive ratio/DOLS target" in message
            failures.append({"draw": draw, "error": message})
    completed = len(annual_rows)
    if completed < max(2, int(np.ceil(.8 * draws))):
        raise RuntimeError(f"Only {completed}/{draws} ratio-ECM bootstrap draws completed; failures must be inspected")
    samples = pd.concat(annual_rows, ignore_index=True)
    intervals = samples.groupby(["year", "channel"]).contribution_pp.quantile([.025, .5, .975]).unstack().reset_index()
    intervals.columns = ["year", "channel", "ci_low", "bootstrap_median", "ci_high"]
    original = annual_decomposition(decompose_ratio_ecm(fit, panel, decomposition_start, decomposition_end))
    intervals = original.merge(intervals, on=["year", "channel"], validate="one_to_one")
    intervals["inference"] = "95% pointwise conditional circular-block residual percentile interval; ratio DOLS and short ECM refitted"
    return {"diagnostics": {"requested_draws": draws, "completed_draws": completed,
            "block_length_quarters": block_length, "seed": seed, "failures": failures,
            "unstable_refit_draws_retained": int(instability), "nonnegative_loading_draws_retained": int(nonnegative_loading),
            "nonpositive_target_failures": int(nonpositive_target),
            "method": "Recursive conditional residual bootstrap; ratio DOLS and short-stage ECM both refitted",
            "limitations": ["Other observed input paths held fixed; no unconditional joint-process bootstrap",
                            "Pointwise percentile intervals; specification and identification uncertainty excluded",
                            "Does not validate cointegration or eliminate residual misspecification for this nonlinear target"]},
            "parameter_draws": pd.DataFrame(parameter_rows), "annual_draws": samples,
            "annual_intervals": intervals}
