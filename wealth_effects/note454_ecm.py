"""Transparent ECM adaptation of Banque de France EcoNotepad 454.

The note discloses the variables and 1990--2019 sample, but not its equations,
lags or coefficients. The log-form Engle--Granger normalization and lag choices
here are explicit research choices, not an exact reproduction of undisclosed
authors' code. No coefficient signs or 2025 contribution target are imposed.
"""
from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy.stats import t
import statsmodels.api as sm
from statsmodels.stats.diagnostic import acorr_breusch_godfrey
from statsmodels.tsa.stattools import adfuller, coint


LONG_COLUMNS = ("logY", "logW80", "logW20")
SHORT_COLUMNS = ("dlogY_ex_transfers", "dlogTransfers", "dlogCredit",
                 "dlogSP500", "dUnemployment")
DEFAULT_SPEC = {"start": "1990Q1", "end": "2019Q4", "short_lags": 1,
                "consumption_lags": 1, "hac_lags": 4,
                "coint_maxlag": 4, "coint_autolag": "aic"}


@dataclass
class ECMFit:
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


def _check_panel(panel):
    if not isinstance(panel.index, pd.PeriodIndex) or panel.index.freqstr != "Q-DEC":
        raise ValueError("A quarterly Q-DEC PeriodIndex is required")
    if panel.index.duplicated().any() or not panel.index.is_monotonic_increasing:
        raise ValueError("Quarterly dates must be unique and sorted")
    if not panel.index.equals(pd.period_range(panel.index.min(), panel.index.max(), freq="Q")):
        raise ValueError("Missing quarter; no interpolation or gap-crossing lags")
    columns = ["logC", "dlogC", *LONG_COLUMNS, *SHORT_COLUMNS]
    if not set(columns).issubset(panel.columns):
        raise ValueError("Missing canonical ECM fields: " + str(sorted(set(columns) - set(panel.columns))))
    if not np.isfinite(panel[["logC", *LONG_COLUMNS]].to_numpy(dtype=float)).all():
        raise ValueError("Nonfinite log-level ECM input; no filling")
    changes = panel[["dlogC", *SHORT_COLUMNS]].to_numpy(dtype=float)
    if not np.isfinite(changes[1:]).all() or np.isinf(changes[0]).any():
        raise ValueError("Nonfinite ECM growth input beyond the initial differencing row; no filling")
    if not np.allclose(panel.dlogC.iloc[1:], panel.logC.diff().iloc[1:], atol=1e-10, rtol=1e-8):
        raise ValueError("dlogC must equal the first difference of logC")


def _spec(spec):
    out = {**DEFAULT_SPEC, **(spec or {})}
    for name in ("short_lags", "consumption_lags", "hac_lags", "coint_maxlag"):
        value = out[name]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(name + " must be a nonnegative integer")
    if pd.Period(out["start"], freq="Q") > pd.Period(out["end"], freq="Q"):
        raise ValueError("Training end precedes start")
    return out


def _ols(y, X, hac_lags=None):
    if not np.isfinite(X.to_numpy()).all() or not np.isfinite(y.to_numpy()).all():
        raise ValueError("Insufficient presample observations or nonfinite design")
    scales = X.std().replace(0., 1.)
    scales["const"] = 1.
    standardized = X / scales
    if np.linalg.matrix_rank(standardized.to_numpy()) != X.shape[1]:
        raise ValueError("Rank-deficient ECM design; no implicit pseudoinverse estimate")
    if len(X) <= X.shape[1] + 10:
        raise ValueError("Too few observations for ECM specification")
    model = sm.OLS(y, standardized, missing="raise")
    result = model.fit() if hac_lags is None else model.fit(
        cov_type="HAC", cov_kwds={"maxlags": hac_lags, "use_correction": True}, use_t=True)
    params = result.params / scales
    covariance = result.cov_params().div(scales, axis=0).div(scales, axis=1)
    return params, covariance, result, float(np.linalg.cond(standardized))


def long_design(panel):
    return sm.add_constant(panel[list(LONG_COLUMNS)], has_constant="add")


def short_design(panel, residual, spec):
    columns = {"const": pd.Series(1., index=panel.index), "ECT_L1": residual.shift(1)}
    for lag in range(1, spec["consumption_lags"] + 1):
        columns[f"consumption_L{lag}"] = panel.dlogC.shift(lag)
    for name in SHORT_COLUMNS:
        for lag in range(spec["short_lags"] + 1):
            columns[f"{name}_L{lag}"] = panel[name].shift(lag)
    return pd.DataFrame(columns, index=panel.index)


def _level_radius(params, consumption_lags):
    coefficients = np.zeros(consumption_lags + 1)
    coefficients[0] = 1. + params.get("ECT_L1", 0.)
    for lag in range(1, consumption_lags + 1):
        phi = params.get(f"consumption_L{lag}", 0.)
        coefficients[lag - 1] += phi
        coefficients[lag] -= phi
    companion = np.zeros((len(coefficients), len(coefficients)))
    companion[0] = coefficients
    if len(coefficients) > 1:
        companion[1:, :-1] = np.eye(len(coefficients) - 1)
    return float(np.max(np.abs(np.linalg.eigvals(companion))))


def fit_ecm(panel, spec=None, compute_diagnostics=True):
    """Estimate both stages on exactly the stated training dates.

    Short-stage HAC intervals condition on the estimated long-run relation.
    Use bootstrap_ecm for uncertainty that refits the generated regressor.
    """
    _check_panel(panel)
    spec = _spec(spec)
    dates = pd.period_range(spec["start"], spec["end"], freq="Q")
    if not dates.isin(panel.index).all():
        raise ValueError("Requested training quarter absent")
    Xlong = long_design(panel)
    long_params, _, long_result, long_condition = _ols(panel.logC.reindex(dates), Xlong.reindex(dates))
    residual = panel.logC - Xlong @ long_params
    Xshort = short_design(panel, residual, spec).reindex(dates)
    y = panel.dlogC.reindex(dates)
    short_params, covariance, result, condition = _ols(y, Xshort, spec["hac_lags"])
    errors = y - Xshort @ short_params
    radius = _level_radius(short_params, spec["consumption_lags"])
    diagnostics = {
        "sample_start": str(dates[0]), "sample_end": str(dates[-1]),
        "observations": len(dates), "long_run_parameters": len(long_params),
        "short_run_parameters": len(short_params), "short_run_df_resid": int(result.df_resid),
        "long_run_condition_number": long_condition, "short_run_condition_number": condition,
        "short_run_r_squared": float(result.rsquared),
        "short_run_rmse_log_growth": float(np.sqrt(np.mean(errors ** 2))),
        "adjustment_coefficient": float(short_params["ECT_L1"]),
        "level_recursion_spectral_radius": radius, "stable_level_recursion": radius < 1.,
        "functional_form_status": "Declared log-form adaptation; Banque de France equations/normalization/lags not disclosed",
        "short_run_inference": "Bartlett HAC with n/(n-k) correction and t critical values; conditional on generated ECT",
        "long_run_inference": "Point estimates; full two-stage conditional residual bootstrap reported separately",
        "identification": "Conditional macro association; no causal instrument or imposed coefficient sign",
    }
    if compute_diagnostics:
        statistic, pvalue, critical = coint(
            panel.loc[dates, "logC"], panel.loc[dates, list(LONG_COLUMNS)],
            trend="c", maxlag=spec["coint_maxlag"], autolag=spec["coint_autolag"])
        bg = acorr_breusch_godfrey(result, nlags=min(4, len(dates) // 10))
        diagnostics.update({
            "engle_granger": {"statistic": float(statistic), "pvalue": float(pvalue),
                "critical_values": dict(zip(("1pct", "5pct", "10pct"), map(float, critical))),
                "reject_no_cointegration_5pct": bool(pvalue < .05 and np.isfinite(statistic)),
                "method": "statsmodels augmented Engle-Granger; MacKinnon cointegration critical values, not ordinary residual ADF p-values",
                "maintained_assumption": "The four log-level series are I(1); test interpretation also requires appropriate deterministic terms"},
            "breusch_godfrey_4lag_pvalue": float(bg[1]),
            "unit_root_diagnostics": {name: {
                "level_adf_pvalue": float(adfuller(panel.loc[dates, name], maxlag=4, autolag="AIC")[1]),
                "difference_adf_pvalue": float(adfuller(panel.loc[dates, name].diff().dropna(), maxlag=4, autolag="AIC")[1])}
                for name in ("logC", *LONG_COLUMNS)},
        })
        if not np.isfinite(statistic):
            diagnostics["engle_granger"]["warning"] = "Numerically collinear levels; cointegration test is not reliable"
        if pvalue >= .05:
            diagnostics["cointegration_warning"] = "No rejection of no cointegration; error-correction interpretation remains unvalidated"
    return ECMFit(long_params, short_params, covariance, Xlong.reindex(dates), Xshort,
                  y, errors, residual, dates, spec, diagnostics, int(result.df_resid))


def coefficient_table(fit):
    """Keep long-run SEs missing until a whole-two-stage bootstrap is supplied."""
    critical = t.ppf(.975, fit.df_resid)
    rows = []
    for name, value in fit.long_run_params.items():
        rows.append({"stage": "long_run", "regressor": name, "estimate": value,
                     "standard_error": np.nan, "ci_low": np.nan, "ci_high": np.nan,
                     "inference": "Whole-two-stage uncertainty supplied by separate bootstrap"})
    for name, value in fit.short_run_params.items():
        se = np.sqrt(max(0., fit.short_run_covariance.loc[name, name]))
        rows.append({"stage": "short_run", "regressor": name, "estimate": value,
                     "standard_error": se, "ci_low": value - critical * se,
                     "ci_high": value + critical * se,
                     "inference": "95% pointwise HAC t interval, conditional on estimated ECT"})
    return pd.DataFrame(rows)


def conditional_predictions(fit, panel, start="2020Q1", end="2025Q4"):
    """Observed-predictor revised-data hindcast, not a real-time forecast."""
    _check_panel(panel)
    dates = pd.period_range(start, end, freq="Q")
    residual = panel.logC - long_design(panel) @ fit.long_run_params
    X = short_design(panel, residual, fit.spec).reindex(dates)
    if not np.isfinite(X.to_numpy()).all():
        raise ValueError("Missing conditional hindcast input")
    prediction = X @ fit.short_run_params
    observed = panel.dlogC.reindex(dates)
    return pd.DataFrame({"quarter": dates.astype(str), "observed_dlogC": observed.to_numpy(),
        "predicted_dlogC": prediction.to_numpy(),
        "observed_annualized_growth_pct": 100. * np.expm1(4. * observed.to_numpy()),
        "predicted_annualized_growth_pct": 100. * np.expm1(4. * prediction.to_numpy()),
        "information_status": "Observed current predictors and lagged C/ECT; revised-data conditional hindcast"})


def annualization_factor(log_growth):
    """Common exact converter ensures contributions sum to annualized growth."""
    values = np.asarray(log_growth, dtype=float)
    result = np.full(values.shape, 400., dtype=float)
    mask = np.abs(values) > 1e-12
    result[mask] = 100. * np.expm1(4. * values[mask]) / values[mask]
    return result


def decompose_ecm(fit, panel, start="1990Q1", end="2025Q4"):
    """Exactly reconcile anchored log-growth and log-level consumption paths.

    All channels receive the consumption-level error-correction and growth-AR
    feedback. Long-run wealth/income enter as changes from the base log levels;
    unanchored lambda*beta*logW contributions would depend on currency units.
    Initial ECT, pre-start short-run drivers and initial growth states are
    reported in initial_state. These are not post-base wealth/income effects.
    """
    _check_panel(panel)
    dates = pd.period_range(start, end, freq="Q")
    base = dates[0] - 1
    if base not in panel.index or not dates.isin(panel.index).all():
        raise ValueError("Decomposition needs the observed base and all requested quarters")
    residual = panel.logC - long_design(panel) @ fit.long_run_params
    X = short_design(panel, residual, fit.spec).reindex(dates)
    if not np.isfinite(X.to_numpy()).all():
        raise ValueError("Missing decomposition presample input")
    channels = ["income", "wealth", "credit", "unemployment", "deterministic", "initial_state", "unexplained"]
    growth = pd.DataFrame(0., index=dates, columns=channels)
    levels = pd.DataFrame(0., index=dates, columns=channels)
    lam = fit.short_run_params["ECT_L1"]
    mapping = {"dlogY_ex_transfers": "income", "dlogTransfers": "income",
               "dlogCredit": "credit", "dlogSP500": "wealth", "dUnemployment": "unemployment"}
    for date in dates:
        innovation = pd.Series(0., index=channels)
        innovation["income"] = -lam * fit.long_run_params["logY"] * (panel.loc[date - 1, "logY"] - panel.loc[base, "logY"])
        innovation["wealth"] = -lam * sum(fit.long_run_params[name] * (panel.loc[date - 1, name] - panel.loc[base, name])
                                           for name in ("logW80", "logW20"))
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
    factor = pd.Series(annualization_factor(panel.dlogC.reindex(dates)), index=dates)
    annualized = growth.mul(factor, axis=0)
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
                         "base_quarter": str(base), "status": "conditional anchored ECM decomposition, not causal attribution"})
    return pd.DataFrame(rows)


def annual_decomposition(decomposition):
    """Chart-2 annual means, with Others merged before absolute-value ratios."""
    frame = decomposition.copy()
    dates = pd.PeriodIndex(frame.quarter, freq="Q")
    frame["year"] = dates.year
    frame["channel"] = frame.channel.replace({"deterministic": "others", "initial_state": "others", "unexplained": "others"})
    quarterly = frame.groupby(["quarter", "year", "channel"], as_index=False).agg(
        annualized_consumption_growth_pp=("annualized_consumption_growth_pp", "sum"),
        observed_annualized_growth_pct=("observed_annualized_growth_pct", "first"))
    result = quarterly.groupby(["year", "channel"], as_index=False).agg(
        contribution_pp=("annualized_consumption_growth_pp", "mean"),
        observed_growth_pct=("observed_annualized_growth_pct", "mean"),
        quarters=("quarter", "nunique"))
    result["measure"] = "Mean quarterly annualized consumption growth/contribution; not calendar-year average growth"
    if (result.quarters != 4).any():
        raise ValueError("Chart-2 annual means require four observed quarters; no partial year relabeling")
    return result


def chart3_indicator(annual):
    """Explicit interpretation of incompletely disclosed absolute ratio.

    Ratio of absolute annual wealth to sum of absolute annual category means,
    not a GDP share. Also export the net-growth-denominator alternative.
    """
    wide = annual.pivot(index="year", columns="channel", values="contribution_pp")
    wealth = wide.wealth.abs()
    gross = wide.abs().sum(axis=1)
    net = wide.sum(axis=1).abs()
    return pd.DataFrame({"year": wide.index,
        "wealth_absolute_contribution_share_pct": (100. * wealth / gross.where(gross > 1e-12)).to_numpy(),
        "wealth_absolute_net_growth_ratio_pct": (100. * wealth / net.where(net > 1e-12)).to_numpy(),
        "wealth_contribution_pp": wide.wealth.to_numpy(),
        "absolute_contribution_denominator_pp": gross.to_numpy(),
        "interpretation": "Declared absolute annual-category ratio; authors' precise aggregation convention unavailable"})


def _circular_blocks(residuals, length, rng):
    n = len(residuals)
    starts = rng.integers(0, n, size=int(np.ceil(n / length)))
    locations = np.concatenate([(start + np.arange(length)) % n for start in starts])[:n]
    return np.asarray(residuals)[locations]


def _simulate_consumption(fit, panel, errors, null=False):
    """Conditional recursive simulation; observed other inputs remain fixed."""
    out = panel.copy()
    if null:
        design = fit.short_X.drop(columns="ECT_L1")
        params, _, _, _ = _ols(fit.y, design, fit.spec["hac_lags"])
        lam = 0.
    else:
        params = fit.short_run_params
        lam = params["ECT_L1"]
    long_prediction = long_design(panel) @ fit.long_run_params
    for j, date in enumerate(fit.index):
        value = params["const"] + errors[j]
        if not null:
            value += lam * (out.loc[date - 1, "logC"] - long_prediction.loc[date - 1])
        for lag in range(1, fit.spec["consumption_lags"] + 1):
            value += params[f"consumption_L{lag}"] * out.loc[date - lag, "dlogC"]
        for name in SHORT_COLUMNS:
            for lag in range(fit.spec["short_lags"] + 1):
                value += params[f"{name}_L{lag}"] * panel.loc[date - lag, name]
        out.loc[date, "dlogC"] = value
        out.loc[date, "logC"] = out.loc[date - 1, "logC"] + value
    # Keep observed post-training levels; repair boundary growth for validation.
    out.loc[out.index[1:], "dlogC"] = out.logC.diff().iloc[1:]
    if "C_real_quarter_bn" in out:
        out["C_real_quarter_bn"] = np.exp(out.logC)
    return out


def bootstrap_ecm(fit, panel, decomposition_start="1990Q1", decomposition_end="2025Q4",
                  draws=399, block_length=4, seed=454):
    """Conditional circular-block residual bootstrap, refitting BOTH stages.

    Simulates consumption through the fitted ECM while holding observed
    income/wealth/credit/market/unemployment paths fixed. Annual decompositions
    are then evaluated on the original observed panel. These percentile bands
    include generated-regressor estimation, but not causal identification or
    unconditional uncertainty in the other input processes. They rely on the
    maintained conditional ECM and suitable residual dependence assumptions.
    """
    if not isinstance(draws, int) or draws < 2 or not isinstance(block_length, int) or not 1 <= block_length <= len(fit.index):
        raise ValueError("Bootstrap needs at least two draws and a valid positive block length")
    rng = np.random.default_rng(seed)
    residuals = fit.residuals.to_numpy() - fit.residuals.mean()
    parameter_rows, annual_rows, failures, instability, nonnegative_loading = [], [], [], 0, 0
    for draw in range(draws):
        try:
            simulated = _simulate_consumption(fit, panel, _circular_blocks(residuals, block_length, rng))
            replicate = fit_ecm(simulated, fit.spec, compute_diagnostics=False)
            annual = annual_decomposition(decompose_ecm(replicate, panel, decomposition_start, decomposition_end))
            for stage, params in (("long_run", replicate.long_run_params), ("short_run", replicate.short_run_params)):
                for name, value in params.items():
                    parameter_rows.append({"draw": draw, "stage": stage, "regressor": name, "estimate": float(value)})
            annual_rows.append(annual.assign(draw=draw))
            instability += not replicate.diagnostics["stable_level_recursion"]
            nonnegative_loading += replicate.short_run_params["ECT_L1"] >= 0.
        except (ValueError, AssertionError, np.linalg.LinAlgError, FloatingPointError) as error:
            failures.append({"draw": draw, "error": str(error)})
    completed = len(annual_rows)
    if completed < max(2, int(np.ceil(.8 * draws))):
        raise RuntimeError(f"Only {completed}/{draws} bootstrap draws completed; failures must be inspected")
    samples = pd.concat(annual_rows, ignore_index=True)
    intervals = samples.groupby(["year", "channel"]).contribution_pp.quantile([.025, .5, .975]).unstack().reset_index()
    intervals.columns = ["year", "channel", "ci_low", "bootstrap_median", "ci_high"]
    original = annual_decomposition(decompose_ecm(fit, panel, decomposition_start, decomposition_end))
    intervals = original.merge(intervals, on=["year", "channel"], validate="one_to_one")
    intervals["inference"] = "95% pointwise conditional circular-block residual percentile interval; both stages refitted"
    return {"diagnostics": {"requested_draws": draws, "completed_draws": completed,
             "block_length_quarters": block_length, "seed": seed, "failures": failures,
             "unstable_refit_draws_retained": int(instability), "nonnegative_loading_draws_retained": int(nonnegative_loading),
             "method": "Recursive conditional ECM residual bootstrap; both long and short stages refitted",
             "limitations": ["Other observed input paths held fixed; no unconditional joint-process bootstrap",
                             "Percentile intervals are pointwise, not simultaneous; identification and specification uncertainty excluded",
                             "Validity depends on the conditional ECM, cointegration and residual weak-dependence assumptions"]},
            "parameter_draws": pd.DataFrame(parameter_rows), "annual_draws": samples,
            "annual_intervals": intervals}


def bootstrap_cointegration_null(fit, panel, draws=199, block_length=4, seed=1454):
    """Supplementary model-dependent no-ECT bootstrap; not a universal EG law.

    The formal primary test is statsmodels.coint with MacKinnon critical values.
    This robustness experiment conditions on observed predictor paths and
    simulates consumption with zero adjustment, creating an additional trend
    through resampled consumption-growth innovations. Critical values depend
    on this maintained conditional null model and block design.
    """
    if draws < 2 or block_length < 1 or block_length > len(fit.index):
        raise ValueError("Invalid null bootstrap settings")
    design = fit.short_X.drop(columns="ECT_L1")
    params, _, _, _ = _ols(fit.y, design, fit.spec["hac_lags"])
    residuals = (fit.y - design @ params).to_numpy()
    residuals -= residuals.mean()
    rng, statistics, failures = np.random.default_rng(seed), [], []
    for draw in range(draws):
        try:
            simulated = _simulate_consumption(fit, panel, _circular_blocks(residuals, block_length, rng), null=True)
            statistic = coint(simulated.loc[fit.index, "logC"], simulated.loc[fit.index, list(LONG_COLUMNS)],
                              trend="c", maxlag=fit.spec["coint_maxlag"], autolag=fit.spec["coint_autolag"])[0]
            if not np.isfinite(statistic):
                raise ValueError("Nonfinite bootstrap cointegration statistic")
            statistics.append(float(statistic))
        except (ValueError, np.linalg.LinAlgError, FloatingPointError) as error:
            failures.append({"draw": draw, "error": str(error)})
    if len(statistics) < max(2, int(np.ceil(.8 * draws))):
        raise RuntimeError("Insufficient completed conditional null bootstrap draws")
    observed = coint(panel.loc[fit.index, "logC"], panel.loc[fit.index, list(LONG_COLUMNS)],
                     trend="c", maxlag=fit.spec["coint_maxlag"], autolag=fit.spec["coint_autolag"])[0]
    pvalue = (1. + np.sum(np.asarray(statistics) <= observed)) / (1. + len(statistics))
    return {"observed_statistic": float(observed), "conditional_bootstrap_pvalue": float(pvalue),
            "critical_values": dict(zip(("1pct", "5pct", "10pct"), map(float, np.quantile(statistics, [.01, .05, .1])))),
            "completed_draws": len(statistics), "requested_draws": draws, "block_length_quarters": block_length,
            "seed": seed, "failures": failures,
            "status": "Supplementary model-dependent conditional no-ECT bootstrap, not an exact author's test"}
