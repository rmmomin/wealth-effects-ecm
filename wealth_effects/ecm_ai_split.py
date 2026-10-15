"""Frozen-coefficient subdivisions of the published ECM wealth channel.

The S&P price control is subdivided using chain-additive market-cap changes,
with its price/capitalization basis retained separately.  The optional
ownership bridge allocates postbaseline cap gains to household income groups.
Neither variant estimates a new equation or changes an original channel total.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.special import exprel

from .ecm_gdp import reconstruct_wealth_draws
from .note454_ecm import (
    DEFAULT_SPEC, SHORT_COLUMNS, annualization_factor, decompose_ecm,
)
from .note454_gdp_draws import _validated_draws
from .note454_ratio_ecm import decompose_ratio_ecm


VARIANTS = ("price_control_only", "ownership_bridge")
COMPONENTS = ("ai_sp", "non_ai_sp", "index_cap_basis", "unallocated_wealth")
MODELS = ("log_ecm", "ratio_ecm")
_GROUPS = (("bottom80", "W80_real_stock_bn", "logW80", "wealth_bottom80"),
           ("top20", "W20_real_stock_bn", "logW20", "wealth_top20"))
_INPUT_COLUMNS = (
    "spx_ai_market_cap_bn", "spx_non_ai_market_cap_bn", "spx_total_market_cap_bn",
    "household_equity_share", "equity_bottom80_share", "equity_top20_share",
)


@dataclass
class FrozenFit:
    """Only the saved coefficients and settings needed by existing accounting."""

    long_run_params: pd.Series
    short_run_params: pd.Series
    spec: dict
    model: str = ""


def load_frozen_fits(coefficients, spec):
    """Restore the two published point coefficient vectors without fitting.

    ``coefficients`` is the published results/coefficients.csv table and
    ``spec`` is the configuration's model object. DOLS difference controls are
    retained in the saved ratio vector, although they do not enter its target.
    """
    required = {"model", "stage", "regressor", "estimate"}
    if not isinstance(coefficients, pd.DataFrame) or not required.issubset(coefficients):
        raise ValueError("Frozen coefficients require model, stage, regressor and estimate")
    frame = coefficients[list(required)].copy()
    if frame.empty or frame.isna().any().any():
        raise ValueError("Missing frozen coefficient value or identifier")
    if frame.duplicated(["model", "stage", "regressor"]).any():
        raise ValueError("Duplicate frozen model/stage/regressor")
    if set(frame.model.unique()) != set(MODELS):
        raise ValueError("Frozen coefficients must contain exactly log_ecm and ratio_ecm")
    if set(frame.stage.unique()) != {"long_run", "short_run"}:
        raise ValueError("Frozen coefficients require long_run and short_run stages")
    try:
        frame["estimate"] = pd.to_numeric(frame.estimate, errors="raise")
    except (TypeError, ValueError) as error:
        raise ValueError("Frozen estimates must be numeric") from error
    if not np.isfinite(frame.estimate.to_numpy(dtype=float)).all():
        raise ValueError("Nonfinite frozen coefficient estimate")
    settings = {**DEFAULT_SPEC, **dict(spec)}
    for name in ("short_lags", "consumption_lags"):
        value = settings[name]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(name + " must be a nonnegative integer")
    expected_short = {"const", "ECT_L1"}
    expected_short.update(f"consumption_L{lag}" for lag in range(1, settings["consumption_lags"] + 1))
    expected_short.update(f"{name}_L{lag}" for name in SHORT_COLUMNS
                          for lag in range(settings["short_lags"] + 1))
    fits = {}
    for model in MODELS:
        part = frame.loc[frame.model == model]
        vectors = {stage: group.set_index("regressor").estimate.astype(float)
                   for stage, group in part.groupby("stage", sort=False)}
        if set(vectors.get("short_run", pd.Series(dtype=float)).index) != expected_short:
            raise ValueError(f"Frozen {model} short-run regressors do not match settings")
        expected_long = ({"const", "logY", "logW80", "logW20"} if model == "log_ecm"
                         else {"const", "wealth_bottom80", "wealth_top20"})
        actual_long = set(vectors.get("long_run", pd.Series(dtype=float)).index)
        if not expected_long.issubset(actual_long) or (model == "log_ecm" and actual_long != expected_long):
            raise ValueError(f"Frozen {model} long-run regressors are incomplete or inconsistent")
        fits[model] = FrozenFit(vectors["long_run"], vectors["short_run"], settings.copy(), model)
    return fits


def _quarterly(frame, label):
    if not isinstance(frame, pd.DataFrame):
        raise ValueError(f"{label} must be a DataFrame")
    out = frame.copy()
    if not isinstance(out.index, pd.PeriodIndex) and "quarter" in out:
        out = out.set_index("quarter")
        out.index = pd.PeriodIndex(out.index, freq="Q")
    if not isinstance(out.index, pd.PeriodIndex) or out.index.freqstr != "Q-DEC":
        raise ValueError(f"{label} requires a quarterly Q-DEC PeriodIndex")
    if out.empty or out.index.duplicated().any() or not out.index.is_monotonic_increasing:
        raise ValueError(f"{label} quarters must be nonempty, unique and sorted")
    if not out.index.equals(pd.period_range(out.index[0], out.index[-1], freq="Q")):
        raise ValueError(f"Missing {label} quarter; no interpolation")
    return out


def logarithmic_mean(first, second):
    """Positive endpoint log mean, including equal endpoints and cancellation.

    No net-change denominator is used. ``exprel`` evaluates the continuous
    limit when the two endpoints coincide.
    """
    first, second = np.broadcast_arrays(np.asarray(first, dtype=float), np.asarray(second, dtype=float))
    if (not np.isfinite(first).all() or not np.isfinite(second).all()
            or (first <= 0).any() or (second <= 0).any()):
        raise ValueError("Logarithmic mean requires positive finite endpoints")
    high, low = np.maximum(first, second), np.minimum(first, second)
    relative = (high - low) / high
    # Use relative differences for nearby endpoints and log differences for
    # widely separated endpoints. The latter also avoids an overflowing ratio.
    gap = np.empty_like(relative)
    nearby = relative < .5
    gap[nearby] = -np.log1p(-relative[nearby])
    gap[~nearby] = np.log(high[~nearby]) - np.log(low[~nearby])
    result = high * exprel(-gap)
    if not np.isfinite(result).all() or (result <= 0).any():
        raise ValueError("Nonfinite logarithmic mean")
    return result


def capitalization_bridge(panel, inputs, split_start="2022Q1", end="2026Q2"):
    """Build the quarter-end cap, lagged ownership and gain-allocation ledger.

    Input ownership shares are contemporaneous source observations: each is
    shifted exactly once before multiplying this quarter's cap change. The
    returned ledger includes the split's preceding anchor with zero drivers
    and zero cumulative gains. All monetary values are billions of dollars.
    Real gain stocks equal cumulative *nominal postbaseline gains* divided by
    the current PCE price; opening wealth's inflation remains unallocated.
    """
    panel, inputs = _quarterly(panel, "panel"), _quarterly(inputs, "capitalization input")
    first, last = pd.Period(split_start, freq="Q"), pd.Period(end, freq="Q")
    if first > last:
        raise ValueError("Split start follows end")
    dates = pd.period_range(first - 1, last, freq="Q")
    if not dates.isin(inputs.index).all() or not dates.isin(panel.index).all():
        raise ValueError("Missing capitalization bridge anchor or requested quarter")
    if not set(_INPUT_COLUMNS).issubset(inputs):
        raise ValueError("Missing capitalization/ownership input fields")
    if not {"pce_price", "dlogSP500"}.issubset(panel):
        raise ValueError("Missing PCE price or original S&P return driver")
    out = inputs.loc[dates, list(_INPUT_COLUMNS)].copy()
    out.index.name = "quarter"
    values = out.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite capitalization/ownership input; no filling")
    caps = out[list(_INPUT_COLUMNS[:3])]
    if (caps.iloc[:, :2] < 0).any().any() or (caps.iloc[:, 2] <= 0).any():
        raise ValueError("Component caps must be nonnegative and total cap positive")
    if not np.allclose(out.spx_ai_market_cap_bn + out.spx_non_ai_market_cap_bn,
                       out.spx_total_market_cap_bn, atol=1e-7, rtol=1e-10):
        raise ValueError("AI and non-AI capitalization must sum to total S&P cap")
    shares = out[list(_INPUT_COLUMNS[3:])]
    if ((shares < 0) | (shares > 1)).any().any():
        raise ValueError("Ownership shares must lie between zero and one")
    if not np.allclose(out.equity_bottom80_share + out.equity_top20_share, 1., atol=1e-10, rtol=1e-10):
        raise ValueError("Income-group equity shares must sum to one")
    price = panel.pce_price.reindex(dates).to_numpy(dtype=float)
    observed = panel.dlogSP500.reindex(dates).to_numpy(dtype=float)
    if not np.isfinite(price).all() or (price <= 0).any() or not np.isfinite(observed[1:]).all():
        raise ValueError("Positive PCE prices and finite original S&P returns required")
    out["pce_price"] = price
    out["original_sp_log_growth"] = np.r_[0., observed[1:]]
    for column in _INPUT_COLUMNS[3:]:
        out[column + "_lagged"] = out[column].shift(1)
        out.iloc[0, out.columns.get_loc(column + "_lagged")] = out.iloc[0][column]
    total = out.spx_total_market_cap_bn.to_numpy()
    means = np.r_[total[0], logarithmic_mean(total[1:], total[:-1])]
    out["total_cap_logarithmic_mean_bn"] = means
    for component, cap_column in (("ai_sp", "spx_ai_market_cap_bn"),
                                  ("non_ai_sp", "spx_non_ai_market_cap_bn")):
        delta = out[cap_column].diff().fillna(0.)
        out[component + "_cap_change_bn"] = delta
        out[component + "_cap_log_growth"] = delta / means
        for group, _, _, _ in _GROUPS:
            gain = delta * out.household_equity_share_lagged * out[f"equity_{group}_share_lagged"]
            out[f"{component}_{group}_nominal_gain_bn"] = gain
            out[f"{component}_{group}_nominal_gain_stock_bn"] = gain.cumsum()
            out[f"{component}_{group}_real_gain_stock_bn"] = gain.cumsum() / price
    out["total_cap_log_growth"] = out.ai_sp_cap_log_growth + out.non_ai_sp_cap_log_growth
    out["index_cap_basis_log_growth"] = out.original_sp_log_growth - out.total_cap_log_growth
    np.testing.assert_allclose(out.total_cap_log_growth.iloc[1:], np.diff(np.log(total)), atol=2e-13, rtol=1e-11)
    if not np.isfinite(out.to_numpy(dtype=float)).all():
        raise ValueError("Nonfinite capitalization bridge ledger")
    out.attrs.update({"split_start": str(first), "anchor": str(first - 1),
                      "ownership_timing": "Previous-quarter observed ownership shares",
                      "real_gain_definition": "Cumulative nominal postbaseline gains divided by current PCE price"})
    return out


def _validate_request(fit, panel, model, variant, start, end, split_start):
    if model not in MODELS:
        raise ValueError("Model must be log_ecm or ratio_ecm")
    if variant not in VARIANTS:
        raise ValueError("Variant must be price_control_only or ownership_bridge")
    if getattr(fit, "model", "") not in ("", model):
        raise ValueError("Frozen fit model differs from requested model")
    panel = _quarterly(panel, "panel")
    dates = pd.period_range(start, end, freq="Q")
    split = pd.Period(split_start, freq="Q")
    if len(dates) == 0 or split < dates[0] or split > dates[-1]:
        raise ValueError("Split start must lie within the decomposition dates")
    required = pd.period_range(dates[0] - 1, dates[-1], freq="Q")
    if not required.isin(panel.index).all():
        raise ValueError("Missing original decomposition base or quarter")
    return panel, dates, split


def _target_components(panel, ledger, vectors, model, variant, dates, split):
    """Return per-vector AI/non-AI target changes at each innovation's t-1."""
    count = len(vectors)
    result = np.zeros((count, len(dates), 2))
    if variant == "price_control_only":
        return result
    active = np.flatnonzero(dates - 1 >= split)
    if len(active) == 0:
        return result
    lag_dates = (dates - 1)[active]
    current = panel.reindex(lag_dates)
    wealth_base = panel.loc[split - 1]
    gains = ledger.reindex(lag_dates)
    if model == "log_ecm":
        for group, wealth, regressor, _ in _GROUPS:
            mean = logarithmic_mean(current[wealth].to_numpy(), wealth_base[wealth])
            beta = vectors[("long_run", regressor)].to_numpy()[:, None]
            for k, component in enumerate(COMPONENTS[:2]):
                result[:, active, k] += beta * gains[f"{component}_{group}_real_gain_stock_bn"].to_numpy()[None, :] / mean
        return result
    alpha = vectors[("long_run", "const")].to_numpy()[:, None]
    betas = {group: vectors[("long_run", ratio_regressor)].to_numpy()[:, None]
             for group, _, _, ratio_regressor in _GROUPS}
    original_base = panel.loc[dates[0] - 1]
    numerators = np.zeros((count, len(active), 2))
    for group, _, _, _ in _GROUPS:
        for k, component in enumerate(COMPONENTS[:2]):
            numerators[:, :, k] += betas[group] * gains[f"{component}_{group}_real_gain_stock_bn"].to_numpy()[None, :] / 4.

    def target(income, transfers, at_current_wealth):
        value = transfers + alpha * (income - transfers)
        for group, wealth, _, _ in _GROUPS:
            stock = current[wealth].to_numpy()[None, :] if at_current_wealth else wealth_base[wealth]
            value = value + betas[group] * stock / 4.
        return value

    # The outer income contexts remain exactly the original two-bundle Shapley
    # contexts. The inner allocation follows a straight wealth path, rather
    # than changing that original income/wealth grouping with a flat Shapley.
    for income, transfers in (
        (original_base.Y_real_quarter_bn, original_base.Transfers_real_quarter_bn),
        (current.Y_real_quarter_bn.to_numpy()[None, :], current.Transfers_real_quarter_bn.to_numpy()[None, :]),
    ):
        mean = logarithmic_mean(target(income, transfers, True), target(income, transfers, False))
        result[:, active, :] += .5 * numerators / mean[:, :, None]
    return result


def _component_paths(panel, ledger, vectors, model, variant, dates, split):
    """Vectorized linear dynamics for AI, other S&P and the index/cap basis."""
    count = len(vectors)
    innovations = np.zeros((count, len(dates), 3))
    lam = vectors[("short_run", "ECT_L1")].to_numpy()
    target = _target_components(panel, ledger, vectors, model, variant, dates, split)
    innovations[:, :, :2] = -lam[:, None, None] * target
    short_lags = int(vectors.attrs["short_lags"])
    for lag in range(short_lags + 1):
        lag_dates = dates - lag
        eligible = np.flatnonzero(lag_dates >= split)
        driver = ledger.reindex(lag_dates[eligible])
        if driver.empty:
            continue
        coefficient = vectors[("short_run", f"dlogSP500_L{lag}")].to_numpy()
        for k, name in enumerate(("ai_sp_cap_log_growth", "non_ai_sp_cap_log_growth", "index_cap_basis_log_growth")):
            innovations[:, eligible, k] += coefficient[:, None] * driver[name].to_numpy()[None, :]
    growth = np.zeros_like(innovations)
    levels = np.zeros_like(innovations)
    level_state = np.zeros((count, 3))
    for t in range(len(dates)):
        growth[:, t, :] = innovations[:, t, :] + lam[:, None] * level_state
        for lag in range(1, int(vectors.attrs["consumption_lags"]) + 1):
            if t >= lag:
                coefficient = vectors[("short_run", f"consumption_L{lag}")].to_numpy()
                growth[:, t, :] += coefficient[:, None] * growth[:, t - lag, :]
        level_state += growth[:, t, :]
        levels[:, t, :] = level_state
    if not np.isfinite(growth).all() or not np.isfinite(levels).all():
        raise ValueError("Nonfinite frozen-vector component path; no draw filtering")
    return growth, levels


def _vectors(fit, parameters=None):
    if parameters is None:
        parameters = pd.DataFrame([
            {"draw": 0, "stage": stage, "regressor": name, "estimate": value}
            for stage, coefficients in (("long_run", fit.long_run_params), ("short_run", fit.short_run_params))
            for name, value in coefficients.items()
        ])
    frame, identifiers = _validated_draws(parameters, fit, None)
    vectors = frame.pivot(index="draw", columns=["stage", "regressor"], values="estimate").reindex(identifiers)
    vectors.attrs.update({"short_lags": fit.spec["short_lags"], "consumption_lags": fit.spec["consumption_lags"]})
    return frame, identifiers, vectors


def decompose_ai(fit, panel, inputs, model, split_start="2022Q1", variant="ownership_bridge",
                 start="1990Q1", end="2026Q2"):
    """Replace only wealth with four exactly reconciling frozen-fit channels."""
    panel, dates, split = _validate_request(fit, panel, model, variant, start, end, split_start)
    ledger = capitalization_bridge(panel, inputs, str(split), str(dates[-1]))
    original = (decompose_ecm if model == "log_ecm" else decompose_ratio_ecm)(fit, panel, str(dates[0]), str(dates[-1]))
    wealth = original.loc[original.channel == "wealth"].set_index("quarter").reindex(dates.astype(str))
    wealth.index.name = "quarter"
    _, _, vectors = _vectors(fit)
    growth, levels = _component_paths(panel, ledger, vectors, model, variant, dates, split)
    growth, levels = growth[0], levels[0]
    residual_growth = wealth.log_growth_contribution.to_numpy() - growth.sum(axis=1)
    residual_level = wealth.log_level_contribution.to_numpy() - levels.sum(axis=1)
    growth = np.column_stack([growth, residual_growth])
    levels = np.column_stack([levels, residual_level])
    factor = annualization_factor(panel.dlogC.reindex(dates).to_numpy())
    status = ("Frozen-coefficient cap-based subdivision; " + variant
              + "; original wealth total preserved; conditional, not causal")
    rows = []
    for k, component in enumerate(COMPONENTS):
        part = wealth.reset_index().copy()
        part["channel"] = component
        part["log_growth_contribution"] = growth[:, k]
        part["log_level_contribution"] = levels[:, k]
        part["annualized_consumption_growth_pp"] = growth[:, k] * factor
        part["status"] = status
        rows.append(part)
    result = pd.concat([original.loc[original.channel != "wealth"], *rows], ignore_index=True)
    result = result[original.columns]
    np.testing.assert_allclose(growth.sum(axis=1), wealth.log_growth_contribution, atol=2e-12, rtol=1e-10)
    np.testing.assert_allclose(levels.sum(axis=1), wealth.log_level_contribution, atol=2e-12, rtol=1e-10)
    result.attrs.update({"model": model, "variant": variant, "split_start": str(split),
                         "split_anchor": str(split - 1), "original_base": str(dates[0] - 1),
                         "ratio_inner_allocation": "Straight-line logarithmic-mean path within original outer income contexts"})
    return result


def bootstrap_ai_components(fit, panel, inputs, parameters, model, split_start="2022Q1",
                            variant="ownership_bridge", start="1990Q1", end="2026Q2"):
    """Recover every saved joint-vector component path, without any refitting.

    Columns ai_sp/non_ai_sp/index_cap_basis/unallocated_wealth/original_wealth
    are quarterly log-consumption-growth contributions. The original long
    decomposition anchor is preserved. Every supplied finite vector is kept.
    """
    panel, dates, split = _validate_request(fit, panel, model, variant, start, end, split_start)
    ledger = capitalization_bridge(panel, inputs, str(split), str(dates[-1]))
    frame, identifiers, vectors = _vectors(fit, parameters)
    original = reconstruct_wealth_draws(fit, panel, frame, model, str(dates[0]), str(dates[-1]), len(identifiers))
    original = original.set_index(["draw", "quarter"]).reindex(
        pd.MultiIndex.from_product([identifiers, dates.astype(str)], names=["draw", "quarter"]))
    original_growth = original.wealth_log_growth_contribution.to_numpy().reshape(len(identifiers), len(dates))
    growth, _ = _component_paths(panel, ledger, vectors, model, variant, dates, split)
    result = pd.DataFrame({"draw": np.repeat(identifiers, len(dates)),
                           "quarter": np.tile(dates.astype(str), len(identifiers))})
    for k, component in enumerate(COMPONENTS[:3]):
        result[component] = growth[:, :, k].ravel()
    result["unallocated_wealth"] = (original_growth - growth.sum(axis=2)).ravel()
    result["original_wealth"] = original_growth.ravel()
    np.testing.assert_allclose(result[list(COMPONENTS)].sum(axis=1), result.original_wealth, atol=2e-12, rtol=1e-10)
    result.attrs.update({"model": model, "variant": variant, "split_start": str(split),
                         "draws": len(identifiers), "estimation": "Saved vectors restored; no refitting",
                         "units": "Quarterly log consumption growth contribution"})
    return result
