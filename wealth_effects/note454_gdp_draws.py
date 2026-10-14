"""Recover quarterly wealth paths from the already saved two-stage ECM draws.

This module does not simulate consumption, resample residuals or estimate new
bootstrap draws. A single original-panel fit supplies the maintained model's
coefficient names and decomposition settings; every saved draw replaces both
coefficient vectors before the existing, tested accounting decomposition runs.
Unstable but finite saved parameter draws are retained. The result is a
conditional consumption input to the separate GDP accounting bridge.
"""
from dataclasses import replace

import numpy as np
import pandas as pd

from .note454_ecm import decompose_ecm, fit_ecm
from .note454_ratio_ecm import decompose_ratio_ecm, fit_ratio_ecm


def _validated_draws(parameter_draws, fit, expected_draws):
    required = {"draw", "stage", "regressor", "estimate"}
    if not isinstance(parameter_draws, pd.DataFrame) or not required.issubset(parameter_draws.columns):
        raise ValueError("Parameter draws require draw, stage, regressor and estimate columns")
    if not parameter_draws.columns.is_unique:
        raise ValueError("Duplicate saved parameter column names")
    if parameter_draws.empty:
        raise ValueError("No saved parameter draws supplied")
    frame = parameter_draws[["draw", "stage", "regressor", "estimate"]].copy()
    if frame.isna().any().any():
        raise ValueError("Missing saved parameter value or identifier; no filling")
    if not frame[["stage", "regressor"]].map(lambda value: isinstance(value, str)).to_numpy().all():
        raise ValueError("Saved stage and regressor names must be strings")
    if frame.draw.map(lambda value: isinstance(value, (bool, np.bool_))).any():
        raise ValueError("Draw identifiers must be nonnegative integers")
    try:
        draw_ids = pd.to_numeric(frame.draw, errors="raise").to_numpy(dtype=float)
        estimates = pd.to_numeric(frame.estimate, errors="raise").to_numpy(dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError("Draw identifiers and estimates must be numeric") from error
    if (not np.isfinite(draw_ids).all() or (draw_ids < 0).any()
            or (draw_ids != np.floor(draw_ids)).any()
            or (draw_ids > np.iinfo(np.int64).max).any()):
        raise ValueError("Draw identifiers must be finite nonnegative integers")
    if not np.isfinite(estimates).all():
        raise ValueError("Nonfinite saved parameter estimate; no draw filtering")
    frame["draw"] = draw_ids.astype(np.int64)
    frame["estimate"] = estimates
    if frame.duplicated(["draw", "stage", "regressor"]).any():
        raise ValueError("Duplicate saved draw/stage/regressor")
    identifiers = np.sort(frame.draw.unique())
    if expected_draws is not None:
        if (isinstance(expected_draws, bool) or not isinstance(expected_draws, int)
                or expected_draws < 1):
            raise ValueError("expected_draws must be a positive integer")
        if len(identifiers) != expected_draws:
            raise ValueError(f"Expected {expected_draws} saved draws, received {len(identifiers)}")
    expected = {(stage, name) for stage, params in (
        ("long_run", fit.long_run_params), ("short_run", fit.short_run_params))
        for name in params.index}
    for draw, group in frame.groupby("draw", sort=True):
        actual = set(zip(group.stage, group.regressor))
        if actual != expected:
            raise ValueError(f"Draw {draw} regressor mismatch: missing {sorted(expected - actual)}, "
                             f"unexpected {sorted(actual - expected)}")
    return frame.sort_values(["draw", "stage", "regressor"]), identifiers


def reconstruct_quarterly_draws(panel, model_config, parameter_draws,
                                model="log_ecm", start="1990Q1", end="2025Q4",
                                *, expected_draws=None):
    """Return draw/quarter/wealth-log-growth rows without another bootstrap.

    ``model_config`` is the original fit specification (the configuration's
    ``model`` object), not the whole publication configuration. ``expected_draws``
    should be the saved run's completed draw count when recovering the entire
    run; it can instead explicitly state the size of a selected draw subset.
    Draw identifiers need not be consecutive, since documented failed draws
    or deliberately selected subsets may have gaps. Every supplied identifier
    must contain exactly both coefficient vectors with no absent regressors.

    Estimates and decomposition inputs are validated rather than imputed.
    A failed decomposition raises with the saved draw identifier: no draw is
    silently discarded and no stability-based selection narrows intervals.
    """
    if isinstance(model_config, dict) and "model" in model_config:
        raise ValueError("model_config must be the model specification object, not the whole publication configuration")
    if model == "log_ecm":
        fit = fit_ecm(panel, model_config, compute_diagnostics=False)
        decompose = decompose_ecm
    elif model == "ratio_ecm":
        fit = fit_ratio_ecm(panel, model_config)
        decompose = decompose_ratio_ecm
    else:
        raise ValueError("model must be 'log_ecm' or 'ratio_ecm'")
    first, last = pd.Period(start, freq="Q"), pd.Period(end, freq="Q")
    if first > last:
        raise ValueError("Decomposition end precedes start")
    dates = pd.period_range(first, last, freq="Q")
    frame, identifiers = _validated_draws(parameter_draws, fit, expected_draws)
    rows = []
    for draw in identifiers:
        group = frame.loc[frame.draw == draw].set_index(["stage", "regressor"])["estimate"]
        restored = replace(fit,
            long_run_params=group.xs("long_run").reindex(fit.long_run_params.index),
            short_run_params=group.xs("short_run").reindex(fit.short_run_params.index))
        try:
            decomposition = decompose(restored, panel, start=str(first), end=str(last))
        except (ValueError, AssertionError, FloatingPointError, np.linalg.LinAlgError) as error:
            raise ValueError(f"Saved {model} draw {draw} could not be decomposed: {error}") from error
        wealth = decomposition.loc[decomposition.channel == "wealth", ["quarter", "log_growth_contribution"]].copy()
        if (len(wealth) != len(dates) or wealth.quarter.duplicated().any()
                or not np.array_equal(wealth.quarter.to_numpy(), dates.astype(str).to_numpy())
                or not np.isfinite(wealth.log_growth_contribution.to_numpy(dtype=float)).all()):
            raise ValueError(f"Saved {model} draw {draw} has incomplete or nonfinite quarterly wealth contributions")
        wealth = wealth.rename(columns={"log_growth_contribution": "wealth_log_growth_contribution"})
        wealth.insert(0, "draw", int(draw))
        rows.append(wealth)
    result = pd.concat(rows, ignore_index=True)
    if len(result) != len(identifiers) * len(dates):
        raise AssertionError("Quarterly reconstruction changed the supplied draw count")
    result.attrs.update({
        "model": model, "draws": len(identifiers), "quarters_per_draw": len(dates),
        "estimation": "Both saved coefficient vectors restored; no bootstrap or draw re-estimation",
        "stability_selection": "All supplied finite parameter draws retained, including unstable refits",
        "units": "Quarterly log consumption growth contribution; not annualized and not GDP growth",
    })
    return result
