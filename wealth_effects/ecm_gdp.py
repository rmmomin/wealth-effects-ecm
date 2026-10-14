"""Portable GDP accounting and saved-draw recovery for the ECM showcase.

No source-repository readers or network calls are used. BEA's published PCE
contribution is allocated by the ECM log-growth shares. This is a conditional
gross consumption allocation, not a causal general-equilibrium GDP estimate.
"""
import numpy as np
import pandas as pd
from scipy.special import exprel

from .note454_gdp_draws import _validated_draws


def _quarter_index(index):
    if not isinstance(index, pd.PeriodIndex) or index.freqstr != "Q-DEC":
        raise ValueError("Quarterly Q-DEC PeriodIndex required")
    if len(index) == 0 or not index.equals(pd.period_range(index.min(), index.max(), freq="Q")):
        raise ValueError("Missing, duplicate, unordered or empty quarter; no interpolation")


def consumption_change_factor(panel):
    """Actual-quarter consumption dollars per unit of allocated log growth."""
    _quarter_index(panel.index)
    C, growth = panel.C_real_quarter_bn, panel.dlogC
    if not np.isfinite(C.to_numpy()).all() or (C <= 0).any():
        raise ValueError("Positive finite consumption levels required")
    if not np.isfinite(growth.iloc[1:].to_numpy()).all():
        raise ValueError("Missing consumption growth after initial row")
    if not np.allclose(np.log(C).diff().iloc[1:], growth.iloc[1:], atol=1e-10, rtol=1e-8):
        raise ValueError("Consumption growth does not match observed levels")
    return C.shift(1) * exprel(growth)


def converters(panel, dates, accounting):
    """Published BEA allocation and an independently price-weighted cross-check."""
    _quarter_index(dates)
    _quarter_index(accounting.index)
    padded = pd.period_range(dates[0] - 1, dates[-1], freq="Q")
    required = ["nominal_gdp_saar_bn", "real_gdp_saar_bn", "published_pce_contribution_pp"]
    if not set(required).issubset(accounting):
        raise ValueError("Missing GDP accounting fields")
    if not padded.isin(panel.index).all() or not padded.isin(accounting.index).all():
        raise ValueError("Missing GDP bridge predecessor or observed quarter")
    nominal = accounting.nominal_gdp_saar_bn.reindex(padded)
    real = accounting.real_gdp_saar_bn.reindex(padded)
    price = panel.pce_price.reindex(padded)
    if not np.isfinite(pd.concat([nominal, real, price], axis=1).to_numpy()).all():
        raise ValueError("Nonfinite GDP/price observation; no filling")
    if (nominal <= 0).any() or (real <= 0).any() or (price <= 0).any():
        raise ValueError("Positive GDP and price levels required")
    link = real / real.shift(1)
    price_link = (nominal / real) / (nominal / real).shift(1)
    annualizer = 1 + link + link ** 2 + link ** 3
    # C is an actual quarterly flow, so convert its change to SAAR once.
    weight = 400 * annualizer * (price / price_link + price.shift(1)) / (2 * nominal.shift(1))
    factor = consumption_change_factor(panel).reindex(dates)
    growth = panel.dlogC.reindex(dates)
    official = accounting.published_pce_contribution_pp.reindex(dates)
    if not np.isfinite(official.to_numpy()).all():
        raise ValueError("Missing published PCE contribution; no substitution")
    if (growth.abs() <= 1e-10).any():
        raise ValueError("Published proportional allocation undefined near zero consumption growth")
    result = pd.DataFrame({
        "consumption_change_factor_bn": factor,
        "gdp_pp_per_consumption_change_quarter_bn": weight.reindex(dates),
        "primary_gdp_pp_per_log_growth": official / growth,
        "approximate_gdp_pp_per_log_growth": weight.reindex(dates) * factor,
        "published_pce_gdp_contribution_pp": official,
        "observed_gdp_growth_saar_pct": (100 * np.expm1(4 * np.log(link))).reindex(dates),
    }, index=dates)
    if not np.isfinite(result.to_numpy()).all():
        raise ValueError("Nonfinite GDP conversion weight")
    return result


def bridge_channels(panel, decomposition, accounting):
    """Reconcile all estimated consumption channels to the published PCE term."""
    required = {"quarter", "channel", "log_growth_contribution"}
    if not required.issubset(decomposition) or decomposition.empty:
        raise ValueError("Missing or empty channel decomposition")
    frame = decomposition.copy()
    if frame.duplicated(["quarter", "channel"]).any():
        raise ValueError("Duplicate quarter/channel")
    frame["quarter"] = pd.PeriodIndex(frame.quarter, freq="Q")
    dates = pd.PeriodIndex(sorted(frame.quarter.unique()), freq="Q")
    sets = frame.groupby("quarter").channel.agg(lambda x: tuple(sorted(x)))
    if sets.nunique() != 1:
        raise ValueError("Missing quarterly channel")
    if not np.isfinite(frame.log_growth_contribution.to_numpy()).all():
        raise ValueError("Nonfinite channel contribution")
    total = frame.groupby("quarter").log_growth_contribution.sum().reindex(dates)
    if not np.allclose(total, panel.dlogC.reindex(dates), atol=1e-10, rtol=1e-9):
        raise ValueError("Channels do not reconcile observed consumption growth")
    conversion = converters(panel, dates, accounting)
    for name in conversion:
        frame[name] = frame.quarter.map(conversion[name])
    frame["real_consumption_change_quarter_bn"] = frame.log_growth_contribution * frame.consumption_change_factor_bn
    frame["gross_gdp_growth_pp"] = frame.log_growth_contribution * frame.primary_gdp_pp_per_log_growth
    frame["approximate_gdp_growth_pp"] = frame.log_growth_contribution * frame.approximate_gdp_pp_per_log_growth
    frame["channel"] = frame.channel.replace({"deterministic": "others", "initial_state": "others", "unexplained": "others"})
    values = ["log_growth_contribution", "real_consumption_change_quarter_bn", "gross_gdp_growth_pp", "approximate_gdp_growth_pp"]
    result = frame.groupby(["quarter", "channel"], as_index=False).agg({
        **{name: "sum" for name in values}, **{name: "first" for name in conversion}})
    np.testing.assert_allclose(result.groupby("quarter").gross_gdp_growth_pp.sum().reindex(dates),
                               conversion.published_pce_gdp_contribution_pp, atol=1e-8, rtol=1e-9)
    result["quarter"] = result.quarter.astype(str)
    result["accounting"] = "Published BEA PCE contribution allocated by ECM log-growth shares"
    result["status"] = "Conditional gross consumption allocation; not a causal GDP estimate"
    return result


def reconstruct_wealth_draws(fit, panel, parameters, model, start, end, expected_draws):
    """Vectorized version of the established anchored wealth-only recursion.

    Every supplied finite parameter vector is retained, including unstable
    refits. The validation is shared with the full-channel draw implementation.
    Ratio targets allocate their nonlinear interaction by the same two-order
    Shapley rule; all corners must be positive, without clipping.
    """
    frame, identifiers = _validated_draws(parameters, fit, expected_draws)
    dates = pd.period_range(start, end, freq="Q")
    base = dates[0] - 1
    if base not in panel.index or not dates.isin(panel.index).all():
        raise ValueError("Missing decomposition base or quarter")
    vectors = frame.pivot(index="draw", columns=["stage", "regressor"], values="estimate").reindex(identifiers)
    lam = vectors[("short_run", "ECT_L1")].to_numpy()[:, None]
    if model == "log_ecm":
        change = np.zeros((len(identifiers), len(dates)))
        for name in ("logW80", "logW20"):
            beta = vectors[("long_run", name)].to_numpy()[:, None]
            change += beta * (panel[name].reindex(dates - 1).to_numpy()[None, :] - panel.loc[base, name])
    elif model == "ratio_ecm":
        alpha = vectors[("long_run", "const")].to_numpy()[:, None]
        b80 = vectors[("long_run", "wealth_bottom80")].to_numpy()[:, None]
        b20 = vectors[("long_run", "wealth_top20")].to_numpy()[:, None]
        def target(income, transfers, w80, w20):
            annual = 4 * transfers + alpha * (4 * income - 4 * transfers) + b80 * w80 + b20 * w20
            if not np.isfinite(annual).all() or (annual <= 0).any():
                failed = identifiers[np.any(~np.isfinite(annual) | (annual <= 0), axis=1)]
                raise ValueError(f"Nonpositive or nonfinite ratio target in saved draws {failed.tolist()}")
            return np.log(annual / 4)
        b = panel.loc[base]
        current = panel.reindex(dates - 1)
        Y, T = current.Y_real_quarter_bn.to_numpy()[None, :], current.Transfers_real_quarter_bn.to_numpy()[None, :]
        w80, w20 = current.W80_real_stock_bn.to_numpy()[None, :], current.W20_real_stock_bn.to_numpy()[None, :]
        # Existing full decomposition also validates target levels on the whole panel.
        target(panel.Y_real_quarter_bn.to_numpy()[None, :], panel.Transfers_real_quarter_bn.to_numpy()[None, :],
               panel.W80_real_stock_bn.to_numpy()[None, :], panel.W20_real_stock_bn.to_numpy()[None, :])
        f00 = target(b.Y_real_quarter_bn, b.Transfers_real_quarter_bn, b.W80_real_stock_bn, b.W20_real_stock_bn)
        f10 = target(Y, T, b.W80_real_stock_bn, b.W20_real_stock_bn)
        f01 = target(b.Y_real_quarter_bn, b.Transfers_real_quarter_bn, w80, w20)
        f11 = target(Y, T, w80, w20)
        change = .5 * ((f01 - f00) + (f11 - f10))
    else:
        raise ValueError("Model must be log_ecm or ratio_ecm")
    innovations = -lam * change
    for lag in range(fit.spec["short_lags"] + 1):
        values = panel.dlogSP500.reindex(dates - lag).to_numpy()
        if not np.isfinite(values).all():
            raise ValueError("Missing S&P presample driver")
        values = np.where(dates - lag >= dates[0], values, 0.)
        innovations += vectors[("short_run", f"dlogSP500_L{lag}")].to_numpy()[:, None] * values[None, :]
    growth = np.zeros_like(innovations)
    level = np.zeros(len(identifiers))
    for j in range(len(dates)):
        growth[:, j] = innovations[:, j] + lam[:, 0] * level
        for lag in range(1, fit.spec["consumption_lags"] + 1):
            if j >= lag:
                growth[:, j] += vectors[("short_run", f"consumption_L{lag}")].to_numpy() * growth[:, j - lag]
        level += growth[:, j]
    if not np.isfinite(growth).all():
        raise ValueError("Nonfinite saved-draw wealth path; no draw filtering")
    return pd.DataFrame({"draw": np.repeat(identifiers, len(dates)),
        "quarter": np.tile(dates.astype(str).to_numpy(), len(identifiers)),
        "wealth_log_growth_contribution": growth.ravel()})


def temporal_summary(table, real_gdp, value_column="gross_gdp_growth_pp", group_columns=("model",)):
    """Aggregate each joint path before inference; label annual and H1 measures."""
    groups = list(group_columns)
    iterator = table.groupby(groups, sort=True) if groups else [((), table)]
    rows = []
    for key, group in iterator:
        key = key if isinstance(key, tuple) else (key,)
        group = group.copy()
        group.index = pd.PeriodIndex(group.quarter, freq="Q")
        group = group.sort_index()
        dates = group.index
        _quarter_index(dates)
        if not np.isfinite(group[value_column].to_numpy()).all():
            raise ValueError("Nonfinite GDP allocation")
        padded = pd.period_range(dates[0] - 1, dates[-1], freq="Q")
        real = real_gdp.reindex(padded)
        if not np.isfinite(real.to_numpy()).all() or (real <= 0).any():
            raise ValueError("Missing or nonpositive real GDP")
        link = real / real.shift(1)
        annualizer = 1 + link + link ** 2 + link ** 3
        increment = (real.shift(1) / (100 * annualizer)).reindex(dates)
        allocation = (group[value_column] * increment).cumsum()
        actual = (100 * np.expm1(4 * np.log(link))).reindex(dates)
        for year in sorted(set(dates.year)):
            current, previous = dates[dates.year == year], dates[dates.year == year - 1]
            if len(current) != 4 and list(current.quarter) != [1, 2]:
                raise ValueError("Only complete years or explicitly labeled H1 supported")
            row = {**dict(zip(groups, key)), "period": str(year) if len(current) == 4 else f"{year} H1",
                "quarters": len(current),
                "mean_gross_quarterly_saar_equivalent_pp": float(group.loc[current, value_column].mean()),
                "mean_observed_quarterly_saar_growth_pct": float(actual.loc[current].mean()),
                "calendar_gross_equivalent_pp": np.nan, "calendar_observed_growth_pct": np.nan,
                "mean_measure": "Mean quarterly annualized rates; not calendar-year growth",
                "calendar_measure": "Unavailable: full previous-year allocation absent"}
            if len(previous) == 4:
                previous = previous[:len(current)]
                denominator = real.reindex(previous).mean()
                row["calendar_gross_equivalent_pp"] = float(100 * (allocation.loc[current].mean() - allocation.loc[previous].mean()) / denominator)
                row["calendar_observed_growth_pct"] = float(100 * (real.reindex(current).mean() / denominator - 1))
                row["calendar_measure"] = "Calendar-year average growth" if len(current) == 4 else "H1 over previous-year H1 growth"
            rows.append(row)
    return pd.DataFrame(rows)
