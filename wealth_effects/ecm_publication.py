"""Reproduce the portable quarterly ECM wealth/GDP showcase from shipped data."""
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd

from .ecm_gdp import bridge_channels, converters, reconstruct_wealth_draws, temporal_summary
from .note454_ecm import DEFAULT_SPEC, bootstrap_ecm, coefficient_table, decompose_ecm, fit_ecm
from .note454_ratio_ecm import bootstrap_ratio_ecm, decompose_ratio_ecm, fit_ratio_ecm

ROOT = Path(__file__).resolve().parents[1]
MODELS = ("ratio_ecm", "log_ecm")
COLORS = {"ratio_ecm": "#ad7416", "log_ecm": "#355f92"}
LABELS = {"ratio_ecm": "Ratio/DOLS ECM", "log_ecm": "Log ECM"}


def _hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def _read_quarterly(path):
    frame = pd.read_csv(path, index_col="quarter")
    frame.index = pd.PeriodIndex(frame.index, freq="Q")
    frame.index.name = "quarter"
    return frame


def _bands(frame, keys, value, prefix):
    result = frame.groupby(keys)[value].quantile([.025, .5, .975]).unstack().reset_index()
    result.columns = [*keys, prefix + "ci_low_pp", prefix + "bootstrap_median_pp", prefix + "ci_high_pp"]
    return result


def build_ecm(output_dir=ROOT / "results", config_path=ROOT / "config/ecm.json", rebuild_bootstrap=False):
    """Fit both equations and produce tables, charts and a relative-path manifest.

    The default restores all saved joint coefficient draws and makes no API
    calls. ``rebuild_bootstrap`` resamples and refits both stages under the
    documented conditional residual design; original shipped inputs remain
    unchanged. Failed rebuild draws raise instead of silently changing count.
    """
    config_path, output = Path(config_path).resolve(), Path(output_dir).resolve()
    config = json.loads(config_path.read_text())
    if config.get("confidence_level", .95) != .95:
        raise ValueError("The publication supports the specified 95% intervals")
    if not rebuild_bootstrap:
        fit_settings = {name: config["model"].get(name, DEFAULT_SPEC[name])
                        for name in ("start", "end", "short_lags", "consumption_lags", "hac_lags")}
        expected_settings = {name: DEFAULT_SPEC[name] for name in fit_settings}
        expected_bootstrap = {"draws": 399, "block_length": 4, "seeds": {"log_ecm": 454, "ratio_ecm": 2454}}
        if (fit_settings != expected_settings or config["model"].get("ratio_leads_lags", 1) != 1
                or config["bootstrap"] != expected_bootstrap):
            raise ValueError("Changed estimation/bootstrap settings require --rebuild-bootstrap; shipped draws belong to the documented original specification")
    paths = {name: ROOT / value for name, value in config["data"].items()}
    if output == ROOT or output.is_relative_to(ROOT / "data"):
        raise ValueError("Outputs must preserve repository files and shipped data")
    panel, accounting = _read_quarterly(paths["panel"]), _read_quarterly(paths["gdp"])
    start, end = config["decomposition_start"], config["data_end"]
    if str(panel.index[-1]) != end:
        raise ValueError("Configured data end differs from shipped observed panel")
    dates = pd.period_range(start, end, freq="Q")
    conversion = converters(panel, dates, accounting)
    real = accounting.real_gdp_saar_bn
    parameters = pd.read_csv(paths["bootstrap"])
    if set(parameters.model.unique()) != set(MODELS):
        raise ValueError("Saved bootstrap must contain exactly both specifications")
    output.mkdir(parents=True, exist_ok=True)
    tables, decompositions, coeffs, samples, diagnostics, rebuilt = [], [], [], [], {}, []
    allchannels = []
    for model in MODELS:
        fit = fit_ratio_ecm(panel, config["model"]) if model == "ratio_ecm" else fit_ecm(panel, config["model"])
        function = decompose_ratio_ecm if model == "ratio_ecm" else decompose_ecm
        decomposition = function(fit, panel, start=start, end=end).assign(model=model)
        channel = bridge_channels(panel, decomposition, accounting).assign(model=model)
        wealth = channel.loc[channel.channel == "wealth"].copy()
        total = channel.groupby("quarter").gross_gdp_growth_pp.sum()
        wealth["arithmetic_remainder_after_wealth_pp"] = wealth.observed_gdp_growth_saar_pct - wealth.gross_gdp_growth_pp
        wealth["non_pce_arithmetic_remainder_pp"] = wealth.observed_gdp_growth_saar_pct - wealth.quarter.map(total)
        wealth["wealth_consumption_growth_pp"] = decomposition.loc[decomposition.channel == "wealth", "annualized_consumption_growth_pp"].to_numpy()
        wealth["observed_consumption_growth_saar_pct"] = decomposition.loc[decomposition.channel == "wealth", "observed_annualized_growth_pct"].to_numpy()
        coefficient = coefficient_table(fit).assign(model=model)
        pdraws = parameters.loc[parameters.model == model].drop(columns="model")
        bootstrap_status = {"method": "Restore saved joint vectors, with both stages originally refitted",
            "requested_draws": config["bootstrap"]["draws"], "block_length_quarters": config["bootstrap"]["block_length"],
            "seed": config["bootstrap"]["seeds"][model], "failures": [],
            "inference_status": "Exploratory conditional percentile intervals; unresolved ECM diagnostics, fixed predictor paths and model form"}
        if rebuild_bootstrap:
            last_full = pd.Period(end, freq="Q")
            if last_full.quarter != 4:
                last_full = pd.Period(f"{last_full.year - 1}Q4", freq="Q")
            method = bootstrap_ratio_ecm if model == "ratio_ecm" else bootstrap_ecm
            boot = method(fit, panel, decomposition_start=start, decomposition_end=str(last_full),
                draws=config["bootstrap"]["draws"], block_length=config["bootstrap"]["block_length"],
                seed=config["bootstrap"]["seeds"][model])
            bootstrap_status = boot["diagnostics"]
            if bootstrap_status["completed_draws"] != config["bootstrap"]["draws"]:
                raise RuntimeError("Bootstrap rebuild contains failures; inspect diagnostics before publishing")
            pdraws = boot["parameter_draws"]
            rebuilt.append(pdraws.assign(model=model))
        draw = reconstruct_wealth_draws(fit, panel, pdraws, model, start, end, config["bootstrap"]["draws"])
        draw["model"] = model
        draw["gross_gdp_growth_pp"] = draw.wealth_log_growth_contribution * draw.quarter.map(
            pd.Series(conversion.primary_gdp_pp_per_log_growth.to_numpy(), index=dates.astype(str)))
        band = _bands(draw, ["model", "quarter"], "gross_gdp_growth_pp", "gross_")
        wealth = wealth.merge(band, on=["model", "quarter"], validate="one_to_one")
        param_bands = pdraws.groupby(["stage", "regressor"]).estimate.quantile([.025, .5, .975]).unstack().reset_index()
        param_bands.columns = ["stage", "regressor", "bootstrap_ci_low", "bootstrap_median", "bootstrap_ci_high"]
        coefficient = coefficient.merge(param_bands, on=["stage", "regressor"], validate="one_to_one")
        coefficient["units"] = np.where((coefficient.stage == "long_run") & coefficient.regressor.isin(["wealth_bottom80", "wealth_top20"]),
            "Annual consumption dollars per wealth-stock dollar", "Log/ratio equation coefficient; see methodology")
        diagnostics[model] = {**fit.diagnostics, "bootstrap": bootstrap_status,
            "saved_draws_recovered": int(pdraws.draw.nunique()),
            "nonnegative_adjustment_draws_retained": int(pdraws.query("stage == 'short_run' and regressor == 'ECT_L1'").estimate.ge(0).sum()),
            "maximum_consumption_reconciliation_error": float(abs(decomposition.groupby("quarter").log_growth_contribution.sum().to_numpy() - panel.dlogC.reindex(dates).to_numpy()).max()),
            "maximum_pce_gdp_reconciliation_error_pp": float(abs(total.reindex(dates.astype(str)).to_numpy() - conversion.published_pce_gdp_contribution_pp.to_numpy()).max())}
        tables.append(wealth); coeffs.append(coefficient); allchannels.append(channel)
        decompositions.append(decomposition); samples.append(draw)
    quarterly = pd.concat(tables, ignore_index=True)
    channels = pd.concat(allchannels, ignore_index=True)
    draws = pd.concat(samples, ignore_index=True)
    periods = temporal_summary(quarterly, real)
    period_draws = temporal_summary(draws, real, group_columns=("model", "draw"))
    for column, prefix in (("mean_gross_quarterly_saar_equivalent_pp", "mean_"), ("calendar_gross_equivalent_pp", "calendar_")):
        periods = periods.merge(_bands(period_draws, ["model", "period"], column, prefix),
            on=["model", "period"], validate="one_to_one")
    consumption = pd.concat(decompositions, ignore_index=True)
    annual = consumption.copy()
    annual["year"] = pd.PeriodIndex(annual.quarter, freq="Q").year
    annual["channel"] = annual.channel.replace({"deterministic": "others", "initial_state": "others", "unexplained": "others"})
    annual = annual.groupby(["model", "year", "quarter", "channel"], as_index=False).annualized_consumption_growth_pp.sum()
    annual = annual.groupby(["model", "year", "channel"], as_index=False).agg(
        contribution_pp=("annualized_consumption_growth_pp", "mean"), quarters=("quarter", "nunique"))
    annual["period"] = np.where(annual.quarters == 4, annual.year.astype(str), annual.year.astype(str) + " H1")
    exports = {"coefficients.csv": pd.concat(coeffs, ignore_index=True), "quarterly_gdp_growth.csv": quarterly,
        "channel_gdp_growth.csv": channels, "period_summary.csv": periods,
        "quarterly_consumption_contributions.csv": consumption, "annual_consumption_contributions.csv": annual}
    for name, frame in exports.items():
        frame.to_csv(output / name, index=False, float_format="%.15g")
    if rebuilt:
        pd.concat(rebuilt, ignore_index=True).to_csv(output / "bootstrap_parameters_rebuilt.csv", index=False, float_format="%.15g")
    _json(output / "diagnostics.json", {"models": diagnostics, "observed_data_end": end,
        "api_calls": 0, "estimation_window": config["model"],
        "gdp_vintage": "Frozen BEA workbook published August 26, 2026; not the latest GDP revision",
        "replication_status": "Actual-data methodology adaptation; exact Banque de France specification and numerical replication unverified",
        "uncertainty": "Pointwise conditional residual percentile intervals; neither causal identification nor general-equilibrium uncertainty included"})
    _plots(output, quarterly, periods, annual, config.get("display_start", "2023Q1"))
    _write_results(output, quarterly, periods)
    sources = [ROOT / "scripts/run_ecm.py", Path(__file__).resolve(), ROOT / "wealth_effects/ecm_gdp.py",
        ROOT / "wealth_effects/note454_ecm.py", ROOT / "wealth_effects/note454_ratio.py",
        ROOT / "wealth_effects/note454_ratio_ecm.py", ROOT / "wealth_effects/note454_gdp_draws.py"]
    config_key = str(config_path.relative_to(ROOT)) if config_path.is_relative_to(ROOT) else "external_configuration"
    manifest = {"created_utc": datetime.now(timezone.utc).isoformat(), "configuration": config,
        "input_sha256": {str(path.relative_to(ROOT)): _hash(path) for path in paths.values()},
        "configuration_sha256": {config_key: _hash(config_path)},
        "source_sha256": {str(path.relative_to(ROOT)): _hash(path) for path in sources},
        "dependencies": {name: version(name) for name in ("numpy", "pandas", "scipy", "statsmodels", "matplotlib")},
        "bootstrap_rebuilt": bool(rebuild_bootstrap),
        "output_sha256": {str(path.relative_to(output)): _hash(path) for path in sorted(output.iterdir())
            if path.is_file() and path.name != "run_manifest.json"}}
    _json(output / "run_manifest.json", manifest)
    return {"data_end": end, "output_directory": str(output), "quarterly_wealth_gdp_pp": quarterly.loc[
        quarterly.quarter.isin(["2026Q1", "2026Q2"]), ["quarter", "model", "gross_gdp_growth_pp"]].to_dict("records")}


def _plots(output, quarterly, periods, annual, display_start):
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "ecm-matplotlib-cache"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
        "axes.spines.right": False, "axes.titleweight": "bold", "pdf.fonttype": 42})
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    for ax, model in zip(axes, MODELS):
        table = quarterly[(quarterly.model == model) & (quarterly.quarter >= display_start)].copy()
        x = np.arange(len(table))
        ax.fill_between(x, table.gross_ci_low_pp, table.gross_ci_high_pp, color=COLORS[model], alpha=.19, label="Exploratory 95% interval")
        ax.plot(x, table.gross_gdp_growth_pp, marker="o", color=COLORS[model], lw=2, label="Wealth-associated PCE allocation")
        ax.plot(x, table.observed_gdp_growth_saar_pct, color="#51565c", lw=1.3, ls="--", label="Observed real GDP growth")
        ax.axhline(0, lw=.7, color="#555"); ax.grid(axis="y", alpha=.16)
        ax.set_ylabel("Annualized pp / %"); ax.set_title(LABELS[model], loc="left")
        ax.set_xticks(x, table.quarter, rotation=45, ha="right")
    axes[0].legend(loc="upper left", frameon=False, fontsize=8)
    fig.suptitle("Quarterly wealth contributions to real GDP growth through 2026Q2", fontsize=14)
    fig.text(.5, .012, "Conditional gross consumption allocation; frozen 1990–2019 coefficients. August 2026 BEA vintage.\n"
             "Unresolved ECM diagnostics; intervals exclude causal identification, imports and general-equilibrium offsets.", ha="center", fontsize=8)
    fig.tight_layout(rect=(0, .065, 1, .95)); _save(fig, output, "quarterly_gdp_wealth", plt)
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=True)
    labels = ["2023", "2024", "2025", "2026 H1"]
    for ax, model in zip(axes, MODELS):
        table = periods[periods.model == model].set_index("period").reindex(labels)
        x = np.arange(len(table))
        ax.bar(x - .17, table.mean_gross_quarterly_saar_equivalent_pp, width=.32, color=COLORS[model], label="Mean quarterly annualized")
        ax.bar(x + .17, table.calendar_gross_equivalent_pp, width=.32, color="#78909c", label="Calendar year / H1 over H1")
        ax.vlines(x - .17, table.mean_ci_low_pp, table.mean_ci_high_pp, lw=1, color="#292929")
        ax.vlines(x + .17, table.calendar_ci_low_pp, table.calendar_ci_high_pp, lw=1, color="#292929")
        ax.set_xticks(x, labels); ax.set_title(LABELS[model]); ax.axhline(0, color="#555", lw=.7); ax.grid(axis="y", alpha=.16)
    axes[0].set_ylabel("Wealth-associated real GDP growth, pp"); axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Annual and half-year GDP growth allocations", fontsize=14)
    fig.text(.5, .012, "2026 H1 contains Q1/Q2 only; H1 over H1 compares average GDP with the prior H1.\n"
             "Exploratory conditional intervals; unresolved ECM diagnostics. Gross consumption allocation, not causal GDP effects.", ha="center", fontsize=8)
    fig.tight_layout(rect=(0, .07, 1, .95)); _save(fig, output, "annual_gdp_wealth", plt)
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    channels = ["wealth", "income", "credit", "unemployment", "others"]
    colors = ["#bd8b27", "#486e9c", "#6e9a93", "#bd6262", "#c7cbd0"]
    labels = ["2020", "2021", "2022", "2023", "2024", "2025", "2026 H1"]
    for ax, model in zip(axes, MODELS):
        table = annual[annual.model == model].pivot(index="period", columns="channel", values="contribution_pp").reindex(labels)
        x, positive, negative = np.arange(len(table)), np.zeros(len(table)), np.zeros(len(table))
        for channel, color in zip(channels, colors):
            values = table[channel].to_numpy()
            ax.bar(x, np.maximum(values, 0), bottom=positive, color=color, label=channel.title())
            ax.bar(x, np.minimum(values, 0), bottom=negative, color=color)
            positive += np.maximum(values, 0); negative += np.minimum(values, 0)
        ax.plot(x, table.sum(axis=1), color="#222", marker="o", label="Observed consumption growth")
        ax.set_title(LABELS[model], loc="left"); ax.set_ylabel("Annualized pp / %")
        ax.axhline(0, color="#555", lw=.7); ax.grid(axis="y", alpha=.16); ax.set_xticks(x, labels)
    axes[0].legend(frameon=False, ncol=3, fontsize=8, loc="upper left")
    fig.suptitle("ECM consumption growth decomposition", fontsize=14)
    fig.text(.5, .012, "Mean quarterly annualized rates; 2026 H1 is a two-quarter mean. Others includes residuals and initial conditions.\n"
             "Actual-data adaptation; Banque de France's undisclosed equation is not exactly replicated. ECM diagnostics remain unresolved.", ha="center", fontsize=8)
    fig.tight_layout(rect=(0, .07, 1, .95)); _save(fig, output, "consumption_decomposition", plt)


def _save(fig, output, name, plt):
    fig.savefig(output / (name + ".png"), dpi=220, bbox_inches="tight")
    fig.savefig(output / (name + ".pdf"), bbox_inches="tight")
    plt.close(fig)


def _write_results(output, quarterly, periods):
    recent = quarterly[quarterly.quarter.isin(["2026Q1", "2026Q2"])]
    rows = "\n".join(f"| {r.quarter} | {LABELS[r.model]} | {r.gross_gdp_growth_pp:.3f} | [{r.gross_ci_low_pp:.3f}, {r.gross_ci_high_pp:.3f}] |" for r in recent.itertuples())
    h1 = periods[periods.period == "2026 H1"]
    hrows = "\n".join(f"| {LABELS[r.model]} | {r.mean_gross_quarterly_saar_equivalent_pp:.3f} | {r.calendar_gross_equivalent_pp:.3f} |" for r in h1.itertuples())
    (output / "RESULTS.md").write_text(f"""# Quarterly ECM results through 2026Q2

Both specifications are estimated on observed **1990Q1–2019Q4 quarterly data**, with coefficients held fixed through **2026Q2**. All 399 supplied joint bootstrap vectors per model are retained. Default reproduction uses only the shipped data and makes no API calls.

Wealth-associated contributions to **quarter-over-quarter annualized real GDP growth**, percentage points:

| Quarter | Specification | Contribution, pp | Exploratory 95% interval |
| --- | --- | ---: | --- |
{rows}

| Specification | 2026 H1 mean quarterly annualized contribution, pp | 2026 H1 over 2025 H1 allocation, pp |
| --- | ---: | ---: |
{hrows}

The H1 mean averages two annualized quarterly contributions. H1 over H1 instead compares average GDP with the prior H1; neither is a full-year 2026 estimate. Period uncertainty is computed by aggregating each joint path before taking quantiles.

[Quarterly GDP chart](quarterly_gdp_wealth.png), [annual/H1 chart](annual_gdp_wealth.png), [consumption decomposition](consumption_decomposition.png), [quarterly estimates](quarterly_gdp_growth.csv), [period summaries](period_summary.csv), [coefficients](coefficients.csv), [diagnostics](diagnostics.json).

The primary bridge allocates BEA's published PCE GDP contribution in proportion to each ECM log-consumption-growth channel. Channels reconcile the PCE contribution, while the arithmetic non-PCE remainder is not explained by this model. A price-weighted quantity approximation is exported as a cross-check. Consumption flows are actual quarterly flows; wealth stocks are not annualized; GDP flows are SAAR. The annualization and dollar conversion are applied separately.

**Interpretation remains conditional.** This is an actual-data methodology adaptation, not verified numerical replication of Banque de France's undisclosed specification. The log equation does not reject no cointegration; the nonlinear ratio target has no formal cointegration validation; both short-run forms retain residual serial correlation. Intervals are exploratory conditional residual-bootstrap intervals and exclude model-form uncertainty and causal identification. The gross consumption allocation does not estimate induced imports, monetary or other general-equilibrium offsets. GDP observations retain the August26,2026 BEA workbook vintage rather than the latest revision.

Reproduce from the repository root:

```bash
python scripts/run_ecm.py
python -m unittest discover -s tests -p 'test_public_ecm.py' -v
```

The shipped input, configuration, source and generated output hashes are recorded with relative paths in [run_manifest.json](run_manifest.json). Rebuild the whole two-stage bootstrap with `python scripts/run_ecm.py --rebuild-bootstrap`; the shipped input vectors are preserved.
""")
