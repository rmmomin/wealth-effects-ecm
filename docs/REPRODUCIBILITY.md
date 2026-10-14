# Reproduce the quarterly ECM calculations

The public repository is self-contained for estimating both ECMs and
reconstructing their quarterly consumption and GDP contribution histories.
It uses the supplied frozen derived observations and saved bootstrap
parameters. No private sibling checkout, API credentials or provider raw
archives are needed.

## Environment and commands

Use Python 3.12 or newer in a clean environment:

```bash
git clone https://github.com/rmmomin/wealth-effects-ecm.git
cd wealth-effects-ecm
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-ecm.txt
.venv/bin/python scripts/run_ecm.py
.venv/bin/python -m unittest discover -s tests -v
```

On Windows use the corresponding `.venv\Scripts\python.exe` executable.
The dependency file specifies the small numerical stack needed for ECM work;
the public runner does not require the SSJ toolkit or the earlier structural
research environment.

The default run estimates point coefficients from the observed panel and
restores the supplied joint bootstrap vectors for uncertainty. To repeat
the conditional bootstrap estimation:

```bash
.venv/bin/python scripts/run_ecm.py --rebuild-bootstrap
```

The bootstrap refits both stages for each draw and is more expensive than the
default run. Its seed, block length and draw count are explicit in the
configuration. All non-consumption paths stay fixed, so the intervals are
conditional on the observed covariates and functional form.

## Configuration and inputs

`config/ecm.json` supplies the estimation and decomposition horizons, lag
orders, HAC bandwidth and bootstrap settings. The maintained training period
ends in 2019Q4. Post-2019 contributions through 2026Q2 use those fixed point
coefficients and corresponding saved pre-2020 bootstrap coefficient vectors.

| File | Purpose |
| --- | --- |
| `data/quarterly_panel.csv` | Observed quarterly levels and model-ready transforms |
| `data/gdp_accounting.csv` | Published GDP/PCE accounting inputs and cross-check inputs |
| `data/bootstrap_parameters.csv` | Saved joint long-run and short-run coefficient draws |
| `data/sources.json` | Sources, vintages and transformations |

Quarterly dates must be unique, sorted and complete. Missing observations,
nonfinite inputs, inconsistent draw schemas and invalid nonlinear targets
raise errors. The runner does not fill a missing wealth or income variable
with an unrelated series.

The unemployment input retains the documented two-month 2025Q4 average.
All 2026Q1/Q2 monthly observations are available. See
[data sources](DATA_SOURCES.md) for this observed gap and the frozen vintages.

## Outputs and conventions

The runner writes `results/`, including coefficient tables, quarterly channel
decompositions, quarterly and period GDP summaries, diagnostics, figures and
validation records. File names and column labels distinguish unannualized
quarterly log consumption growth from annualized percentage-point
contributions.

The main GDP result proportionally allocates BEA's published PCE contribution
using the model's consumption-growth channels. Its total reconciles to the
published PCE contribution. It is a gross conditional allocation and does
not estimate induced imports or general-equilibrium offsets.

The 2026 H1 mean averages two quarterly annualized contributions. H1-over-H1
compares the corresponding average levels. Neither is a full-year estimate.
Period confidence intervals aggregate each joint parameter draw first and
then compute quantiles; they do not average quarterly marginal endpoints.

## Verification

The public tests cover household-sector data units as they enter the ECM,
training-window isolation, regression schema, anchored growth/level
reconciliation, nonlinear Shapley allocation, saved two-stage coefficient
reconstruction and GDP accounting. The publication validation also checks
the requested observed horizon and finite complete exports.

Minor floating-point differences can arise from numerical library or BLAS
versions. Compare coefficients and reconciliation errors with numerical
tolerances rather than expecting every CSV byte or PNG to match across
platforms. The maintained dependency versions and deterministic bootstrap
seeds support reproducibility; the saved parameters avoid regenerating
uncertainty during the default run.

The tests establish numerical correctness of the implemented method. They
do not establish cointegration or causal identification: the log
cointegration failure, residual serial correlation in both specifications,
and unvalidated nonlinear ratio target remain documented in
[the methodology](ECM_METHODOLOGY.md).

Updating inputs to a newer vintage constitutes a separate estimation
exercise. The public workflow intentionally uses its supplied snapshot;
preserve that snapshot and provenance before replacing any historical data.
