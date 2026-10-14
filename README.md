# Wealth effects: quarterly ECM estimates

Estimate the relationship between U.S. household wealth and consumption with
quarterly error-correction models, then allocate the estimated consumption
channel to real GDP growth. Observed inputs and results run through **2026Q2**.
The estimation window is **1990Q1–2019Q4**; coefficients stay fixed afterward.

This is a reproducible methodological adaptation of Banque de France
[Eco Notepad 454](https://www.banque-france.fr/system/files/2026-06/Billet_454_VE.pdf),
with a second long-run specification motivated by
[Beach, Gamber and Moran (2025)](https://www.federalreserve.gov/econres/notes/feds-notes/wealth-heterogeneity-and-consumer-spending-20250805.html).
The note does not publish its full equation, coefficients or replication code,
so agreement with a headline result does not establish exact replication.

![Quarterly wealth-associated contribution to real GDP growth](results/quarterly_gdp_wealth.png)

Wealth-associated contributions to **quarter-over-quarter annualized real GDP
growth**, in percentage points:

| Quarter or period | Ratio/DOLS ECM | Log ECM |
| --- | ---: | ---: |
| 2026Q1 | 0.561 | 0.214 |
| 2026Q2 | 0.469 | 0.163 |
| 2026 H1 mean of quarterly contributions | 0.515 | 0.188 |
| H1 exploratory 95% interval | [0.183, 0.723] | [-0.087, 0.340] |

The ratio specification is shown first for comparison with the note; the log
specification provides a functional-form sensitivity. Neither is established
as the authors' exact model. The H1 mean averages Q1 and Q2 annualized
contributions; it is not a full-year 2026 estimate.

The figures are **conditional consumption-channel allocations**, not causal
GDP effects. The GDP bridge uses BEA's published PCE contribution and preserves
its accounting total. Induced imports and general-equilibrium feedbacks are
not identified. GDP/PCE data retain the **August 26, 2026 BEA vintage**.

Econometric checks also qualify the interpretation: the log model's
Engle–Granger test does not reject no cointegration (p ≈ 0.883); both models
retain residual serial correlation (four-lag Breusch–Godfrey p ≈ 0.008 and
0.006). The nonlinear ratio target has no formal cointegration validation.
Bootstrap intervals are exploratory and conditional on observed inputs and
the maintained model.

## Reproduce

Requires Python 3.12 or newer. The supplied derived quarterly inputs and saved
joint bootstrap parameters allow an offline run without API credentials or a
private source repository.

```bash
git clone https://github.com/rmmomin/wealth-effects-ecm.git
cd wealth-effects-ecm
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-ecm.txt
.venv/bin/python scripts/run_ecm.py
.venv/bin/python -m unittest discover -s tests -v
```

To regenerate the conditional two-stage bootstrap rather than reuse its saved
parameter vectors:

```bash
.venv/bin/python scripts/run_ecm.py --rebuild-bootstrap
```

See [reproduction details](docs/REPRODUCIBILITY.md) for units, horizon,
configuration and verification.

## Models and results

Both specifications estimate consumption growth with a lagged equilibrium
error, consumption dynamics, and current/lagged income excluding transfers,
transfers, credit, S&P 500 prices and unemployment changes. They differ in the
long-run target:

- **Log ECM:** consumption is linear in the logs of income and the wealth of
  the bottom 80% and top 20% of households ranked by **income**.
- **Ratio/DOLS ECM:** consumption net of transfers is normalized by income
  net of transfers. Wealth slopes are annual consumption dollars per wealth
  dollar. The long-run transfer MPC is assumed to equal one.

The wealth groups cover total net worth, including financial and nonfinancial
assets less liabilities. These equations do not separately identify housing,
equity and bond MPCs. The S&P term is an additional short-run price-index
control. [Full functional forms and attribution equations](docs/ECM_METHODOLOGY.md)
explain this distinction and the quarterly/annualized conversions.

| Resource | Contents |
| --- | --- |
| [Quarterly GDP chart](results/quarterly_gdp_wealth.pdf) | Estimates and exploratory intervals |
| [Quarterly estimates](results/quarterly_gdp_growth.csv) | Both specifications through 2026Q2 |
| [Period summaries](results/period_summary.csv) | Annual and H1 growth conventions |
| [Model coefficients](results/coefficients.csv) | Estimated long-run and short-run parameters |
| [Diagnostics](results/diagnostics.json) | Tests, stability and model status |
| [Data sources](docs/DATA_SOURCES.md) | Series, transformations, vintages and gaps |
| [Local archive policy](docs/LOCAL_ARCHIVE.md) | Preservation of the earlier research working tree |

## Repository layout

```text
config/ecm.json                 estimation and publication settings
data/                           derived inputs and saved bootstrap parameters
wealth_effects/                  reusable ECM and GDP-accounting modules
scripts/run_ecm.py               portable research runner
tests/                          numerical and accounting validation
results/                        tables, diagnostics and publication charts
docs/                           methodology, sources and reproduction guide
```

Provider source documents and raw archives are excluded from this public
release. Source links and factual derived inputs support the calculations;
their inclusion does not grant rights to upstream material. See
[data provenance](data/sources.json).
