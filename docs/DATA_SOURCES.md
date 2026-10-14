# Data sources and transformations

The public release supplies the derived factual inputs required to reproduce
the quarterly ECM and GDP-accounting results. It does not include provider raw
archives, author PDFs, credentials or material from the private source
repository. [Source provenance](../data/sources.json) records the retained
source dates and transformations; reproducibility uses the supplied frozen
inputs rather than downloading current revisions.

## Series

| Input | Provider and identifier | Working transformation |
| --- | --- | --- |
| Real consumption | BEA NIPA 1.1.6, `DPCERX` | Chained 2017-dollar PCE SAAR / 4 |
| Nominal consumption | BEA NIPA 1.1.5, `DPCERC` | Supplies PCE implicit price with real PCE |
| Disposable income | BEA NIPA 2.1, `A067RC` | Nominal DPI SAAR / 4 / PCE price |
| Transfers | BEA NIPA 2.1, `A577RC` | Personal current transfer receipts SAAR / 4 / PCE price |
| Income excluding transfers | Derived BEA values | Real DPI minus real personal current transfer receipts |
| Income-group wealth | Federal Reserve DFA, `dfa-income-levels-detail.csv` | Net-worth stocks summed within the income groups below, then deflated by PCE price |
| Credit | Federal Reserve Financial Accounts, `FL154135005.Q` | Household and nonprofit loans liability stock, deflated by PCE price |
| Unemployment | BLS CPS, `LNS14000000` | Seasonally adjusted monthly rate, quarterly arithmetic mean |
| S&P 500 prices | Public Yahoo Finance `^GSPC` observations | Last monthly close in each quarter; nominal price-index level |
| GDP and PCE contribution | BEA NIPA GDP tables | Published quarterly annualized GDP growth and PCE contribution |

BEA workbook monetary series are converted from millions to billions before
forming working variables. The PCE price is nominal PCE / real PCE, with 2017
equal to one. It deflates monetary stocks but not the S&P price index.

Provider references:

- [BEA National Income and Product Accounts](https://www.bea.gov/products/national-income-and-product-accounts)
- [Federal Reserve Financial Accounts](https://www.federalreserve.gov/releases/z1/)
- [Federal Reserve Distributional Financial Accounts](https://www.federalreserve.gov/releases/z1/dataviz/dfa/)
- [BLS unemployment series](https://data.bls.gov/timeseries/LNS14000000)
- [Public S&P 500 price history](https://finance.yahoo.com/quote/%5EGSPC/history/)

## Income-group definitions

The bottom 80% sums DFA categories `pct00to20`, `pct20to40`, `pct40to60` and
`pct60to80`. The top 20% sums `pct80to99` and `pct99to100`. Those are official
income-percentile categories. They are not wealth rankings and do not follow
the same households through time.

DFA observations are official quarterly distribution estimates constructed
from Survey of Consumer Finances distributional information and Financial
Accounts aggregates, with interpolation/extrapolation between survey waves.
They are not a quarterly household-panel survey. DFA wealth uses the
household sector; the credit control covers households and nonprofits. This
boundary difference is retained explicitly.

Net worth combines financial assets, real estate and other assets less
liabilities. Using its aggregate value in an ECM does not independently
identify an MPC for every underlying asset class.

## Frequency and observed gaps

The panel begins in 1989Q3 to supply presample quarterly differences and lags.
The 120-quarter short-run estimation window is 1990Q1–2019Q4; the DOLS stage
uses 117 observations after within-window boundary trimming. The reported
contribution horizon is 1990Q1–2026Q2.

Monetary flows are actual quarterly levels, stocks are quarter-end levels,
and first differences are unannualized natural-log changes. Unemployment
changes are percentage points. Annualized model charts apply their
conversion only after quarterly estimation and decomposition.

October 2025 unemployment is absent in the retained public observations.
The 2025Q4 value therefore averages the two available months; no missing
monthly value is fabricated. Each 2026Q1/Q2 unemployment quarter has all
three observed months. This difference is disclosed rather than presented
as a fully observed three-month historical quarter.

S&P data are price-index closes and do not include dividends. This series
is a public substitute for the licensed LSEG Datastream input identified in
the note. The note's precise valuation transform is undisclosed, so this is
a methodological input difference, not an exact provider replication.

## Vintages and publication boundary

BEA consumption, income and GDP retain the **August 26, 2026 workbook
publication**. Financial Accounts credit retains the **September 11, 2026**
archive. The DFA input and incremental 2026 unemployment/S&P observations
were retrieved in October 2026. Historical training observations remain
frozen when the new quarters are appended. Those source vintages differ from
the note's June 2026 publication and from later official GDP revisions.

`quarterly_panel.csv` contains the model-ready observed and transformed
series. `gdp_accounting.csv` contains the quarterly accounting inputs.
`bootstrap_parameters.csv` contains the joint coefficient vectors used for
conditional uncertainty. These files allow an offline calculation; raw
provider bytes are excluded from the public Git history.

Source attribution does not relicense provider content, trademarks, index
data or published papers. Consult the linked providers for their applicable
terms. The public numerical series and model outputs are research inputs and
results; no right to upstream source material is granted by this repository.
