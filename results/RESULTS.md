# Quarterly ECM results through 2026Q2

Both specifications are estimated on observed **1990Q1–2019Q4 quarterly data**, with coefficients held fixed through **2026Q2**. All 399 supplied joint bootstrap vectors per model are retained. Default reproduction uses only the shipped data and makes no API calls.

Wealth-associated contributions to **quarter-over-quarter annualized real GDP growth**, percentage points:

| Quarter | Specification | Contribution, pp | Exploratory 95% interval |
| --- | --- | ---: | --- |
| 2026Q1 | Ratio/DOLS ECM | 0.561 | [0.193, 0.824] |
| 2026Q2 | Ratio/DOLS ECM | 0.469 | [0.008, 0.779] |
| 2026Q1 | Log ECM | 0.214 | [-0.085, 0.409] |
| 2026Q2 | Log ECM | 0.163 | [-0.284, 0.492] |

| Specification | 2026 H1 mean quarterly annualized contribution, pp | 2026 H1 over 2025 H1 allocation, pp |
| --- | ---: | ---: |
| Log ECM | 0.188 | 0.400 |
| Ratio/DOLS ECM | 0.515 | 0.723 |

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
