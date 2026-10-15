# GDP and wealth contribution charts

All figures use the Ratio/DOLS ECM ownership bridge and display 2023Q1–2026Q2.
The calculation keeps its 2021Q4 split baseline and original 1989Q4 ECM anchor.
No model is refitted. The national-accounts data retain the original August26,2026 vintage.

1. **Stacked real GDP growth.** Columns stack the arithmetic GDP contribution excluding wealth with the estimated wealth contribution. Diamonds show observed quarter-over-quarter annualized GDP growth. Opposite signs use diverging stacks, so the diamond, rather than the top positive segment, identifies net growth. This is an accounting decomposition, not an identified causal or general-equilibrium counterfactual.
2. **Wealth contribution: AI, non-AI and other.** AI-linked S&P, the remaining S&P and other wealth sum to the original total wealth contribution. Other combines the index/capitalization basis with unallocated wealth, including non-S&P assets/liabilities, inherited effects and mapping residuals. It is not a housing estimate. Signed stacks retain negative components; diamonds show the net total.

3. **Wealth contribution as a percentage of observed growth.** The line is `100 × wealth contribution (pp) / observed GDP growth (%)`. The denominator retains its sign. A positive wealth contribution during a GDP contraction produces a negative ratio because wealth cushions the contraction. Shares above100% reflect a negative contribution from the rest of GDP. No ratios are clipped or absolute-valued. At effectively zero GDP growth (absolute annualized growth at most1e-10%), the ratio is undefined and exported blank; line connections are broken around negative-growth quarters. This ratio is not an identified causal fraction and becomes sensitive near zero GDP growth.

The CSV contains derived model results and public GDP growth only. The excluded AI/non-AI capitalization observations and detailed gain ledger are not read or exported.

```bash
python scripts/plot_ecm_columns.py
```

[Stacked GDP decomposition](gdp_with_without_wealth.png), [three-part wealth decomposition](wealth_ai_non_ai_other.png), [percentage of GDP growth](wealth_share_of_gdp_growth.png), [chart values](column_chart_data.csv). PDF versions are provided for presentations.
