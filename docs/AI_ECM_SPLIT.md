# AI-linked and other S&P wealth contributions, 2022–2026Q2

This extension subdivides the published ECM wealth contribution using the
S&P-specific AI-linked and remaining S&P market-capitalization series from
the separate AI wealth calculation. It uses the **existing estimated
coefficients**, with no new fit and no separately estimated AI wealth MPC.
Both the log ECM and ratio/DOLS ECM retain their 1990Q1–2019Q4 estimates.
The [original ECM methodology](ECM_METHODOLOGY.md) describes those estimates,
their diagnostics and the historical consumption accounting.

The primary calculation attributes changes after a **2021Q4 split baseline**
and reports **2022Q1–2026Q2**. A sensitivity starts in 2023Q1, with a 2022Q4
split baseline. These dates govern the new AI subdivision only. The full ECM
decomposition still starts in 1990Q1, with its original **1989Q4 anchor**.
Inherited wealth effects are retained rather than resetting the original
consumption dynamics at the start of the AI calculation.

## What the two variants measure

| Variant | S&P short-run control | Long-run net-worth channel |
| --- | --- | --- |
| `price_control_only` | Split between AI-linked S&P capitalization, other S&P capitalization and an index/capitalization basis | Retained in unallocated wealth |
| `ownership_bridge` | The same exact short-run partition | Also allocates postbaseline capitalization changes using lagged household ownership and income-group equity shares |

The ownership bridge is the main explanatory version. The price-control-only
version shows what can be partitioned without mapping market capitalization
into household net worth. The difference between them is a declared
allocation assumption, not evidence of an additional estimated coefficient.

Every version replaces the existing wealth channel with four subchannels:

- `ai_sp`: AI-linked **S&P** capitalization inputs and, where enabled, their
  mapped household wealth increments.
- `non_ai_sp`: the remaining **S&P** capitalization inputs and their mapped
  household wealth increments.
- `index_cap_basis`: the difference between the ECM's existing S&P price
  growth control and growth in the supplied total S&P capitalization.
- `unallocated_wealth`: the remainder needed to preserve the original wealth
  channel, including other assets, inherited effects and ownership-mapping
  differences.

The last channel is not a housing estimate, and `non_ai_sp` does not mean all
non-AI household wealth. Income, credit, unemployment, deterministic terms,
initial conditions and unexplained innovations keep their original values.
No newly calculated wealth response is added on top of the old wealth total.

## Inputs, provenance and units

The AI-linked and other S&P capitalization observations are **excluded from
the published repository**, together with the detailed capitalization and
ownership ledger. To reproduce this optional extension, provide a local
input file at `data/spx_ai_split_inputs.csv`, or change its configured path.
Source details and transformations are recorded in
[spx_ai_split_sources.json](../data/spx_ai_split_sources.json).

The required CSV header is:

```text
quarter,spx_ai_market_cap_bn,spx_total_market_cap_bn,spx_non_ai_market_cap_bn,household_equity_share,equity_bottom80_share,equity_top20_share
```

Supply complete quarter-end observations from **2021Q4 through 2026Q2**.
Capitalizations are nominal USD billions. AI-linked and other S&P values
must sum to total S&P capitalization. Ownership shares must lie between zero
and one, and the two income-group shares must sum to one. Store
contemporaneous ownership shares; the implementation shifts them once before
use. The local input and detailed output ledger are ignored by Git.

The capitalization source is a **user-supplied workbook** from the earlier
AI wealth analysis. Its upstream vendor, constituent definitions and
redistribution terms have not been independently established. The new series
therefore cannot be described as independently verified public data. Neither
the original workbook nor its derived capitalization observations are
distributed. The original ECM's public BEA, Federal
Reserve, BLS and S&P price inputs remain unchanged.

Use the workbook's `spx_ai_mkt_cap`, rather than its broader `ai_mkt_cap`
basket. In nominal USD billions, define:

```math
M_t^{A}=\mathrm{spx\_ai\_mkt\_cap}_t,\qquad
M_t=\frac{\mathrm{spx\_tot\_mkt\_cap}_t}{1000},\qquad
M_t^{N}=M_t-M_t^{A}.
```

The user confirmed that the source workbook's total S&P series requires
division by 1,000; the AI series is already in billions. The local CSV must
contain the converted values, so the runner does not apply that conversion
again. Missing quarters,
negative component capitalizations, inconsistent sums or invalid ownership
shares raise errors; no interpolation or unrelated replacement series is
used.

The ownership inputs use the existing Federal Reserve
[Financial Accounts](https://www.federalreserve.gov/releases/z1/) household
direct-and-indirect equity ownership proxy and
[Distributional Financial Accounts](https://www.federalreserve.gov/releases/z1/dataviz/dfa/)
corporate-equity and mutual-fund holdings. The bottom 80% and top 20% are
groups in the **income distribution**, not the wealth distribution or fixed
household cohorts. Contemporaneous source shares are retained in the input
file and shifted once, inside the calculation, before use.

The ECM consumption, income and GDP inputs retain the **August 26, 2026 BEA
vintage**. The earlier AI calculation used a **September 9, 2026 BEA
snapshot**; results from the two calculations are not automatically on the
same national-accounts vintage. This extension does not revise the original
ECM to force agreement with the earlier AI exercise.

## Additive partition of the S&P control

The fitted short-run ECM uses changes in the nominal S&P **price index**
$`S_t`$, rather than changes in market capitalization. Replacing that
regressor would change the estimated model. Instead, partition its observed
log change and retain the difference explicitly.

For positive endpoints, define the logarithmic mean:

```math
L(x,y)=
\begin{cases}
\displaystyle\frac{x-y}{\log x-\log y},&x\ne y,\\[6pt]
x,&x=y.
\end{cases}
```

For each quarter after the split baseline, let:

```math
g_{A,t}=\frac{M_t^A-M_{t-1}^A}{L(M_t,M_{t-1})},\qquad
g_{N,t}=\frac{M_t^N-M_{t-1}^N}{L(M_t,M_{t-1})}.
```

Then define the price/capitalization basis:

```math
g_{B,t}=\Delta\log S_t-g_{A,t}-g_{N,t},
\qquad
g_{A,t}+g_{N,t}=\Delta\log M_t.
```

Thus the three terms sum exactly to the original S&P price-control input.
The logarithmic mean has a continuous equal-endpoint limit: the calculation
remains finite when AI and other S&P changes offset and total capitalization
is unchanged. It does not divide component changes by the net total
capitalization change.

Each partition uses the same fitted current and lagged S&P coefficients.
The basis can reflect price-index versus capitalization construction,
issuance, buybacks, membership changes and source differences; the supplied
data do not identify those items separately. Market-cap changes are therefore
**capitalization-based proxies**, not an independently identified series of
valuation-only capital gains. Dividends and cash-flow effects are not added
as separate shocks.

## Ownership bridge into real household wealth

Let $`b`$ be the split baseline, $`h_{t-1}`$ the lagged household
direct-and-indirect equity ownership share, and $`\omega_{g,t-1}`$ the lagged
DFA corporate-equity and mutual-fund share of income group
$`g\in\{80,20\}`$. For capitalization component $`k\in\{A,N\}`$, allocate
the nominal change as:

```math
G_{g,k,t}=h_{t-1}\,\omega_{g,t-1}\,
          (M_t^k-M_{t-1}^k),\qquad t>b.
```

Cumulative nominal postbaseline gains and their current real value are:

```math
H_{g,k,t}=\sum_{s=b+1}^{t}G_{g,k,s},\qquad
A_{g,k,t}=\frac{H_{g,k,t}}{P_t},\qquad
H_{g,k,b}=A_{g,k,b}=0.
```

Here $`P_t`$ is the ECM's PCE implicit price, with 2017 equal to one.
Cumulative **nominal** gains are deflated once at the current price. This is
a real stock increment, so it is not divided by four. Inflation erosion of
the opening portfolio remains in unallocated wealth. The mapping does not
claim that cumulative allocated gains equal the observed change in household
net worth.

The bridge assumes the same AI-versus-other-S&P portfolio composition within
both income groups. The data do not observe group-specific AI holdings.
The aggregate household ownership proxy and DFA allocation weights also have
different sector and instrument boundaries: the former includes indirect
ownership, including defined-contribution pension exposure, whereas the DFA
corporate-equity/mutual-fund distribution does not identify all pension
equity look-through. Full pension balances are not added as though they were
entirely equities. These are exposure-calibration assumptions, held fixed
when quantifying coefficient uncertainty.

## Long-run allocation under each functional form

For `price_control_only`, set all new long-run target components to zero.
For `ownership_bridge`, the following formulas allocate the mapped real
wealth increments. They are accounting allocations of the existing target,
not counterfactual household optimization or distinct AI/non-AI MPC estimates.

### Log ECM

Let $`W_{g,t}`$ be observed real group net worth and $`\widehat b_g`$ the
saved long-run log-wealth coefficient. The allocated target component is:

```math
s_{k,t}^{\log}
=\sum_{g\in\{80,20\}}
 \widehat b_g\frac{A_{g,k,t}}{L(W_{g,t},W_{g,b})}.
```

This logarithmic-mean allocation follows the straight path between observed
baseline and current wealth. Applying the same denominator to all pieces of
a group's wealth change makes their log contributions additive. The
unallocated target retains the difference from the original wealth target,
which is still measured relative to 1989Q4.

### Ratio/DOLS ECM

Write the existing quarterly equilibrium target as:

```math
C^{\ast}(I,W)=T+\widehat\alpha(Y-T)
 +\frac{\widehat\beta_{80}W_{80}
        +\widehat\beta_{20}W_{20}}{4},\qquad I=(Y,T).
```

The divisor four converts the estimated annual-consumption wealth slopes to
actual quarterly flows; it does not annualize the wealth stock. The original
decomposition first allocates the change in $`\log C^{\ast}`$ between income
and wealth using two income contexts: the original 1989Q4 income bundle
$`I_0`$ and the current bundle $`I_t`$. Preserve that outer allocation and
subdivide wealth within each context:

```math
s_{k,t}^{ratio}
=\frac12\sum_{I\in\{I_0,I_t\}}
 \frac{\displaystyle
       \sum_{g\in\{80,20\}}\widehat\beta_g A_{g,k,t}/4}
      {L\!\left(C^{\ast}(I,W_t),C^{\ast}(I,W_b)\right)}.
```

This is a straight-line, logarithmic-mean allocation **within** each original
income context. It is not a new flat Shapley allocation across income, AI,
other S&P and residual wealth, which would change the original income
channel. Both endpoint targets must be positive for every retained parameter
vector. Invalid targets raise errors rather than being clipped, and no
bootstrap draw is silently discarded.

## Timing, recursion and reconciliation

For component $`k\in\{A,N\}`$, the new forcing term is:

```math
\nu_{k,t}=-\widehat\lambda s_{k,t-1}
 +\widehat\theta_{S,0}g_{k,t}
 +\widehat\theta_{S,1}g_{k,t-1}.
```

The basis uses only its corresponding short-run terms. New drivers and
states are zero before the chosen split begins, so earlier S&P inputs stay
in the unallocated channel. With the 2021Q4 baseline, the current short-run
component can respond in **2022Q1**; the first new long-run target enters in
**2022Q2** because the ECM uses the lagged equilibrium error.

Propagate each new forcing term with the existing fitted dynamics:

```math
\Delta z_{k,t}=\widehat\lambda z_{k,t-1}
 +\widehat\phi\Delta z_{k,t-1}+\nu_{k,t},\qquad
z_{k,t}=z_{k,t-1}+\Delta z_{k,t}.
```

Define the unallocated wealth path by subtracting the three new paths from
the original wealth path. At every quarter, for both log-growth and log-level
contributions:

```math
z_{W,t}=z_{A,t}+z_{N,t}+z_{B,t}+z_{R,t},\qquad
\Delta z_{W,t}=\Delta z_{A,t}+\Delta z_{N,t}
                +\Delta z_{B,t}+\Delta z_{R,t}.
```

This identity also holds for every saved joint coefficient draw. The
identity preserves original wealth totals without assuming the capitalization
bridge explains every observed dollar of household net worth.

## Allocation to real GDP growth

Let $`h_t^{PCE}`$ be BEA's published PCE contribution to quarterly annualized
real GDP growth, and $`\Delta c_t`$ observed quarterly log consumption growth.
Use the unchanged proportional bridge:

```math
h_{k,t}^{GDP}=h_t^{PCE}\frac{\Delta z_{k,t}}{\Delta c_t}.
```

The four new GDP allocations therefore sum to the original wealth-associated
GDP allocation. These are **gross consumption-channel allocations**, in
percentage points of quarter-over-quarter annualized real GDP growth. They
are not identified causal AI effects, net GDP effects or separate production
contributions. Induced imports, displacement, monetary-policy responses and
other general-equilibrium offsets are not estimated by this bridge.

The denominator uses observed aggregate consumption growth; division is
rejected at effectively zero growth and can remain sensitive near zero.
The original national-accounts total supplies the accounting benchmark,
rather than adding nonadditive chained-dollar component levels. A 2026 H1
mean averages the Q1 and Q2 annualized contributions; it is not growth in
half-year average GDP or a full-year 2026 forecast.

## Uncertainty and interpretation

The point calculations load `results/coefficients.csv`. Conditional
uncertainty reuses the **399 saved joint two-stage parameter vectors per
model** in `data/bootstrap_parameters.csv`, preserving the pairing of
long-run and short-run coefficients. It does not refit either model,
resample the capitalization series or generate a new post-2022 regression.
Each draw applies the complete mapping and recursion, and each reported
period interval is calculated from that period's draw-level aggregate.

The intervals are exploratory conditional coefficient intervals. They do not
include uncertainty about the AI classification, capitalization source,
ownership incidence, cross-group portfolio assumptions, baseline choice,
data revisions or model selection. They also inherit the original ECM's
weak cointegration evidence and residual serial correlation. Comparing the
two functional forms, the two bridge variants and the alternative split
baseline provides sensitivity information, not identification of a causal
AI-specific wealth coefficient.

The older AI calculation uses a broader AI basket and externally chosen
wealth-consumption parameters. This extension uses its **S&P-specific
capitalization inputs** inside the historically estimated ECM accounting.
Different baskets, baselines, exposure mappings, coefficients, adjustment
dynamics, GDP vintages and netting conventions can all produce different
results. Numerical agreement would not validate the ownership assumptions;
disagreement would not by itself establish that one estimated an AI causal
effect and the other did not.

## Chart presentation

The main charts display **2023Q1–2026Q2** and focus on the **ratio/DOLS ECM**.
`display_start` and `display_model` in `config/ai_split.json` control only
presentation. The primary ownership-bridge attribution retains its 2021Q4
baseline, and all quarterly estimates from 2022Q1 and the log-model sensitivity
remain in the CSV exports. A 2023Q1 display cutoff does not reset the model
or discard the effects of 2022 changes.

## Reproduce and inspect

From the repository root, after installing `requirements-ecm.txt`:

```bash
python scripts/run_ai_ecm_split.py --output results/ai_split
python -m unittest discover -s tests -v
```

[config/ai_split.json](../config/ai_split.json) controls paths, reporting
horizons, split baselines and variants. The run uses locally supplied AI/S&P
inputs and the committed saved coefficients. It requires no API credentials
or network download once the local input is available. The original ECM
outputs are not overwritten, and the baseline ECM runner remains fully
reproducible from the published data alone.

The [AI split results folder](../results/ai_split/) publishes quarterly
consumption/GDP allocations, period comparisons, conditional intervals,
publication figures and a provenance manifest. These are model outputs;
the underlying AI/non-AI capitalization observations are not included. A
local run also generates `capitalization_ownership_ledger.csv`, exposing the
lagged ownership weights, nominal gains, real gain stocks and
price/capitalization basis for auditing. That detailed ledger remains local
and ignored by Git. Automated checks cover additive
capitalization accounting, timing and ownership lags, unchanged original
channels, preservation of wealth and GDP totals, and reconciliation across
saved parameter vectors.
