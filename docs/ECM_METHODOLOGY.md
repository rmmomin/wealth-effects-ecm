# Quarterly wealth–consumption ECM methodology

This repository estimates two unrestricted, two-stage error-correction
models. The estimation window is 1990Q1–2019Q4. Observed inputs extend through
2026Q2, and post-2019 calculations keep the pre-2020 coefficients fixed.
This is a methodological adaptation of
[Banque de France Eco Notepad 454](https://www.banque-france.fr/system/files/2026-06/Billet_454_VE.pdf).
Its complete author specification, coefficients and data vintage are
unavailable, which prevents an exact numerical replication claim.

## Variables and units

Let $C_t,Y_t,T_t$ be actual quarterly real consumption, disposable income and
personal current transfer receipts, in billions of PCE purchasing-power
dollars. Let $W_{80,t},W_{20,t}$ be real quarter-end net-worth stocks for
households in the bottom 80% and top 20% of the **income distribution**. These
are income-percentile groups, not wealth-percentile groups or fixed cohorts.

The real flows are nominal BEA seasonally adjusted annual rates divided by
four and by the aggregate PCE implicit price. Real consumption comes from
chained-dollar PCE divided by four. The same PCE price deflates wealth and
credit stocks; stocks are not divided by four. Lowercase letters denote
natural logs. $\Delta$ denotes an unannualized quarterly first difference.

## Log specification

The first stage fits:

$$
c_t=a+b_Y y_t+b_{80}w_{80,t}+b_{20}w_{20,t}+e_t.
$$

The error-correction term is the generated residual:

$$
ECT_t^{\log}=c_t-\hat a-\hat b_Y y_t
 -\hat b_{80}w_{80,t}-\hat b_{20}w_{20,t}.
$$

The long-run wealth coefficients are elasticities. A dollar MPC depends on the
consumption/wealth ratio, and therefore is not a constant under this form.
Ordinary least squares estimates the first stage without coefficient-sign
restrictions or target matching.

## Ratio/DOLS specification

The second specification follows the normalization discussed by
[Beach, Gamber and Moran (2025)](https://www.federalreserve.gov/econres/notes/feds-notes/wealth-heterogeneity-and-consumer-spending-20250805.html).
Define annualized flow levels $C_t^A=4C_t$, $Y_t^A=4Y_t$,
$T_t^A=4T_t$, and

$$
R_t=\frac{C_t^A-T_t^A}{Y_t^A-T_t^A},\qquad
Q_{j,t}=\frac{W_{j,t}}{Y_t^A-T_t^A},\quad j\in\{80,20\}.
$$

The DOLS stage estimates:

$$
R_t=\alpha+\beta_{80}Q_{80,t}+\beta_{20}Q_{20,t}
 +\sum_{j\in\{80,20\}}\sum_{k=-1}^{1}
   \delta_{j,k}\Delta Q_{j,t-k}+u_t.
$$

Differences and leads/lags are formed within the training window. Trimming
unavailable boundary rows leaves 117 observations, from 1990Q3 to 2019Q3.
No 2020 observation enters a lead. The difference terms are estimation
controls and do not enter the static equilibrium target:

$$
C_t^{\ast}=T_t+\alpha(Y_t-T_t)
 +\frac{\beta_{80}W_{80,t}+\beta_{20}W_{20,t}}{4}.
$$

The wealth slopes measure annual real consumption dollars per real
wealth-stock dollar; the corresponding quarterly slopes are $\beta_j/4$.
The normalization imposes a long-run transfer MPC of one when nontransfer
income $Y_t-T_t$ and wealth are held fixed. That assumption does not constrain
the separately estimated short-run transfer coefficients.
The second-stage equilibrium error is:

$$
ECT_t^{ratio}=\log C_t-\log C_t^{\ast}.
$$

All relevant target values must be positive. Invalid targets raise errors;
they are not clipped to manufacture a logarithm. The ratio target is a
declared sensitivity, not the disclosed author equation of the Banque de
France note.

## Shared short-run equation

For each long-run specification, a separate OLS regression estimates:

$$
\Delta c_t=\kappa+\lambda ECT_{t-1}
 +\phi\Delta c_{t-1}
 +\theta_0'x_t+\theta_1'x_{t-1}+\varepsilon_t,
$$

where

$$
x_t=\begin{bmatrix}
\Delta\log(Y_t-T_t)\\
\Delta\log T_t\\
\Delta\log Credit_t\\
\Delta\log SP500_t\\
\Delta Unemployment_t
\end{bmatrix}.
$$

Credit is the real stock of household and nonprofit loans. S&P 500 is a
nominal quarter-end price index, not a total-return index or market
capitalization measure. Unemployment changes are percentage points. No
separate interest-rate, housing-price or general-equilibrium equation is
estimated here. Household net worth includes those asset classes in aggregate.

Pointwise coefficient inference uses Bartlett HAC standard errors with four
lags and a finite-sample correction. Conditional short-stage HAC inference
does not account for the generated first-stage target; the joint bootstrap
provides the separate whole-fit uncertainty calculation.

## Historical channel accounting

The decomposition is anchored at 1989Q4 and recursively reconstructs observed
consumption growth and the change in its log level. It separates income
(including transfers), wealth, credit, unemployment, deterministic terms,
initial state and unexplained innovations. Its dynamic recursion is:

$$
\Delta z_{k,t}=\lambda z_{k,t-1}
 +\phi\Delta z_{k,t-1}+\nu_{k,t},\qquad
z_{k,t}=z_{k,t-1}+\Delta z_{k,t}.
$$

The channel innovations $\nu_{k,t}$ sum to the model's observed innovation,
including the residual. In the log model, for example, the long-run wealth
innovation is:

$$
-\lambda\{b_{80}(w_{80,t-1}-w_{80,0})
             +b_{20}(w_{20,t-1}-w_{20,0})\},
$$

with the fitted current/lagged S&P growth terms added to that channel. Income
uses the analogous long-run income deviation plus short-run nontransfer
income and transfer terms. Presample short-run inputs and consumption lags
belong to the initial-state channel. This anchoring avoids assigning
arbitrary absolute log levels to wealth contributions.

The ratio target is nonlinear after taking logs. Its change from the baseline
is allocated between two bundles—income/transfers and wealth—using a
two-bundle Shapley calculation. If $f(I,W)=\log C^{\ast}(I,W)$, the wealth share is:

$$
s_{W,t}=\tfrac12\{f(I_0,W_t)-f(I_0,W_0)
                     +f(I_t,W_t)-f(I_t,W_0)\}.
$$

The income share uses the analogous expression, and the two sum exactly to
$f(I_t,W_t)-f(I_0,W_0)$. The wealth innovation uses
$-\lambda s_{W,t-1}$ plus its S&P terms. This is an explicit accounting
allocation of nonlinear interactions, not a causal intervention or a
uniquely identified decomposition.

For charting, each log-growth channel is multiplied by the same factor:

$$
K_t=100\frac{\exp(4\Delta c_t)-1}{\Delta c_t},
\qquad K_t\to400\ \text{as}\ \Delta c_t\to0.
$$

The resulting percentage-point channels sum to the exact annualized
consumption growth rate. Annual chart values average the four quarterly
annualized contributions; this differs from growth in annual-average levels.

## GDP bridge

Let $h_t^{PCE}$ be BEA's published PCE contribution to annualized quarterly
real GDP growth. The primary allocation is:

$$
h_{k,t}^{GDP}=h_t^{PCE}\frac{\Delta z_{k,t}}{\Delta c_t}.
$$

Thus $\sum_k h_{k,t}^{GDP}=h_t^{PCE}$. The published PCE contribution, rather
than a sum of chained-dollar component levels, supplies the accounting total.
The method is undefined at zero observed consumption growth. The implementation
raises when its absolute log change is at most $10^{-10}$. Small denominators
above that threshold can still make allocations sensitive. A price-weighted
approximation is exported as a cross-check.

H1 mean values average the Q1 and Q2 annualized contributions. For the separate
level-based period measures, let $G_t$ be BEA real GDP at an annual rate,
$r_t=G_t/G_{t-1}$, and construct a cumulative allocation path:

$$
A_{k,t}=A_{k,t-1}
 +\frac{G_{t-1}h_{k,t}^{GDP}}{100(1+r_t+r_t^2+r_t^3)}.
$$

The annual or H1-over-H1 percentage-point contribution is 100 times the
difference in average $A_k$ between the two matching periods, divided by
previous-period average real GDP. This is a separate accounting-equivalent
measure. Each bootstrap path is aggregated before quantiles are computed;
marginal interval endpoints are never averaged to create a period interval.

This bridge allocates a gross consumption channel. It does not estimate the
imports induced by that consumption, investment displacement, monetary
policy responses or other general-equilibrium offsets. The result therefore
does not identify a causal contribution of wealth to GDP growth.

## Uncertainty and econometric limitations

The maintained bootstrap holds all non-consumption observed paths fixed,
resamples centered short-run residuals in circular four-quarter blocks,
simulates consumption recursively, and refits both stages. Each model has
399 saved joint coefficient vectors. The historical observed panel is then
decomposed with each vector and converted to GDP units before intervals are
computed. The two finite unstable log refits are retained; failed new target
calculations are reported rather than silently removing a draw.

The log model's Engle–Granger test does not reject no cointegration
(p approximately 0.883). Its residual serial-correlation test and the ratio
model's test reject at conventional levels. No standard generated-residual
ADF p-value is presented as valid cointegration evidence for the nonlinear
ratio/DOLS target. These diagnostics leave the economic interpretation of
long-run equilibrium adjustment unresolved. Bootstrap intervals are
exploratory, model-conditional uncertainty measures, not identification-robust
causal intervals.

The publication runner checks exact growth/level accounting, quarterly units,
complete inputs and saved parameter schemas. Tests cover the numerical
identities, training-window isolation, Shapley allocation and bootstrap
reconstruction. Passing those checks verifies implementation; it does not
resolve the model's empirical diagnostics.
