"""Budget, chronology and nonlinear-allocation checks for ratio-target ECM."""
import unittest

import numpy as np
import pandas as pd

from wealth_effects.note454_ecm import annual_decomposition
from wealth_effects.note454_ratio_ecm import (
    _target, bootstrap_ratio_ecm, decompose_ratio_ecm, fit_ratio_ecm,
    ratio_log_target, shapley_target_changes,
)


def ratio_panel(seed=727):
    rng = np.random.default_rng(seed)
    dates = pd.period_range("1985Q1", "2025Q4", freq="Q")
    n = len(dates)
    income = 200. + np.cumsum(1. + rng.normal(0., 2., n))
    transfers = 40. + np.cumsum(.2 + rng.normal(0., .8, n))
    w80 = 1500. + np.cumsum(12. + rng.normal(0., 30., n))
    w20 = 2200. + np.cumsum(15. + rng.normal(0., 50., n))
    target = transfers + .75 * (income - transfers) + (.03 * w80 + .01 * w20) / 4.
    error = np.zeros(n)
    for i in range(1, n):
        error[i] = .6 * error[i - 1] + rng.normal(0., .12)
    consumption = target + error
    credit = 700. + np.cumsum(2. + rng.normal(0., 5., n))
    stocks = np.exp(5. + np.cumsum(.006 + rng.normal(0., .04, n)))
    unemployment = 5. + rng.normal(0., .3, n)
    frame = pd.DataFrame({"C_real_quarter_bn": consumption, "Y_real_quarter_bn": income,
        "Transfers_real_quarter_bn": transfers, "W80_real_stock_bn": w80,
        "W20_real_stock_bn": w20, "logC": np.log(consumption), "logY": np.log(income),
        "logW80": np.log(w80), "logW20": np.log(w20)}, index=dates)
    frame["dlogC"] = frame.logC.diff()
    for name, values in (("dlogY_ex_transfers", income - transfers), ("dlogTransfers", transfers),
                         ("dlogCredit", credit), ("dlogSP500", stocks)):
        frame[name] = np.diff(np.log(values), prepend=np.nan)
    frame["dUnemployment"] = np.diff(unemployment, prepend=np.nan)
    return frame


class RatioECMTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.panel = ratio_panel()
        cls.fit = fit_ratio_ecm(cls.panel)

    def test_annual_flow_stock_conversion_and_exact_shapley_sum(self):
        params = pd.Series({"const": .75, "wealth_bottom80": .03, "wealth_top20": .01})
        quarter = _target(params, 200., 40., 1500., 2200.)
        self.assertAlmostEqual(quarter, 40. + .75 * 160. + (.03 * 1500. + .01 * 2200.) / 4.)
        dates = self.panel.index[self.panel.index >= pd.Period("1990Q1", freq="Q")]
        allocation = shapley_target_changes(params, self.panel, "1989Q4", dates)
        target = ratio_log_target(params, self.panel)
        np.testing.assert_allclose(allocation.income_target_log_change + allocation.wealth_target_log_change,
                                  target.reindex(dates) - target.loc["1989Q4"], atol=1e-12)

    def test_ratio_ecm_growth_and_levels_reconcile(self):
        out = decompose_ratio_ecm(self.fit, self.panel)
        totals = out.groupby("quarter").sum(numeric_only=True)
        dates = pd.PeriodIndex(totals.index, freq="Q")
        np.testing.assert_allclose(totals.log_growth_contribution, self.panel.dlogC.reindex(dates), atol=1e-10)
        np.testing.assert_allclose(totals.log_level_contribution,
                                  self.panel.logC.reindex(dates) - self.panel.loc["1989Q4", "logC"], atol=1e-10)
        annual = annual_decomposition(out)
        np.testing.assert_allclose(annual.groupby("year").contribution_pp.sum(),
                                  annual.groupby("year").observed_growth_pct.first(), atol=1e-8)
        self.assertEqual(self.fit.diagnostics["observations"], 120)
        self.assertEqual(self.fit.diagnostics["ratio_long_run"]["observations"], 117)

    def test_holdout_outcomes_cannot_leak_through_dols_leads(self):
        altered = self.panel.copy()
        mask = altered.index >= pd.Period("2020Q1", freq="Q")
        altered.loc[mask, "C_real_quarter_bn"] *= 1.2
        altered["logC"] = np.log(altered.C_real_quarter_bn)
        altered["dlogC"] = altered.logC.diff()
        changed = fit_ratio_ecm(altered)
        np.testing.assert_allclose(changed.long_run_params, self.fit.long_run_params, atol=1e-12)
        np.testing.assert_allclose(changed.short_run_params, self.fit.short_run_params, atol=1e-12)

    def test_currency_unit_invariance_and_nonpositive_target_rejection(self):
        altered = self.panel.copy()
        for name in ("C_real_quarter_bn", "Y_real_quarter_bn", "Transfers_real_quarter_bn",
                     "W80_real_stock_bn", "W20_real_stock_bn"):
            altered[name] *= 1000.
        for name in ("logC", "logY", "logW80", "logW20"):
            altered[name] += np.log(1000.)
        changed = fit_ratio_ecm(altered)
        a = decompose_ratio_ecm(self.fit, self.panel)
        b = decompose_ratio_ecm(changed, altered)
        np.testing.assert_allclose(a.annualized_consumption_growth_pp, b.annualized_consumption_growth_pp,
                                  atol=1e-7, rtol=1e-7)
        with self.assertRaisesRegex(ValueError, "Nonpositive"):
            _target(pd.Series({"const": -10., "wealth_bottom80": 0., "wealth_top20": 0.}),
                    200., 40., 1500., 2200.)

    def test_whole_ratio_bootstrap_refits_both_stages_reproducibly(self):
        a = bootstrap_ratio_ecm(self.fit, self.panel, draws=6, seed=2454)
        b = bootstrap_ratio_ecm(self.fit, self.panel, draws=6, seed=2454)
        pd.testing.assert_frame_equal(a["parameter_draws"], b["parameter_draws"])
        self.assertEqual(a["diagnostics"]["completed_draws"], 6)
        self.assertGreater(a["parameter_draws"].query("stage == 'long_run' and regressor == 'wealth_bottom80'").estimate.std(), 0.)
        self.assertGreater(a["parameter_draws"].query("stage == 'short_run' and regressor == 'ECT_L1'").estimate.std(), 0.)
        self.assertTrue((a["annual_intervals"].ci_high >= a["annual_intervals"].ci_low).all())


if __name__ == "__main__":
    unittest.main()
