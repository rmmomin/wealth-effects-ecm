"""Independent synthetic checks for the note-454 ECM adaptation."""
import unittest

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint

from wealth_effects.note454_ecm import (
    annual_decomposition, annualization_factor, bootstrap_cointegration_null,
    bootstrap_ecm, chart3_indicator, conditional_predictions, decompose_ecm,
    fit_ecm,
)


def synthetic_panel(seed=454):
    """Cointegrated observations from a known stable conditional ECM, not data."""
    rng = np.random.default_rng(seed)
    index = pd.period_range("1985Q1", "2025Q4", freq="Q")
    n = len(index)
    y = 2. + np.cumsum(.002 + rng.normal(0., .005, n))
    w80 = 5. + np.cumsum(.004 + rng.normal(0., .025, n))
    w20 = 7. + np.cumsum(.004 + rng.normal(0., .035, n))
    transfers = 1. + np.cumsum(.002 + rng.normal(0., .014, n))
    credit = 4. + np.cumsum(.002 + rng.normal(0., .018, n))
    stocks = 5. + np.cumsum(.004 + rng.normal(0., .04, n))
    unemployment = np.zeros(n)
    for i in range(1, n):
        unemployment[i] = .65 * unemployment[i - 1] + rng.normal(0., .15)
    changes = np.column_stack([np.diff(v, prepend=np.nan) for v in (y, transfers, credit, stocks, unemployment)])
    target = 1.3 + .8 * y + .12 * w80 - .04 * w20
    consumption = target.copy()
    consumption[:2] += rng.normal(0., .002, 2)
    current = np.array([.5, .04, .01, .01, -.005])
    lagged = np.array([.1, -.02, .03, .01, -.01])
    for i in range(2, n):
        growth = (-.35 * (consumption[i - 1] - target[i - 1])
                  + .15 * (consumption[i - 1] - consumption[i - 2])
                  + changes[i] @ current + changes[i - 1] @ lagged
                  + rng.normal(0., .0008))
        consumption[i] = consumption[i - 1] + growth
    return pd.DataFrame({"logC": consumption, "logY": y, "logW80": w80,
        "logW20": w20, "dlogC": np.diff(consumption, prepend=np.nan),
        "dlogY_ex_transfers": changes[:, 0], "dlogTransfers": changes[:, 1],
        "dlogCredit": changes[:, 2], "dlogSP500": changes[:, 3],
        "dUnemployment": changes[:, 4]}, index=index)


class Note454ECMTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.panel = synthetic_panel()
        cls.fit = fit_ecm(cls.panel)

    def test_known_cointegration_and_unrestricted_negative_coefficient(self):
        self.assertLess(self.fit.long_run_params.logW20, 0.)
        self.assertLess(self.fit.short_run_params["ECT_L1"], 0.)
        self.assertTrue(self.fit.diagnostics["stable_level_recursion"])
        self.assertTrue(self.fit.diagnostics["engle_granger"]["reject_no_cointegration_5pct"])
        self.assertAlmostEqual(self.fit.long_run_params.logY, .8, delta=.15)
        covariance = self.fit.short_run_covariance.to_numpy()
        np.testing.assert_allclose(covariance, covariance.T, atol=1e-12)
        self.assertGreater(np.linalg.eigvalsh(covariance).min(), -1e-12)

    def test_formal_cointegration_distribution_matches_maintained_toolkit(self):
        dates = self.fit.index
        statistic, pvalue, _ = coint(self.panel.loc[dates, "logC"],
                                    self.panel.loc[dates, ["logY", "logW80", "logW20"]],
                                    trend="c", maxlag=4, autolag="aic")
        self.assertAlmostEqual(statistic, self.fit.diagnostics["engle_granger"]["statistic"])
        self.assertAlmostEqual(pvalue, self.fit.diagnostics["engle_granger"]["pvalue"])

    def test_growth_levels_and_annualized_contributions_reconcile(self):
        out = decompose_ecm(self.fit, self.panel)
        grouped = out.groupby("quarter").sum(numeric_only=True)
        dates = pd.PeriodIndex(grouped.index, freq="Q")
        np.testing.assert_allclose(grouped.log_growth_contribution, self.panel.dlogC.reindex(dates), atol=1e-10)
        np.testing.assert_allclose(grouped.log_level_contribution,
            self.panel.logC.reindex(dates) - self.panel.loc["1989Q4", "logC"], atol=1e-10)
        np.testing.assert_allclose(grouped.annualized_consumption_growth_pp,
                                  100. * np.expm1(4. * self.panel.dlogC.reindex(dates)), atol=1e-8)
        annual = annual_decomposition(out)
        by_year = annual.groupby("year")
        np.testing.assert_allclose(by_year.contribution_pp.sum(), by_year.observed_growth_pct.first(), atol=1e-8)

    def test_anchored_channels_invariant_to_currency_unit_rescaling(self):
        changed = self.panel.copy()
        for name in ("logY", "logW80", "logW20"):
            changed[name] += np.log(1000.)
        altered = fit_ecm(changed)
        original = decompose_ecm(self.fit, self.panel)
        replacement = decompose_ecm(altered, changed)
        np.testing.assert_allclose(original.annualized_consumption_growth_pp,
                                  replacement.annualized_consumption_growth_pp, atol=1e-7, rtol=1e-7)

    def test_holdout_changes_cannot_change_training_estimates(self):
        altered = self.panel.copy()
        mask = altered.index >= pd.Period("2020Q1", freq="Q")
        altered.loc[mask, "logC"] += np.arange(mask.sum()) * .001
        altered["dlogC"] = altered.logC.diff()
        updated = fit_ecm(altered)
        np.testing.assert_allclose(self.fit.long_run_params, updated.long_run_params, atol=1e-12)
        np.testing.assert_allclose(self.fit.short_run_params, updated.short_run_params, atol=1e-12)
        prediction = conditional_predictions(self.fit, self.panel)
        self.assertEqual(len(prediction), 24)
        self.assertEqual(prediction.quarter.iloc[0], "2020Q1")

    def test_zero_growth_converter_and_chart3_denominators(self):
        np.testing.assert_allclose(annualization_factor([0., 1e-14]), [400., 400.])
        np.testing.assert_allclose(annualization_factor([.01]), [100. * np.expm1(.04) / .01])
        annual = pd.DataFrame({"year": [2025] * 5,
            "channel": ["wealth", "income", "credit", "unemployment", "others"],
            "contribution_pp": [1., 1., -.5, 0., 0.]})
        result = chart3_indicator(annual).iloc[0]
        self.assertAlmostEqual(result.wealth_absolute_contribution_share_pct, 40.)
        self.assertAlmostEqual(result.wealth_absolute_net_growth_ratio_pct, 100. / 1.5)

    def test_two_stage_bootstrap_is_reproducible_and_refits_both_stages(self):
        first = bootstrap_ecm(self.fit, self.panel, draws=8, block_length=4, seed=71)
        second = bootstrap_ecm(self.fit, self.panel, draws=8, block_length=4, seed=71)
        pd.testing.assert_frame_equal(first["parameter_draws"], second["parameter_draws"])
        self.assertEqual(first["diagnostics"]["completed_draws"], 8)
        long_draws = first["parameter_draws"].query("stage == 'long_run' and regressor == 'logY'")
        short_draws = first["parameter_draws"].query("stage == 'short_run' and regressor == 'ECT_L1'")
        self.assertGreater(long_draws.estimate.std(), 0.)
        self.assertGreater(short_draws.estimate.std(), 0.)
        self.assertTrue((first["annual_intervals"].ci_high >= first["annual_intervals"].ci_low).all())

    def test_conditional_null_cointegration_bootstrap(self):
        result = bootstrap_cointegration_null(self.fit, self.panel, draws=8, seed=17)
        self.assertEqual(result["completed_draws"], 8)
        self.assertGreater(result["conditional_bootstrap_pvalue"], 0.)
        self.assertLessEqual(result["conditional_bootstrap_pvalue"], 1.)

    def test_gaps_and_inconsistent_growth_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "Missing quarter"):
            fit_ecm(self.panel.drop(index=pd.Period("2005Q2", freq="Q")))
        altered = self.panel.copy()
        altered.loc["2005Q2", "dlogC"] += .1
        with self.assertRaisesRegex(ValueError, "dlogC"):
            fit_ecm(altered)


if __name__ == "__main__":
    unittest.main()
