"""Numerical and training-isolation checks for the auxiliary ratio/DOLS fit."""
from unittest import TestCase, main

import numpy as np
import pandas as pd

from wealth_effects.note454_ratio import fit_ratio_dols, REQUIRED_LEVELS


def ratio_fixture(dols=False, noise=0.):
    rng = np.random.default_rng(454)
    dates = pd.period_range("1989Q3", "2025Q4", freq="Q")
    n = len(dates)
    annual_income = 400. + .9 * np.arange(n)
    annual_transfers = 40. + .2 * np.arange(n)
    denominator = annual_income - annual_transfers
    bottom = 2. + .03 * np.arange(n) + rng.normal(0., .2, n)
    top = 4. + .025 * np.arange(n) + rng.normal(0., .3, n)
    coefficients = {"const": .65, "wealth_bottom80": .065, "wealth_top20": .009}
    ratio = coefficients["const"] + coefficients["wealth_bottom80"] * bottom + coefficients["wealth_top20"] * top
    if dols:
        for name, values in (("wealth_bottom80", bottom), ("wealth_top20", top)):
            change = np.r_[0., np.diff(values)]
            for offset, value in ((-1, .003), (0, -.005), (1, .002)):
                label = f"lead{-offset}" if offset < 0 else f"lag{offset}"
                coefficients[f"d_{name}_{label}"] = value
                ratio += value * np.roll(change, offset)
    ratio += rng.normal(0., noise, n)
    annual_consumption = annual_transfers + denominator * ratio
    panel = pd.DataFrame({"C_real_quarter_bn": annual_consumption / 4.,
        "Y_real_quarter_bn": annual_income / 4.,
        "Transfers_real_quarter_bn": annual_transfers / 4.,
        "W80_real_stock_bn": denominator * bottom,
        "W20_real_stock_bn": denominator * top}, index=dates)
    return panel, coefficients


class Note454RatioTests(TestCase):
    def test_ratio_ols_recovers_annual_dollar_mpcs_with_unannualized_stocks(self):
        panel, expected = ratio_fixture()
        result = fit_ratio_dols(panel, leads_lags=0)
        np.testing.assert_allclose(result["params"], pd.Series(expected).reindex(result["params"].index), atol=1e-12)
        coefficients = result["coefficients"].set_index("regressor")
        for group, value in (("wealth_bottom80", .065), ("wealth_top20", .009)):
            self.assertAlmostEqual(coefficients.loc[group, "annual_consumption_mpc"], value, places=12)
            self.assertAlmostEqual(coefficients.loc[group, "quarterly_consumption_mpc"], value / 4., places=12)
            self.assertAlmostEqual(coefficients.loc[group, "quarterly_mpc_ci_high"], coefficients.loc[group, "ci_high"] / 4., places=12)
        self.assertEqual(result["diagnostics"]["observations"], 120)
        self.assertEqual(result["diagnostics"]["transfer_mpc"], 1.)
        self.assertIn("Maintained assumption", result["diagnostics"]["transfer_mpc_status"])
        fitted = result["fitted"]
        np.testing.assert_allclose(fitted.implied_long_run_real_consumption_saar_bn,
                                   fitted.observed_real_consumption_saar_bn, atol=1e-10)
        self.assertTrue(coefficients.loc["const", ["annual_consumption_mpc", "quarterly_consumption_mpc"]].isna().all())

    def test_dols_recovers_core_and_nuisance_slopes_on_disclosed_trimmed_sample(self):
        panel, expected = ratio_fixture(dols=True)
        result = fit_ratio_dols(panel, leads_lags=1)
        np.testing.assert_allclose(result["params"], pd.Series(expected).reindex(result["params"].index), atol=1e-12)
        self.assertEqual(result["diagnostics"]["effective_sample_start"], "1990Q3")
        self.assertEqual(result["diagnostics"]["effective_sample_end"], "2019Q3")
        self.assertEqual(result["diagnostics"]["observations"], 117)
        self.assertEqual(result["diagnostics"]["removed_boundary_quarters"], ["1990Q1", "1990Q2", "2019Q4"])
        self.assertEqual(result["X"].shape, (117, 9))
        np.testing.assert_allclose(result["X"] @ result["params"], result["y"], atol=1e-13)
        self.assertTrue((result["coefficients"].role == "DOLS_nuisance_difference_control").any())
        self.assertGreater((result["fitted"].fitted_long_run_ratio - result["fitted"].fitted_ratio_including_dols_terms).abs().max(), .001)

    def test_all_difference_levels_and_leads_remain_inside_training_window(self):
        panel, _ = ratio_fixture(dols=True, noise=.002)
        original = fit_ratio_dols(panel)
        changed = panel.copy()
        changed.loc[:"1989Q4", list(REQUIRED_LEVELS)] *= 1000.
        changed.loc["2020Q1":, list(REQUIRED_LEVELS)] = np.nan
        second = fit_ratio_dols(changed)
        truncated = fit_ratio_dols(panel.loc["1990Q1":"2019Q4"])
        for result in (second, truncated):
            np.testing.assert_allclose(result["X"], original["X"], rtol=0., atol=0.)
            np.testing.assert_allclose(result["params"], original["params"], rtol=0., atol=0.)
            np.testing.assert_allclose(result["covariance"], original["covariance"], rtol=0., atol=0.)
        window = panel.loc["1990Q1":"2019Q4"]
        annual_nontransfer = 4. * (window.Y_real_quarter_bn - window.Transfers_real_quarter_bn)
        wealth_ratio = window.W80_real_stock_bn / annual_nontransfer
        last = pd.Period("2019Q3", freq="Q")
        self.assertAlmostEqual(original["X"].loc[last, "d_wealth_bottom80_lead1"],
                               wealth_ratio.loc["2019Q4"] - wealth_ratio.loc[last], places=12)
        self.assertNotIn(pd.Period("2019Q4", freq="Q"), original["index"])

    def test_ratios_are_invariant_to_common_currency_and_deflator_rescaling(self):
        panel, _ = ratio_fixture(dols=True, noise=.002)
        original = fit_ratio_dols(panel)
        common = pd.Series(1. + .01 * np.arange(len(panel)), index=panel.index)
        changed = panel.mul(common, axis=0)
        second = fit_ratio_dols(changed)
        np.testing.assert_allclose(second["X"], original["X"], rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(second["params"], original["params"], rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(second["covariance"], original["covariance"], rtol=1e-9, atol=1e-12)

    def test_estimated_wealth_slopes_are_not_sign_constrained(self):
        panel, _ = ratio_fixture()
        panel.C_real_quarter_bn -= .075 * panel.W80_real_stock_bn / 4.
        result = fit_ratio_dols(panel, leads_lags=0)
        self.assertAlmostEqual(result["params"]["wealth_bottom80"], -.01, places=12)
        coefficient = result["coefficients"].set_index("regressor").loc["wealth_bottom80"]
        self.assertLess(coefficient.annual_consumption_mpc, 0.)
        self.assertLess(coefficient.quarterly_consumption_mpc, 0.)

    def test_hac_covariance_matches_bartlett_sandwich_in_original_parameter_units(self):
        panel, _ = ratio_fixture(dols=True, noise=.002)
        result = fit_ratio_dols(panel)
        X, y = result["X"].to_numpy(), result["y"].to_numpy()
        residual = y - X @ result["params"].to_numpy()
        scores = X * residual[:, None]
        meat = scores.T @ scores
        for lag in range(1, 5):
            cross = scores[lag:].T @ scores[:-lag]
            meat += (1. - lag / 5.) * (cross + cross.T)
        bread = np.linalg.inv(X.T @ X)
        expected = bread @ meat @ bread * len(X) / (len(X) - X.shape[1])
        np.testing.assert_allclose(result["covariance"], expected, rtol=1e-8, atol=1e-12)
        np.testing.assert_allclose(result["covariance"], result["covariance"].T, atol=1e-12)
        self.assertGreaterEqual(np.linalg.eigvalsh(result["covariance"]).min(), -1e-12)

    def test_invalid_missing_rank_deficient_and_zero_income_inputs_fail(self):
        panel, _ = ratio_fixture(noise=.002)
        for name in ("leads_lags", "hac_lags"):
            for value in (-1, True, .5):
                with self.subTest(name=name, value=value):
                    with self.assertRaises(ValueError):
                        fit_ratio_dols(panel, **{name: value})
        with self.assertRaisesRegex(ValueError, "Missing quarter"):
            fit_ratio_dols(panel.drop(panel.index[20]))
        with self.assertRaisesRegex(ValueError, "quarterly Q-DEC"):
            fit_ratio_dols(panel.set_axis(panel.index.to_timestamp()))
        collinear = panel.copy()
        collinear.W20_real_stock_bn = collinear.W80_real_stock_bn
        with self.assertRaisesRegex(ValueError, "Rank-deficient"):
            fit_ratio_dols(collinear)
        zero_income = panel.copy()
        zero_income.Transfers_real_quarter_bn = zero_income.Y_real_quarter_bn
        with self.assertRaisesRegex(ValueError, "excluding transfers must be positive"):
            fit_ratio_dols(zero_income)
        missing = panel.copy()
        missing.loc["2010Q1", "W80_real_stock_bn"] = np.nan
        with self.assertRaisesRegex(ValueError, "Nonfinite"):
            fit_ratio_dols(missing)
        with self.assertRaisesRegex(ValueError, "Too few observations"):
            fit_ratio_dols(panel, start="2010Q1", end="2013Q4")
        with self.assertRaisesRegex(ValueError, "HAC bandwidth"):
            fit_ratio_dols(panel, hac_lags=117)


if __name__ == "__main__":
    main()
