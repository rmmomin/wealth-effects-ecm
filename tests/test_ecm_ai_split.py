"""Fixed-coefficient AI attribution preserves the published ECM accounting."""
from contextlib import ExitStack
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from unittest import TestCase
from unittest.mock import patch

import numpy as np
import pandas as pd

from wealth_effects import ecm_ai_split as ai
from wealth_effects.ecm_gdp import bridge_channels
from wealth_effects.note454_ecm import decompose_ecm
from wealth_effects.note454_ratio_ecm import decompose_ratio_ecm


ROOT = Path(__file__).resolve().parents[1]
MEASURES = ("log_growth_contribution", "log_level_contribution",
            "annualized_consumption_growth_pp")
COMPONENTS = ("ai_sp", "non_ai_sp", "index_cap_basis", "unallocated_wealth")
VARIANTS = ("price_control_only", "ownership_bridge")


def quarterly_file(name):
    frame = pd.read_csv(ROOT / "data" / name, index_col="quarter")
    frame.index = pd.PeriodIndex(frame.index, freq="Q")
    return frame


def synthetic_capitalization_inputs():
    """Deterministic test-only portfolios; these are not historical observations.

    Core accounting tests must run without redistributing the private AI/other
    market-cap series. No source record is used to construct this fixture.
    """
    dates = pd.period_range("2021Q4", "2026Q2", freq="Q")
    time = np.arange(len(dates), dtype=float)
    total = 35_000. * np.exp(.025 * time + .03 * np.sin(time))
    ai_share = .3 + .003 * time
    bottom_share = .15 + .002 * np.cos(time)
    return pd.DataFrame({
        "spx_ai_market_cap_bn": total * ai_share,
        "spx_non_ai_market_cap_bn": total * (1. - ai_share),
        "spx_total_market_cap_bn": total,
        "household_equity_share": .62 + .005 * np.sin(time),
        "equity_bottom80_share": bottom_share,
        "equity_top20_share": 1. - bottom_share,
    }, index=pd.PeriodIndex(dates, name="quarter"))


def by_channel(frame, measure="log_growth_contribution"):
    return frame.pivot(index="quarter", columns="channel", values=measure).sort_index()


def saved_draw_table(fit, draw=0):
    return pd.DataFrame([
        {"draw": draw, "stage": stage, "regressor": regressor, "estimate": value}
        for stage, parameters in (("long_run", fit.long_run_params),
                                  ("short_run", fit.short_run_params))
        for regressor, value in parameters.items()
    ])


class FixedCoefficientAISplitTests(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.panel = quarterly_file("quarterly_panel.csv")
        cls.accounting = quarterly_file("gdp_accounting.csv")
        cls.inputs = synthetic_capitalization_inputs()
        cls.spec = json.loads((ROOT / "config/ecm.json").read_text())["model"]
        cls.coefficients = pd.read_csv(ROOT / "results/coefficients.csv")
        cls.parameters = pd.read_csv(ROOT / "data/bootstrap_parameters.csv")
        cls.fits = ai.load_frozen_fits(cls.coefficients, cls.spec)
        cls.original = {
            model: (decompose_ecm if model == "log_ecm" else decompose_ratio_ecm)(
                fit, cls.panel, start="1990Q1", end="2026Q2")
            for model, fit in cls.fits.items()
        }
        cls.split = {
            (model, variant): ai.decompose_ai(
                fit, cls.panel, cls.inputs, model, variant=variant)
            for model, fit in cls.fits.items() for variant in VARIANTS
        }

    def test_frozen_parameters_are_the_published_vectors(self):
        for model, fit in self.fits.items():
            with self.subTest(model=model):
                source = self.coefficients[self.coefficients.model == model]
                for stage, parameters in (("long_run", fit.long_run_params),
                                           ("short_run", fit.short_run_params)):
                    expected = source[source.stage == stage].set_index("regressor").estimate
                    self.assertEqual(set(parameters.index), set(expected.index))
                    np.testing.assert_array_equal(parameters.sort_index(), expected.sort_index())
                self.assertEqual(fit.spec["start"], "1990Q1")
                self.assertEqual(fit.spec["end"], "2019Q4")

    def test_synthetic_portfolio_components_reconcile_original_wealth_and_unchanged_channels(self):
        for (model, variant), split in self.split.items():
            with self.subTest(model=model, variant=variant):
                self.assertFalse(split.duplicated(["quarter", "channel"]).any())
                self.assertEqual(set(split.base_quarter), {"1989Q4"})
                self.assertEqual(split.quarter.nunique(), 146)
                self.assertEqual(set(split.channel),
                    (set(self.original[model].channel) - {"wealth"}) | set(COMPONENTS))
                for measure in MEASURES:
                    original = by_channel(self.original[model], measure)
                    observed = by_channel(split, measure)
                    np.testing.assert_allclose(observed[list(COMPONENTS)].sum(axis=1),
                        original.wealth, atol=2e-11, rtol=2e-10)
                    other = original.columns.drop("wealth")
                    np.testing.assert_array_equal(observed[other], original[other])
                    np.testing.assert_allclose(observed.sum(axis=1), original.sum(axis=1),
                        atol=2e-11, rtol=2e-10)

    def test_pce_dollars_and_published_gdp_contribution_still_reconcile(self):
        for (model, variant), split in self.split.items():
            with self.subTest(model=model, variant=variant):
                table = bridge_channels(self.panel, split, self.accounting)
                original = bridge_channels(self.panel, self.original[model], self.accounting)
                sums = table.groupby("quarter").sum(numeric_only=True)
                dates = pd.PeriodIndex(sums.index, freq="Q")
                np.testing.assert_allclose(sums.real_consumption_change_quarter_bn,
                    self.panel.C_real_quarter_bn.diff().reindex(dates), atol=2e-9, rtol=1e-10)
                np.testing.assert_allclose(sums.gross_gdp_growth_pp,
                    self.accounting.published_pce_contribution_pp.reindex(dates),
                    atol=1e-8, rtol=1e-9)
                actual = table.pivot(index="quarter", columns="channel", values="gross_gdp_growth_pp")
                prior = original.pivot(index="quarter", columns="channel", values="gross_gdp_growth_pp")
                np.testing.assert_allclose(actual[list(COMPONENTS)].sum(axis=1), prior.wealth,
                    atol=2e-9, rtol=1e-9)
                np.testing.assert_array_equal(actual[prior.columns.drop("wealth")],
                    prior.drop(columns="wealth"))

    def test_split_starts_in_2022_and_preserves_the_original_anchor(self):
        ledger = ai.capitalization_bridge(self.panel, self.inputs)
        self.assertEqual(ledger.index[0], pd.Period("2021Q4", freq="Q"))
        self.assertEqual(ledger.index[-1], pd.Period("2026Q2", freq="Q"))
        gain_fields = [name for name in ledger if "gain" in name or name.endswith("_log_growth")]
        np.testing.assert_array_equal(ledger.iloc[0][gain_fields].astype(float), 0.)
        for (model, variant), split in self.split.items():
            with self.subTest(model=model, variant=variant):
                values = by_channel(split)
                np.testing.assert_array_equal(values.loc[:"2021Q4", list(COMPONENTS[:3])], 0.)
                np.testing.assert_allclose(values.loc[:"2021Q4", "unallocated_wealth"],
                    by_channel(self.original[model]).loc[:"2021Q4", "wealth"], atol=1e-12)

    def test_zero_ai_capitalization_has_zero_attributed_ai_channel(self):
        inputs = self.inputs.copy()
        inputs["spx_ai_market_cap_bn"] = 0.
        inputs["spx_non_ai_market_cap_bn"] = inputs.spx_total_market_cap_bn
        for model, fit in self.fits.items():
            for variant in VARIANTS:
                with self.subTest(model=model, variant=variant):
                    result = ai.decompose_ai(fit, self.panel, inputs, model, variant=variant)
                    np.testing.assert_array_equal(by_channel(result).ai_sp, 0.)

    def test_zero_household_exposure_removes_only_the_ownership_bridge(self):
        inputs = self.inputs.copy()
        inputs["household_equity_share"] = 0.
        for model, fit in self.fits.items():
            with self.subTest(model=model):
                price_only = ai.decompose_ai(fit, self.panel, inputs, model,
                    variant="price_control_only")
                ownership = ai.decompose_ai(fit, self.panel, inputs, model,
                    variant="ownership_bridge")
                for measure in MEASURES:
                    np.testing.assert_allclose(by_channel(price_only, measure),
                        by_channel(ownership, measure), atol=2e-12, rtol=1e-10)

    def test_swapping_basket_labels_swaps_only_the_two_basket_channels(self):
        swapped = self.inputs.copy()
        swapped[["spx_ai_market_cap_bn", "spx_non_ai_market_cap_bn"]] = self.inputs[
            ["spx_non_ai_market_cap_bn", "spx_ai_market_cap_bn"]].to_numpy()
        for (model, variant), original in self.split.items():
            with self.subTest(model=model, variant=variant):
                result = ai.decompose_ai(self.fits[model], self.panel, swapped, model, variant=variant)
                for measure in MEASURES:
                    expected = by_channel(original, measure).rename(
                        columns={"ai_sp": "non_ai_sp", "non_ai_sp": "ai_sp"}).sort_index(axis=1)
                    np.testing.assert_allclose(by_channel(result, measure).sort_index(axis=1),
                        expected, atol=2e-12, rtol=1e-10)

    def test_offsetting_gains_have_finite_signed_allocations(self):
        dates = pd.period_range("2021Q4", "2022Q2", freq="Q")
        inputs = pd.DataFrame({
            "spx_ai_market_cap_bn": [100., 120., 100.],
            "spx_non_ai_market_cap_bn": [900., 880., 900.],
            "spx_total_market_cap_bn": 1000.,
            "household_equity_share": .5,
            "equity_bottom80_share": .25,
            "equity_top20_share": .75,
        }, index=dates)
        ledger = ai.capitalization_bridge(self.panel, inputs, end="2022Q2")
        numeric = ledger.select_dtypes(include=[np.number])
        self.assertTrue(np.isfinite(numeric.to_numpy()).all())
        np.testing.assert_allclose(ledger.loc["2022Q1", ["ai_sp_cap_log_growth",
            "non_ai_sp_cap_log_growth"]], [.02, -.02], atol=1e-14)
        np.testing.assert_allclose(ledger.loc["2022Q1", ["ai_sp_bottom80_nominal_gain_bn",
            "ai_sp_top20_nominal_gain_bn"]], [2.5, 7.5], atol=1e-14)
        stock_fields = [name for name in ledger if name.endswith("gain_stock_bn")]
        np.testing.assert_allclose(ledger.loc["2022Q2", stock_fields], 0., atol=1e-14)
        np.testing.assert_allclose(ledger.loc["2022Q1":, "index_cap_basis_log_growth"],
            self.panel.loc["2022Q1":"2022Q2", "dlogSP500"], atol=1e-14)

    def test_ownership_uses_previous_quarter_shares_once(self):
        inputs = self.inputs.loc["2021Q4":"2022Q1"].copy()
        inputs["spx_ai_market_cap_bn"] = [100., 120.]
        inputs["spx_non_ai_market_cap_bn"] = [900., 880.]
        inputs["spx_total_market_cap_bn"] = 1000.
        inputs["household_equity_share"] = [.5, .8]
        inputs["equity_bottom80_share"] = [.2, .7]
        inputs["equity_top20_share"] = [.8, .3]
        ledger = ai.capitalization_bridge(self.panel, inputs, end="2022Q1")
        np.testing.assert_allclose(ledger.loc["2022Q1", ["ai_sp_bottom80_nominal_gain_bn",
            "ai_sp_top20_nominal_gain_bn"]], [2., 8.], atol=1e-14)
        np.testing.assert_allclose(ledger.loc["2022Q1", "ai_sp_bottom80_real_gain_stock_bn"],
            2. / self.panel.loc["2022Q1", "pce_price"], atol=1e-14)

    def test_long_run_wealth_effect_starts_one_quarter_after_the_gain(self):
        for model, fit in self.fits.items():
            with self.subTest(model=model):
                short = fit.short_run_params.copy() * 0.
                short["ECT_L1"] = -.2
                isolated = replace(fit, short_run_params=short)
                result = by_channel(ai.decompose_ai(isolated, self.panel, self.inputs, model))
                self.assertEqual(result.loc["2022Q1", "ai_sp"], 0.)
                self.assertGreater(abs(result.loc["2022Q2", "ai_sp"]), 1e-12)

    def test_first_long_run_response_uses_stock_units_and_original_income_contexts(self):
        # A $20bn cap gain gives the two income groups $2bn/$8bn at the
        # previous quarter's .5 household ownership and .2/.8 group shares.
        inputs = self.inputs.loc["2021Q4":"2022Q2"].copy()
        inputs["spx_ai_market_cap_bn"] = [100., 120., 120.]
        inputs["spx_non_ai_market_cap_bn"] = [900., 880., 880.]
        inputs["spx_total_market_cap_bn"] = 1000.
        inputs["household_equity_share"] = .5
        inputs["equity_bottom80_share"] = .2
        inputs["equity_top20_share"] = .8
        current = self.panel.loc["2022Q1"]
        wealth_base = self.panel.loc["2021Q4"]
        original_base = self.panel.loc["1989Q4"]
        real_gain = np.array([2., 8.]) / current.pce_price

        def endpoint_mean(first, second):
            return (first - second) / (np.log(first) - np.log(second))

        for model, fit in self.fits.items():
            with self.subTest(model=model):
                short = fit.short_run_params.copy() * 0.
                short["ECT_L1"] = -.2
                isolated = replace(fit, short_run_params=short)
                params = fit.long_run_params
                if model == "log_ecm":
                    means = endpoint_mean(current[["W80_real_stock_bn", "W20_real_stock_bn"]].to_numpy(),
                        wealth_base[["W80_real_stock_bn", "W20_real_stock_bn"]].to_numpy())
                    target = np.sum(params[["logW80", "logW20"]].to_numpy() * real_gain / means)
                else:
                    betas = params[["wealth_bottom80", "wealth_top20"]].to_numpy()
                    numerator = np.dot(betas, real_gain) / 4.
                    target = 0.
                    for income_context in (original_base, current):
                        income = income_context.Transfers_real_quarter_bn + params["const"] * (
                            income_context.Y_real_quarter_bn - income_context.Transfers_real_quarter_bn)
                        at_current = income + np.dot(betas,
                            current[["W80_real_stock_bn", "W20_real_stock_bn"]].to_numpy()) / 4.
                        at_base = income + np.dot(betas,
                            wealth_base[["W80_real_stock_bn", "W20_real_stock_bn"]].to_numpy()) / 4.
                        target += .5 * numerator / endpoint_mean(at_current, at_base)
                result = by_channel(ai.decompose_ai(isolated, self.panel, inputs, model, end="2022Q2"))
                np.testing.assert_allclose(result.loc["2022Q2", "ai_sp"], .2 * target,
                    atol=2e-13, rtol=1e-10)
                price_only = by_channel(ai.decompose_ai(isolated, self.panel, inputs, model,
                    variant="price_control_only", end="2022Q2"))
                np.testing.assert_array_equal(price_only.ai_sp, 0.)

    def test_saved_draw_components_reconcile_without_refitting_or_filtering(self):
        for model, fit in self.fits.items():
            selected = self.parameters[(self.parameters.model == model) &
                self.parameters.draw.isin([0, 198])].drop(columns="model")
            for variant in VARIANTS:
                with self.subTest(model=model, variant=variant):
                    draws = ai.bootstrap_ai_components(fit, self.panel, self.inputs,
                        selected, model, variant=variant)
                    self.assertEqual(set(draws.draw), {0, 198})
                    self.assertEqual(len(draws), 2 * 146)
                    self.assertFalse(draws.duplicated(["draw", "quarter"]).any())
                    np.testing.assert_allclose(draws[list(COMPONENTS)].sum(axis=1),
                        draws.original_wealth, atol=2e-11, rtol=1e-9)
                    for draw, parameters in selected.groupby("draw"):
                        vector = parameters.set_index(["stage", "regressor"]).estimate
                        restored = replace(fit, long_run_params=vector.xs("long_run"),
                            short_run_params=vector.xs("short_run"))
                        point = by_channel(ai.decompose_ai(restored, self.panel,
                            self.inputs, model, variant=variant))
                        actual = draws[draws.draw == draw].set_index("quarter").sort_index()
                        np.testing.assert_allclose(actual[list(COMPONENTS)], point[list(COMPONENTS)],
                            atol=2e-11, rtol=1e-9)

    def test_attribution_never_calls_an_estimator(self):
        targets = ["statsmodels.regression.linear_model.OLS.fit",
                   "wealth_effects.note454_ecm.fit_ecm",
                   "wealth_effects.note454_ecm._ols",
                   "wealth_effects.note454_ratio_ecm.fit_ratio_ecm",
                   "wealth_effects.note454_ratio_ecm.fit_ratio_dols"]
        with ExitStack() as stack:
            for target in targets:
                stack.enter_context(patch(target, side_effect=AssertionError("Unexpected refit")))
            fits = ai.load_frozen_fits(self.coefficients, self.spec)
            for model, fit in fits.items():
                ai.decompose_ai(fit, self.panel, self.inputs, model)
                ai.bootstrap_ai_components(fit, self.panel, self.inputs,
                    saved_draw_table(fit), model)

    def test_finite_unstable_saved_draw_is_retained(self):
        fit = self.fits["log_ecm"]
        short = fit.short_run_params.copy()
        short["ECT_L1"] = .001
        unstable = replace(fit, short_run_params=short)
        result = ai.bootstrap_ai_components(fit, self.panel, self.inputs,
            saved_draw_table(unstable, draw=937), "log_ecm")
        self.assertEqual(set(result.draw), {937})
        self.assertEqual(len(result), 146)
        self.assertTrue(np.isfinite(result[list(COMPONENTS)].to_numpy()).all())
        expected = by_channel(ai.decompose_ai(unstable, self.panel, self.inputs, "log_ecm"))
        np.testing.assert_allclose(result.set_index("quarter")[list(COMPONENTS)],
            expected[list(COMPONENTS)], atol=2e-11, rtol=1e-9)

    def test_invalid_capitalization_inputs_are_rejected_without_filling(self):
        cases = {}
        bad_units = self.inputs.copy()
        bad_units["spx_total_market_cap_bn"] *= 1000.
        cases["unconverted_total_units"] = bad_units
        negative = self.inputs.copy()
        negative.iloc[1, negative.columns.get_loc("spx_ai_market_cap_bn")] = -1.
        cases["negative_cap"] = negative
        missing = self.inputs.copy()
        missing.iloc[1, missing.columns.get_loc("household_equity_share")] = np.nan
        cases["missing_share"] = missing
        shares = self.inputs.copy()
        shares["equity_top20_share"] = .5
        shares["equity_bottom80_share"] = .6
        cases["income_shares_not_exhaustive"] = shares
        cases["missing_anchor"] = self.inputs.iloc[1:]
        cases["missing_quarter"] = self.inputs.drop(self.inputs.index[3])
        for name, inputs in cases.items():
            with self.subTest(case=name), self.assertRaises(ValueError):
                ai.capitalization_bridge(self.panel, inputs)

    def test_incomplete_saved_parameter_vectors_are_rejected(self):
        fit = self.fits["log_ecm"]
        incomplete = saved_draw_table(fit).iloc[1:]
        with self.assertRaises(ValueError):
            ai.bootstrap_ai_components(fit, self.panel, self.inputs, incomplete, "log_ecm")
        duplicate = pd.concat([saved_draw_table(fit), saved_draw_table(fit).iloc[:1]])
        with self.assertRaises(ValueError):
            ai.bootstrap_ai_components(fit, self.panel, self.inputs, duplicate, "log_ecm")


class AISplitPublicationTests(TestCase):
    @classmethod
    def setUpClass(cls):
        from wealth_effects import ecm_ai_publication

        cls.publication = ecm_ai_publication
        cls.config = json.loads((ROOT / "config/ai_split.json").read_text())
        cls.panel = quarterly_file("quarterly_panel.csv")
        cls.accounting = quarterly_file("gdp_accounting.csv")
        cls.has_private_inputs = (ROOT / cls.config["data"]["capitalization"]).is_file()
        cls.inputs = (quarterly_file("spx_ai_split_inputs.csv") if cls.has_private_inputs
            else synthetic_capitalization_inputs())
        cls.parameters = pd.read_csv(ROOT / "data/bootstrap_parameters.csv")
        cls.fits = ai.load_frozen_fits(pd.read_csv(ROOT / "results/coefficients.csv"),
            json.loads((ROOT / "config/ecm.json").read_text())["model"])
        cls.shipped = ROOT / "results/ai_split"
        cls.folder = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.folder.cleanup)
        cls.portable = (Path(cls.folder.name) / "portable").resolve()
        relative_paths = {name for name in cls.config["data"].values()
            if (ROOT / name).is_file()} | {
            "config/ai_split.json", "config/ecm.json", "scripts/run_ai_ecm_split.py"}
        relative_paths.update(str(path.relative_to(ROOT)) for path in (ROOT / "results").iterdir()
            if path.is_file())
        relative_paths.update("wealth_effects/" + name + ".py" for name in (
            "__init__", "ecm_ai_publication", "ecm_ai_split", "ecm_publication", "ecm_gdp",
            "note454_ecm", "note454_ratio", "note454_ratio_ecm", "note454_gdp_draws"))
        for name in sorted(relative_paths):
            target = cls.portable / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / name, target)
        if not cls.has_private_inputs:
            name = cls.config["data"]["capitalization"]
            target = cls.portable / name
            target.parent.mkdir(parents=True, exist_ok=True)
            cls.inputs.to_csv(target, index_label="quarter", float_format="%.17g")
            relative_paths.add(name)
        cls.before = {name: hashlib.sha256((cls.portable / name).read_bytes()).hexdigest()
            for name in relative_paths}
        # Execute the actual CLI inside a minimal clone. In public CI its
        # private input is replaced by the explicitly synthetic fixture above.
        # Any fit call raises; only optional figure production is suppressed.
        script = cls.portable / "scripts/run_ai_ecm_split.py"
        code = f"""
import json
from pathlib import Path
import runpy
import sys
from contextlib import ExitStack
from unittest.mock import patch
sys.path.insert(0, {str(cls.portable)!r})
from wealth_effects import ecm_ai_publication as publication
assert publication.ROOT == Path({str(cls.portable)!r})
with ExitStack() as stack:
    stack.enter_context(patch.object(publication, '_plots'))
    for target in ('statsmodels.regression.linear_model.OLS.fit',
                   'wealth_effects.note454_ecm.fit_ecm',
                   'wealth_effects.note454_ecm._ols',
                   'wealth_effects.note454_ratio_ecm.fit_ratio_ecm',
                   'wealth_effects.note454_ratio_ecm.fit_ratio_dols'):
        stack.enter_context(patch(target, side_effect=AssertionError('Unexpected runner refit')))
    sys.argv = [{str(script)!r}]
    runpy.run_path({str(script)!r}, run_name='__main__')
"""
        completed = subprocess.run([sys.executable, "-c", code], cwd=cls.portable,
            text=True, capture_output=True, timeout=60)
        if completed.returncode:
            raise AssertionError(completed.stdout + "\n" + completed.stderr)
        cls.runner_summary = json.loads(completed.stdout)
        cls.rebuilt = cls.portable / "results/ai_split"

    def test_portable_runner_preserves_all_inputs_and_baseline_artifacts_without_refitting(self):
        self.assertFalse((self.portable / "replications").exists())
        self.assertEqual(self.runner_summary["new_estimation_calls"], 0)
        self.assertEqual(self.runner_summary["checks"], 8)
        for name, before in self.before.items():
            with self.subTest(file=name):
                self.assertEqual(hashlib.sha256((self.portable / name).read_bytes()).hexdigest(), before)
        manifest = json.loads((self.rebuilt / "run_manifest.json").read_text())
        self.assertEqual(manifest["new_estimation_calls"], 0)
        for name, digest in manifest["input_sha256"].items():
            self.assertEqual(digest, self.before[name])
        for name, digest in manifest["output_sha256"].items():
            self.assertEqual(hashlib.sha256((self.rebuilt / name).read_bytes()).hexdigest(), digest)

    def test_shipped_numerical_artifacts_match_the_portable_rebuild(self):
        if not self.has_private_inputs:
            self.skipTest("Historical AI/other market-cap input is intentionally excluded from the public repository")
        for name in ("quarterly_wealth_split.csv",
                     "quarterly_all_channels.csv", "period_summary.csv"):
            with self.subTest(file=name):
                pd.testing.assert_frame_equal(pd.read_csv(self.rebuilt / name),
                    pd.read_csv(self.shipped / name), check_exact=False, atol=1e-12, rtol=1e-10)

    def test_shipped_quarterly_and_period_tables_reconcile_every_scenario(self):
        keys = ["model", "variant", "split_start"]
        quarterly = pd.read_csv(self.shipped / "quarterly_wealth_split.csv")
        periods = pd.read_csv(self.shipped / "period_summary.csv")
        channels = pd.read_csv(self.shipped / "quarterly_all_channels.csv")
        self.assertEqual(len(quarterly[keys].drop_duplicates()), 8)
        self.assertFalse(quarterly.duplicated(keys + ["quarter", "channel"]).any())
        self.assertFalse(periods.duplicated(keys + ["period", "channel"]).any())
        original = pd.read_csv(ROOT / "results/channel_gdp_growth.csv")
        for scenario, frame in quarterly.groupby(keys):
            with self.subTest(scenario=scenario):
                values = frame.pivot(index="quarter", columns="channel", values="gross_gdp_growth_pp")
                np.testing.assert_allclose(values[list(COMPONENTS)].sum(axis=1),
                    values.original_wealth, atol=2e-11, rtol=1e-9)
                baseline = original[(original.model == scenario[0]) & (original.channel == "wealth")]
                baseline = baseline.set_index("quarter").reindex(values.index)
                np.testing.assert_allclose(values.original_wealth, baseline.gross_gdp_growth_pp,
                    atol=2e-11, rtol=1e-9)
        totals = channels.groupby(keys + ["quarter"]).gross_gdp_growth_pp.sum()
        np.testing.assert_allclose(totals, [self.accounting.loc[pd.Period(quarter, freq="Q"),
            "published_pce_contribution_pp"] for *_, quarter in totals.index], atol=1e-8, rtol=1e-9)
        actual = periods.set_index(keys + ["period", "channel"]).sort_index()
        quarterly["period"] = quarterly.quarter.map(
            lambda value: "2026 H1" if value.startswith("2026") else value[:4])
        expected = quarterly.groupby(keys + ["period", "channel"]).gross_gdp_growth_pp.mean()
        np.testing.assert_allclose(actual.mean_quarterly_annualized_gdp_pp, expected,
            atol=2e-12, rtol=1e-10)
        np.testing.assert_array_equal(actual.quarters,
            [2 if period == "2026 H1" else 4 for *_, period, channel in actual.index])

    def test_h1_intervals_aggregate_joint_quarterly_draws_before_quantiles(self):
        if not self.has_private_inputs:
            self.skipTest("Historical intervals require the locally supplied AI/other market-cap input")
        periods = pd.read_csv(self.shipped / "period_summary.csv")
        baseline = pd.read_csv(ROOT / "results/period_summary.csv").query("period == '2026 H1'")
        quarters = ["2026Q1", "2026Q2"]
        factor = self.accounting.loc[quarters, "published_pce_contribution_pp"].to_numpy() / self.panel.loc[
            quarters, "dlogC"].to_numpy()
        for model, fit in self.fits.items():
            selected = self.parameters[self.parameters.model == model].drop(columns="model")
            for variant in VARIANTS:
                for split in ("2022Q1", "2023Q1"):
                    with self.subTest(model=model, variant=variant, split=split):
                        draw = ai.bootstrap_ai_components(fit, self.panel, self.inputs, selected,
                            model, split_start=split, variant=variant)
                        draw = draw[draw.quarter.isin(quarters)].sort_values(["draw", "quarter"])
                        # One H1 contribution per joint vector; never average marginal endpoints.
                        values = draw[[*COMPONENTS, "original_wealth"]].to_numpy().reshape(399, 2, 5)
                        means = (values * factor[None, :, None]).mean(axis=1)
                        expected = np.quantile(means, [.025, .5, .975], axis=0).T
                        actual = periods[(periods.model == model) & (periods.variant == variant) &
                            (periods.split_start == split) & (periods.period == "2026 H1")]
                        actual = actual.set_index("channel").loc[[*COMPONENTS, "original_wealth"]]
                        np.testing.assert_allclose(actual[["ci_low_pp", "bootstrap_median_pp", "ci_high_pp"]],
                            expected, atol=2e-11, rtol=1e-9)
                        previous = baseline[baseline.model == model].iloc[0]
                        np.testing.assert_allclose(actual.loc["original_wealth", "mean_quarterly_annualized_gdp_pp"],
                            previous.mean_gross_quarterly_saar_equivalent_pp, atol=2e-11, rtol=1e-9)
                        np.testing.assert_allclose(expected[-1], [previous.mean_ci_low_pp,
                            previous.mean_bootstrap_median_pp, previous.mean_ci_high_pp],
                            atol=2e-11, rtol=1e-9)

    def test_missing_private_input_fails_helpfully_before_creating_outputs(self):
        input_path = self.portable / self.config["data"]["capitalization"]
        output = Path(self.folder.name) / "missing-input-output"
        parked = input_path.with_suffix(".parked")
        input_path.rename(parked)
        try:
            completed = subprocess.run([sys.executable,
                str(self.portable / "scripts/run_ai_ecm_split.py"), "--output", str(output)],
                cwd=self.portable, text=True, capture_output=True, timeout=30)
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("AI/non-AI market-cap inputs are excluded from the public repository",
                completed.stderr)
            self.assertIn("docs/AI_ECM_SPLIT.md", completed.stderr)
            self.assertFalse(output.exists())
        finally:
            parked.rename(input_path)

    def test_invalid_configuration_and_protected_output_leave_old_results_intact(self):
        protected = [ROOT / "results", ROOT / "data", ROOT / "config"]
        baseline_paths = [ROOT / "results/coefficients.csv", ROOT / "results/period_summary.csv",
                          ROOT / "data/quarterly_panel.csv", ROOT / "config/ecm.json"]
        before = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in baseline_paths}
        with patch.object(self.publication, "_plots"):
            for output in protected:
                with self.subTest(output=str(output)), self.assertRaises(ValueError):
                    self.publication.build_ai_split(output)
            for setting, value in (("fit_new_parameters", True), ("market_cap_total_divisor", 1),
                                   ("decomposition_start", "2020Q1")):
                with self.subTest(setting=setting), tempfile.TemporaryDirectory() as folder:
                    config = {**self.config, setting: value}
                    config_path = Path(folder) / "invalid.json"
                    config_path.write_text(json.dumps(config))
                    output = Path(folder) / "output"
                    with patch.object(self.publication, "ROOT", self.portable), self.assertRaises(ValueError):
                        self.publication.build_ai_split(output, config_path)
                    self.assertFalse(output.exists())
        for path, digest in before.items():
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), digest)
