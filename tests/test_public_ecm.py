"""Portable ECM checks using shipped public data only, without the legacy repo."""
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from unittest import TestCase

import numpy as np
import pandas as pd

from wealth_effects.ecm_gdp import bridge_channels, consumption_change_factor, reconstruct_wealth_draws, temporal_summary
from wealth_effects.ecm_publication import build_ecm
from wealth_effects.note454_ecm import decompose_ecm, fit_ecm
from wealth_effects.note454_ratio_ecm import decompose_ratio_ecm, fit_ratio_ecm

ROOT = Path(__file__).resolve().parents[1]


def read_quarterly(name):
    result = pd.read_csv(ROOT / "data" / name, index_col="quarter")
    result.index = pd.PeriodIndex(result.index, freq="Q")
    return result


def parameter_table(fit, draw=0):
    return pd.DataFrame([{"draw": draw, "stage": stage, "regressor": name, "estimate": value}
        for stage, params in (("long_run", fit.long_run_params), ("short_run", fit.short_run_params))
        for name, value in params.items()])


class PublicECMTests(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.panel = read_quarterly("quarterly_panel.csv")
        cls.accounting = read_quarterly("gdp_accounting.csv")
        cls.config = json.loads((ROOT / "config/ecm.json").read_text())
        cls.parameters = pd.read_csv(ROOT / "data/bootstrap_parameters.csv")
        cls.fits = {"log_ecm": fit_ecm(cls.panel, cls.config["model"]),
                    "ratio_ecm": fit_ratio_ecm(cls.panel, cls.config["model"])}
        cls.decompositions = {model: (decompose_ecm if model == "log_ecm" else decompose_ratio_ecm)(
            fit, cls.panel, start=cls.config["decomposition_start"], end=cls.config["data_end"])
            for model, fit in cls.fits.items()}
        cls.channels = {model: bridge_channels(cls.panel, decomposition, cls.accounting)
            for model, decomposition in cls.decompositions.items()}

    def test_observed_quarterly_data_horizon_and_training_counts(self):
        self.assertEqual(str(self.panel.index[0]), "1989Q3")
        self.assertEqual(str(self.panel.index[-1]), "2026Q2")
        self.assertEqual(len(self.panel), 148)
        for fit in self.fits.values():
            self.assertEqual(len(fit.index), 120)
            self.assertEqual(str(fit.index[-1]), "2019Q4")
        self.assertEqual(self.fits["ratio_ecm"].ratio_fit["diagnostics"]["observations"], 117)
        self.assertEqual(self.parameters.groupby("model").draw.nunique().to_dict(),
                         {"log_ecm": 399, "ratio_ecm": 399})

    def test_actual_dollars_and_all_channels_reconcile(self):
        for model, table in self.channels.items():
            with self.subTest(model=model):
                sums = table.groupby("quarter").sum(numeric_only=True)
                dates = pd.PeriodIndex(sums.index, freq="Q")
                np.testing.assert_allclose(sums.real_consumption_change_quarter_bn,
                    self.panel.C_real_quarter_bn.diff().reindex(dates), atol=2e-9, rtol=1e-10)
                np.testing.assert_allclose(sums.gross_gdp_growth_pp,
                    self.accounting.published_pce_contribution_pp.reindex(dates), atol=1e-8, rtol=1e-9)

    def test_previously_published_2026_point_estimates_are_preserved(self):
        # Independent frozen results, recorded before the portable runner existed.
        expected = {"ratio_ecm": [0.560684429812927, 0.469117297158909],
                    "log_ecm": [0.213975506896723, 0.162782023466562]}
        for model, values in expected.items():
            table = self.channels[model].query("channel == 'wealth'").set_index("quarter")
            np.testing.assert_allclose(table.loc[["2026Q1", "2026Q2"], "gross_gdp_growth_pp"],
                                       values, atol=2e-8, rtol=1e-8)

    def test_new_observations_cannot_change_training_coefficients(self):
        altered = self.panel.copy()
        post = altered.index >= pd.Period("2020Q1", freq="Q")
        altered.loc[post, "C_real_quarter_bn"] *= 1.02
        altered["logC"] = np.log(altered.C_real_quarter_bn)
        altered["dlogC"] = altered.logC.diff()
        altered.loc[post, "W80_real_stock_bn"] *= 1.1
        altered["logW80"] = np.log(altered.W80_real_stock_bn)
        for model, original in self.fits.items():
            fit = fit_ecm(altered, self.config["model"], compute_diagnostics=False) if model == "log_ecm" else fit_ratio_ecm(altered, self.config["model"])
            np.testing.assert_allclose(fit.long_run_params, original.long_run_params, atol=1e-12)
            np.testing.assert_allclose(fit.short_run_params, original.short_run_params, atol=1e-12)

    def test_fast_draw_recursion_matches_full_channel_methods(self):
        for model, fit in self.fits.items():
            selected = self.parameters[(self.parameters.model == model) & self.parameters.draw.isin([0, 198, 398])].drop(columns="model")
            fast = reconstruct_wealth_draws(fit, self.panel, selected, model, "1990Q1", "2026Q2", 3)
            function = decompose_ecm if model == "log_ecm" else decompose_ratio_ecm
            for draw, params in selected.groupby("draw"):
                params = params.set_index(["stage", "regressor"]).estimate
                restored = replace(fit, long_run_params=params.xs("long_run").reindex(fit.long_run_params.index),
                    short_run_params=params.xs("short_run").reindex(fit.short_run_params.index))
                full = function(restored, self.panel, start="1990Q1", end="2026Q2").query("channel == 'wealth'")
                np.testing.assert_allclose(fast.loc[fast.draw == draw, "wealth_log_growth_contribution"],
                    full.log_growth_contribution, atol=2e-12, rtol=1e-10)

    def test_unstable_but_finite_draw_is_retained(self):
        fit = self.fits["log_ecm"]
        unstable = replace(fit, short_run_params=fit.short_run_params.copy())
        unstable.short_run_params["ECT_L1"] = .001
        fast = reconstruct_wealth_draws(fit, self.panel, parameter_table(unstable), "log_ecm", "1990Q1", "2026Q2", 1)
        full = decompose_ecm(unstable, self.panel, start="1990Q1", end="2026Q2").query("channel == 'wealth'")
        np.testing.assert_allclose(fast.wealth_log_growth_contribution, full.log_growth_contribution, atol=1e-12)

    def test_incomplete_saved_draw_is_rejected(self):
        fit = self.fits["log_ecm"]
        with self.assertRaisesRegex(ValueError, "regressor mismatch"):
            reconstruct_wealth_draws(fit, self.panel, parameter_table(fit).iloc[1:],
                                    "log_ecm", "1990Q1", "2026Q2", 1)
        with self.assertRaisesRegex(ValueError, "Expected 2"):
            reconstruct_wealth_draws(fit, self.panel, parameter_table(fit),
                                    "log_ecm", "1990Q1", "2026Q2", 2)

    def test_changed_estimation_cannot_reuse_incompatible_saved_bootstrap(self):
        config = json.loads(json.dumps(self.config))
        config["model"]["start"] = "1995Q1"
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "config.json"
            path.write_text(json.dumps(config))
            with self.assertRaisesRegex(ValueError, "require --rebuild-bootstrap"):
                build_ecm(Path(folder) / "results", path)

    def test_ratio_target_does_not_clip_nonpositive_draws(self):
        fit = self.fits["ratio_ecm"]
        parameters = parameter_table(fit)
        parameters.loc[(parameters.stage == "long_run") & (parameters.regressor == "const"), "estimate"] = -100.
        with self.assertRaisesRegex(ValueError, "Nonpositive"):
            reconstruct_wealth_draws(fit, self.panel, parameters, "ratio_ecm", "1990Q1", "2026Q2", 1)

    def test_calendar_and_h1_aggregation_recovers_observed_growth(self):
        dates = pd.period_range("1990Q1", "2026Q2", freq="Q")
        real = self.accounting.real_gdp_saar_bn
        growth = 100 * np.expm1(4 * np.log(real / real.shift()))
        table = pd.DataFrame({"quarter": dates.astype(str), "model": "all_gdp",
                              "gross_gdp_growth_pp": growth.reindex(dates).to_numpy()})
        periods = temporal_summary(table, real).set_index("period")
        np.testing.assert_allclose(periods.calendar_gross_equivalent_pp.dropna(),
                                   periods.calendar_observed_growth_pct.dropna(), atol=1e-10)
        self.assertEqual(periods.loc["2026 H1", "quarters"], 2)
        self.assertIn("H1 over", periods.loc["2026 H1", "calendar_measure"])

    def test_quantity_conversion_uses_actual_quarter_flows(self):
        factor = consumption_change_factor(self.panel)
        np.testing.assert_allclose((factor * self.panel.dlogC).iloc[1:],
                                   self.panel.C_real_quarter_bn.diff().iloc[1:], atol=2e-10)

    def test_public_copy_imports_and_runs_without_original_repository(self):
        files = ["wealth_effects/ecm_gdp.py", "wealth_effects/ecm_publication.py",
            "wealth_effects/note454_ecm.py", "wealth_effects/note454_ratio.py",
            "wealth_effects/note454_ratio_ecm.py", "wealth_effects/note454_gdp_draws.py",
            "data/quarterly_panel.csv", "data/gdp_accounting.csv", "data/bootstrap_parameters.csv", "config/ecm.json"]
        with tempfile.TemporaryDirectory() as folder:
            public = Path(folder)
            for name in files:
                target = public / name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(ROOT / name, target)
            (public / "wealth_effects/__init__.py").write_text("")
            code = """from pathlib import Path
import pandas as pd
from wealth_effects.ecm_publication import ROOT
from wealth_effects.ecm_gdp import bridge_channels
from wealth_effects.note454_ecm import fit_ecm, decompose_ecm
assert ROOT == Path.cwd()
p = pd.read_csv('data/quarterly_panel.csv', index_col='quarter'); p.index = pd.PeriodIndex(p.index, freq='Q')
a = pd.read_csv('data/gdp_accounting.csv', index_col='quarter'); a.index = pd.PeriodIndex(a.index, freq='Q')
f = fit_ecm(p, compute_diagnostics=False)
r = bridge_channels(p, decompose_ecm(f,p,end='2026Q2'), a)
assert r.quarter.max() == '2026Q2'
import sys
assert 'wealth_effects.legacy' not in sys.modules
"""
            env = {**os.environ, "PYTHONPATH": str(public), "WEALTH_DATA_REPOSITORY": "/nonexistent/source/repo"}
            run = subprocess.run([sys.executable, "-c", code], cwd=public, env=env,
                capture_output=True, text=True, timeout=45)
            self.assertEqual(run.returncode, 0, run.stderr)

    def test_showcase_manifest_has_relative_paths_and_valid_hashes(self):
        path = ROOT / "results/run_manifest.json"
        if not path.exists():
            self.skipTest("Run python scripts/run_ecm.py to validate generated artifacts")
        manifest = json.loads(path.read_text())
        for field in ("input_sha256", "source_sha256", "configuration_sha256"):
            for name, expected in manifest[field].items():
                self.assertFalse(Path(name).is_absolute())
                self.assertEqual(hashlib.sha256((ROOT / name).read_bytes()).hexdigest(), expected)
        for name, expected in manifest["output_sha256"].items():
            self.assertFalse(Path(name).is_absolute())
            self.assertEqual(hashlib.sha256((ROOT / "results" / name).read_bytes()).hexdigest(), expected)
