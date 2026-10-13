# -*- coding: utf-8 -*-
"""固定 γ=0.70 的 SRTS + LR 稳健性实验协议测试。"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import run_srts_lr_fixed_gamma070_standard5 as experiment


class FixedGammaProtocolTests(unittest.TestCase):
    def test_all_five_folds_use_pre_specified_gamma_070(self):
        self.assertEqual(experiment.LOCKED_GAMMAS, (0.70,) * 5)
        config = experiment.protocol_config(
            Path("/tmp/fixed-gamma-output"),
            {"eval_batch_size": 64, "eig_batch_size": 64},
        )
        self.assertEqual(config["gammas"], [0.70] * 5)
        self.assertEqual(config["gamma_selection"], "pre-specified robustness analysis")
        self.assertFalse(config["gamma_selected_with_test_data"])

    def test_classifier_matches_locked_srts_lr(self):
        model = experiment.linear.make_model("lr")
        parameters = model.get_params()
        expected = {
            "penalty": "l2",
            "C": 1.0,
            "solver": "liblinear",
            "dual": True,
            "max_iter": 5000,
            "random_state": 42,
        }
        self.assertEqual(
            {key: parameters[key] for key in expected}, expected
        )
        config = experiment.protocol_config(
            Path("/tmp/fixed-gamma-output"),
            {"eval_batch_size": 64, "eig_batch_size": 64},
        )
        self.assertEqual(config["model_hyperparameters"], expected)
        self.assertFalse(config["classifier_specific_tuning"])

    def test_result_directory_is_independent_and_non_overwriting(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            summary_path = output / "models" / "lr" / "summary.json"
            summary_path.parent.mkdir(parents=True)
            summary_path.write_text("{}", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                experiment.assert_result_not_complete(output)


if __name__ == "__main__":
    unittest.main(verbosity=2)
