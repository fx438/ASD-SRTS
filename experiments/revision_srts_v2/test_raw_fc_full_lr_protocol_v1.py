# -*- coding: utf-8 -*-
"""原始 FC 全维 + LR 对齐实验的协议单元测试。"""
from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

import run_raw_fc_full_lr_standard5 as experiment


class RawFcFullProtocolTests(unittest.TestCase):
    def test_raw_fc_svec_uses_diagonal_then_sqrt2_upper(self):
        time_series = np.asarray(
            [
                [
                    [1.0, -1.0, 1.0, -1.0],
                    [1.0, 1.0, -1.0, -1.0],
                    [0.5, -0.5, -0.5, 0.5],
                ],
                [
                    [2.0, 0.0, -2.0, 0.0],
                    [0.0, 2.0, 0.0, -2.0],
                    [1.0, 1.0, -1.0, -1.0],
                ],
            ],
            dtype=np.float32,
        )
        actual = experiment.raw_fc_svec(
            time_series, batch_size=1, device="cpu"
        )
        expected = []
        rows, cols = np.triu_indices(3, k=1)
        for subject in time_series:
            correlation = subject @ subject.T / subject.shape[1]
            expected.append(np.concatenate([
                np.diag(correlation),
                correlation[rows, cols] * math.sqrt(2.0),
            ]))
        np.testing.assert_allclose(
            actual,
            np.asarray(expected, dtype=np.float32),
            rtol=1e-6,
            atol=1e-6,
        )
        self.assertEqual(actual.shape, (2, 6))
        self.assertEqual(actual.dtype, np.float32)

    def test_cc400_svec_dimension_is_77028(self):
        rows, cols = experiment.svec_indices(392)
        self.assertEqual(len(rows), 392 * 391 // 2)
        self.assertEqual(len(cols), 392 * 391 // 2)
        self.assertEqual(experiment.svec_dimension(392), 77028)
        self.assertTrue(np.all(rows < cols))

    def test_lr_matches_locked_srts_lr_hyperparameters(self):
        parameters = experiment.make_model().get_params()
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

    def test_protocol_changes_only_representation(self):
        config = experiment.protocol_config(Path("/tmp/test-output"))
        self.assertEqual(config["seed"], 42)
        self.assertEqual(config["per_fold_sizes"], [602, 86, 172])
        self.assertEqual(config["feature_dim"], 77028)
        self.assertIsNone(config["feature_selection"])
        self.assertEqual(config["fisher_z"], False)
        self.assertEqual(config["standardization"], "StandardScaler, training fold only")
        self.assertEqual(
            config["model_hyperparameters"],
            {
                "penalty": "l2",
                "C": 1.0,
                "solver": "liblinear",
                "dual": True,
                "max_iter": 5000,
                "random_state": 42,
            },
        )
        self.assertFalse(config["classifier_specific_tuning"])

    def test_save_npy_new_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "features.npy"
            experiment.save_npy_new(path, np.zeros((2, 3), dtype=np.float32))
            with self.assertRaises(FileExistsError):
                experiment.save_npy_new(
                    path, np.ones((2, 3), dtype=np.float32)
                )
            np.testing.assert_array_equal(
                np.load(path, allow_pickle=False),
                np.zeros((2, 3), dtype=np.float32),
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
