# -*- coding: utf-8 -*-
"""γ–AIRM 分析的数学与协议测试。"""
from __future__ import annotations

import unittest

import numpy as np
import torch

import analyze_gamma_airm_v1 as analysis


class GammaAirmTests(unittest.TestCase):
    def test_frobenius_norm_of_tangent_equals_airm_distance(self):
        tangent = torch.tensor(
            [
                [[3.0, 4.0], [4.0, 0.0]],
                [[1.0, 0.0], [0.0, -2.0]],
            ],
            dtype=torch.float64,
        )
        actual = analysis.frobenius_distances(tangent)
        expected = np.asarray([np.sqrt(41.0), np.sqrt(5.0)])
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)

    def test_distance_summary_uses_subject_values(self):
        actual = analysis.summarize_distances([1.0, 2.0, 3.0, 4.0])
        self.assertEqual(actual["n"], 4)
        self.assertEqual(actual["mean"], 2.5)
        self.assertAlmostEqual(actual["sample_sd"], np.std([1, 2, 3, 4], ddof=1))
        self.assertEqual(actual["median"], 2.5)
        self.assertEqual(actual["q25"], 1.75)
        self.assertEqual(actual["q75"], 3.25)

    def test_protocol_uses_validation_only_and_marks_095_as_audit(self):
        protocol = analysis.protocol_config()
        self.assertEqual(
            protocol["core_gamma_grid"],
            [0.05, 0.10, 0.20, 0.30, 0.50, 0.70, 0.80, 0.90],
        )
        self.assertEqual(protocol["audit_gamma"], 0.95)
        self.assertEqual(protocol["target_partition"], "validation only")
        self.assertEqual(protocol["reference_fit_partition"], "training only")
        self.assertFalse(protocol["uses_test_predictions_or_metrics"])
        self.assertFalse(protocol["used_to_change_locked_gamma_selection"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
