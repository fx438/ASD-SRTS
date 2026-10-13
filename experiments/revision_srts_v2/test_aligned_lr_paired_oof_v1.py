# -*- coding: utf-8 -*-
"""严格对齐 LR 表示比较的配对 OOF 统计协议测试。"""
from __future__ import annotations

import unittest

import numpy as np

import compare_aligned_lr_oof_v1 as comparison


class AlignedLrOofComparisonTests(unittest.TestCase):
    def test_align_oof_requires_identical_subject_label_and_fold(self):
        raw = [
            {"subject_index": 1, "label": 0, "fold": 0, "score": 0.2},
            {"subject_index": 0, "label": 1, "fold": 1, "score": 0.8},
        ]
        srts = [
            {"subject_index": 0, "label": 1, "fold": 1, "score": 0.9},
            {"subject_index": 1, "label": 0, "fold": 0, "score": 0.1},
        ]
        labels, folds, probabilities = comparison.align_oof_rows(raw, srts)
        np.testing.assert_array_equal(labels, [1, 0])
        np.testing.assert_array_equal(folds, [1, 0])
        np.testing.assert_allclose(probabilities["raw_fc"], [0.8, 0.2])
        np.testing.assert_allclose(probabilities["srts"], [0.9, 0.1])

        srts[0]["label"] = 0
        with self.assertRaises(AssertionError):
            comparison.align_oof_rows(raw, srts)

    def test_predictions_use_each_models_saved_validation_threshold(self):
        folds = [
            {
                "test_indices": [0, 2],
                "test": {"threshold": 0.6},
            },
            {
                "test_indices": [1, 3],
                "test": {"threshold": 0.4},
            },
        ]
        scores = np.asarray([0.7, 0.3, 0.5, 0.5])
        actual = comparison.predictions_from_folds(scores, folds, 4)
        np.testing.assert_array_equal(actual, [1, 0, 0, 1])

    def test_protocol_fixes_bootstrap_before_results(self):
        protocol = comparison.protocol_config(20000, 20260824)
        self.assertEqual(protocol["n_bootstrap"], 20000)
        self.assertEqual(protocol["seed"], 20260824)
        self.assertEqual(
            protocol["resampling"],
            "paired subject bootstrap stratified within fold and class",
        )
        self.assertIn("fixed five fitted fold models", protocol["estimand"])
        self.assertFalse(protocol["used_for_model_selection"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
