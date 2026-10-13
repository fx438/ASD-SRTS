# -*- coding: utf-8 -*-
"""定位原始 FC svec 对角不等于 1 的数据来源，不修改任何结果。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

SOURCE_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = SOURCE_ROOT / 'revision_srts_v2'
RESULT_ROOT = (
    SOURCE_ROOT / 'results_revision_v2' / 'raw_fc_full_lr_standard5_seed42_v1'
)
sys.path.insert(0, str(SOURCE_ROOT))
import dataset_cc400 as dataset  # noqa: E402


def main():
    time_series, labels, sites = dataset.load_cc400(t_use=146, verbose=False)
    features = np.load(
        RESULT_ROOT / 'features' / 'raw_fc_full_svec_float32.npy',
        mmap_mode='r', allow_pickle=False,
    )
    n_roi, n_time = time_series.shape[1:]
    cached_diagonal = np.asarray(features[:, :n_roi], dtype=np.float64)
    direct_diagonal = np.mean(
        np.asarray(time_series, dtype=np.float64) ** 2, axis=2
    )
    standard_deviation = np.std(
        np.asarray(time_series, dtype=np.float64), axis=2, ddof=0
    )
    means = np.mean(np.asarray(time_series, dtype=np.float64), axis=2)
    bad = np.argwhere(np.abs(cached_diagonal - 1.0) > 1e-5)
    zeros = np.argwhere(cached_diagonal == 0.0)
    report = {
        'shape': list(time_series.shape),
        'cached_diagonal_min': float(cached_diagonal.min()),
        'cached_diagonal_max': float(cached_diagonal.max()),
        'cached_diagonal_max_abs_error_from_direct_second_moment': float(
            np.max(np.abs(cached_diagonal - direct_diagonal))
        ),
        'time_series_mean_max_abs': float(np.max(np.abs(means))),
        'standardized_time_series_sd_min': float(standard_deviation.min()),
        'standardized_time_series_sd_max': float(standard_deviation.max()),
        'diagonal_abs_error_gt_1e_5_count': int(len(bad)),
        'exact_zero_diagonal_count': int(len(zeros)),
        'affected_subject_count': int(len(np.unique(bad[:, 0]))) if len(bad) else 0,
        'affected_roi_count': int(len(np.unique(bad[:, 1]))) if len(bad) else 0,
        'first_50_bad_subject_roi': bad[:50].tolist(),
        'first_50_zero_subject_roi': zeros[:50].tolist(),
        'zero_locations_time_series_min_max_sd': [
            {
                'subject_index': int(subject),
                'roi_index': int(roi),
                'minimum': float(time_series[subject, roi].min()),
                'maximum': float(time_series[subject, roi].max()),
                'mean': float(time_series[subject, roi].mean()),
                'sd_ddof0': float(time_series[subject, roi].std(ddof=0)),
            }
            for subject, roi in zeros[:50]
        ],
        'interpretation': (
            'dataset_cc400 uses (a-mean)/(std_ddof0+1e-6); a per-subject '
            'constant ROI therefore becomes all zeros and has FC diagonal zero'
        ),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
