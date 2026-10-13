# -*- coding: utf-8 -*-
"""Paired bootstrap on pooled out-of-fold predictions (Section 2.9).

Participants are resampled within each fold and class; the same resample is applied to both
score vectors, so the interval is for the paired AUC difference.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score

N_BOOT = 20000
BOOT_SEED = 20260824


def paired_bootstrap_auc(y, fold, score_a, score_b, n_boot=N_BOOT, seed=BOOT_SEED):
    """AUC(score_a) - AUC(score_b) with a percentile 95% CI."""
    y, fold = np.asarray(y), np.asarray(fold)
    a, b = np.asarray(score_a), np.asarray(score_b)
    rng = np.random.default_rng(seed)
    groups = [np.where((fold == k) & (y == c))[0] for k in np.unique(fold) for c in (0, 1)]
    deltas = np.empty(n_boot)
    for i in range(n_boot):
        idx = np.concatenate([g[rng.integers(0, len(g), len(g))] for g in groups])
        deltas[i] = roc_auc_score(y[idx], a[idx]) - roc_auc_score(y[idx], b[idx])
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return {'delta_auc': float(roc_auc_score(y, a) - roc_auc_score(y, b)),
            'ci95': (float(lo), float(hi)), 'p_le_0': float((deltas <= 0).mean())}
