# -*- coding: utf-8 -*-
"""Shrinkage-regularized tangent-space (SRTS) connectivity.

Reference NumPy implementation of Section 2.3 of the paper:

    C_i            Pearson correlation of the ROI time series (rank <= T - 1)
    S_i  = (1 - gamma) C_i + gamma I                              linear shrinkage, Eq. (4)
    R    = mean of S_i over the training participants              training-fold reference
    T_i  = log(R^{-1/2} S_i R^{-1/2})                              whitening + matrix logarithm
    x_i  = svec(T_i) = [diag(T_i), sqrt(2) * triu(T_i, k=1)]       isometric vectorization

The experiment scripts in ``experiments/`` use an equivalent PyTorch implementation
(``experiments/revision_srts_v2/run_revision.py``); ``tests/test_core.py`` checks that both agree.
"""
from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

GAMMA_GRID = (0.05, 0.10, 0.20, 0.30, 0.50, 0.70, 0.80, 0.90)
EIG_FLOOR = 1e-12


def correlation_matrices(time_series):
    """Pearson correlation matrices from ROI time series.

    time_series: array (n_subjects, n_roi, n_timepoints), each ROI already z-scored over time
    (as in the paper), or any real array, in which case it is standardized here.
    """
    x = np.asarray(time_series, dtype=np.float64)
    x = x - x.mean(-1, keepdims=True)
    x = x / (x.std(-1, keepdims=True) + 1e-6)
    c = x @ np.swapaxes(x, -1, -2) / x.shape[-1]
    c = 0.5 * (c + np.swapaxes(c, -1, -2))
    return np.clip(c, -1.0, 1.0)


def shrink(c, gamma):
    """Linear shrinkage toward the identity: (1 - gamma) C + gamma I."""
    if not 0.0 < gamma <= 1.0:
        raise ValueError('gamma must lie in (0, 1]')
    c = np.asarray(c, dtype=np.float64)
    return (1.0 - gamma) * c + gamma * np.eye(c.shape[-1])


def _sym_function(m, fn, floor=EIG_FLOOR):
    m = 0.5 * (m + np.swapaxes(m, -1, -2))
    w, v = np.linalg.eigh(m)
    return (v * fn(np.maximum(w, floor))[..., None, :]) @ np.swapaxes(v, -1, -2)


def svec(t):
    """Isometric vectorization: diagonal followed by sqrt(2) times the strict upper triangle."""
    t = np.asarray(t)
    n = t.shape[-1]
    rows, cols = np.triu_indices(n, k=1)
    diag = np.diagonal(t, axis1=-2, axis2=-1)
    return np.concatenate([diag, np.sqrt(2.0) * t[..., rows, cols]], axis=-1)


class SRTS(TransformerMixin, BaseEstimator):
    """Shrinkage-regularized tangent-space representation (scikit-learn style).

    Fit on training participants only; ``transform`` then maps any participant to the tangent
    space at the training reference. Input: correlation matrices (n, N, N) or, with
    ``from_time_series=True``, time series (n, N, T).
    """

    def __init__(self, gamma=0.9, vectorize=True, from_time_series=False):
        self.gamma = gamma
        self.vectorize = vectorize
        self.from_time_series = from_time_series

    def _shrunk(self, x):
        c = correlation_matrices(x) if self.from_time_series else np.asarray(x, dtype=np.float64)
        return shrink(c, self.gamma)

    def fit(self, x, y=None):
        reference = self._shrunk(x).mean(0)
        self.reference_ = reference
        self.reference_isqrt_ = _sym_function(reference, lambda w: w ** -0.5)
        return self

    def transform(self, x):
        s = self._shrunk(x)
        white = self.reference_isqrt_ @ s @ self.reference_isqrt_
        tangent = _sym_function(white, np.log)
        return svec(tangent) if self.vectorize else tangent

    def fit_transform(self, x, y=None):
        return self.fit(x).transform(x)
