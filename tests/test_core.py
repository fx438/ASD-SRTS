# -*- coding: utf-8 -*-
"""Checks for the reference implementation in srts/.

Run with:  python -m pytest tests
The equivalence test also needs PyTorch, because it compares against the implementation used
for the experiments (experiments/revision_srts_v2/run_revision.py).
"""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from srts import SRTS, correlation_matrices, energy_shares, paired_bootstrap_auc, shrink, svec  # noqa: E402

N_ROI, T = 30, 12          # more regions than time points: rank-deficient, as in the paper


@pytest.fixture(scope='module')
def series():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((40, N_ROI, T))
    return (x - x.mean(-1, keepdims=True)) / x.std(-1, keepdims=True)


def test_correlation_is_rank_deficient(series):
    c = correlation_matrices(series)
    assert np.allclose(np.diagonal(c, axis1=1, axis2=2), 1.0, atol=1e-5)
    assert np.linalg.matrix_rank(c[0], tol=1e-8) <= T - 1
    assert np.linalg.eigvalsh(shrink(c[0], 0.5)).min() >= 0.5 - 1e-10


def test_svec_is_isometric():
    rng = np.random.default_rng(1)
    a = rng.standard_normal((N_ROI, N_ROI))
    a = a + a.T
    assert np.isclose(np.dot(svec(a), svec(a)), np.sum(a * a))


def test_reference_maps_to_zero(series):
    c = correlation_matrices(series)
    model = SRTS(gamma=0.9, vectorize=False).fit(c[:30])
    whitened_ref = model.reference_isqrt_ @ model.reference_ @ model.reference_isqrt_
    assert np.allclose(whitened_ref, np.eye(N_ROI), atol=1e-8)      # log(I) = 0 at the reference
    assert np.abs(model.transform(c[:30]).mean(0)).max() < 0.5      # training tangents centre near 0


def test_null_space_energy_is_lower_with_strong_shrinkage(series):
    c = correlation_matrices(series)
    weak = SRTS(gamma=0.05, vectorize=False).fit(c[:30]).transform(c[30:])
    strong = SRTS(gamma=0.9, vectorize=False).fit(c[:30]).transform(c[30:])
    e_weak = np.mean([energy_shares(ci, ti)[0] for ci, ti in zip(c[30:], weak)])
    e_strong = np.mean([energy_shares(ci, ti)[0] for ci, ti in zip(c[30:], strong)])
    assert e_strong < e_weak


def test_matches_experiment_implementation(series):
    torch = pytest.importorskip('torch')
    sys.path.insert(0, str(ROOT / 'experiments' / 'revision_srts_v2'))
    import run_revision as rr

    c = correlation_matrices(series)
    train, target = np.arange(30), np.arange(40)
    ref = SRTS(gamma=0.7, vectorize=False).fit(c[train]).transform(c[target])
    base = rr.shrink(torch.from_numpy(c.astype(np.float32)), 0.7)
    exp, _ = rr.tangent_transform(base, train, target)
    assert np.allclose(ref, exp.numpy(), atol=1e-4)


def test_paired_bootstrap_sign():
    rng = np.random.default_rng(2)
    y = np.repeat([0, 1], 100)
    fold = np.tile(np.arange(5), 40)
    good = y + rng.normal(0, 0.8, 200)
    bad = y + rng.normal(0, 3.0, 200)
    res = paired_bootstrap_auc(y, fold, good, bad, n_boot=500)
    assert res['delta_auc'] > 0 and res['ci95'][0] > 0
