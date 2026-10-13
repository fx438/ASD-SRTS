# -*- coding: utf-8 -*-
"""不依赖真实数据的快速回归测试。"""
import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold, train_test_split

from model_revision import RevisionBrainNetMLP
from run_revision import fisher_z_offdiag, build_splits, matrix_function, tangent_transform


def test_standard_five_fold():
    labels = np.array([0] * 443 + [1] * 417, dtype=np.int64)
    splits = build_splits(labels, seed=42)
    seen = np.zeros(len(labels), dtype=np.int64)
    for item in splits:
        tr, va, te = map(np.asarray, (item['train'], item['val'], item['test']))
        assert (len(tr), len(va), len(te)) == (602, 86, 172)
        assert not set(tr) & set(va)
        assert not set(tr) & set(te)
        assert not set(va) & set(te)
        seen[te] += 1
    assert np.all(seen == 1)


def test_fisher_diagonal_and_symmetry():
    c = torch.tensor([[[1.0, 0.5], [0.5, 1.0]]])
    z = fisher_z_offdiag(c)
    assert torch.allclose(torch.diagonal(z, dim1=-2, dim2=-1), torch.zeros(1, 2))
    assert torch.allclose(z, z.transpose(-1, -2))
    assert torch.allclose(z[0, 0, 1], torch.atanh(torch.tensor(0.5)))


def test_tangent_and_model_shapes():
    torch.manual_seed(1)
    a = torch.randn(10, 6, 20)
    a = (a - a.mean(-1, keepdim=True)) / (a.std(-1, keepdim=True, unbiased=False) + 1e-6)
    c = a @ a.transpose(-1, -2) / a.shape[-1]
    eye = torch.eye(6)
    spd = 0.8 * c + 0.2 * eye
    tan, diag = tangent_transform(spd, np.arange(6), np.arange(10), diag_positions={6, 7})
    assert tan.shape == (10, 6, 6)
    assert torch.isfinite(tan).all()
    assert len(diag) == 2

    model = RevisionBrainNetMLP(n_roi=6, hidden=4, k=5)
    logits = model(a, tan)
    assert logits.shape == (10, 2)
    spatial, spectral = model.encode_branches(tan, a)
    lam = torch.full((10, 1), 0.7)
    perm = torch.arange(9, -1, -1)
    mixed = model.classify_embeddings(
        lam * spatial + (1 - lam) * spatial[perm],
        lam * spectral + (1 - lam) * spectral[perm],
    )
    assert mixed.shape == (10, 2)


if __name__ == '__main__':
    test_standard_five_fold()
    test_fisher_diagonal_and_symmetry()
    test_tangent_and_model_shapes()
    print('all revision tests passed')
