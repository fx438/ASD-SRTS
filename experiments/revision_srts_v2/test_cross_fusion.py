# -*- coding: utf-8 -*-
"""残差双分支交互融合候选模块的局部测试。"""
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from model_cross_fusion import CrossFusionBrainNetMLP
from model_revision import RevisionBrainNetMLP


def test_initial_function_matches_base():
    n_roi, hidden, k = 12, 8, 6
    ts = torch.randn(4, n_roi, 20)
    fc = torch.randn(4, n_roi, n_roi)

    torch.manual_seed(123)
    base = RevisionBrainNetMLP(n_roi=n_roi, hidden=hidden, drop=0.0, k=k)
    torch.manual_seed(123)
    candidate = CrossFusionBrainNetMLP(n_roi=n_roi, hidden=hidden, drop=0.0, k=k)
    base.eval()
    candidate.eval()

    with torch.no_grad():
        base_output = base(ts, fc)
        candidate_output = candidate(ts, fc)
    assert torch.equal(base.freq_proj[0].weight, candidate.freq_proj[0].weight)
    assert torch.equal(base.spatial_proj[0].weight, candidate.spatial_proj[0].weight)
    assert torch.equal(base.decoder[1].weight, candidate.decoder[1].weight)
    assert torch.allclose(base_output, candidate_output, atol=1e-7, rtol=0.0)


def test_cross_fusion_receives_gradient():
    model = CrossFusionBrainNetMLP(n_roi=10, hidden=8, drop=0.0, k=5)
    with torch.no_grad():
        model.cross_mlp[-1].weight.normal_(0.0, 0.01)
    ts = torch.randn(5, 10, 18)
    fc = torch.randn(5, 10, 10)
    output = model(ts, fc)
    assert output.shape == (5, 2)
    output.square().mean().backward()
    assert model.cross_mlp[0].weight.grad is not None
    assert model.residual_gate[0].weight.grad is not None
    assert torch.isfinite(model.cross_mlp[0].weight.grad).all()
    assert torch.isfinite(model.residual_gate[0].weight.grad).all()


if __name__ == '__main__':
    test_initial_function_matches_base()
    test_cross_fusion_receives_gradient()
    print('all cross-fusion tests passed')
