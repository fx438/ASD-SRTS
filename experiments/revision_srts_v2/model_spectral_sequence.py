# -*- coding: utf-8 -*-
"""带深度频谱序列编码模块的 2025 BrainNetMLP。

保留原空间分支和分类头。频谱分支在原逐 bin ROI 投影后增加深度可分离一维卷积，
并使用样本自适应注意力汇聚 39 个 bin。卷积残差末层和注意力打分均零初始化，
因此模型初始函数严格退化为原 BrainNetMLP 的简单均值汇聚。
"""
import torch
import torch.nn as nn


class SpectralSequenceBrainNetMLP(nn.Module):
    """残差频谱卷积与自适应注意力增强的 BrainNetMLP。"""

    def __init__(self, n_roi=392, hidden=32, drop=0.0, k=40, num_classes=2):
        super().__init__()
        rows, cols = torch.triu_indices(n_roi, n_roi, offset=0)
        self.register_buffer('rows', rows)
        self.register_buffer('cols', cols)
        self.k = k

        # 与原 BrainNetMLP 相同的共同层及创建顺序。
        self.freq_proj = nn.Sequential(nn.Linear(n_roi, hidden), nn.Dropout(drop))
        self.spatial_proj = nn.Sequential(nn.Linear(rows.numel(), hidden), nn.Dropout(drop))
        self.decoder = nn.Sequential(nn.GELU(), nn.Linear(hidden * 2, num_classes))

        self.spectral_depthwise = nn.Conv1d(
            hidden, hidden, kernel_size=3, padding=1, groups=hidden
        )
        self.spectral_pointwise = nn.Conv1d(hidden, hidden, kernel_size=1)
        self.spectral_activation = nn.GELU()
        self.spectral_norm = nn.LayerNorm(hidden)
        self.spectral_score = nn.Linear(hidden, 1, bias=False)

        # 初始时局部残差为零、注意力为均匀分布，精确复现基线均值汇聚。
        nn.init.zeros_(self.spectral_pointwise.weight)
        nn.init.zeros_(self.spectral_pointwise.bias)
        nn.init.zeros_(self.spectral_score.weight)

    def encode_branches(self, fc, ts_roi_time):
        spatial = self.spatial_proj(fc[:, self.rows, self.cols])

        ts_time_roi = ts_roi_time.transpose(1, 2)
        spectrum = torch.abs(torch.fft.rfft(ts_time_roi, dim=1))[:, 1:self.k, :]
        bins = self.freq_proj(spectrum)
        local = self.spectral_depthwise(bins.transpose(1, 2))
        local = self.spectral_pointwise(self.spectral_activation(local)).transpose(1, 2)
        bins = bins + local
        weights = torch.softmax(self.spectral_score(self.spectral_norm(bins)), dim=1)
        spectral = (weights * bins).sum(dim=1)
        return spatial, spectral

    def classify_embeddings(self, spatial, spectral):
        return self.decoder(torch.cat((spatial, spectral), dim=1))

    def forward(self, ts_roi_time, fc):
        spatial, spectral = self.encode_branches(fc, ts_roi_time)
        return self.classify_embeddings(spatial, spectral)
