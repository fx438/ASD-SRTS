# -*- coding: utf-8 -*-
"""SRTS 后的双分支残差交互融合候选模块。

该文件只定义一个预先固定的候选结构。空间和频谱编码器、基础分类头与
RevisionBrainNetMLP 保持一致；新增模块根据两分支的乘积与绝对差构造交互项，
再以门控残差方式修正原始拼接表征。残差末层零初始化，使模型初始函数与基础
BrainNetMLP 完全一致，避免仅由初始化变化造成不公平比较。
"""
import torch
import torch.nn as nn


class CrossFusionBrainNetMLP(nn.Module):
    """带双分支残差交互融合的 BrainNetMLP。"""

    def __init__(self, n_roi=392, hidden=32, drop=0.0, k=40, num_classes=2):
        super().__init__()
        rows, cols = torch.triu_indices(n_roi, n_roi, offset=0)
        self.register_buffer('rows', rows)
        self.register_buffer('cols', cols)
        self.k = k

        # 以下三层的创建顺序与 RevisionBrainNetMLP 一致，以锁定共同参数初始化。
        self.freq_proj = nn.Sequential(nn.Linear(n_roi, hidden), nn.Dropout(drop))
        self.spatial_proj = nn.Sequential(nn.Linear(rows.numel(), hidden), nn.Dropout(drop))
        self.decoder = nn.Sequential(nn.GELU(), nn.Linear(hidden * 2, num_classes))

        self.spatial_norm = nn.LayerNorm(hidden)
        self.spectral_norm = nn.LayerNorm(hidden)
        self.cross_mlp = nn.Sequential(
            nn.Linear(hidden * 2, hidden * 2),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(hidden * 2, hidden * 2),
        )
        self.residual_gate = nn.Sequential(
            nn.Linear(hidden * 4, hidden * 2),
            nn.Sigmoid(),
        )

        # 候选模型从基础模型的函数出发，训练后才允许交互残差偏离零。
        nn.init.zeros_(self.cross_mlp[-1].weight)
        nn.init.zeros_(self.cross_mlp[-1].bias)

    def encode_branches(self, fc, ts_roi_time):
        spatial = self.spatial_proj(fc[:, self.rows, self.cols])
        ts_time_roi = ts_roi_time.transpose(1, 2)
        spectrum = torch.abs(torch.fft.rfft(ts_time_roi, dim=1))[:, 1:self.k, :]
        spectral = self.freq_proj(spectrum).mean(1)
        return spatial, spectral

    def fuse_embeddings(self, spatial, spectral):
        spatial_unit = self.spatial_norm(spatial)
        spectral_unit = self.spectral_norm(spectral)
        cross = torch.cat(
            (spatial_unit * spectral_unit, torch.abs(spatial_unit - spectral_unit)),
            dim=1,
        )
        delta = self.cross_mlp(cross)
        gate = self.residual_gate(torch.cat((spatial_unit, spectral_unit, cross), dim=1))
        base = torch.cat((spatial, spectral), dim=1)
        return base + gate * delta

    def classify_embeddings(self, spatial, spectral):
        return self.decoder(self.fuse_embeddings(spatial, spectral))

    def forward(self, ts_roi_time, fc):
        spatial, spectral = self.encode_branches(fc, ts_roi_time)
        return self.classify_embeddings(spatial, spectral)
