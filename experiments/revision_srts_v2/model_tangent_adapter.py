# -*- coding: utf-8 -*-
"""带轻量切线潜表征残差适配器的 2025 BrainNetMLP。"""
import torch
import torch.nn as nn


class TangentAdapterBrainNetMLP(nn.Module):
    """以低秩残差 MLP 非线性修正 SRTS 空间潜表征。"""

    def __init__(self, n_roi=392, hidden=32, drop=0.0, k=40, num_classes=2):
        super().__init__()
        rows, cols = torch.triu_indices(n_roi, n_roi, offset=0)
        self.register_buffer('rows', rows)
        self.register_buffer('cols', cols)
        self.k = k

        # 共同层及创建顺序与原 BrainNetMLP 完全一致。
        self.freq_proj = nn.Sequential(nn.Linear(n_roi, hidden), nn.Dropout(drop))
        self.spatial_proj = nn.Sequential(nn.Linear(rows.numel(), hidden), nn.Dropout(drop))
        self.decoder = nn.Sequential(nn.GELU(), nn.Linear(hidden * 2, num_classes))

        bottleneck = max(4, hidden // 4)
        self.tangent_adapter = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, bottleneck),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(bottleneck, hidden),
        )
        nn.init.zeros_(self.tangent_adapter[-1].weight)
        nn.init.zeros_(self.tangent_adapter[-1].bias)

    def encode_branches(self, fc, ts_roi_time):
        spatial = self.spatial_proj(fc[:, self.rows, self.cols])
        spatial = spatial + self.tangent_adapter(spatial)

        ts_time_roi = ts_roi_time.transpose(1, 2)
        spectrum = torch.abs(torch.fft.rfft(ts_time_roi, dim=1))[:, 1:self.k, :]
        spectral = self.freq_proj(spectrum).mean(1)
        return spatial, spectral

    def classify_embeddings(self, spatial, spectral):
        return self.decoder(torch.cat((spatial, spectral), dim=1))

    def forward(self, ts_roi_time, fc):
        spatial, spectral = self.encode_branches(fc, ts_roi_time)
        return self.classify_embeddings(spatial, spectral)
