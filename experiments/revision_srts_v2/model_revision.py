# -*- coding: utf-8 -*-
"""修订实验的纯净 BrainNetMLP 骨干。

旧 SSJR 的 ROI 门控与四段频谱权重不在本文件中。空间表示由训练脚本预先计算，
因此原始 FC、各切线空间分支在训练时使用完全一致的干净空间输入；高斯噪声只作用
于所有方法共有的频谱分支。新的模块二是训练期双分支潜表征 Mixup，不增加参数。
"""
import torch
import torch.nn as nn


class RevisionBrainNetMLP(nn.Module):
    def __init__(self, n_roi=392, hidden=32, drop=0.0, k=40, num_classes=2):
        super().__init__()
        rows, cols = torch.triu_indices(n_roi, n_roi, offset=0)
        self.register_buffer('rows', rows)
        self.register_buffer('cols', cols)
        self.k = k
        self.freq_proj = nn.Sequential(nn.Linear(n_roi, hidden), nn.Dropout(drop))
        self.spatial_proj = nn.Sequential(nn.Linear(rows.numel(), hidden), nn.Dropout(drop))
        self.decoder = nn.Sequential(nn.GELU(), nn.Linear(hidden * 2, num_classes))

    def encode_branches(self, fc, ts_roi_time):
        """返回空间和频谱两个 32 维潜表征，供普通训练或一致 Mixup 使用。"""
        spatial = self.spatial_proj(fc[:, self.rows, self.cols])
        ts_time_roi = ts_roi_time.transpose(1, 2)
        spectrum = torch.abs(torch.fft.rfft(ts_time_roi, dim=1))[:, 1:self.k, :]
        spectral = self.freq_proj(spectrum).mean(1)
        return spatial, spectral

    def classify_embeddings(self, spatial, spectral):
        return self.decoder(torch.cat((spatial, spectral), dim=1))

    def forward(self, ts_roi_time, fc):
        spatial, spectral = self.encode_branches(fc, ts_roi_time)
        return self.classify_embeddings(spatial, spectral)
