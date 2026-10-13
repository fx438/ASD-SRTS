# -*- coding: utf-8 -*-
"""SRTS 切线空间 ROI 交互编码增强的 BrainNetMLP。

保留 2025 BrainNetMLP 的空间投影、频谱投影和分类头；新增单层多头自注意力，
将切线矩阵的每一行视为一个 ROI 连接轮廓，并把全局 ROI 交互上下文以零初始化
残差加入空间表征。该模块不包含旧 SSJR 的逐 ROI 标量门控或频段重加权。
"""
import torch
import torch.nn as nn


class ROIInteractionBrainNetMLP(nn.Module):
    """带轻量切线空间 ROI 交互残差的 BrainNetMLP。"""

    def __init__(self, n_roi=392, hidden=32, drop=0.0, k=40, num_classes=2):
        super().__init__()
        rows, cols = torch.triu_indices(n_roi, n_roi, offset=0)
        self.register_buffer('rows', rows)
        self.register_buffer('cols', cols)
        self.k = k

        # 创建顺序与原 BrainNetMLP 完全一致，以锁定共同参数的随机初始化。
        self.freq_proj = nn.Sequential(nn.Linear(n_roi, hidden), nn.Dropout(drop))
        self.spatial_proj = nn.Sequential(nn.Linear(rows.numel(), hidden), nn.Dropout(drop))
        self.decoder = nn.Sequential(nn.GELU(), nn.Linear(hidden * 2, num_classes))

        self.roi_projection = nn.Linear(n_roi, hidden)
        self.roi_identity = nn.Parameter(torch.empty(1, n_roi, hidden))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=4,
            dim_feedforward=hidden * 2,
            dropout=0.10,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.roi_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=1,
            enable_nested_tensor=False,
        )
        self.roi_context_norm = nn.LayerNorm(hidden)
        self.roi_residual = nn.Linear(hidden, hidden)

        nn.init.normal_(self.roi_identity, mean=0.0, std=0.02)
        nn.init.zeros_(self.roi_residual.weight)
        nn.init.zeros_(self.roi_residual.bias)

    def encode_branches(self, fc, ts_roi_time):
        spatial = self.spatial_proj(fc[:, self.rows, self.cols])

        roi_tokens = self.roi_projection(fc) + self.roi_identity
        roi_tokens = self.roi_encoder(roi_tokens)
        roi_context = self.roi_context_norm(roi_tokens.mean(dim=1))
        spatial = spatial + self.roi_residual(roi_context)

        ts_time_roi = ts_roi_time.transpose(1, 2)
        spectrum = torch.abs(torch.fft.rfft(ts_time_roi, dim=1))[:, 1:self.k, :]
        spectral = self.freq_proj(spectrum).mean(1)
        return spatial, spectral

    def classify_embeddings(self, spatial, spectral):
        return self.decoder(torch.cat((spatial, spectral), dim=1))

    def forward(self, ts_roi_time, fc):
        spatial, spectral = self.encode_branches(fc, ts_roi_time)
        return self.classify_embeddings(spatial, spectral)
