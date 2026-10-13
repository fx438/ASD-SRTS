"""为切线谱能量诊断补充 Ledoit-Wolf 一列，并合并进已有报告。

测量定义与 analyze_tangent_spectrum_v2.py 完全一致：第 1 折验证被试，
参考矩阵由同折 602 名训练被试估计，不涉及测试被试。
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault('OMP_NUM_THREADS', '8')
os.environ.setdefault('MKL_NUM_THREADS', '8')

import numpy as np
import torch
from sklearn.covariance import LedoitWolf

torch.set_num_threads(8)

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'results_revision_v2/standard5_seed42'
OUTPUT = ROOT / 'results_revision_v2/tangent_spectrum_diagnostic_v1'
NUMERIC_FLOOR = 1e-12
RANK_TOL = 1e-8
BATCH = 8


def summarize(values) -> dict:
    values = np.asarray(values, dtype=np.float64)
    return {
        'mean': float(values.mean()),
        'sample_sd': float(values.std(ddof=1)) if values.size > 1 else 0.0,
        'min': float(values.min()),
        'median': float(np.median(values)),
        'max': float(values.max()),
    }


def corr_batch(ts_block):
    block = ts_block.double()
    c = block @ block.transpose(-1, -2) / block.shape[-1]
    c = (c + c.transpose(-1, -2)) * 0.5
    return c.clamp(-1.0, 1.0)


def ledoit_wolf_correlation(ts_block):
    """逐被试 Ledoit-Wolf 估计，归一化为相关矩阵；返回矩阵与收缩系数。"""
    out, coefficients = [], []
    for i in range(ts_block.shape[0]):
        subject = ts_block[i].numpy().astype(np.float64)      # (n_roi, t)
        estimator = LedoitWolf(
            assume_centered=True, store_precision=False
        ).fit(subject.T)
        covariance = estimator.covariance_
        scale = np.sqrt(np.clip(np.diag(covariance), 1e-12, None))
        correlation = covariance / np.outer(scale, scale)
        correlation = (correlation + correlation.T) * 0.5
        np.fill_diagonal(correlation, 1.0)
        out.append(correlation)
        coefficients.append(float(estimator.shrinkage_))
    return torch.as_tensor(np.stack(out)).double(), coefficients


def matrix_isqrt(m):
    eigval, eigvec = torch.linalg.eigh(m)
    inv_sqrt = eigval.clamp_min(NUMERIC_FLOOR).rsqrt()
    return (eigvec * inv_sqrt.unsqueeze(-2)) @ eigvec.transpose(-1, -2)


def main() -> None:
    sys.path.insert(0, str(ROOT))
    import dataset_cc400 as dataset

    ts_all, _, _ = dataset.load_cc400(t_use=146, verbose=False)
    fold = json.loads((BASE / 'splits_manifest.json').read_text())['splits'][0]
    train = np.asarray(fold['train'], dtype=np.int64)
    val = np.asarray(fold['val'], dtype=np.int64)
    ts_train = torch.as_tensor(ts_all[train]).float()
    ts_val = torch.as_tensor(ts_all[val]).float()
    del ts_all
    n_roi = ts_train.shape[1]

    null_bases, range_bases = [], []
    for start in range(0, len(val), BATCH):
        c = corr_batch(ts_val[start:start + BATCH])
        eigval, eigvec = torch.linalg.eigh(c)
        for i in range(c.shape[0]):
            keep = eigval[i] > RANK_TOL
            null_bases.append(eigvec[i][:, ~keep].clone())
            range_bases.append(eigvec[i][:, keep].clone())
        del c, eigval, eigvec

    reference = torch.zeros(n_roi, n_roi, dtype=torch.float64)
    all_coefficients = []
    for start in range(0, len(train), BATCH):
        block, coefficients = ledoit_wolf_correlation(ts_train[start:start + BATCH])
        reference += block.sum(0)
        all_coefficients.extend(coefficients)
        del block
    reference /= len(train)
    reference_isqrt = matrix_isqrt(reference)

    norms, null_frac, range_frac, cross_frac = [], [], [], []
    abs_log_max, eta_min, eta_max, null_rms, range_rms = [], [], [], [], []
    val_coefficients = []

    for start in range(0, len(val), BATCH):
        base, coefficients = ledoit_wolf_correlation(ts_val[start:start + BATCH])
        val_coefficients.extend(coefficients)
        whitened = reference_isqrt @ base @ reference_isqrt
        whitened = (whitened + whitened.transpose(-1, -2)) * 0.5
        eta, vec = torch.linalg.eigh(whitened)
        log_eta = eta.clamp_min(NUMERIC_FLOOR).log()
        tangent = (vec * log_eta.unsqueeze(-2)) @ vec.transpose(-1, -2)

        for i in range(tangent.shape[0]):
            idx = start + i
            s = tangent[i]
            total = float((s * s).sum())
            u_null, u_range = null_bases[idx], range_bases[idx]
            block_null = u_null.transpose(0, 1) @ s @ u_null
            block_range = u_range.transpose(0, 1) @ s @ u_range
            e_null = float((block_null * block_null).sum())
            e_range = float((block_range * block_range).sum())
            norms.append(total ** 0.5)
            null_frac.append(e_null / total)
            range_frac.append(e_range / total)
            cross_frac.append(max(0.0, 1.0 - (e_null + e_range) / total))
            null_rms.append((e_null / max(1, u_null.shape[1] ** 2)) ** 0.5)
            range_rms.append((e_range / max(1, u_range.shape[1] ** 2)) ** 0.5)
            abs_log_max.append(float(log_eta[i].abs().max()))
            eta_min.append(float(eta[i].min()))
            eta_max.append(float(eta[i].max()))
        del base, whitened, eta, vec, log_eta, tangent

    entry = {
        'parameter': float(np.mean(val_coefficients)),
        'parameter_name': 'mean Ledoit-Wolf shrinkage coefficient (validation)',
        'shrinkage_coefficient_train': summarize(all_coefficients),
        'shrinkage_coefficient_validation': summarize(val_coefficients),
        'tangent_frobenius_norm': summarize(norms),
        'energy_fraction_null_block': summarize(null_frac),
        'energy_fraction_range_block': summarize(range_frac),
        'energy_fraction_cross_block': summarize(cross_frac),
        'per_direction_rms_null_block': summarize(null_rms),
        'per_direction_rms_range_block': summarize(range_rms),
        'abs_log_eigenvalue_max': summarize(abs_log_max),
        'whitened_eigenvalue_min': summarize(eta_min),
        'whitened_eigenvalue_max': summarize(eta_max),
    }

    report_file = OUTPUT / 'report.json'
    report = json.loads(report_file.read_text(encoding='utf-8'))
    report['schemes']['ledoit_wolf'] = entry
    report['artifact_version'] = 'tangent-spectrum-diagnostic-v1.1'
    report['scheme_order'] = ['eigclip', 'ledoit_wolf', 'shrink']
    with report_file.open('w', encoding='utf-8') as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write('\n')

    print('Ledoit-Wolf 收缩系数（验证）:',
          json.dumps(entry['shrinkage_coefficient_validation'], ensure_ascii=False))
    print()
    print('%-32s %14s %14s %14s' % ('指标', '截断1e-6', 'Ledoit-Wolf', '收缩γ=0.9'))
    for key in ('tangent_frobenius_norm', 'energy_fraction_null_block',
                'energy_fraction_range_block', 'abs_log_eigenvalue_max',
                'whitened_eigenvalue_min'):
        print('%-32s %14.6g %14.6g %14.6g' % (
            key,
            report['schemes']['eigclip'][key]['mean'],
            entry[key]['mean'],
            report['schemes']['shrink'][key]['mean'],
        ))
    print()
    print('updated:', report_file)


if __name__ == '__main__':
    main()
