"""相关矩阵有效秩的容差敏感性与谱衰减诊断。

目的：判定第 1 折验证被试相关矩阵的实测秩约 62 是数据本身的有效秩，
还是判定容差造成的假象。只读数据，不涉及模型与测试被试。
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

torch.set_num_threads(8)

BASE = Path(__file__).resolve().parents[1] / 'results_revision_v2' / 'standard5_seed42'
OUTPUT = Path(__file__).resolve().parents[1] / 'results_revision_v2' / 'effective_rank_diagnostic_v1'
TOLERANCES = [1e-12, 1e-10, 1e-8, 1e-6, 1e-4, 1e-3, 1e-2, 1e-1]
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


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import dataset_cc400 as dataset

    ts_all, labels, sites = dataset.load_cc400(t_use=146, verbose=False)
    manifest = json.loads((BASE / 'splits_manifest.json').read_text())
    val = np.asarray(manifest['splits'][0]['val'], dtype=np.int64)
    ts_val = torch.as_tensor(ts_all[val]).float()
    site_val = np.asarray(sites, dtype=object)[val]
    del ts_all

    n_roi = ts_val.shape[1]
    t_use = ts_val.shape[2]

    all_eig = []
    for start in range(0, len(val), BATCH):
        block = ts_val[start:start + BATCH].double()
        c = block @ block.transpose(-1, -2) / block.shape[-1]
        c = (c + c.transpose(-1, -2)) * 0.5
        eigval = torch.linalg.eigvalsh(c.clamp(-1.0, 1.0))
        all_eig.append(eigval.cpu().numpy())
        del block, c, eigval
    eig = np.concatenate(all_eig)                 # (n_val, n_roi) 升序

    desc = np.sort(eig, axis=1)[:, ::-1]          # 降序
    total = desc.sum(axis=1, keepdims=True)
    cumulative = np.cumsum(desc, axis=1) / total

    rank_by_tol = {
        f'{tol:.0e}': summarize((eig > tol).sum(axis=1))
        for tol in TOLERANCES
    }
    rank_relative = {
        f'{rel:.0e}': summarize((eig > rel * desc[:, [0]]).sum(axis=1))
        for rel in (1e-10, 1e-8, 1e-6, 1e-4, 1e-2)
    }
    energy_rank = {
        f'{q:.2f}': summarize((cumulative < q).sum(axis=1) + 1)
        for q in (0.90, 0.95, 0.99, 0.999)
    }
    participation_ratio = (desc.sum(axis=1) ** 2) / (desc ** 2).sum(axis=1)

    # 谱衰减：前若干个特征值占总能量比例
    fractions = {
        f'top_{k}': summarize(cumulative[:, k - 1])
        for k in (1, 5, 10, 20, 50, 62, 100, 145)
    }

    # 站点差异（有效秩是否与站点相关）
    by_site = {}
    rank_1e8 = (eig > 1e-8).sum(axis=1)
    for site in sorted(set(site_val.tolist())):
        idx = np.where(site_val == site)[0]
        by_site[site] = {
            'n': int(len(idx)),
            'rank_tol_1e-8': summarize(rank_1e8[idx]),
        }

    report = {
        'status': 'completed',
        'artifact_version': 'effective-rank-diagnostic-v1',
        'analysis_role': (
            'determine whether the measured correlation-matrix rank of about 62 '
            'is a genuine effective rank of the preprocessed time series or an '
            'artifact of the numerical tolerance'
        ),
        'uses_test_subjects': False,
        'models_retrained': False,
        'fold': 0,
        'n_validation_measured': int(len(val)),
        'n_roi': int(n_roi),
        't_use': int(t_use),
        'theoretical_rank_upper_bound': int(t_use - 1),
        'rank_by_absolute_tolerance': rank_by_tol,
        'rank_by_relative_tolerance': rank_relative,
        'rank_capturing_energy_quantile': energy_rank,
        'cumulative_energy_at_top_k': fractions,
        'participation_ratio': summarize(participation_ratio),
        'largest_eigenvalue': summarize(desc[:, 0]),
        'eigenvalue_at_index_62': summarize(desc[:, 61]),
        'eigenvalue_at_index_145': summarize(desc[:, 144]),
        'by_site_rank_tol_1e-8': by_site,
        'interpretation_constraint': (
            'Descriptive spectral diagnostic on fold-1 validation subjects only.'
        ),
    }

    out_file = OUTPUT / 'report.json'
    with out_file.open('w', encoding='utf-8') as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write('\n')

    print('理论秩上界 T-1 =', t_use - 1, '  ROI 数 =', n_roi)
    print()
    print('--- 按绝对容差判定的秩 ---')
    for tol, v in rank_by_tol.items():
        print('  tol=%-8s mean=%7.2f sd=%6.2f min=%5.0f max=%5.0f'
              % (tol, v['mean'], v['sample_sd'], v['min'], v['max']))
    print()
    print('--- 按相对最大特征值容差判定的秩 ---')
    for tol, v in rank_relative.items():
        print('  rel=%-8s mean=%7.2f sd=%6.2f min=%5.0f max=%5.0f'
              % (tol, v['mean'], v['sample_sd'], v['min'], v['max']))
    print()
    print('--- 捕获指定能量所需的特征值个数 ---')
    for q, v in energy_rank.items():
        print('  q=%-6s mean=%7.2f sd=%6.2f min=%5.0f max=%5.0f'
              % (q, v['mean'], v['sample_sd'], v['min'], v['max']))
    print()
    print('--- 前 k 个特征值累计能量占比 ---')
    for k, v in fractions.items():
        print('  %-8s mean=%.4f sd=%.4f' % (k, v['mean'], v['sample_sd']))
    print()
    print('participation ratio :', json.dumps(report['participation_ratio']))
    print('lambda_max          :', json.dumps(report['largest_eigenvalue']))
    print('lambda_[62]         :', json.dumps(report['eigenvalue_at_index_62']))
    print('lambda_[145]        :', json.dumps(report['eigenvalue_at_index_145']))
    print()
    print('written:', out_file)


if __name__ == '__main__':
    main()
