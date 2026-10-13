"""切线矩阵谱能量诊断：检验特征值截断为何在切线空间中失效。

测量对象为第 1 折的验证被试；参考矩阵由同折 602 名训练被试估计。
不读取任何测试折被试，不产生测试预测，不修改已锁定结果。

核心测量：把切线矩阵投影到原始相关矩阵的零空间与值域空间，比较两类
方向携带的 Frobenius 能量。零空间方向在原始数据中不含信息；若其在切线
空间中占据可观能量，即说明该正定化方式把无信息方向放大到了与有信息
方向可比的量级。

容器内存上限为 2 GiB，故全部按批流式计算。
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
OUTPUT = Path(__file__).resolve().parents[1] / 'results_revision_v2' / 'tangent_spectrum_diagnostic_v1'
EIGCLIP_FLOOR = 1e-6
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
    """标准化时间序列 -> 双精度相关矩阵。"""
    block = ts_block.double()
    c = block @ block.transpose(-1, -2) / block.shape[-1]
    c = (c + c.transpose(-1, -2)) * 0.5
    return c.clamp(-1.0, 1.0)


def positive_definitize(c, scheme, gamma):
    if scheme == 'shrink':
        eye = torch.eye(c.shape[-1], dtype=c.dtype)
        return (1.0 - gamma) * c + gamma * eye
    eigval, eigvec = torch.linalg.eigh(c)
    eigval = eigval.clamp_min(EIGCLIP_FLOOR)
    return (eigvec * eigval.unsqueeze(-2)) @ eigvec.transpose(-1, -2)


def matrix_isqrt(m):
    eigval, eigvec = torch.linalg.eigh(m)
    inv_sqrt = eigval.clamp_min(NUMERIC_FLOOR).rsqrt()
    return (eigvec * inv_sqrt.unsqueeze(-2)) @ eigvec.transpose(-1, -2)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import dataset_cc400 as dataset

    ts_all, labels, _ = dataset.load_cc400(t_use=146, verbose=False)
    manifest = json.loads((BASE / 'splits_manifest.json').read_text())
    fold = manifest['splits'][0]
    train = np.asarray(fold['train'], dtype=np.int64)
    val = np.asarray(fold['val'], dtype=np.int64)
    gamma = json.loads((BASE / 'summary.json').read_text())['selected_gammas'][0]

    ts_train = torch.as_tensor(ts_all[train]).float()
    ts_val = torch.as_tensor(ts_all[val]).float()
    del ts_all
    n_roi = ts_train.shape[1]

    # 原始相关矩阵的秩与零空间基（仅验证被试）
    null_bases, range_bases, ranks = [], [], []
    for start in range(0, len(val), BATCH):
        c = corr_batch(ts_val[start:start + BATCH])
        eigval, eigvec = torch.linalg.eigh(c)
        for i in range(c.shape[0]):
            keep = eigval[i] > RANK_TOL
            ranks.append(int(keep.sum()))
            null_bases.append(eigvec[i][:, ~keep].clone())
            range_bases.append(eigvec[i][:, keep].clone())
        del c, eigval, eigvec

    results = {}
    for scheme in ('shrink', 'eigclip'):
        reference = torch.zeros(n_roi, n_roi, dtype=torch.float64)
        for start in range(0, len(train), BATCH):
            c = corr_batch(ts_train[start:start + BATCH])
            reference += positive_definitize(c, scheme, gamma).sum(0)
            del c
        reference /= len(train)
        reference_isqrt = matrix_isqrt(reference)

        norms, null_frac, range_frac, cross_frac = [], [], [], []
        abs_log_max, eta_min, eta_max = [], [], []
        null_rms, range_rms = [], []

        for start in range(0, len(val), BATCH):
            c = corr_batch(ts_val[start:start + BATCH])
            base = positive_definitize(c, scheme, gamma)
            whitened = reference_isqrt @ base @ reference_isqrt
            whitened = (whitened + whitened.transpose(-1, -2)) * 0.5
            eta, vec = torch.linalg.eigh(whitened)
            log_eta = eta.clamp_min(NUMERIC_FLOOR).log()
            tangent = (vec * log_eta.unsqueeze(-2)) @ vec.transpose(-1, -2)

            for i in range(tangent.shape[0]):
                idx = start + i
                s = tangent[i]
                total = float((s * s).sum())
                u_null = null_bases[idx]
                u_range = range_bases[idx]
                block_null = u_null.transpose(0, 1) @ s @ u_null
                block_range = u_range.transpose(0, 1) @ s @ u_range
                e_null = float((block_null * block_null).sum())
                e_range = float((block_range * block_range).sum())
                norms.append(total ** 0.5)
                null_frac.append(e_null / total)
                range_frac.append(e_range / total)
                cross_frac.append(max(0.0, 1.0 - (e_null + e_range) / total))
                # 按方向对数归一，消除两个子空间维数不同带来的影响
                null_rms.append((e_null / max(1, u_null.shape[1] ** 2)) ** 0.5)
                range_rms.append((e_range / max(1, u_range.shape[1] ** 2)) ** 0.5)
                abs_log_max.append(float(log_eta[i].abs().max()))
                eta_min.append(float(eta[i].min()))
                eta_max.append(float(eta[i].max()))
            del c, base, whitened, eta, vec, log_eta, tangent

        results[scheme] = {
            'parameter': gamma if scheme == 'shrink' else EIGCLIP_FLOOR,
            'parameter_name': 'gamma' if scheme == 'shrink' else 'eigenvalue floor',
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
        del reference, reference_isqrt

    report = {
        'status': 'completed',
        'artifact_version': 'tangent-spectrum-diagnostic-v1',
        'analysis_role': (
            'mechanism diagnostic: energy carried by raw null-space directions '
            'after tangent mapping under two positive-definitization schemes'
        ),
        'uses_test_subjects': False,
        'models_retrained': False,
        'fold': 0,
        'n_train_for_reference': int(len(train)),
        'n_validation_measured': int(len(val)),
        'locked_gamma_fold1': gamma,
        'eigclip_floor': EIGCLIP_FLOOR,
        'numeric_floor': NUMERIC_FLOOR,
        'rank_tolerance': RANK_TOL,
        'n_roi': int(n_roi),
        'raw_correlation_rank': summarize(ranks),
        'raw_null_dimension': summarize(n_roi - np.asarray(ranks)),
        'block_definition': (
            'null block = projection of the tangent matrix onto eigenvectors of '
            'the raw correlation matrix with eigenvalue <= 1e-8; range block = '
            'the complementary eigenvectors; fractions are of squared '
            'Frobenius norm'
        ),
        'schemes': results,
        'interpretation_constraint': (
            'Descriptive spectral diagnostic on fold-1 validation subjects only. '
            'It explains a mechanism; it is not a performance measurement.'
        ),
    }

    out_file = OUTPUT / 'report.json'
    with out_file.open('w', encoding='utf-8') as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write('\n')

    print('rank      :', json.dumps(report['raw_correlation_rank']))
    print('null dim  :', json.dumps(report['raw_null_dimension']))
    for scheme, res in results.items():
        print()
        print('==', scheme, res['parameter_name'], '=', res['parameter'])
        for key in (
            'tangent_frobenius_norm',
            'energy_fraction_null_block',
            'energy_fraction_range_block',
            'per_direction_rms_null_block',
            'per_direction_rms_range_block',
            'abs_log_eigenvalue_max',
            'whitened_eigenvalue_min',
        ):
            v = res[key]
            print('   %-32s mean=%12.6g sd=%10.4g min=%12.6g max=%12.6g'
                  % (key, v['mean'], v['sample_sd'], v['min'], v['max']))
    print()
    print('written:', out_file)


if __name__ == '__main__':
    main()
