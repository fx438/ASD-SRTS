# -*- coding: utf-8 -*-
"""分析收缩强度 γ、验证集 AIRM 距离与验证 AUC 的对应关系。

每折参考矩阵仅由该折训练集的收缩 FC 计算；AIRM 仅针对该折验证样本。核心
γ 网格的验证 AUC 读取已锁定主实验，γ=0.95 只计算几何/数值审计量，不训练
分类器、不评估测试集，也不改变正式 γ 选择。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import scipy
import torch
from scipy.stats import pearsonr, spearmanr

SOURCE_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = SOURCE_ROOT / 'revision_srts_v2'
MAIN_ROOT = SOURCE_ROOT / 'results_revision_v2' / 'standard5_seed42'
DEFAULT_OUTPUT = (
    SOURCE_ROOT / 'results_revision_v2'
    / 'gamma_airm_validation_standard5_seed42_v1'
)
CORE_GAMMAS = (0.05, 0.10, 0.20, 0.30, 0.50, 0.70, 0.80, 0.90)
AUDIT_GAMMA = 0.95
sys.path.insert(0, str(SCRIPT_ROOT))
sys.path.insert(0, str(SOURCE_ROOT))

import dataset_cc400 as dataset  # noqa: E402
import run_revision as rr  # noqa: E402


def read_json(path):
    with Path(path).open(encoding='utf-8') as handle:
        return json.load(handle)


def protocol_config():
    return {
        'seed': 42,
        't_use': 146,
        'folds': 5,
        'per_fold_sizes': [602, 86, 172],
        'core_gamma_grid': list(CORE_GAMMAS),
        'audit_gamma': AUDIT_GAMMA,
        'target_partition': 'validation only',
        'reference_fit_partition': 'training only',
        'reference_matrix': 'arithmetic mean of shrunk training FC only',
        'distance': (
            'affine-invariant Riemannian distance, equal to the Frobenius '
            'norm of log(R^(-1/2) C_tilde R^(-1/2))'
        ),
        'validation_auc_source': (
            'locked BrainNetMLP gamma trials from the standard-five-fold '
            'main experiment; unavailable by design for gamma=0.95 here'
        ),
        'uses_test_predictions_or_metrics': False,
        'used_to_change_locked_gamma_selection': False,
    }


def frobenius_distances(tangent):
    if not isinstance(tangent, torch.Tensor):
        tangent = torch.as_tensor(tangent)
    if tangent.ndim != 3 or tangent.shape[-1] != tangent.shape[-2]:
        raise ValueError('tangent 必须为 [N, ROI, ROI] 方阵批次')
    return torch.linalg.matrix_norm(
        tangent, ord='fro', dim=(-2, -1)
    ).detach().cpu().numpy().astype(np.float64, copy=False)


def summarize_distances(values):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or len(values) == 0:
        raise ValueError('距离必须是一维非空数组')
    if not np.isfinite(values).all():
        raise ValueError('距离包含非有限值')
    return {
        'n': int(len(values)),
        'mean': float(values.mean()),
        'sample_sd': float(values.std(ddof=1)) if len(values) > 1 else 0.0,
        'median': float(np.median(values)),
        'q25': float(np.quantile(values, 0.25)),
        'q75': float(np.quantile(values, 0.75)),
        'minimum': float(values.min()),
        'maximum': float(values.max()),
    }


def locked_validation_auc(fold_record, gamma):
    matches = [
        trial for trial in fold_record['gamma_trials']
        if np.isclose(float(trial['gamma']), gamma, rtol=0.0, atol=1e-12)
    ]
    if len(matches) != 1:
        raise AssertionError(f'Fold {fold_record["fold"] + 1} gamma={gamma} 记录异常')
    if 'test' in matches[0]:
        raise AssertionError('gamma trial 不应包含测试字段')
    return float(matches[0]['best_val_auc'])


def aggregate_records(records):
    by_gamma = []
    for gamma in (*CORE_GAMMAS, AUDIT_GAMMA):
        subset = [
            record for record in records
            if np.isclose(record['gamma'], gamma, rtol=0.0, atol=1e-12)
        ]
        if len(subset) != 5:
            raise AssertionError(f'gamma={gamma} 不是五折记录')
        fold_distance_means = np.asarray(
            [record['airm']['mean'] for record in subset], dtype=np.float64
        )
        pooled_distances = np.concatenate([
            np.asarray(record['subject_distances'], dtype=np.float64)
            for record in subset
        ])
        validation_auc = [record['validation_auc'] for record in subset]
        item = {
            'gamma': float(gamma),
            'role': 'core_grid' if gamma in CORE_GAMMAS else 'audit_only',
            'airm_fold_mean_values': fold_distance_means.tolist(),
            'airm_fold_mean': float(fold_distance_means.mean()),
            'airm_fold_mean_sample_sd': float(
                fold_distance_means.std(ddof=1)
            ),
            'airm_all_validation_occurrences': summarize_distances(
                pooled_distances
            ),
            'validation_auc_fold_values': validation_auc,
            'validation_auc_mean': None,
            'validation_auc_sample_sd': None,
        }
        if gamma in CORE_GAMMAS:
            auc = np.asarray(validation_auc, dtype=np.float64)
            item['validation_auc_mean'] = float(auc.mean())
            item['validation_auc_sample_sd'] = float(auc.std(ddof=1))
        elif any(value is not None for value in validation_auc):
            raise AssertionError('gamma=0.95 不应在本分析产生验证 AUC')
        by_gamma.append(item)

    core = [record for record in records if record['role'] == 'core_grid']
    distances = np.asarray([record['airm']['mean'] for record in core])
    auc = np.asarray([record['validation_auc'] for record in core])
    pearson = pearsonr(distances, auc)
    spearman = spearmanr(distances, auc)
    association = {
        'unit': '40 fold-by-gamma validation records',
        'n': int(len(core)),
        'pearson_r': float(pearson.statistic),
        'pearson_two_sided_p': float(pearson.pvalue),
        'spearman_rho': float(spearman.statistic),
        'spearman_two_sided_p': float(spearman.pvalue),
        'interpretation_scope': (
            'descriptive association across locked validation trials; not a '
            'causal estimate and not a test-set selection rule'
        ),
    }
    return by_gamma, association


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main():
    args = parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f'拒绝覆盖已有目录：{output}')
    started = time.time()

    main_config = read_json(MAIN_ROOT / 'config.json')
    main_summary = read_json(MAIN_ROOT / 'summary.json')
    split_document = read_json(MAIN_ROOT / 'splits_manifest.json')
    fold_records = [
        read_json(MAIN_ROOT / 'folds' / f'fold_{fold}.json')
        for fold in range(1, 6)
    ]
    if [float(value) for value in main_summary['selected_gammas']] != (
        [0.90, 0.90, 0.70, 0.50, 0.90]
    ):
        raise AssertionError('正式主实验 gamma 已改变')
    for fold in fold_records:
        trial_gammas = tuple(float(row['gamma']) for row in fold['gamma_trials'])
        if trial_gammas != CORE_GAMMAS:
            raise AssertionError('正式 gamma 网格与锁定值不一致')

    time_series, labels, sites = dataset.load_cc400(t_use=146, verbose=True)
    manifest = rr.load_manifest(146, labels, sites)
    if manifest != split_document['manifest']:
        raise AssertionError('身份清单与锁定主实验不一致')
    splits = rr.build_splits(labels, seed=42)
    if splits != split_document['splits']:
        raise AssertionError('划分与锁定主实验不一致')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type != 'cuda':
        raise RuntimeError('AIRM 批量分析需要 CUDA')
    x = torch.from_numpy(time_series).to(device)
    with torch.no_grad():
        raw_correlation = rr.corr_matrix_all(
            x, int(main_config['eval_batch_size'])
        )
    print(f'[GPU] raw FC ready {tuple(raw_correlation.shape)}', flush=True)

    records = []
    for gamma in (*CORE_GAMMAS, AUDIT_GAMMA):
        with torch.no_grad():
            shrunk = rr.shrink(raw_correlation, gamma)
        for fold_number, (split, main_fold) in enumerate(
            zip(splits, fold_records), start=1
        ):
            train = split['train']
            validation = split['val']
            if set(validation) & set(split['test']):
                raise AssertionError('验证集与测试集重叠')
            with torch.no_grad():
                tangent, numerics = rr.tangent_transform(
                    shrunk,
                    train,
                    validation,
                    diag_positions=validation,
                    batch_size=int(main_config['eig_batch_size']),
                )
                distances = frobenius_distances(tangent)
            del tangent
            torch.cuda.empty_cache()
            if len(distances) != 86:
                raise AssertionError('每折验证 AIRM 距离不是 86 个')
            validation_auc = (
                locked_validation_auc(main_fold, gamma)
                if gamma in CORE_GAMMAS else None
            )
            role = 'core_grid' if gamma in CORE_GAMMAS else 'audit_only'
            record = {
                'fold': fold_number - 1,
                'gamma': float(gamma),
                'role': role,
                'reference_fit_indices': train,
                'reference_uses_training_only': True,
                'target_partition': 'validation',
                'target_indices': validation,
                'test_indices_or_metrics_used': False,
                'validation_auc': validation_auc,
                'validation_auc_source': (
                    str(MAIN_ROOT / 'folds' / f'fold_{fold_number}.json')
                    if validation_auc is not None else None
                ),
                'airm': summarize_distances(distances),
                'subject_distances': distances.tolist(),
                'numerical_diagnostics': numerics,
            }
            records.append(record)
            print(
                f'[fold {fold_number}/5 gamma={gamma:.2f} {role}] '
                f'AIRM={record["airm"]["mean"]:.4f} '
                f'val_auc={validation_auc}',
                flush=True,
            )
        del shrunk
        torch.cuda.empty_cache()

    aggregate, association = aggregate_records(records)
    output.mkdir(parents=True, exist_ok=False)
    config = protocol_config()
    result = {
        'protocol': config,
        'records': records,
        'by_gamma': aggregate,
        'airm_validation_auc_association': association,
        'runtime_seconds': float(time.time() - started),
    }
    provenance = {
        'script_sha256': rr.sha256_file(Path(__file__).resolve()),
        'data_cache_sha256': rr.sha256_file(
            SOURCE_ROOT / 'cc400_cache_t146.npz'
        ),
        'main_config_sha256': rr.sha256_file(MAIN_ROOT / 'config.json'),
        'main_summary_sha256': rr.sha256_file(MAIN_ROOT / 'summary.json'),
        'split_manifest_sha256': rr.sha256_file(
            MAIN_ROOT / 'splits_manifest.json'
        ),
        'fold_sha256': {
            str(fold): rr.sha256_file(
                MAIN_ROOT / 'folds' / f'fold_{fold}.json'
            )
            for fold in range(1, 6)
        },
        'numpy_version': np.__version__,
        'scipy_version': scipy.__version__,
        'torch_version': torch.__version__,
    }
    rr.write_json_new(output / 'config.json', config)
    rr.write_json_new(output / 'results.json', result)
    rr.write_json_new(output / 'provenance.json', provenance)
    print(json.dumps({
        'status': 'passed',
        'output': str(output),
        'association': association,
        'by_gamma': [
            {
                'gamma': row['gamma'],
                'role': row['role'],
                'airm_mean': row['airm_fold_mean'],
                'validation_auc_mean': row['validation_auc_mean'],
            }
            for row in aggregate
        ],
    }, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
