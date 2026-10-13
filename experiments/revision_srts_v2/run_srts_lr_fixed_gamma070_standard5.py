# -*- coding: utf-8 -*-
"""标准五折 SRTS + LR 的预设统一 γ=0.70 稳健性分析。

本分析不重新选择 γ：五折均在查看测试结果前固定为 0.70。除收缩强度外，
划分、SRTS 特征定义、训练折参考矩阵、StandardScaler、LR 参数和验证集阈值
逻辑均与已锁定的 SRTS + LR 主实验一致。结果写入独立目录并拒绝覆盖。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import scipy
import sklearn
import torch

SOURCE_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = SOURCE_ROOT / 'revision_srts_v2'
MAIN_RESULT = SOURCE_ROOT / 'results_revision_v2' / 'standard5_seed42'
LOCKED_SRTS_RESULT = (
    SOURCE_ROOT / 'results_revision_v2' / 'srts_linear_standard5_seed42_v1'
)
DEFAULT_OUTPUT = (
    SOURCE_ROOT / 'results_revision_v2'
    / 'srts_lr_fixed_gamma070_standard5_seed42_v1'
)
LOCKED_GAMMAS = (0.70,) * 5
MAIN_SELECTED_GAMMAS = (0.90, 0.90, 0.70, 0.50, 0.90)
MODEL_KEY = 'lr'
MODEL_DISPLAY = 'SRTS + Logistic Regression, fixed gamma=0.70'

sys.path.insert(0, str(SCRIPT_ROOT))
sys.path.insert(0, str(SOURCE_ROOT))

import dataset_cc400 as dataset  # noqa: E402
import run_revision as rr  # noqa: E402
import run_srts_linear_standard5 as linear  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--resume', action='store_true')
    return parser.parse_args()


def read_json(path):
    with Path(path).open(encoding='utf-8') as handle:
        return json.load(handle)


def assert_result_not_complete(output):
    summary_path = Path(output) / 'models' / MODEL_KEY / 'summary.json'
    if summary_path.exists():
        raise FileExistsError(f'拒绝覆盖已完成结果：{summary_path}')


def protocol_config(output, main_config):
    return {
        'analysis_role': 'pre-specified fixed-gamma robustness analysis',
        'seed': 42,
        't_use': 146,
        'per_fold_sizes': [602, 86, 172],
        'gammas': list(LOCKED_GAMMAS),
        'gamma_selection': 'pre-specified robustness analysis',
        'gamma_selected_with_test_data': False,
        'comparison_result': str(
            LOCKED_SRTS_RESULT / 'models' / 'lr' / 'summary.json'
        ),
        'reference_matrix': 'arithmetic mean of shrunk training FC only',
        'feature': (
            'SRTS tangent svec: diagonal + sqrt(2) times strict upper triangle'
        ),
        'feature_dim': 77028,
        'feature_selection': None,
        'standardization': 'StandardScaler, training fold only',
        'threshold_selection': (
            'maximum validation accuracy, tie closest to 0.5'
        ),
        'model_hyperparameters': {
            'penalty': 'l2', 'C': 1.0, 'solver': 'liblinear',
            'dual': True, 'max_iter': 5000, 'random_state': 42,
        },
        'classifier_specific_tuning': False,
        'main_split_manifest': str(MAIN_RESULT / 'splits_manifest.json'),
        'eval_batch_size': int(main_config['eval_batch_size']),
        'eig_batch_size': int(main_config['eig_batch_size']),
        'output': str(output),
    }


def summarize(fold_results, manifest, runtime_seconds):
    summary = {
        'model_key': MODEL_KEY,
        'model_name': MODEL_DISPLAY,
        'analysis_role': 'pre-specified fixed-gamma robustness analysis',
        'protocol': {
            'folds': 5,
            'per_fold_sizes': [602, 86, 172],
            'test_union_n': 860,
            'test_count_per_subject': 1,
            'seed': 42,
            'gammas': list(LOCKED_GAMMAS),
            'gamma_selected_with_test_data': False,
            'sd_definition': 'sample SD across five folds (ddof=1)',
        },
        'metrics': {},
        'runtime_seconds': float(runtime_seconds),
        'oof_predictions': [],
    }
    for metric in linear.METRICS:
        values = np.asarray(
            [fold['test'][metric] for fold in fold_results],
            dtype=np.float64,
        )
        summary['metrics'][metric] = {
            'mean': float(values.mean()),
            'sample_sd': float(values.std(ddof=1)),
            'fold_values': values.tolist(),
        }
    validation_auc = np.asarray(
        [fold['validation']['auc'] for fold in fold_results],
        dtype=np.float64,
    )
    summary['validation_auc'] = {
        'mean': float(validation_auc.mean()),
        'sample_sd': float(validation_auc.std(ddof=1)),
        'fold_values': validation_auc.tolist(),
    }
    seen = np.zeros(len(manifest), dtype=np.int64)
    for fold in fold_results:
        indices = fold['test_indices']
        seen[indices] += 1
        for position, index in enumerate(indices):
            summary['oof_predictions'].append({
                'subject_index': index,
                'sub_id': manifest[index]['sub_id'],
                'file_id': manifest[index]['file_id'],
                'site_id': manifest[index]['site_id'],
                'label': manifest[index]['label'],
                'fold': fold['fold'],
                'score': fold['test']['scores'][position],
            })
    if not np.all(seen == 1):
        raise RuntimeError('固定 gamma 分析的 OOF 测试覆盖不是每人一次')
    summary['oof_predictions'].sort(key=lambda row: row['subject_index'])
    return summary


def main(args):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    assert_result_not_complete(output)

    main_config = read_json(MAIN_RESULT / 'config.json')
    main_summary = read_json(MAIN_RESULT / 'summary.json')
    split_payload = read_json(MAIN_RESULT / 'splits_manifest.json')
    locked_config = read_json(LOCKED_SRTS_RESULT / 'config.json')
    if tuple(main_summary['selected_gammas']) != MAIN_SELECTED_GAMMAS:
        raise RuntimeError('正式主实验 gamma 已改变，拒绝继续')

    config = protocol_config(output, main_config)
    if config['model_hyperparameters'] != locked_config['model_hyperparameters']['lr']:
        raise RuntimeError('LR 参数与已锁定 SRTS + LR 不一致')
    if config['per_fold_sizes'] != locked_config['per_fold_sizes']:
        raise RuntimeError('折规模与已锁定 SRTS + LR 不一致')
    config_path = output / 'config.json'
    if not config_path.exists():
        rr.write_json_new(config_path, config)
    elif read_json(config_path) != config:
        raise RuntimeError('当前配置与已有 config.json 不一致')

    started = time.time()
    time_series, labels, sites = dataset.load_cc400(t_use=146, verbose=True)
    manifest = rr.load_manifest(146, labels, sites)
    if manifest != split_payload['manifest']:
        raise RuntimeError('当前身份清单与主实验不一致')
    splits = rr.build_splits(labels, seed=42)
    if splits != split_payload['splits']:
        raise RuntimeError('当前划分与锁定主实验不一致')

    provenance = {
        'script_sha256': rr.sha256_file(Path(__file__).resolve()),
        'linear_helper_script_sha256': rr.sha256_file(
            SCRIPT_ROOT / 'run_srts_linear_standard5.py'
        ),
        'data_cache_sha256': rr.sha256_file(
            SOURCE_ROOT / 'cc400_cache_t146.npz'
        ),
        'main_summary_sha256': rr.sha256_file(MAIN_RESULT / 'summary.json'),
        'split_manifest_sha256': rr.sha256_file(
            MAIN_RESULT / 'splits_manifest.json'
        ),
        'locked_srts_lr_summary_sha256': rr.sha256_file(
            LOCKED_SRTS_RESULT / 'models' / 'lr' / 'summary.json'
        ),
        'numpy_version': np.__version__,
        'scipy_version': scipy.__version__,
        'sklearn_version': sklearn.__version__,
        'torch_version': torch.__version__,
    }
    provenance_path = output / 'provenance.json'
    if not provenance_path.exists():
        rr.write_json_new(provenance_path, provenance)
    elif read_json(provenance_path) != provenance:
        raise RuntimeError('源码、数据、划分或正式结果来源已改变')

    feature_cache = linear.TangentFeatureCache(
        time_series,
        labels,
        splits,
        output,
        main_config['eval_batch_size'],
        main_config['eig_batch_size'],
    )
    fold_results = []
    for fold_number, (split, gamma) in enumerate(
        zip(splits, LOCKED_GAMMAS), start=1
    ):
        fold_path = (
            output / 'models' / MODEL_KEY / 'folds'
            / f'fold_{fold_number}.json'
        )
        if fold_path.exists():
            if not args.resume:
                raise FileExistsError(f'已有折结果：{fold_path}')
            fold_result = read_json(fold_path)
            if not np.isclose(fold_result['gamma'], gamma, rtol=0.0, atol=0.0):
                raise RuntimeError(f'已有 Fold {fold_number} gamma 不一致')
            print(f'[resume] fixed gamma fold {fold_number}', flush=True)
        else:
            fold_result = linear.run_fold(
                fold_number,
                split,
                gamma,
                feature_cache,
                labels,
                output,
                MODEL_KEY,
            )
        fold_results.append(fold_result)

    result = summarize(fold_results, manifest, time.time() - started)
    model_summary_path = output / 'models' / MODEL_KEY / 'summary.json'
    rr.write_json_new(model_summary_path, result)
    rr.write_json_new(output / 'summary.json', result)
    metrics = result['metrics']
    print(
        f'COMPLETE {MODEL_DISPLAY} '
        f'AUC={metrics["auc"]["mean"]:.4f}±'
        f'{metrics["auc"]["sample_sd"]:.4f} '
        f'ACC={metrics["accuracy"]["mean"]:.4f}±'
        f'{metrics["accuracy"]["sample_sd"]:.4f}',
        flush=True,
    )


if __name__ == '__main__':
    main(parse_args())
