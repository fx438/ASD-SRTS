# -*- coding: utf-8 -*-
"""在锁定五折上复核 gamma=0.90，并仅用训练/验证数据扩展到 gamma=0.95。

本脚本不计算、读取或保存任何新增测试集预测。原正式结果保持只读，审计产物写入
独立目录。0.90 重跑用于验证审计流程是否逐值复现原实验；0.95 用于判断原网格
是否在上界截断。验证集配对 bootstrap 仅描述固定训练结果下的 AUC 差异。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

SOURCE_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = SOURCE_ROOT / 'revision_srts_v2'
MAIN_RESULT = SOURCE_ROOT / 'results_revision_v2' / 'standard5_seed42'
DEFAULT_OUTPUT = (
    SOURCE_ROOT / 'results_revision_v2' / 'gamma_boundary_audit_standard5_seed42_v1'
)
sys.path.insert(0, str(SCRIPT_ROOT))
sys.path.insert(0, str(SOURCE_ROOT))

import dataset_cc400 as dataset  # noqa: E402
import run_revision as rr  # noqa: E402

AUDIT_GAMMAS = (0.90, 0.95)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--bootstrap-repetitions', type=int, default=20_000)
    parser.add_argument('--bootstrap-seed', type=int, default=20260824)
    return parser.parse_args()


def read_json(path):
    with Path(path).open(encoding='utf-8') as fh:
        return json.load(fh)


def paired_stratified_bootstrap_auc(labels, prob_a, prob_b, repetitions, seed):
    """在同一验证折和类别内配对重采样，估计 AUC(a)-AUC(b)。"""
    labels = np.asarray(labels, dtype=np.int64)
    prob_a = np.asarray(prob_a, dtype=np.float64)
    prob_b = np.asarray(prob_b, dtype=np.float64)
    by_class = [np.flatnonzero(labels == c) for c in (0, 1)]
    if any(len(idx) == 0 for idx in by_class):
        raise ValueError('验证折必须同时包含两个类别')
    rng = np.random.default_rng(seed)
    differences = np.empty(repetitions, dtype=np.float64)
    for b in range(repetitions):
        sample = np.concatenate([
            rng.choice(idx, size=len(idx), replace=True) for idx in by_class
        ])
        differences[b] = (
            roc_auc_score(labels[sample], prob_a[sample])
            - roc_auc_score(labels[sample], prob_b[sample])
        )
    return {
        'estimate': float(roc_auc_score(labels, prob_a) - roc_auc_score(labels, prob_b)),
        'ci95_percentile': np.quantile(differences, [0.025, 0.975]).tolist(),
        'probability_difference_gt_zero': float(np.mean(differences > 0.0)),
        'repetitions': int(repetitions),
        'seed': int(seed),
        'estimand': 'fixed-model validation AUC difference; no retraining uncertainty',
    }


def numeric_summary(rows):
    return {
        'n_validation_subjects': len(rows),
        'min_eigenvalue_mean': float(np.mean([x['min_eigenvalue'] for x in rows])),
        'min_eigenvalue_min': float(np.min([x['min_eigenvalue'] for x in rows])),
        'log10_condition_number_mean': float(
            np.mean([x['log10_condition_number'] for x in rows])
        ),
        'log10_condition_number_max': float(
            np.max([x['log10_condition_number'] for x in rows])
        ),
    }


def build_train_args(main_config):
    keys = (
        'hidden', 'drop', 'k', 'lr', 'weight_decay', 'epochs', 'min_epochs',
        'patience', 'batch_size', 'eval_batch_size', 'eig_batch_size', 'noise',
    )
    return SimpleNamespace(**{key: main_config[key] for key in keys})


def run_one_fold(
    fold_number, split, x, y, raw_corr, train_args, output,
    bootstrap_repetitions, bootstrap_seed,
):
    fold_index = int(split['fold'])
    tr = np.asarray(split['train'], dtype=np.int64)
    va = np.asarray(split['val'], dtype=np.int64)
    if fold_number != fold_index + 1:
        raise RuntimeError('折编号与划分清单不一致')
    fold_seed = 42 + fold_index
    train_val = np.concatenate([tr, va])
    trials = []
    print(f'\n=== gamma boundary audit fold {fold_number}/5 ===', flush=True)
    for gamma in AUDIT_GAMMAS:
        with torch.no_grad():
            fc_train_val, diagnostics = rr.tangent_transform(
                rr.shrink(raw_corr, gamma), tr, train_val,
                diag_positions=set(va.tolist()),
                batch_size=train_args.eig_batch_size,
            )
        trained = rr.train_model(
            x[tr], fc_train_val[:len(tr)], y[tr],
            x[va], fc_train_val[len(tr):], y[va],
            train_args, seed=fold_seed, mixup=False,
        )
        trial = {
            'gamma': gamma,
            'best_val_auc': trained['best_val_auc'],
            'epochs_run': trained['epochs_run'],
            'n_parameters': trained['n_parameters'],
            'validation_labels': trained['val']['labels'],
            'validation_probabilities': trained['val']['probs'],
            'validation_subject_indices': va.tolist(),
            'numerics': diagnostics,
            'numerics_summary': numeric_summary(diagnostics),
        }
        trials.append(trial)
        print(
            f'[fold {fold_number}] gamma={gamma:.2f} '
            f'val_auc={trained["best_val_auc"]:.6f} '
            f'ep={trained["epochs_run"]}',
            flush=True,
        )
        del trained, fc_train_val
        torch.cuda.empty_cache()

    old_fold = read_json(MAIN_RESULT / 'folds' / f'fold_{fold_number}.json')
    old_trial_090 = next(x for x in old_fold['gamma_trials'] if x['gamma'] == 0.90)
    reproduction_error = float(trials[0]['best_val_auc'] - old_trial_090['best_val_auc'])
    if abs(reproduction_error) > 1e-10:
        raise RuntimeError(
            f'fold {fold_number} gamma=0.90 未复现：差值 {reproduction_error:+.12g}'
        )
    bootstrap = paired_stratified_bootstrap_auc(
        trials[0]['validation_labels'],
        trials[1]['validation_probabilities'],
        trials[0]['validation_probabilities'],
        repetitions=bootstrap_repetitions,
        seed=bootstrap_seed + fold_index,
    )
    result = {
        'fold': fold_index,
        'sizes_used': {'train': len(tr), 'validation': len(va), 'test_evaluated': 0},
        'selection_uses_test_data': False,
        'test_predictions_computed': False,
        'reference_selected_gamma': old_fold['selected_gamma'],
        'reference_gamma_090_auc': old_trial_090['best_val_auc'],
        'gamma_090_reproduction_error': reproduction_error,
        'audit_trials': trials,
        'delta_auc_095_minus_090': float(
            trials[1]['best_val_auc'] - trials[0]['best_val_auc']
        ),
        'paired_validation_bootstrap_095_minus_090': bootstrap,
    }
    rr.write_json_new(output / 'folds' / f'fold_{fold_number}.json', result)
    return result


def summarize(fold_results):
    merged_folds = []
    changed = 0
    upper_boundary = 0
    for fold_number, audit in enumerate(fold_results, start=1):
        old = read_json(MAIN_RESULT / 'folds' / f'fold_{fold_number}.json')
        curve = [
            {'gamma': float(x['gamma']), 'validation_auc': float(x['best_val_auc'])}
            for x in old['gamma_trials']
        ]
        auc_095 = float(audit['audit_trials'][1]['best_val_auc'])
        curve.append({'gamma': 0.95, 'validation_auc': auc_095})
        curve.sort(key=lambda x: x['gamma'])
        ranked = sorted(curve, key=lambda x: (-x['validation_auc'], x['gamma']))
        expanded_selected = ranked[0]
        changed_fold = not np.isclose(
            expanded_selected['gamma'], old['selected_gamma'], atol=0.0, rtol=0.0
        )
        changed += int(changed_fold)
        boundary_fold = expanded_selected['gamma'] == 0.95
        upper_boundary += int(boundary_fold)
        merged_folds.append({
            'fold': fold_number - 1,
            'original_selected_gamma': float(old['selected_gamma']),
            'expanded_selected_gamma': float(expanded_selected['gamma']),
            'selection_changed_after_adding_095': bool(changed_fold),
            'expanded_peak_at_upper_boundary': bool(boundary_fold),
            'top_minus_second_validation_auc': float(
                ranked[0]['validation_auc'] - ranked[1]['validation_auc']
            ),
            'gammas_within_0_005_of_best': [
                x['gamma'] for x in curve
                if expanded_selected['validation_auc'] - x['validation_auc'] <= 0.005 + 1e-12
            ],
            'validation_curve': curve,
            'paired_validation_bootstrap_095_minus_090': audit[
                'paired_validation_bootstrap_095_minus_090'
            ],
        })
    return {
        'analysis': 'gamma upper-bound audit on locked standard five-fold validation sets',
        'seed': 42,
        'original_grid': list(rr.GAMMAS),
        'audited_extension': [0.95],
        'folds_with_changed_selection': changed,
        'folds_selecting_new_upper_boundary_095': upper_boundary,
        'test_evaluated': False,
        'interpretation_guardrail': (
            '结果只用于判断候选网格边界和验证选择稳定性；不得据此读取或重算测试指标。'
        ),
        'folds': merged_folds,
    }


def main(args):
    if args.bootstrap_repetitions <= 0:
        raise ValueError('bootstrap 次数必须为正数')
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    final_path = output / 'summary.json'
    if final_path.exists():
        raise FileExistsError(f'拒绝覆盖已完成审计：{final_path}')

    main_config = read_json(MAIN_RESULT / 'config.json')
    split_payload = read_json(MAIN_RESULT / 'splits_manifest.json')
    if main_config['seed'] != 42 or main_config['t_use'] != 146:
        raise RuntimeError('主实验配置不是锁定的 seed=42、t_use=146')
    config = {
        'seed': 42,
        't_use': 146,
        'audit_gammas': list(AUDIT_GAMMAS),
        'main_result_read_only': str(MAIN_RESULT),
        'output': str(output),
        'bootstrap_repetitions': args.bootstrap_repetitions,
        'bootstrap_seed': args.bootstrap_seed,
        'test_evaluation_enabled': False,
        'training_hyperparameters': {
            key: main_config[key] for key in (
                'hidden', 'drop', 'k', 'lr', 'weight_decay', 'epochs',
                'min_epochs', 'patience', 'batch_size', 'eval_batch_size',
                'eig_batch_size', 'noise',
            )
        },
    }
    config_path = output / 'config.json'
    if not config_path.exists():
        rr.write_json_new(config_path, config)
    elif not args.resume:
        raise FileExistsError(f'目录已有配置；断点续跑请加 --resume：{config_path}')
    elif read_json(config_path) != config:
        raise RuntimeError('续跑配置与已有 config.json 不一致')

    started = time.time()
    ts, labels, sites = dataset.load_cc400(t_use=146, verbose=True)
    manifest = rr.load_manifest(146, labels, sites)
    if manifest != split_payload['manifest']:
        raise RuntimeError('当前数据身份清单与主实验不一致')
    rebuilt = rr.build_splits(labels, seed=42)
    if rebuilt != split_payload['splits']:
        raise RuntimeError('当前五折划分与主实验不一致')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type != 'cuda':
        raise RuntimeError('gamma 审计要求 CUDA')
    x = torch.from_numpy(ts).to(device)
    y = torch.from_numpy(labels).to(device)
    with torch.no_grad():
        raw_corr = rr.corr_matrix_all(x, main_config['eval_batch_size'])
    print(f'device={device}, X={tuple(x.shape)}, test evaluation disabled', flush=True)

    train_args = build_train_args(main_config)
    fold_results = []
    for fold_number, split in enumerate(split_payload['splits'], start=1):
        fold_path = output / 'folds' / f'fold_{fold_number}.json'
        if fold_path.exists():
            if not args.resume:
                raise FileExistsError(f'已有折结果：{fold_path}')
            result = read_json(fold_path)
            print(f'[resume] fold {fold_number}', flush=True)
        else:
            result = run_one_fold(
                fold_number, split, x, y, raw_corr, train_args, output,
                args.bootstrap_repetitions, args.bootstrap_seed,
            )
        fold_results.append(result)

    summary = summarize(fold_results)
    summary['runtime_seconds'] = float(time.time() - started)
    summary['all_gamma_090_reproduced_exactly'] = bool(all(
        abs(x['gamma_090_reproduction_error']) <= 1e-10 for x in fold_results
    ))
    rr.write_json_new(final_path, summary)
    print('\n=== gamma boundary audit complete ===', flush=True)
    for fold in summary['folds']:
        print(
            f"fold {fold['fold'] + 1}: "
            f"old={fold['original_selected_gamma']:.2f}, "
            f"expanded={fold['expanded_selected_gamma']:.2f}, "
            f"margin={fold['top_minus_second_validation_auc']:.6f}",
            flush=True,
        )
    print(f'DONE -> {final_path}', flush=True)


if __name__ == '__main__':
    main(parse_args())
