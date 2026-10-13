# -*- coding: utf-8 -*-
"""严格对齐的 raw FC 全维 LR 与 SRTS LR 配对 OOF 比较。

两项实验使用相同五折、特征维度、StandardScaler、LR 参数和验证集阈值规则。
本脚本只读取已锁定结果，以折内、类别分层的被试级配对 bootstrap 估计 pooled
OOF 指标差的条件不确定性；不重新训练，也不用于选择表示或分类器。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

SOURCE_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = SOURCE_ROOT / 'revision_srts_v2'
RESULT_ROOT = SOURCE_ROOT / 'results_revision_v2'
RAW_ROOT = RESULT_ROOT / 'raw_fc_full_lr_standard5_seed42_v1'
SRTS_ROOT = RESULT_ROOT / 'srts_linear_standard5_seed42_v1'
DEFAULT_OUTPUT = RESULT_ROOT / 'aligned_lr_representation_comparison_v1'
MODEL_KEY = 'lr'
N_SUBJECTS = 860

sys.path.insert(0, str(SCRIPT_ROOT))

import postprocess_revision as post  # noqa: E402
import run_revision as rr  # noqa: E402


def read_json(path):
    with Path(path).open(encoding='utf-8') as handle:
        return json.load(handle)


def write_text_new(path, text):
    path = Path(path)
    with path.open('x', encoding='utf-8', newline='\n') as handle:
        handle.write(text)


def protocol_config(n_bootstrap, seed):
    return {
        'comparison': 'SRTS LR minus raw Pearson FC full-svec LR',
        'n_bootstrap': int(n_bootstrap),
        'seed': int(seed),
        'resampling': (
            'paired subject bootstrap stratified within fold and class'
        ),
        'estimand': (
            'pooled OOF metric difference conditional on the fixed five '
            'fitted fold models'
        ),
        'used_for_model_selection': False,
        'limitations': (
            'does not include uncertainty from re-splitting, re-training, '
            'or random initialization'
        ),
    }


def align_oof_rows(raw_rows, srts_rows):
    raw_by_index = {int(row['subject_index']): row for row in raw_rows}
    srts_by_index = {int(row['subject_index']): row for row in srts_rows}
    if len(raw_by_index) != len(raw_rows) or len(srts_by_index) != len(srts_rows):
        raise AssertionError('OOF subject_index 存在重复')
    if set(raw_by_index) != set(srts_by_index):
        raise AssertionError('两项 OOF 的被试集合不一致')
    indices = sorted(raw_by_index)
    labels = []
    folds = []
    raw_scores = []
    srts_scores = []
    for index in indices:
        raw = raw_by_index[index]
        srts = srts_by_index[index]
        if int(raw['label']) != int(srts['label']):
            raise AssertionError(f'被试 {index} 标签不一致')
        if int(raw['fold']) != int(srts['fold']):
            raise AssertionError(f'被试 {index} OOF 折不一致')
        labels.append(int(raw['label']))
        folds.append(int(raw['fold']))
        raw_scores.append(float(raw['score']))
        srts_scores.append(float(srts['score']))
    return (
        np.asarray(labels, dtype=np.int64),
        np.asarray(folds, dtype=np.int64),
        {
            'raw_fc': np.asarray(raw_scores, dtype=np.float64),
            'srts': np.asarray(srts_scores, dtype=np.float64),
        },
    )


def predictions_from_folds(scores, fold_records, n_subjects):
    scores = np.asarray(scores, dtype=np.float64)
    if len(scores) != n_subjects:
        raise AssertionError('分数长度与被试数不一致')
    predictions = np.zeros(n_subjects, dtype=np.int64)
    seen = np.zeros(n_subjects, dtype=np.int64)
    for fold in fold_records:
        indices = np.asarray(fold['test_indices'], dtype=np.int64)
        threshold = float(fold['test']['threshold'])
        predictions[indices] = (scores[indices] >= threshold).astype(np.int64)
        seen[indices] += 1
    if not np.all(seen == 1):
        raise AssertionError('五个测试折没有让每名被试恰好预测一次')
    return predictions


def load_folds(root):
    return [
        read_json(root / 'models' / MODEL_KEY / 'folds' / f'fold_{fold}.json')
        for fold in range(1, 6)
    ]


def validate_alignment(raw_config, srts_config, raw_folds, srts_folds):
    checks = {
        'same_seed': raw_config['seed'] == srts_config['seed'] == 42,
        'same_fold_sizes': (
            raw_config['per_fold_sizes']
            == srts_config['per_fold_sizes']
            == [602, 86, 172]
        ),
        'same_feature_dim': (
            raw_config['feature_dim'] == srts_config['feature_dim'] == 77028
        ),
        'same_feature_selection': (
            raw_config['feature_selection'] is None
            and srts_config['feature_selection'] is None
        ),
        'same_standardization': (
            raw_config['standardization'] == srts_config['standardization']
        ),
        'same_threshold_rule': (
            raw_config['threshold_selection']
            == srts_config['threshold_selection']
        ),
        'same_lr_hyperparameters': (
            raw_config['model_hyperparameters']
            == srts_config['model_hyperparameters']['lr']
        ),
        'classifier_specific_tuning_absent': (
            raw_config['classifier_specific_tuning'] is False
            and srts_config['classifier_specific_tuning'] is False
        ),
        'same_train_validation_test_indices': all(
            raw['train_indices'] == srts['train_indices']
            and raw['validation_indices'] == srts['validation_indices']
            and raw['test_indices'] == srts['test_indices']
            for raw, srts in zip(raw_folds, srts_folds)
        ),
        'only_representation_changed': (
            raw_config['representation'] == 'raw Pearson FC full svec'
            and raw_config['tangent_mapping'] is False
            and raw_config['shrinkage'] is None
            and srts_config['feature'].startswith('SRTS tangent svec')
        ),
    }
    if not all(checks.values()):
        failed = [key for key, passed in checks.items() if not passed]
        raise AssertionError('协议对齐失败：' + ', '.join(failed))
    return checks


def metric_summary(summary, metric):
    values = summary['metrics'][metric]
    return {
        'mean': float(values['mean']),
        'sample_sd': float(values['sample_sd']),
        'fold_values': [float(value) for value in values['fold_values']],
    }


def five_fold_comparison(raw_summary, srts_summary, metric):
    raw = np.asarray(
        raw_summary['metrics'][metric]['fold_values'], dtype=np.float64
    )
    srts = np.asarray(
        srts_summary['metrics'][metric]['fold_values'], dtype=np.float64
    )
    differences = srts - raw
    return {
        'raw_fc': metric_summary(raw_summary, metric),
        'srts': metric_summary(srts_summary, metric),
        'difference_of_fold_means': float(srts.mean() - raw.mean()),
        'paired_fold_differences': differences.tolist(),
        'paired_fold_difference_sample_sd': float(differences.std(ddof=1)),
        'note': (
            'descriptive five-fold summary; distinct from the pooled OOF '
            'bootstrap estimand'
        ),
    }


def make_markdown(report):
    fold_auc = report['five_fold_comparison']['auc']
    fold_acc = report['five_fold_comparison']['accuracy']
    comparison = report['paired_pooled_oof_bootstrap']['comparisons'][0]
    pooled_auc = comparison['metrics']['auc']
    pooled_acc = comparison['metrics']['accuracy']
    return '\n'.join([
        '# 严格对齐 LR 表示比较',
        '',
        '> 两项实验使用相同锁定五折、77,028 维特征、训练折 '
        '`StandardScaler`、固定 LR 参数及验证集阈值规则；唯一研究变量为 FC 表示。',
        '',
        '## 五折测试结果',
        '',
        '| 表示 | AUC | ACC |',
        '|---|---:|---:|',
        f'| 原始 Pearson FC 全维 svec | '
        f'{fold_auc["raw_fc"]["mean"]:.4f}±'
        f'{fold_auc["raw_fc"]["sample_sd"]:.4f} | '
        f'{fold_acc["raw_fc"]["mean"]:.4f}±'
        f'{fold_acc["raw_fc"]["sample_sd"]:.4f} |',
        f'| SRTS | {fold_auc["srts"]["mean"]:.4f}±'
        f'{fold_auc["srts"]["sample_sd"]:.4f} | '
        f'{fold_acc["srts"]["mean"]:.4f}±'
        f'{fold_acc["srts"]["sample_sd"]:.4f} |',
        '',
        f'- 五折 AUC 均值描述性差：'
        f'{fold_auc["difference_of_fold_means"]:+.4f}。',
        f'- 五折 ACC 均值描述性差：'
        f'{fold_acc["difference_of_fold_means"]:+.4f}。',
        '',
        '## 配对 pooled OOF bootstrap',
        '',
        f'- ΔAUC：{pooled_auc["point_difference"]:+.5f}，95% CI '
        f'[{pooled_auc["percentile_95_ci"][0]:+.5f}, '
        f'{pooled_auc["percentile_95_ci"][1]:+.5f}]。',
        f'- ΔACC：{pooled_acc["point_difference"]:+.5f}，95% CI '
        f'[{pooled_acc["percentile_95_ci"][0]:+.5f}, '
        f'{pooled_acc["percentile_95_ci"][1]:+.5f}]。',
        '',
        '说明：bootstrap 在折内按类别分层并对被试成对重采样；CI 条件于固定的'
        '五个已训练模型，不包含重新划分、重新训练或随机初始化的不确定性。'
        'pooled OOF 差与五折均值差是不同估计量，不能混用。',
        '',
    ])


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw-root', type=Path, default=RAW_ROOT)
    parser.add_argument('--srts-root', type=Path, default=SRTS_ROOT)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--bootstrap', type=int, default=20000)
    parser.add_argument('--seed', type=int, default=20260824)
    return parser.parse_args()


def main():
    args = parse_args()
    raw_root = args.raw_root.resolve()
    srts_root = args.srts_root.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)

    raw_config = read_json(raw_root / 'config.json')
    srts_config = read_json(srts_root / 'config.json')
    raw_summary = read_json(raw_root / 'models' / MODEL_KEY / 'summary.json')
    srts_summary = read_json(srts_root / 'models' / MODEL_KEY / 'summary.json')
    raw_folds = load_folds(raw_root)
    srts_folds = load_folds(srts_root)
    alignment_checks = validate_alignment(
        raw_config, srts_config, raw_folds, srts_folds
    )

    labels, fold_ids, probabilities = align_oof_rows(
        raw_summary['oof_predictions'], srts_summary['oof_predictions']
    )
    if len(labels) != N_SUBJECTS:
        raise AssertionError('对齐后的 OOF 不是 860 人')
    if int((labels == 1).sum()) != 417 or int((labels == 0).sum()) != 443:
        raise AssertionError('OOF 类别计数不是 ASD=417、TC=443')
    if set(np.unique(fold_ids).tolist()) != set(range(5)):
        raise AssertionError('OOF 折编号不是 0 到 4')
    if not all(np.isfinite(scores).all() for scores in probabilities.values()):
        raise AssertionError('OOF 分数包含非有限值')

    predictions = {
        'raw_fc': predictions_from_folds(
            probabilities['raw_fc'], raw_folds, N_SUBJECTS
        ),
        'srts': predictions_from_folds(
            probabilities['srts'], srts_folds, N_SUBJECTS
        ),
    }
    bootstrap = post.paired_bootstrap(
        labels,
        fold_ids,
        probabilities,
        predictions,
        [('srts', 'raw_fc')],
        n_bootstrap=args.bootstrap,
        seed=args.seed,
    )
    report = {
        'protocol': protocol_config(args.bootstrap, args.seed),
        'alignment_checks': alignment_checks,
        'oof_alignment': {
            'n_subjects': int(len(labels)),
            'asd_n': int((labels == 1).sum()),
            'tc_n': int((labels == 0).sum()),
            'fold_ids': sorted(np.unique(fold_ids).tolist()),
            'same_subject_label_fold_alignment': True,
        },
        'five_fold_comparison': {
            'auc': five_fold_comparison(raw_summary, srts_summary, 'auc'),
            'accuracy': five_fold_comparison(
                raw_summary, srts_summary, 'accuracy'
            ),
        },
        'paired_pooled_oof_bootstrap': bootstrap,
        'input_sha256': {
            'raw_summary': rr.sha256_file(
                raw_root / 'models' / MODEL_KEY / 'summary.json'
            ),
            'srts_summary': rr.sha256_file(
                srts_root / 'models' / MODEL_KEY / 'summary.json'
            ),
            **{
                f'raw_fold_{fold}': rr.sha256_file(
                    raw_root / 'models' / MODEL_KEY / 'folds'
                    / f'fold_{fold}.json'
                )
                for fold in range(1, 6)
            },
            **{
                f'srts_fold_{fold}': rr.sha256_file(
                    srts_root / 'models' / MODEL_KEY / 'folds'
                    / f'fold_{fold}.json'
                )
                for fold in range(1, 6)
            },
        },
        'provenance': {
            'script_sha256': rr.sha256_file(Path(__file__).resolve()),
            'bootstrap_helper_script_sha256': rr.sha256_file(
                SCRIPT_ROOT / 'postprocess_revision.py'
            ),
        },
    }
    rr.write_json_new(output / 'report.json', report)
    write_text_new(output / 'report.md', make_markdown(report))
    print(json.dumps({
        'status': 'passed',
        'output': str(output),
        'five_fold_auc_difference': report['five_fold_comparison']['auc'][
            'difference_of_fold_means'
        ],
        'pooled_oof_auc': bootstrap['comparisons'][0]['metrics']['auc'],
        'pooled_oof_accuracy': bootstrap['comparisons'][0]['metrics']['accuracy'],
    }, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
