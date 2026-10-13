# -*- coding: utf-8 -*-
"""对齐 SRTS 与原始 FC 的锁定测试 OOF，进行站点和 TR 描述性分析。"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import shutil
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import sklearn
from sklearn.metrics import accuracy_score, roc_auc_score

DEFAULT_ROOT = Path(__file__).resolve().parent
DEFAULT_RESULT_ROOT = DEFAULT_ROOT / 'results_revision_v2'
DEFAULT_OUTPUT = (
    DEFAULT_RESULT_ROOT / 'site_tr_oof_stratified_standard5_seed42_v1'
)


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path):
    with Path(path).open(encoding='utf-8') as handle:
        return json.load(handle)


def _read_csv(path):
    with Path(path).open(newline='', encoding='utf-8-sig') as handle:
        return list(csv.DictReader(handle))


def _finite_score(value, label):
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f'{label} 必须为有限值')
    return result


def _validate_thresholds(thresholds, folds, model_name):
    expected = set(folds)
    actual = {int(key) for key in thresholds}
    if actual != expected:
        raise ValueError(
            f'{model_name} 阈值折集合不一致：实际 {sorted(actual)}，'
            f'预期 {sorted(expected)}'
        )
    clean = {}
    for fold in sorted(expected):
        value = _finite_score(thresholds[fold], f'{model_name} fold={fold} 阈值')
        if not 0.0 <= value <= 1.0:
            raise ValueError(f'{model_name} fold={fold} 阈值超出 [0, 1]')
        clean[fold] = value
    return clean


def _index_oof(rows, model_name):
    indexed = {}
    for source in rows:
        required = {
            'subject_index', 'sub_id', 'file_id', 'site_id',
            'label', 'fold', 'score',
        }
        missing = required.difference(source)
        if missing:
            raise ValueError(f'{model_name} OOF 缺少字段：{sorted(missing)}')
        index = int(source['subject_index'])
        if index in indexed:
            raise ValueError(f'{model_name} OOF subject_index={index} 重复')
        row = dict(source)
        row['subject_index'] = index
        row['sub_id'] = int(row['sub_id'])
        row['label'] = int(row['label'])
        row['fold'] = int(row['fold'])
        row['score'] = _finite_score(row['score'], f'{model_name} OOF score')
        if row['label'] not in (0, 1):
            raise ValueError(f'{model_name} OOF 标签必须为 0/1')
        indexed[index] = row
    return indexed


def _index_phenotypic(rows):
    indexed = {}
    for row in rows:
        try:
            sub_id = int(float(str(row.get('SUB_ID', '')).strip()))
        except ValueError:
            continue
        if sub_id in indexed:
            raise ValueError(f'表型文件 SUB_ID={sub_id} 重复')
        indexed[sub_id] = row
    return indexed


def _match_tr_rule(manifest_row, rules):
    site = str(manifest_row['site_id'])
    file_id = str(manifest_row['file_id'])
    matches = []
    for rule in rules:
        if str(rule.get('site_id')) != site:
            continue
        prefix = rule.get('file_id_prefix')
        if prefix is None or file_id.startswith(str(prefix)):
            matches.append(rule)
    if len(matches) != 1:
        raise ValueError(
            f'subject_index={manifest_row["index"]} 的 TR 规则匹配数为 '
            f'{len(matches)}（site={site}, file_id={file_id}）'
        )
    rule = matches[0]
    tr = _finite_score(rule.get('tr_seconds'), 'TR')
    if tr <= 0.0:
        raise ValueError('TR 必须大于 0')
    return rule, tr


def _validate_tr_map(tr_map):
    if not isinstance(tr_map, dict) or not tr_map.get('rules'):
        raise ValueError('TR 映射必须包含非空 rules')
    if not str(tr_map.get('official_index_url', '')).startswith('https://'):
        raise ValueError('TR 映射缺少官方 HTTPS 索引页')
    rule_ids = set()
    for rule in tr_map['rules']:
        required = {
            'rule_id', 'site_id', 'file_id_prefix', 'tr_seconds',
            'source_url', 'source_sha256',
        }
        missing = required.difference(rule)
        if missing:
            raise ValueError(f'TR 规则缺少字段：{sorted(missing)}')
        rule_id = str(rule['rule_id'])
        if rule_id in rule_ids:
            raise ValueError(f'TR 规则 ID 重复：{rule_id}')
        rule_ids.add(rule_id)
        if not str(rule['source_url']).startswith('https://'):
            raise ValueError(f'TR 规则 {rule_id} 的来源不是 HTTPS URL')
        source_hash = str(rule['source_sha256'])
        if len(source_hash) != 64 or any(
            char not in '0123456789abcdef' for char in source_hash.lower()
        ):
            raise ValueError(f'TR 规则 {rule_id} 的 PDF SHA-256 无效')
        if _finite_score(rule['tr_seconds'], 'TR') <= 0.0:
            raise ValueError(f'TR 规则 {rule_id} 的 TR 无效')


def _auc_or_none(labels, scores):
    if len(set(labels)) < 2:
        return None
    return float(roc_auc_score(labels, scores))


def _metric_row(rows, group_kind, group_value):
    labels = [int(row['label']) for row in rows]
    srts_scores = [float(row['srts_score']) for row in rows]
    raw_scores = [float(row['raw_score']) for row in rows]
    srts_pred = [int(row['srts_predicted_label']) for row in rows]
    raw_pred = [int(row['raw_predicted_label']) for row in rows]
    srts_auc = _auc_or_none(labels, srts_scores)
    raw_auc = _auc_or_none(labels, raw_scores)
    available = srts_auc is not None and raw_auc is not None
    return {
        'group_kind': group_kind,
        'group_value': str(group_value),
        'n': len(rows),
        'n_asd': int(sum(labels)),
        'n_tc': int(len(labels) - sum(labels)),
        'auc_available': available,
        'auc_unavailable_reason': None if available else 'single class',
        'srts_auc': srts_auc,
        'raw_auc': raw_auc,
        'delta_auc': None if not available else float(srts_auc - raw_auc),
        'srts_accuracy': float(accuracy_score(labels, srts_pred)),
        'raw_accuracy': float(accuracy_score(labels, raw_pred)),
        'delta_accuracy': float(
            accuracy_score(labels, srts_pred) - accuracy_score(labels, raw_pred)
        ),
    }


def summarize_groups(subject_rows, group_field, group_kind):
    grouped = defaultdict(list)
    for row in subject_rows:
        grouped[row[group_field]].append(row)
    keys = sorted(grouped, key=float) if group_kind == 'TR' else sorted(grouped)
    result = []
    for key in keys:
        value = f'{float(key):.3f}' if group_kind == 'TR' else str(key)
        result.append(_metric_row(grouped[key], group_kind, value))
    return result


def _weighted_available_mean(rows, field):
    available = [row for row in rows if row[field] is not None]
    if not available:
        return None
    weights = np.asarray([row['n'] for row in available], dtype=np.float64)
    values = np.asarray([row[field] for row in available], dtype=np.float64)
    return float(np.average(values, weights=weights))


def _site_influence(subject_rows, site_metrics):
    result = []
    for site_row in site_metrics:
        site = site_row['group_value']
        retained = [row for row in subject_rows if row['site_id'] != site]
        metric = _metric_row(retained, 'leave_one_site_out', site)
        metric['excluded_site'] = site
        metric['n_excluded'] = site_row['n']
        metric['n_remaining'] = len(retained)
        result.append(metric)
    return result


def build_analysis(
    manifest,
    phenotypic_rows,
    srts_oof,
    raw_oof,
    srts_thresholds,
    raw_thresholds,
    tr_map,
):
    """构造对齐的被试、站点和 TR 描述性 OOF 分析。"""
    _validate_tr_map(tr_map)
    if not manifest:
        raise ValueError('manifest 为空')
    manifest_by_index = {}
    for source in manifest:
        row = dict(source)
        index = int(row['index'])
        if index in manifest_by_index:
            raise ValueError(f'manifest index={index} 重复')
        row['index'] = index
        row['sub_id'] = int(row['sub_id'])
        row['label'] = int(row['label'])
        row['dx_group'] = int(row['dx_group'])
        manifest_by_index[index] = row
    expected_indices = set(range(len(manifest)))
    if set(manifest_by_index) != expected_indices:
        raise ValueError('manifest index 必须连续覆盖 0..N-1')

    srts_by_index = _index_oof(srts_oof, 'SRTS')
    raw_by_index = _index_oof(raw_oof, 'raw FC')
    if set(srts_by_index) != expected_indices or set(raw_by_index) != expected_indices:
        raise ValueError('两个 OOF 都必须完整覆盖 manifest')
    folds = sorted({row['fold'] for row in srts_by_index.values()})
    if folds != sorted({row['fold'] for row in raw_by_index.values()}):
        raise ValueError('SRTS 与 raw FC 的 OOF 折集合不一致')
    srts_thresholds = _validate_thresholds(srts_thresholds, folds, 'SRTS')
    raw_thresholds = _validate_thresholds(raw_thresholds, folds, 'raw FC')
    pheno_by_sub = _index_phenotypic(phenotypic_rows)

    subjects = []
    for index in sorted(expected_indices):
        manifest_row = manifest_by_index[index]
        srts = srts_by_index[index]
        raw = raw_by_index[index]
        identity_fields = ('sub_id', 'file_id', 'site_id', 'label')
        mismatch = [
            field for field in identity_fields
            if str(srts[field]) != str(raw[field])
            or str(srts[field]) != str(manifest_row[
                'index' if field == 'subject_index' else field
            ])
        ]
        if srts['fold'] != raw['fold']:
            mismatch.append('fold')
        if mismatch:
            raise ValueError(
                f'subject_index={index} 的 SRTS/raw/manifest OOF 身份或折不一致：'
                + ', '.join(sorted(set(mismatch)))
            )
        pheno = pheno_by_sub.get(manifest_row['sub_id'])
        if pheno is None:
            raise ValueError(f'SUB_ID={manifest_row["sub_id"]} 缺少表型记录')
        pheno_site = str(pheno.get('SITE_ID', '')).upper()
        pheno_dx = int(float(str(pheno.get('DX_GROUP', '')).strip()))
        if pheno_site != str(manifest_row['site_id']).upper() or (
            pheno_dx != manifest_row['dx_group']
        ):
            raise ValueError(
                f'SUB_ID={manifest_row["sub_id"]} 的表型站点或诊断不一致'
            )
        rule, tr = _match_tr_rule(manifest_row, tr_map['rules'])
        fold = srts['fold']
        srts_threshold = srts_thresholds[fold]
        raw_threshold = raw_thresholds[fold]
        subjects.append({
            'subject_index': index,
            'sub_id': manifest_row['sub_id'],
            'file_id': str(manifest_row['file_id']),
            'site_id': str(manifest_row['site_id']),
            'label': int(manifest_row['label']),
            'diagnosis': 'ASD' if manifest_row['label'] == 1 else 'TC',
            'fold': fold,
            'tr_seconds': tr,
            'tr_group': f'{tr:.3f}',
            'tr_rule_id': str(rule['rule_id']),
            'n_timepoints_original': int(manifest_row['n_timepoints_original']),
            'srts_score': srts['score'],
            'srts_threshold': srts_threshold,
            'srts_predicted_label': int(srts['score'] >= srts_threshold),
            'raw_score': raw['score'],
            'raw_threshold': raw_threshold,
            'raw_predicted_label': int(raw['score'] >= raw_threshold),
        })

    site_metrics = summarize_groups(subjects, 'site_id', 'site')
    tr_metrics = summarize_groups(subjects, 'tr_group', 'TR')
    overall = _metric_row(subjects, 'overall', 'all subjects')
    influence = _site_influence(subjects, site_metrics)
    available_sites = [row for row in site_metrics if row['auc_available']]
    available_tr = [row for row in tr_metrics if row['auc_available']]
    influence_delta = [row['delta_auc'] for row in influence]
    result = {
        'protocol': {
            'analysis_scope': 'locked test OOF only',
            'models_retrained': False,
            'threshold_source': 'each model validation partition in each fold',
            'thresholds_test_selected': False,
            'group_analysis_role': 'descriptive robustness analysis',
            'site_influence_role': (
                'prediction-level leave-one-site-out sensitivity without retraining'
            ),
            'uses_dsm_iv_tr_as_repetition_time': False,
            'dsm_iv_tr_meaning': 'DSM-IV diagnostic category, not scanner TR',
            'tr_source': 'official ABIDE-I functional scan protocol PDFs',
            'auc_scope': 'pooled OOF within each stratum',
            'accuracy_scope': 'locked fold-specific validation thresholds',
            'inferential_claims': False,
        },
        'subjects': subjects,
        'overall_metrics': overall,
        'site_metrics': site_metrics,
        'tr_metrics': tr_metrics,
        'site_influence': influence,
        'stratified_summary': {
            'n_subjects': len(subjects),
            'n_sites': len(site_metrics),
            'n_tr_groups': len(tr_metrics),
            'site_auc_available_n': len(available_sites),
            'site_delta_auc_positive_n': sum(
                row['delta_auc'] > 0.0 for row in available_sites
            ),
            'site_delta_auc_zero_n': sum(
                math.isclose(row['delta_auc'], 0.0, abs_tol=1e-15)
                for row in available_sites
            ),
            'site_delta_auc_negative_n': sum(
                row['delta_auc'] < 0.0 for row in available_sites
            ),
            'site_subject_weighted_delta_auc': _weighted_available_mean(
                site_metrics, 'delta_auc'
            ),
            'site_subject_weighted_delta_accuracy': _weighted_available_mean(
                site_metrics, 'delta_accuracy'
            ),
            'tr_auc_available_n': len(available_tr),
            'tr_delta_auc_positive_n': sum(
                row['delta_auc'] > 0.0 for row in available_tr
            ),
            'tr_delta_auc_zero_n': sum(
                math.isclose(row['delta_auc'], 0.0, abs_tol=1e-15)
                for row in available_tr
            ),
            'tr_delta_auc_negative_n': sum(
                row['delta_auc'] < 0.0 for row in available_tr
            ),
            'leave_one_site_out_delta_auc_min': float(min(influence_delta)),
            'leave_one_site_out_delta_auc_max': float(max(influence_delta)),
            'leave_one_site_out_delta_auc_positive_all': all(
                value > 0.0 for value in influence_delta
            ),
        },
        'tr_mapping_audit': {
            'version': tr_map.get('version'),
            'official_index_url': tr_map['official_index_url'],
            'rule_count': len(tr_map['rules']),
            'used_rule_ids': sorted({row['tr_rule_id'] for row in subjects}),
            'unmatched_subjects': 0,
            'ambiguous_subjects': 0,
            'tr_groups_seconds': sorted({row['tr_seconds'] for row in subjects}),
            'all_subjects_mapped': True,
        },
        'phenotypic_audit': {
            'manifest_subjects_matched': len(subjects),
            'missing_subjects': 0,
            'duplicate_subjects': 0,
            'site_or_diagnosis_conflicts': 0,
        },
    }
    return result


def _assert_fold_json_against_oof(fold_data, fold_number, oof_by_index, model_name):
    if int(fold_data['fold']) + 1 != fold_number:
        raise ValueError(f'{model_name} fold_{fold_number}.json 折编号不一致')
    validation_threshold = float(fold_data['validation']['threshold'])
    test_threshold = float(fold_data['test']['threshold'])
    if not math.isclose(validation_threshold, test_threshold, rel_tol=0.0, abs_tol=0.0):
        raise ValueError(f'{model_name} fold={fold_number} 测试阈值不是锁定验证阈值')
    indices = [int(value) for value in fold_data['test_indices']]
    scores = [float(value) for value in fold_data['test']['scores']]
    labels = [int(value) for value in fold_data['test']['labels']]
    if not (len(indices) == len(scores) == len(labels)):
        raise ValueError(f'{model_name} fold={fold_number} 测试数组长度不一致')
    for index, score, label in zip(indices, scores, labels):
        oof = oof_by_index.get(index)
        if oof is None or oof['fold'] != fold_number:
            raise ValueError(f'{model_name} fold={fold_number} 与 OOF 索引不一致')
        if oof['label'] != label or not math.isclose(
            oof['score'], score, rel_tol=0.0, abs_tol=1e-15
        ):
            raise ValueError(f'{model_name} fold={fold_number} 与 OOF 数值不一致')
    return validation_threshold, indices


def load_model_bundle(result_dir, model_name):
    result_dir = Path(result_dir)
    summary_path = result_dir / 'models' / 'lr' / 'summary.json'
    summary = _read_json(summary_path)
    source_oof = summary['oof_predictions']
    source_folds = {int(row['fold']) for row in source_oof}
    if source_folds == set(range(5)):
        oof = [
            {**row, 'fold': int(row['fold']) + 1}
            for row in source_oof
        ]
    elif source_folds == set(range(1, 6)):
        oof = [dict(row) for row in source_oof]
    else:
        raise ValueError(
            f'{model_name} OOF 折编号既不是 0..4，也不是 1..5：'
            f'{sorted(source_folds)}'
        )
    oof_by_index = _index_oof(oof, model_name)
    thresholds = {}
    test_indices = {}
    source_paths = [summary_path]
    for fold_number in range(1, 6):
        path = result_dir / 'models' / 'lr' / 'folds' / f'fold_{fold_number}.json'
        data = _read_json(path)
        threshold, indices = _assert_fold_json_against_oof(
            data, fold_number, oof_by_index, model_name
        )
        thresholds[fold_number] = threshold
        test_indices[fold_number] = indices
        source_paths.append(path)
    union = [index for fold in sorted(test_indices) for index in test_indices[fold]]
    if len(union) != len(set(union)) or set(union) != set(oof_by_index):
        raise ValueError(f'{model_name} 五折测试索引未恰好覆盖 OOF 一次')
    return {
        'oof': oof,
        'thresholds': thresholds,
        'summary': summary,
        'test_indices': test_indices,
        'source_paths': source_paths,
    }


def validate_split_manifest(split_data, model_bundles):
    manifest = split_data['manifest']
    splits = split_data['splits']
    if len(splits) != 5:
        raise ValueError('标准五折 manifest 必须有 5 折')
    test_counts = Counter()
    for split in splits:
        fold = int(split['fold']) + 1
        test = [int(value) for value in split['test']]
        test_counts.update(test)
        for model_name, bundle in model_bundles.items():
            if test != bundle['test_indices'][fold]:
                raise ValueError(
                    f'{model_name} fold={fold} 测试索引与 splits_manifest 不一致'
                )
    expected = set(range(len(manifest)))
    if set(test_counts) != expected or set(test_counts.values()) != {1}:
        raise ValueError('五折测试集没有恰好覆盖全部被试一次')


def _csv_value(value):
    if value is None:
        return ''
    if isinstance(value, bool):
        return str(value).lower()
    return value


def _write_csv(path, rows, fields):
    with Path(path).open('x', newline='', encoding='utf-8-sig') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _csv_value(row.get(field)) for field in fields})


def _json_without_subjects(result):
    return {key: value for key, value in result.items() if key != 'subjects'}


def _fmt(value):
    return 'NA' if value is None else f'{float(value):.4f}'


def _metrics_table(rows):
    lines = [
        '| 分层 | n（ASD/TC） | SRTS AUC | raw FC AUC | ΔAUC | '
        'SRTS ACC | raw FC ACC | ΔACC |',
        '|---|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for row in rows:
        lines.append(
            f'| {row["group_value"]} | {row["n"]} '
            f'（{row["n_asd"]}/{row["n_tc"]}） | '
            f'{_fmt(row["srts_auc"])} | {_fmt(row["raw_auc"])} | '
            f'{_fmt(row["delta_auc"])} | {_fmt(row["srts_accuracy"])} | '
            f'{_fmt(row["raw_accuracy"])} | {_fmt(row["delta_accuracy"])} |'
        )
    return '\n'.join(lines)


def render_markdown(result):
    report = _json_without_subjects(result)
    overall = report['overall_metrics']
    summary = report['stratified_summary']
    return (
        '# SRTS 与原始 FC 的站点/TR 分层 OOF 描述性分析\n\n'
        '本分析只使用 5 个锁定模型产生的测试集 OOF 预测，不重新训练模型，'
        '也不据此选择表示、γ、分类器或阈值。ACC 使用各模型在各折验证集上'
        '预先锁定的阈值。站点/TR 子组结果仅用于描述性稳健性分析。\n\n'
        '## 总体 pooled OOF\n\n'
        f'- 被试：{overall["n"]}（ASD {overall["n_asd"]}，TC {overall["n_tc"]}）\n'
        f'- SRTS AUC：{_fmt(overall["srts_auc"])}；raw FC AUC：'
        f'{_fmt(overall["raw_auc"])}；ΔAUC：{_fmt(overall["delta_auc"])}\n'
        f'- SRTS ACC：{_fmt(overall["srts_accuracy"])}；raw FC ACC：'
        f'{_fmt(overall["raw_accuracy"])}；ΔACC：{_fmt(overall["delta_accuracy"])}\n\n'
        '## 站点分层\n\n'
        f'{_metrics_table(report["site_metrics"])}\n\n'
        '## TR 分层\n\n'
        f'{_metrics_table(report["tr_metrics"])}\n\n'
        '## 单站点影响敏感性\n\n'
        f'- 可计算 AUC 的站点中，ΔAUC 为正/零/负的站点数：'
        f'{summary["site_delta_auc_positive_n"]}/'
        f'{summary["site_delta_auc_zero_n"]}/'
        f'{summary["site_delta_auc_negative_n"]}。\n'
        f'- 逐一排除一个站点但不重新训练时，pooled OOF ΔAUC 范围：'
        f'{_fmt(summary["leave_one_site_out_delta_auc_min"])}～'
        f'{_fmt(summary["leave_one_site_out_delta_auc_max"])}。\n'
        f'- 排除任一单站点后 ΔAUC 是否均为正：'
        f'{"是" if summary["leave_one_site_out_delta_auc_positive_all"] else "否"}。\n\n'
        '## TR 来源与解释边界\n\n'
        '- TR 来自 ABIDE-I 官方各站点功能扫描协议 PDF；CMU 根据 FILE_ID 中的 '
        '`CMU_a_`/`CMU_b_` 子协议分别映射。\n'
        '- 表型字段 `DSM_IV_TR` 表示 DSM-IV 诊断类别，不是 repetition time，'
        '本分析未将其用作扫描 TR。\n'
        '- 小样本站点的 AUC 可能不稳定；本表不提供子组显著性检验，不支持因果'
        '解释，也不是站点外泛化或 LOSO 重新训练实验。\n'
    )


def write_bundle(result, output_dir, provenance_sources, script_path, tr_map):
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f'拒绝覆盖已存在目录：{output_dir}')
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(
        prefix=f'.{output_dir.name}.', dir=output_dir.parent
    ))
    try:
        paths = {
            'subjects': staging / 'subject_level_oof_site_tr_v1.csv',
            'sites': staging / 'site_metrics_v1.csv',
            'tr': staging / 'tr_metrics_v1.csv',
            'influence': staging / 'leave_one_site_out_oof_sensitivity_v1.csv',
            'report_json': staging / 'report.json',
            'report_md': staging / 'report.md',
            'provenance': staging / 'provenance.json',
        }
        subject_fields = [
            'subject_index', 'sub_id', 'file_id', 'site_id', 'label',
            'diagnosis', 'fold', 'tr_seconds', 'tr_group', 'tr_rule_id',
            'n_timepoints_original', 'srts_score', 'srts_threshold',
            'srts_predicted_label', 'raw_score', 'raw_threshold',
            'raw_predicted_label',
        ]
        metric_fields = [
            'group_kind', 'group_value', 'n', 'n_asd', 'n_tc',
            'auc_available', 'auc_unavailable_reason', 'srts_auc', 'raw_auc',
            'delta_auc', 'srts_accuracy', 'raw_accuracy', 'delta_accuracy',
        ]
        influence_fields = metric_fields + [
            'excluded_site', 'n_excluded', 'n_remaining'
        ]
        _write_csv(paths['subjects'], result['subjects'], subject_fields)
        _write_csv(paths['sites'], result['site_metrics'], metric_fields)
        _write_csv(paths['tr'], result['tr_metrics'], metric_fields)
        _write_csv(paths['influence'], result['site_influence'], influence_fields)
        paths['report_json'].write_text(
            json.dumps(_json_without_subjects(result), ensure_ascii=False, indent=2)
            + '\n',
            encoding='utf-8',
        )
        paths['report_md'].write_text(render_markdown(result), encoding='utf-8')
        unique_sources = []
        seen = set()
        for source in [*provenance_sources, Path(script_path)]:
            resolved = str(Path(source).resolve())
            if resolved not in seen:
                unique_sources.append(Path(source))
                seen.add(resolved)
        provenance = {
            'artifact_version': 'site-tr-oof-stratified-v1',
            'analysis_scope': 'locked test OOF only',
            'models_retrained': False,
            'uses_test_predictions': True,
            'used_for_model_selection': False,
            'uses_dsm_iv_tr_as_repetition_time': False,
            'tr_source_type': 'official ABIDE-I functional protocol PDFs',
            'official_index_url': tr_map['official_index_url'],
            'tr_rules': tr_map['rules'],
            'input_files': [
                {'path': str(path.resolve()), 'sha256': _sha256(path)}
                for path in unique_sources
            ],
            'output_sha256': {
                key: _sha256(path)
                for key, path in paths.items() if key != 'provenance'
            },
            'software': {
                'python': platform.python_version(),
                'numpy': np.__version__,
                'scikit_learn': sklearn.__version__,
                'platform': platform.platform(),
            },
        }
        paths['provenance'].write_text(
            json.dumps(provenance, ensure_ascii=False, indent=2) + '\n',
            encoding='utf-8',
        )
        staging.rename(output_dir)
        return {key: output_dir / path.name for key, path in paths.items()}
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--manifest',
        default=DEFAULT_RESULT_ROOT / 'standard5_seed42' / 'splits_manifest.json',
        type=Path,
    )
    parser.add_argument(
        '--phenotypic', default=Path(os.environ.get('ABIDE_CC400_DIR', Path(__file__).resolve().parents[1] / 'data' / 'ABIDE_CC400_Complete')) / 'Phenotypic.csv',
        type=Path,
    )
    parser.add_argument(
        '--srts-result',
        default=DEFAULT_RESULT_ROOT / 'srts_linear_standard5_seed42_v1',
        type=Path,
    )
    parser.add_argument(
        '--raw-result',
        default=DEFAULT_RESULT_ROOT / 'raw_fc_full_lr_standard5_seed42_v1',
        type=Path,
    )
    parser.add_argument(
        '--tr-map', default=DEFAULT_ROOT / 'tr_acquisition_map_abide1_v1.json',
        type=Path,
    )
    parser.add_argument('--output', default=DEFAULT_OUTPUT, type=Path)
    return parser.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    split_data = _read_json(args.manifest)
    phenotypic = _read_csv(args.phenotypic)
    tr_map = _read_json(args.tr_map)
    srts = load_model_bundle(args.srts_result, 'SRTS')
    raw = load_model_bundle(args.raw_result, 'raw FC')
    validate_split_manifest(split_data, {'SRTS': srts, 'raw FC': raw})
    result = build_analysis(
        manifest=split_data['manifest'],
        phenotypic_rows=phenotypic,
        srts_oof=srts['oof'],
        raw_oof=raw['oof'],
        srts_thresholds=srts['thresholds'],
        raw_thresholds=raw['thresholds'],
        tr_map=tr_map,
    )
    source_paths = [
        args.manifest, args.phenotypic, args.tr_map,
        *srts['source_paths'], *raw['source_paths'],
    ]
    outputs = write_bundle(
        result=result,
        output_dir=args.output,
        provenance_sources=source_paths,
        script_path=Path(__file__),
        tr_map=tr_map,
    )
    print(json.dumps({
        'status': 'passed',
        'output': str(args.output),
        'outputs': {key: str(path) for key, path in outputs.items()},
        'overall_metrics': result['overall_metrics'],
        'stratified_summary': result['stratified_summary'],
        'tr_mapping_audit': result['tr_mapping_audit'],
    }, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
