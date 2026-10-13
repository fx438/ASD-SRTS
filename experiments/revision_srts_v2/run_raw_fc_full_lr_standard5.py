# -*- coding: utf-8 -*-
"""锁定标准五折上的原始 FC 全维 svec + Logistic Regression 对齐实验。

与 SRTS + LR 使用相同的 602/86/172 划分、77,028 维 svec、训练折
StandardScaler、固定 LR 超参数及验证集阈值逻辑。唯一研究变量是表示：本脚本
使用未经收缩、未经 Fisher-Z、未经特征选择的原始 Pearson FC。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import scipy
import sklearn
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

SOURCE_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = SOURCE_ROOT / 'revision_srts_v2'
MAIN_RESULT = SOURCE_ROOT / 'results_revision_v2' / 'standard5_seed42'
SRTS_LINEAR_RESULT = (
    SOURCE_ROOT / 'results_revision_v2' / 'srts_linear_standard5_seed42_v1'
)
DEFAULT_OUTPUT = (
    SOURCE_ROOT / 'results_revision_v2' / 'raw_fc_full_lr_standard5_seed42_v1'
)
sys.path.insert(0, str(SCRIPT_ROOT))
sys.path.insert(0, str(SOURCE_ROOT))

import dataset_cc400 as dataset  # noqa: E402
import run_revision as rr  # noqa: E402

MODEL_KEY = 'lr'
MODEL_DISPLAY = 'raw FC full svec + Logistic Regression'
METRICS = (
    'auc', 'accuracy', 'sensitivity', 'specificity', 'precision', 'f1',
    'acc_at_half',
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--resume', action='store_true')
    return parser.parse_args()


def read_json(path):
    with Path(path).open(encoding='utf-8') as handle:
        return json.load(handle)


def array_sha256(array):
    contiguous = np.ascontiguousarray(array)
    return hashlib.sha256(contiguous.view(np.uint8)).hexdigest()


def save_npy_new(path, array):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f'拒绝覆盖：{path}')
    temporary = path.with_suffix('.tmp.npy')
    if temporary.exists():
        raise FileExistsError(f'存在未确认的临时文件：{temporary}')
    np.save(temporary, array, allow_pickle=False)
    temporary.replace(path)


def svec_indices(n_roi):
    rows, cols = np.triu_indices(n_roi, k=1)
    return rows.astype(np.int64), cols.astype(np.int64)


def svec_dimension(n_roi):
    return n_roi + n_roi * (n_roi - 1) // 2


def raw_fc_svec(time_series, batch_size=64, device=None):
    """计算原始 Pearson FC 的 svec：对角后接 sqrt(2)×严格上三角。"""
    time_series = np.asarray(time_series, dtype=np.float32)
    if time_series.ndim != 3:
        raise ValueError('time_series 必须为 [N, ROI, T]')
    n_subjects, n_roi, n_time = time_series.shape
    if batch_size <= 0:
        raise ValueError('batch_size 必须为正整数')
    device = torch.device(
        device if device is not None
        else ('cuda' if torch.cuda.is_available() else 'cpu')
    )
    rows_np, cols_np = svec_indices(n_roi)
    rows = torch.as_tensor(rows_np, device=device, dtype=torch.long)
    cols = torch.as_tensor(cols_np, device=device, dtype=torch.long)
    output = np.empty(
        (n_subjects, svec_dimension(n_roi)), dtype=np.float32
    )
    with torch.no_grad():
        for start in range(0, n_subjects, batch_size):
            stop = min(start + batch_size, n_subjects)
            batch = torch.from_numpy(
                np.ascontiguousarray(time_series[start:stop])
            ).to(device)
            correlation = batch @ batch.transpose(-1, -2) / n_time
            diagonal = torch.diagonal(correlation, dim1=-2, dim2=-1)
            upper = correlation[:, rows, cols] * np.sqrt(2.0)
            output[start:stop] = torch.cat(
                [diagonal, upper], dim=1
            ).cpu().numpy().astype(np.float32, copy=False)
            print(f'[raw FC] {stop}/{n_subjects}', flush=True)
    if not np.isfinite(output).all():
        raise RuntimeError('原始 FC svec 包含非有限值')
    return output


def make_model():
    return LogisticRegression(
        penalty='l2', C=1.0, solver='liblinear', dual=True,
        max_iter=5000, random_state=42,
    )


def evaluate_scores(labels, scores, threshold):
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    predictions = (scores >= threshold).astype(np.int64)
    tp = int(((predictions == 1) & (labels == 1)).sum())
    tn = int(((predictions == 0) & (labels == 0)).sum())
    fp = int(((predictions == 1) & (labels == 0)).sum())
    fn = int(((predictions == 0) & (labels == 1)).sum())
    sensitivity = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    return {
        'auc': float(roc_auc_score(labels, scores)),
        'accuracy': float((predictions == labels).mean()),
        'sensitivity': float(sensitivity),
        'specificity': float(specificity),
        'precision': float(precision),
        'f1': float(
            2 * precision * sensitivity / (precision + sensitivity)
        ) if precision + sensitivity else 0.0,
        'acc_at_half': float(
            ((scores >= 0.5).astype(np.int64) == labels).mean()
        ),
        'threshold': float(threshold),
        'scores': scores.tolist(),
        'labels': labels.tolist(),
    }


def fit_diagnostics(classifier, caught_warnings):
    diagnostics = {
        'warnings': [
            {'category': item.category.__name__, 'message': str(item.message)}
            for item in caught_warnings
        ],
    }
    for attribute in ('n_iter_', 'n_features_in_', 'classes_'):
        if hasattr(classifier, attribute):
            diagnostics[attribute] = np.asarray(
                getattr(classifier, attribute)
            ).tolist()
    if hasattr(classifier, 'coef_'):
        coefficient = np.asarray(classifier.coef_, dtype=np.float64)
        diagnostics['coefficient_sha256'] = array_sha256(coefficient)
        diagnostics['coefficient_l2_norm'] = float(np.linalg.norm(coefficient))
    if hasattr(classifier, 'intercept_'):
        diagnostics['intercept'] = np.asarray(
            classifier.intercept_, dtype=np.float64
        ).tolist()
    return diagnostics


def protocol_config(output):
    return {
        'seed': 42,
        't_use': 146,
        'per_fold_sizes': [602, 86, 172],
        'representation': 'raw Pearson FC full svec',
        'feature': (
            'raw Pearson FC svec: diagonal + sqrt(2) times strict upper triangle'
        ),
        'feature_dim': 77028,
        'fisher_z': False,
        'shrinkage': None,
        'tangent_mapping': False,
        'reference_matrix': None,
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
        'aligned_srts_result': str(SRTS_LINEAR_RESULT / 'models' / 'lr' / 'summary.json'),
        'main_split_manifest': str(MAIN_RESULT / 'splits_manifest.json'),
        'output': str(output),
    }


def load_or_create_features(time_series, output, batch_size):
    feature_root = output / 'features'
    feature_path = feature_root / 'raw_fc_full_svec_float32.npy'
    metadata_path = feature_root / 'raw_fc_full_svec_float32.json'
    n_roi = time_series.shape[1]
    expected_shape = (len(time_series), svec_dimension(n_roi))
    rows, cols = svec_indices(n_roi)
    expected_metadata = {
        'representation': 'raw Pearson FC full svec',
        'correlation': 'X @ X.T / T',
        'fisher_z': False,
        'shrinkage': None,
        'tangent_mapping': False,
        'feature_selection': None,
        'vectorization': (
            'svec: diagonal followed by sqrt(2) times strict upper triangle'
        ),
        'shape': list(expected_shape),
        'dtype': 'float32',
        'upper_triangle_rows_sha256': array_sha256(rows),
        'upper_triangle_cols_sha256': array_sha256(cols),
    }
    if feature_path.exists() or metadata_path.exists():
        if not feature_path.exists() or not metadata_path.exists():
            raise RuntimeError('原始 FC 特征缓存不完整')
        metadata = read_json(metadata_path)
        for key, value in expected_metadata.items():
            if metadata.get(key) != value:
                raise RuntimeError(f'特征缓存元数据不一致：{key}')
        features = np.load(feature_path, mmap_mode='r', allow_pickle=False)
        if features.shape != expected_shape or features.dtype != np.float32:
            raise RuntimeError('原始 FC 特征缓存形状或类型异常')
        if not np.isfinite(features).all():
            raise RuntimeError('原始 FC 特征缓存包含非有限值')
        if rr.sha256_file(feature_path) != metadata['file_sha256']:
            raise RuntimeError('原始 FC 特征缓存哈希不一致')
        print(f'[cache] {feature_path} {features.shape}', flush=True)
        return features, metadata

    features = raw_fc_svec(time_series, batch_size=batch_size)
    save_npy_new(feature_path, features)
    metadata = {
        **expected_metadata,
        'file_sha256': rr.sha256_file(feature_path),
    }
    rr.write_json_new(metadata_path, metadata)
    print(f'[cache] saved {feature_path} {features.shape}', flush=True)
    return np.load(feature_path, mmap_mode='r', allow_pickle=False), metadata


def run_fold(fold_number, split, features, feature_metadata, labels, output):
    train = np.asarray(split['train'], dtype=np.int64)
    validation_indices = np.asarray(split['val'], dtype=np.int64)
    test = np.asarray(split['test'], dtype=np.int64)
    if (len(train), len(validation_indices), len(test)) != (602, 86, 172):
        raise RuntimeError('划分规模不是锁定的 602/86/172')

    scaler = StandardScaler().fit(np.asarray(features[train]))
    x_train = scaler.transform(np.asarray(features[train]))
    x_validation = scaler.transform(np.asarray(features[validation_indices]))
    classifier = make_model()
    fit_started = time.time()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        classifier.fit(x_train, labels[train])
    fit_seconds = time.time() - fit_started
    validation_scores = classifier.predict_proba(x_validation)[:, 1]
    threshold = rr.best_threshold(
        validation_scores, labels[validation_indices]
    )
    validation = {
        'auc': float(roc_auc_score(
            labels[validation_indices], validation_scores
        )),
        'threshold': float(threshold),
        'scores': validation_scores.tolist(),
        'labels': labels[validation_indices].tolist(),
    }

    # 仅在模型与验证阈值锁定后变换并评估测试折。
    x_test = scaler.transform(np.asarray(features[test]))
    test_scores = classifier.predict_proba(x_test)[:, 1]
    tested = evaluate_scores(labels[test], test_scores, threshold)
    fold_result = {
        'fold': fold_number - 1,
        'model_key': MODEL_KEY,
        'model_name': MODEL_DISPLAY,
        'sizes': [len(train), len(validation_indices), len(test)],
        'class_counts': split['class_counts'],
        'train_indices': train.tolist(),
        'validation_indices': validation_indices.tolist(),
        'test_indices': test.tolist(),
        'representation': {
            'name': 'raw Pearson FC full svec',
            'fisher_z': False,
            'shrinkage': None,
            'tangent_mapping': False,
            'feature_selection': None,
            'feature_dim': int(features.shape[1]),
            'feature_file_sha256': feature_metadata['file_sha256'],
        },
        'standard_scaler': {
            'fit_indices': train.tolist(),
            'fit_on_training_only': True,
            'mean_sha256': array_sha256(scaler.mean_.astype(np.float64)),
            'scale_sha256': array_sha256(scaler.scale_.astype(np.float64)),
        },
        'model_fit_seconds': float(fit_seconds),
        'fit_diagnostics': fit_diagnostics(classifier, caught),
        'validation': validation,
        'test': tested,
    }
    fold_path = output / 'models' / MODEL_KEY / 'folds' / f'fold_{fold_number}.json'
    rr.write_json_new(fold_path, fold_result)
    print(
        f'[{MODEL_DISPLAY} fold {fold_number}/5] '
        f'val_auc={validation["auc"]:.4f} '
        f'test_auc={tested["auc"]:.4f} '
        f'test_acc={tested["accuracy"]:.4f} fit={fit_seconds:.1f}s',
        flush=True,
    )
    return fold_result


def summarize(fold_results, manifest, runtime_seconds):
    summary = {
        'model_key': MODEL_KEY,
        'model_name': MODEL_DISPLAY,
        'protocol': {
            'folds': 5,
            'per_fold_sizes': [602, 86, 172],
            'test_union_n': 860,
            'test_count_per_subject': 1,
            'seed': 42,
            'feature_dim': 77028,
            'feature_selection': None,
            'sd_definition': 'sample SD across five folds (ddof=1)',
        },
        'metrics': {},
        'runtime_seconds': float(runtime_seconds),
        'oof_predictions': [],
    }
    for metric in METRICS:
        values = np.asarray([
            fold['test'][metric] for fold in fold_results
        ], dtype=np.float64)
        summary['metrics'][metric] = {
            'mean': float(values.mean()),
            'sample_sd': float(values.std(ddof=1)),
            'fold_values': values.tolist(),
        }
    validation_auc = np.asarray([
        fold['validation']['auc'] for fold in fold_results
    ], dtype=np.float64)
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
        raise RuntimeError('原始 FC 全维 LR 的 OOF 测试覆盖不是每人一次')
    summary['oof_predictions'].sort(key=lambda row: row['subject_index'])
    return summary


def main(args):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    summary_path = output / 'models' / MODEL_KEY / 'summary.json'
    if summary_path.exists():
        raise FileExistsError(f'拒绝覆盖已完成结果：{summary_path}')

    split_payload = read_json(MAIN_RESULT / 'splits_manifest.json')
    config = protocol_config(output)
    config_path = output / 'config.json'
    if not config_path.exists():
        rr.write_json_new(config_path, config)
    elif read_json(config_path) != config:
        raise RuntimeError('当前配置与已有 config.json 不一致')

    srts_config = read_json(SRTS_LINEAR_RESULT / 'config.json')
    if config['model_hyperparameters'] != srts_config['model_hyperparameters']['lr']:
        raise RuntimeError('LR 超参数与 SRTS + LR 不一致')
    if config['per_fold_sizes'] != srts_config['per_fold_sizes']:
        raise RuntimeError('折规模与 SRTS + LR 不一致')

    started = time.time()
    time_series, labels, sites = dataset.load_cc400(t_use=146, verbose=True)
    manifest = rr.load_manifest(146, labels, sites)
    if manifest != split_payload['manifest']:
        raise RuntimeError('当前身份清单与主实验不一致')
    splits = rr.build_splits(labels, seed=42)
    if splits != split_payload['splits']:
        raise RuntimeError('当前划分与锁定主实验不一致')

    features, feature_metadata = load_or_create_features(
        time_series, output, args.batch_size
    )
    if features.shape != (860, 77028):
        raise RuntimeError('原始 FC 全维特征形状不是 [860, 77028]')

    provenance = {
        'script_sha256': rr.sha256_file(Path(__file__).resolve()),
        'data_cache_sha256': rr.sha256_file(
            SOURCE_ROOT / 'cc400_cache_t146.npz'
        ),
        'main_summary_sha256': rr.sha256_file(MAIN_RESULT / 'summary.json'),
        'split_manifest_sha256': rr.sha256_file(
            MAIN_RESULT / 'splits_manifest.json'
        ),
        'srts_lr_summary_sha256': rr.sha256_file(
            SRTS_LINEAR_RESULT / 'models' / 'lr' / 'summary.json'
        ),
        'numpy_version': np.__version__,
        'scipy_version': scipy.__version__,
        'sklearn_version': sklearn.__version__,
        'torch_version': torch.__version__,
        'feature_matrix_shape': list(features.shape),
        'feature_matrix_sha256': feature_metadata['file_sha256'],
    }
    provenance_path = output / 'provenance.json'
    if not provenance_path.exists():
        rr.write_json_new(provenance_path, provenance)
    elif read_json(provenance_path) != provenance:
        raise RuntimeError('源码、数据、划分或特征与已有来源记录不一致')

    fold_results = []
    for fold_number, split in enumerate(splits, start=1):
        fold_path = output / 'models' / MODEL_KEY / 'folds' / f'fold_{fold_number}.json'
        if fold_path.exists():
            if not args.resume:
                raise FileExistsError(f'已有折结果：{fold_path}')
            fold_result = read_json(fold_path)
            print(f'[resume] fold {fold_number}', flush=True)
        else:
            fold_result = run_fold(
                fold_number, split, features, feature_metadata,
                labels, output,
            )
        fold_results.append(fold_result)

    result = summarize(fold_results, manifest, time.time() - started)
    rr.write_json_new(summary_path, result)
    # 根目录汇总便于与其他独立实验目录统一读取。
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
