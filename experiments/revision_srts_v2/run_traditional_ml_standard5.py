# -*- coding: utf-8 -*-
"""在锁定标准五折上逐组重跑传统 FC 分类器。

沿用旧稿流程：Fisher-Z 非对角连接、训练折内 t 检验选 top-1200、训练折内
StandardScaler、验证集选分类阈值。仅替换为 602/86/172 的锁定划分；不针对
任何分类器单独调参。强制依次完成 LR、KNN、普通 MLP、RF、RBF-SVM。
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
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

SOURCE_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = SOURCE_ROOT / 'revision_srts_v2'
MAIN_RESULT = SOURCE_ROOT / 'results_revision_v2' / 'standard5_seed42'
DEFAULT_OUTPUT = (
    SOURCE_ROOT / 'results_revision_v2' / 'traditional_ml_standard5_seed42_v1'
)
sys.path.insert(0, str(SCRIPT_ROOT))
sys.path.insert(0, str(SOURCE_ROOT))

import dataset_cc400 as dataset  # noqa: E402
import run_revision as rr  # noqa: E402

TOPK = 1200
MODEL_RANDOM_STATE = 0
RUN_ORDER = ('lr', 'knn', 'mlp', 'rf', 'svm')
MODEL_DISPLAY = {
    'lr': 'LogisticRegression',
    'knn': 'KNN',
    'mlp': 'MLP',
    'rf': 'RandomForest',
    'svm': 'RBF-SVM',
}
METRICS = (
    'auc', 'accuracy', 'sensitivity', 'specificity', 'precision', 'f1',
    'acc_at_half',
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--model', choices=RUN_ORDER, required=True)
    parser.add_argument('--resume', action='store_true')
    return parser.parse_args()


def read_json(path):
    with Path(path).open(encoding='utf-8') as fh:
        return json.load(fh)


def array_sha256(array):
    contiguous = np.ascontiguousarray(array)
    return hashlib.sha256(contiguous.view(np.uint8)).hexdigest()


def fc_features(ts):
    """(N, ROI, T) -> Fisher-Z 非对角上三角特征。"""
    n_subjects, n_roi, n_time = ts.shape
    rows, cols = np.triu_indices(n_roi, k=1)
    output = np.empty((n_subjects, len(rows)), dtype=np.float32)
    for i, subject in enumerate(ts):
        corr = subject @ subject.T / n_time
        values = np.clip(corr[rows, cols], -0.999, 0.999)
        output[i] = np.arctanh(values).astype(np.float32, copy=False)
        if (i + 1) % 100 == 0 or i + 1 == n_subjects:
            print(f'[FC] {i + 1}/{n_subjects}', flush=True)
    if not np.isfinite(output).all():
        raise RuntimeError('Fisher-Z 特征包含非有限值')
    return output, rows, cols


def make_model(model_key):
    """保持旧稿模型与超参数，不做分类器特异调参。"""
    factories = {
        'lr': lambda: LogisticRegression(max_iter=2000, C=1.0),
        'knn': lambda: KNeighborsClassifier(n_neighbors=15),
        'mlp': lambda: MLPClassifier(
            hidden_layer_sizes=(64,), max_iter=500,
            random_state=MODEL_RANDOM_STATE,
        ),
        'rf': lambda: RandomForestClassifier(
            n_estimators=500, n_jobs=-1,
            random_state=MODEL_RANDOM_STATE,
        ),
        'svm': lambda: SVC(
            kernel='rbf', C=1.0, gamma='scale', probability=True,
            random_state=MODEL_RANDOM_STATE,
        ),
    }
    return factories[model_key]()


def evaluate_probabilities(labels, probabilities, threshold):
    labels = np.asarray(labels, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    predictions = (probabilities >= threshold).astype(np.int64)
    tp = int(((predictions == 1) & (labels == 1)).sum())
    tn = int(((predictions == 0) & (labels == 0)).sum())
    fp = int(((predictions == 1) & (labels == 0)).sum())
    fn = int(((predictions == 0) & (labels == 1)).sum())
    sensitivity = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    return {
        'auc': float(roc_auc_score(labels, probabilities)),
        'accuracy': float((predictions == labels).mean()),
        'sensitivity': float(sensitivity),
        'specificity': float(specificity),
        'precision': float(precision),
        'f1': float(
            2 * precision * sensitivity / (precision + sensitivity)
        ) if precision + sensitivity else 0.0,
        'acc_at_half': float(
            ((probabilities >= 0.5).astype(np.int64) == labels).mean()
        ),
        'threshold': float(threshold),
        'probabilities': probabilities.tolist(),
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
    return diagnostics


def run_fold(fold_number, split, features, labels, output, model_key):
    tr = np.asarray(split['train'], dtype=np.int64)
    va = np.asarray(split['val'], dtype=np.int64)
    te = np.asarray(split['test'], dtype=np.int64)
    if (len(tr), len(va), len(te)) != (602, 86, 172):
        raise RuntimeError('划分规模不是锁定的 602/86/172')

    # t 检验与标准化均严格只拟合训练折。
    train_features = np.asarray(features[tr])
    t_values, _ = stats.ttest_ind(
        train_features[labels[tr] == 1],
        train_features[labels[tr] == 0],
        axis=0,
    )
    scores = np.abs(np.nan_to_num(t_values))
    selected = np.argsort(-scores, kind='stable')[:TOPK]
    scaler = StandardScaler().fit(train_features[:, selected])
    x_train = scaler.transform(train_features[:, selected])
    x_val = scaler.transform(np.asarray(features[va])[:, selected])
    x_test = scaler.transform(np.asarray(features[te])[:, selected])
    del train_features, t_values, scores

    classifier = make_model(model_key)
    fit_started = time.time()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        classifier.fit(x_train, labels[tr])
    fit_seconds = time.time() - fit_started
    val_probability = classifier.predict_proba(x_val)[:, 1]
    test_probability = classifier.predict_proba(x_test)[:, 1]
    threshold = rr.best_threshold(val_probability, labels[va])
    validation = {
        'auc': float(roc_auc_score(labels[va], val_probability)),
        'threshold': float(threshold),
        'probabilities': val_probability.tolist(),
        'labels': labels[va].tolist(),
    }
    tested = evaluate_probabilities(labels[te], test_probability, threshold)
    fold_result = {
        'fold': fold_number - 1,
        'model_key': model_key,
        'model_name': MODEL_DISPLAY[model_key],
        'sizes': [len(tr), len(va), len(te)],
        'class_counts': split['class_counts'],
        'test_indices': te.tolist(),
        'preprocessing_fit_indices': tr.tolist(),
        'feature_selection': {
            'fit_on_training_only': True,
            'method': 'absolute two-sample t statistic',
            'topk': TOPK,
            'selected_indices': selected.tolist(),
            'selected_indices_sha256': array_sha256(
                selected.astype(np.int64)
            ),
        },
        'standard_scaler': {
            'fit_on_training_only': True,
            'mean_sha256': array_sha256(scaler.mean_.astype(np.float64)),
            'scale_sha256': array_sha256(scaler.scale_.astype(np.float64)),
        },
        'model_fit_seconds': float(fit_seconds),
        'fit_diagnostics': fit_diagnostics(classifier, caught),
        'validation': validation,
        'test': tested,
    }
    fold_path = (
        output / 'models' / model_key / 'folds' / f'fold_{fold_number}.json'
    )
    rr.write_json_new(fold_path, fold_result)
    print(
        f'[{MODEL_DISPLAY[model_key]} fold {fold_number}/5] '
        f'val_auc={validation["auc"]:.4f} '
        f'test_auc={tested["auc"]:.4f} '
        f'test_acc={tested["accuracy"]:.4f} '
        f'fit={fit_seconds:.1f}s',
        flush=True,
    )
    return fold_result


def summarize_model(model_key, fold_results, manifest, runtime_seconds):
    summary = {
        'model_key': model_key,
        'model_name': MODEL_DISPLAY[model_key],
        'protocol': {
            'folds': 5,
            'per_fold_sizes': [602, 86, 172],
            'test_union_n': 860,
            'test_count_per_subject': 1,
            'seed': 42,
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
    val_auc = np.asarray([
        fold['validation']['auc'] for fold in fold_results
    ], dtype=np.float64)
    summary['validation_auc'] = {
        'mean': float(val_auc.mean()),
        'sample_sd': float(val_auc.std(ddof=1)),
        'fold_values': val_auc.tolist(),
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
                'probability': fold['test']['probabilities'][position],
            })
    if not np.all(seen == 1):
        raise RuntimeError('传统方法 OOF 测试覆盖不是每人一次')
    summary['oof_predictions'].sort(key=lambda row: row['subject_index'])
    return summary


def load_or_create_features(ts, output):
    cache_path = output / 'preprocessing' / 'fisher_z_upper_float32.npy'
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    rows, cols = np.triu_indices(ts.shape[1], k=1)
    if cache_path.exists():
        features = np.load(cache_path, mmap_mode='r', allow_pickle=False)
        expected = (len(ts), len(rows))
        if features.shape != expected or features.dtype != np.float32:
            raise RuntimeError('已有 Fisher-Z 特征缓存的形状或类型不一致')
        if not np.isfinite(features).all():
            raise RuntimeError('已有 Fisher-Z 特征缓存包含非有限值')
        print(f'[cache] feature matrix {features.shape}', flush=True)
        return features, rows, cols

    features, rows, cols = fc_features(ts)
    temporary = cache_path.with_suffix('.tmp.npy')
    np.save(temporary, features, allow_pickle=False)
    temporary.replace(cache_path)
    print(f'[cache] saved {cache_path}', flush=True)
    return np.load(cache_path, mmap_mode='r', allow_pickle=False), rows, cols


def maybe_write_aggregate(output):
    summaries = {}
    for key in RUN_ORDER:
        path = output / 'models' / key / 'summary.json'
        if not path.exists():
            return None
        summaries[key] = read_json(path)
    aggregate = {
        'protocol': summaries['lr']['protocol'],
        'run_order': list(RUN_ORDER),
        'models': {
            key: {
                'name': summaries[key]['model_name'],
                'metrics': summaries[key]['metrics'],
                'validation_auc': summaries[key]['validation_auc'],
                'runtime_seconds': summaries[key]['runtime_seconds'],
                'summary_path': str(
                    output / 'models' / key / 'summary.json'
                ),
            }
            for key in RUN_ORDER
        },
    }
    final_path = output / 'summary.json'
    if final_path.exists():
        raise FileExistsError(f'拒绝覆盖已有总汇总：{final_path}')
    rr.write_json_new(final_path, aggregate)
    return final_path


def protocol_config(output):
    return {
        'seed': 42,
        't_use': 146,
        'topk': TOPK,
        'feature': 'Fisher-Z Pearson FC strict upper triangle',
        'feature_selection': (
            'absolute two-sample t statistic, training fold only'
        ),
        'standardization': 'StandardScaler, training fold only',
        'threshold_selection': (
            'maximum validation accuracy, tie closest to 0.5'
        ),
        'model_random_state': MODEL_RANDOM_STATE,
        'model_hyperparameters': {
            'lr': {'C': 1.0, 'max_iter': 2000},
            'knn': {'n_neighbors': 15},
            'mlp': {'hidden_layer_sizes': [64], 'max_iter': 500},
            'rf': {'n_estimators': 500},
            'svm': {
                'kernel': 'rbf', 'C': 1.0,
                'gamma': 'scale', 'probability': True,
            },
        },
        'classifier_specific_tuning': False,
        'run_order': list(RUN_ORDER),
        'main_split_manifest': str(MAIN_RESULT / 'splits_manifest.json'),
        'output': str(output),
    }


def main(args):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    model_key = args.model
    model_root = output / 'models' / model_key
    model_summary_path = model_root / 'summary.json'
    if model_summary_path.exists():
        raise FileExistsError(f'拒绝覆盖已完成结果：{model_summary_path}')

    # 强制按 LR -> KNN -> MLP -> RF -> RBF-SVM 逐组完成和核验。
    model_position = RUN_ORDER.index(model_key)
    for previous in RUN_ORDER[:model_position]:
        previous_summary = output / 'models' / previous / 'summary.json'
        if not previous_summary.exists():
            raise RuntimeError(f'必须先完成并核验前一组：{previous_summary}')

    split_payload = read_json(MAIN_RESULT / 'splits_manifest.json')
    config = protocol_config(output)
    config_path = output / 'config.json'
    if not config_path.exists():
        rr.write_json_new(config_path, config)
    elif read_json(config_path) != config:
        raise RuntimeError('当前配置与已有 config.json 不一致')

    started = time.time()
    ts, labels, sites = dataset.load_cc400(t_use=146, verbose=True)
    manifest = rr.load_manifest(146, labels, sites)
    if manifest != split_payload['manifest']:
        raise RuntimeError('当前身份清单与主实验不一致')
    splits = rr.build_splits(labels, seed=42)
    if splits != split_payload['splits']:
        raise RuntimeError('当前划分与锁定主实验不一致')
    features, rows, cols = load_or_create_features(ts, output)

    provenance_path = output / 'provenance.json'
    provenance = {
        'script_sha256': rr.sha256_file(Path(__file__).resolve()),
        'data_cache_sha256': rr.sha256_file(
            SOURCE_ROOT / 'cc400_cache_t146.npz'
        ),
        'numpy_version': np.__version__,
        'scipy_version': scipy.__version__,
        'sklearn_version': sklearn.__version__,
        'feature_matrix_shape': list(features.shape),
        'feature_matrix_sha256': array_sha256(features),
        'upper_triangle_rows_sha256': array_sha256(
            rows.astype(np.int64)
        ),
        'upper_triangle_cols_sha256': array_sha256(
            cols.astype(np.int64)
        ),
    }
    if not provenance_path.exists():
        rr.write_json_new(provenance_path, provenance)
    elif read_json(provenance_path) != provenance:
        raise RuntimeError('源码、数据或 Fisher-Z 特征与已有来源记录不一致')

    fold_results = []
    for fold_number, split in enumerate(splits, start=1):
        fold_path = model_root / 'folds' / f'fold_{fold_number}.json'
        if fold_path.exists():
            if not args.resume:
                raise FileExistsError(f'已有折结果：{fold_path}')
            fold_result = read_json(fold_path)
            print(f'[resume] {model_key} fold {fold_number}', flush=True)
        else:
            fold_result = run_fold(
                fold_number, split, features, labels, output, model_key
            )
        fold_results.append(fold_result)

    result = summarize_model(
        model_key, fold_results, manifest, time.time() - started
    )
    rr.write_json_new(model_summary_path, result)
    metrics = result['metrics']
    print(
        f'COMPLETE {MODEL_DISPLAY[model_key]} '
        f'AUC={metrics["auc"]["mean"]:.4f}±'
        f'{metrics["auc"]["sample_sd"]:.4f} '
        f'ACC={metrics["accuracy"]["mean"]:.4f}±'
        f'{metrics["accuracy"]["sample_sd"]:.4f}',
        flush=True,
    )
    aggregate_path = maybe_write_aggregate(output)
    if aggregate_path is not None:
        print(f'ALL COMPLETE -> {aggregate_path}', flush=True)
    else:
        print(f'DONE -> {model_summary_path}', flush=True)


if __name__ == '__main__':
    main(parse_args())
