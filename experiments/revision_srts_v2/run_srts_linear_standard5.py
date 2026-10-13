# -*- coding: utf-8 -*-
"""锁定标准五折上的 SRTS 切线特征线性分类器实验。

每折固定使用正式主实验选择的 gamma，参考矩阵仅由 602 人训练集计算；切线
矩阵用保持 Frobenius 内积的 svec（对角 + sqrt(2)×严格上三角）向量化。仅在
训练折拟合 StandardScaler 和线性分类器，验证集只选阈值，测试集最后评估。
LR 与线性 SVM 按顺序独立运行，结果与特征缓存均拒绝覆盖。
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
from sklearn.svm import SVC

SOURCE_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = SOURCE_ROOT / 'revision_srts_v2'
MAIN_RESULT = SOURCE_ROOT / 'results_revision_v2' / 'standard5_seed42'
DEFAULT_OUTPUT = (
    SOURCE_ROOT / 'results_revision_v2' / 'srts_linear_standard5_seed42_v1'
)
sys.path.insert(0, str(SCRIPT_ROOT))
sys.path.insert(0, str(SOURCE_ROOT))

import dataset_cc400 as dataset  # noqa: E402
import run_revision as rr  # noqa: E402

EXPECTED_GAMMAS = (0.90, 0.90, 0.70, 0.50, 0.90)
RUN_ORDER = ('lr', 'svm')
MODEL_DISPLAY = {
    'lr': 'SRTS + Logistic Regression',
    'svm': 'SRTS + linear SVM',
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


def make_model(model_key):
    if model_key == 'lr':
        return LogisticRegression(
            penalty='l2', C=1.0, solver='liblinear', dual=True,
            max_iter=5000, random_state=42,
        )
    return SVC(
        kernel='linear', C=1.0, probability=True, random_state=42,
        cache_size=4096,
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
    for attribute in ('n_iter_', 'n_features_in_', 'classes_', 'n_support_'):
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


class TangentFeatureCache:
    def __init__(self, ts, labels, splits, output, eval_batch_size, eig_batch_size):
        self.ts = ts
        self.labels = labels
        self.splits = splits
        self.output = output
        self.eval_batch_size = eval_batch_size
        self.eig_batch_size = eig_batch_size
        self.n_roi = ts.shape[1]
        self.feature_dim = self.n_roi + self.n_roi * (self.n_roi - 1) // 2
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._x = None
        self._raw_corr = None
        rows, cols = np.triu_indices(self.n_roi, k=1)
        self.rows_np = rows.astype(np.int64)
        self.cols_np = cols.astype(np.int64)

    def _ensure_raw_corr(self):
        if self._raw_corr is not None:
            return
        if self.device.type != 'cuda':
            raise RuntimeError('生成切线特征需要 CUDA')
        self._x = torch.from_numpy(self.ts).to(self.device)
        with torch.no_grad():
            self._raw_corr = rr.corr_matrix_all(
                self._x, self.eval_batch_size
            )
        print(
            f'[GPU] raw FC ready {tuple(self._raw_corr.shape)}',
            flush=True,
        )

    def _svec(self, tangent):
        rows = torch.as_tensor(
            self.rows_np, device=tangent.device, dtype=torch.long
        )
        cols = torch.as_tensor(
            self.cols_np, device=tangent.device, dtype=torch.long
        )
        diagonal = torch.diagonal(tangent, dim1=-2, dim2=-1)
        upper = tangent[:, rows, cols] * np.sqrt(2.0)
        return torch.cat([diagonal, upper], dim=1).cpu().numpy().astype(
            np.float32, copy=False
        )

    def _expected_metadata(self, fold_number, part, target_indices, gamma):
        split = self.splits[fold_number - 1]
        return {
            'fold': fold_number - 1,
            'part': part,
            'gamma': float(gamma),
            'reference_fit_indices': split['train'],
            'target_indices': [int(v) for v in target_indices],
            'n_roi': self.n_roi,
            'feature_dim': self.feature_dim,
            'vectorization': (
                'svec: diagonal followed by sqrt(2) times strict upper triangle'
            ),
            'reference_uses_training_only': True,
        }

    def get(self, fold_number, part, target_indices, gamma, diag_positions=None):
        fold_root = self.output / 'features' / f'fold_{fold_number}'
        feature_path = fold_root / f'{part}.npy'
        metadata_path = fold_root / f'{part}.json'
        expected = self._expected_metadata(
            fold_number, part, target_indices, gamma
        )
        if feature_path.exists() or metadata_path.exists():
            if not feature_path.exists() or not metadata_path.exists():
                raise RuntimeError(f'特征缓存不完整：{fold_root}')
            metadata = read_json(metadata_path)
            for key, value in expected.items():
                if metadata.get(key) != value:
                    raise RuntimeError(f'特征缓存元数据不一致：{metadata_path}::{key}')
            features = np.load(feature_path, mmap_mode='r', allow_pickle=False)
            if features.shape != (len(target_indices), self.feature_dim):
                raise RuntimeError(f'特征缓存形状不一致：{feature_path}')
            if features.dtype != np.float32 or not np.isfinite(features).all():
                raise RuntimeError(f'特征缓存数值异常：{feature_path}')
            if rr.sha256_file(feature_path) != metadata['file_sha256']:
                raise RuntimeError(f'特征缓存哈希不一致：{feature_path}')
            print(f'[cache] {feature_path} {features.shape}', flush=True)
            return features, metadata

        self._ensure_raw_corr()
        split = self.splits[fold_number - 1]
        with torch.no_grad():
            base = rr.shrink(self._raw_corr, gamma)
            tangent, diagnostics = rr.tangent_transform(
                base,
                split['train'],
                target_indices,
                diag_positions=diag_positions,
                batch_size=self.eig_batch_size,
            )
            features = self._svec(tangent)
        del base, tangent
        torch.cuda.empty_cache()
        save_npy_new(feature_path, features)
        metadata = {
            **expected,
            'shape': list(features.shape),
            'dtype': str(features.dtype),
            'file_sha256': rr.sha256_file(feature_path),
            'numerical_diagnostics': diagnostics,
        }
        rr.write_json_new(metadata_path, metadata)
        print(f'[cache] saved {feature_path} {features.shape}', flush=True)
        return np.load(feature_path, mmap_mode='r', allow_pickle=False), metadata


def run_fold(fold_number, split, gamma, feature_cache, labels, output, model_key):
    tr = np.asarray(split['train'], dtype=np.int64)
    va = np.asarray(split['val'], dtype=np.int64)
    te = np.asarray(split['test'], dtype=np.int64)
    if (len(tr), len(va), len(te)) != (602, 86, 172):
        raise RuntimeError('划分规模不是锁定的 602/86/172')
    train_val_indices = np.concatenate([tr, va])

    # 先只构造训练/验证表示、拟合模型并锁定阈值。
    train_val_features, train_val_metadata = feature_cache.get(
        fold_number,
        'train_val',
        train_val_indices,
        gamma,
        diag_positions=va,
    )
    scaler = StandardScaler().fit(np.asarray(train_val_features[:len(tr)]))
    x_train = scaler.transform(np.asarray(train_val_features[:len(tr)]))
    x_val = scaler.transform(np.asarray(train_val_features[len(tr):]))
    classifier = make_model(model_key)
    fit_started = time.time()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        classifier.fit(x_train, labels[tr])
    fit_seconds = time.time() - fit_started
    val_scores = classifier.predict_proba(x_val)[:, 1]
    threshold = rr.best_threshold(val_scores, labels[va])
    validation = {
        'auc': float(roc_auc_score(labels[va], val_scores)),
        'threshold': float(threshold),
        'scores': val_scores.tolist(),
        'labels': labels[va].tolist(),
    }

    # 模型与阈值锁定后才构造并评估测试表示。
    test_features, test_metadata = feature_cache.get(
        fold_number, 'test', te, gamma
    )
    x_test = scaler.transform(np.asarray(test_features))
    test_scores = classifier.predict_proba(x_test)[:, 1]
    tested = evaluate_scores(labels[te], test_scores, threshold)
    fold_result = {
        'fold': fold_number - 1,
        'model_key': model_key,
        'model_name': MODEL_DISPLAY[model_key],
        'gamma': float(gamma),
        'sizes': [len(tr), len(va), len(te)],
        'class_counts': split['class_counts'],
        'train_indices': tr.tolist(),
        'validation_indices': va.tolist(),
        'test_indices': te.tolist(),
        'representation': {
            'name': 'SRTS tangent svec',
            'reference_fit_indices': tr.tolist(),
            'reference_uses_training_only': True,
            'feature_selection': None,
            'feature_dim': feature_cache.feature_dim,
            'train_val_feature_sha256': train_val_metadata['file_sha256'],
            'test_feature_sha256': test_metadata['file_sha256'],
        },
        'standard_scaler': {
            'fit_indices': tr.tolist(),
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
        f'gamma={gamma:.2f} val_auc={validation["auc"]:.4f} '
        f'test_auc={tested["auc"]:.4f} '
        f'test_acc={tested["accuracy"]:.4f} fit={fit_seconds:.1f}s',
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
            'gammas': list(EXPECTED_GAMMAS),
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
                'score': fold['test']['scores'][position],
            })
    if not np.all(seen == 1):
        raise RuntimeError('线性分类器 OOF 测试覆盖不是每人一次')
    summary['oof_predictions'].sort(key=lambda row: row['subject_index'])
    return summary


def protocol_config(output, main_config):
    return {
        'seed': 42,
        't_use': 146,
        'per_fold_sizes': [602, 86, 172],
        'gammas': list(EXPECTED_GAMMAS),
        'gamma_source': str(MAIN_RESULT / 'summary.json'),
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
            'lr': {
                'penalty': 'l2', 'C': 1.0, 'solver': 'liblinear',
                'dual': True, 'max_iter': 5000, 'random_state': 42,
            },
            'svm': {
                'kernel': 'linear', 'C': 1.0, 'probability': True,
                'random_state': 42, 'cache_size': 4096,
            },
        },
        'classifier_specific_tuning': False,
        'run_order': list(RUN_ORDER),
        'main_split_manifest': str(MAIN_RESULT / 'splits_manifest.json'),
        'eval_batch_size': int(main_config['eval_batch_size']),
        'eig_batch_size': int(main_config['eig_batch_size']),
        'output': str(output),
    }


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


def main(args):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    model_key = args.model
    model_root = output / 'models' / model_key
    model_summary_path = model_root / 'summary.json'
    if model_summary_path.exists():
        raise FileExistsError(f'拒绝覆盖已完成结果：{model_summary_path}')
    position = RUN_ORDER.index(model_key)
    for previous in RUN_ORDER[:position]:
        prior = output / 'models' / previous / 'summary.json'
        if not prior.exists():
            raise RuntimeError(f'必须先完成并核验前一组：{prior}')

    main_config = read_json(MAIN_RESULT / 'config.json')
    main_summary = read_json(MAIN_RESULT / 'summary.json')
    split_payload = read_json(MAIN_RESULT / 'splits_manifest.json')
    if tuple(main_summary['selected_gammas']) != EXPECTED_GAMMAS:
        raise RuntimeError('正式主实验 gamma 与锁定值不一致')
    config = protocol_config(output, main_config)
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

    provenance = {
        'script_sha256': rr.sha256_file(Path(__file__).resolve()),
        'data_cache_sha256': rr.sha256_file(
            SOURCE_ROOT / 'cc400_cache_t146.npz'
        ),
        'main_summary_sha256': rr.sha256_file(MAIN_RESULT / 'summary.json'),
        'split_manifest_sha256': rr.sha256_file(
            MAIN_RESULT / 'splits_manifest.json'
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
        raise RuntimeError('源码、数据或正式结果与已有来源记录不一致')

    feature_cache = TangentFeatureCache(
        ts, labels, splits, output,
        main_config['eval_batch_size'], main_config['eig_batch_size'],
    )
    fold_results = []
    for fold_number, (split, gamma) in enumerate(
        zip(splits, EXPECTED_GAMMAS), start=1
    ):
        fold_path = model_root / 'folds' / f'fold_{fold_number}.json'
        if fold_path.exists():
            if not args.resume:
                raise FileExistsError(f'已有折结果：{fold_path}')
            fold_result = read_json(fold_path)
            print(f'[resume] {model_key} fold {fold_number}', flush=True)
        else:
            fold_result = run_fold(
                fold_number, split, gamma, feature_cache,
                labels, output, model_key,
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
    aggregate = maybe_write_aggregate(output)
    if aggregate is not None:
        print(f'ALL COMPLETE -> {aggregate}', flush=True)
    else:
        print(f'DONE -> {model_summary_path}', flush=True)


if __name__ == '__main__':
    main(parse_args())
