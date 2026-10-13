# -*- coding: utf-8 -*-
"""锁定第 5 折 LR/SVM 的 CPU 推理效率复测。

从锁定 SRTS 特征重新拟合原始分类器，以系数哈希和概率证明模型复现，再测量
单样本及 batch=64 CPU 时延。输出独占创建，不覆盖旧结果。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import platform
import time
import warnings
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import scipy
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from threadpoolctl import threadpool_info, threadpool_limits

ROOT = Path(__file__).resolve().parent
RESULT_ROOT = ROOT / 'results_revision_v2'
LINEAR_RESULT = RESULT_ROOT / 'srts_linear_standard5_seed42_v1'
CACHE = ROOT / 'cc400_cache_t146.npz'
FOLD_NUMBER = 5
FEATURE_DIM = 77028
DEFAULT_OUTPUT = (
    RESULT_ROOT / 'efficiency_core_evidence_v1_staging'
    / 'cpu_classifier_measurements.json'
)


def progress_event(stage, **details):
    print(json.dumps({
        'stage': str(stage),
        'at_utc': datetime.now(timezone.utc).isoformat(),
        **details,
    }, ensure_ascii=False, allow_nan=False), flush=True)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(array):
    contiguous = np.ascontiguousarray(array)
    return hashlib.sha256(contiguous.view(np.uint8)).hexdigest()


def summarize_trials(values):
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or len(array) == 0 or not np.isfinite(array).all():
        raise ValueError('测量值必须是一维、非空且全部有限')
    return {
        'n': int(len(array)),
        'values': array.tolist(),
        'mean': float(array.mean()),
        'median': float(np.median(array)),
        'sample_sd': float(array.std(ddof=1)) if len(array) > 1 else None,
        'minimum': float(array.min()),
        'maximum': float(array.max()),
        'sd_definition': 'sample SD (ddof=1)',
    }


def effective_parameter_counts(feature_dim):
    feature_dim = int(feature_dim)
    if feature_dim < 1:
        raise ValueError('特征维数必须为正整数')
    return {
        'lr': feature_dim + 1,
        'svm_linear_decision': feature_dim + 1,
        'svm_probability_calibration': 2,
        'svm_probability_total': feature_dim + 3,
    }


def make_model(model_key):
    if model_key == 'lr':
        return LogisticRegression(
            penalty='l2', C=1.0, solver='liblinear', dual=True,
            max_iter=5000, random_state=42,
        )
    if model_key == 'svm':
        return SVC(
            kernel='linear', C=1.0, probability=True,
            random_state=42, cache_size=4096,
        )
    raise ValueError(model_key)


@contextmanager
def controlled_benchmark_environment(cpu):
    """将计时区间限制为单核心、单线程，并在退出时恢复 affinity。"""
    original_affinity = None
    if hasattr(os, 'sched_getaffinity') and hasattr(os, 'sched_setaffinity'):
        original_affinity = set(os.sched_getaffinity(0))
        if cpu not in original_affinity:
            raise ValueError(f'CPU {cpu} 不在当前 affinity 中')
        os.sched_setaffinity(0, {cpu})
    try:
        with threadpool_limits(limits=1):
            pools = threadpool_info()
            if not pools or any(pool['num_threads'] != 1 for pool in pools):
                raise RuntimeError('无法将全部已加载线程池限制为单线程')
            yield {
                'process_cpu_affinity': (
                    sorted(os.sched_getaffinity(0))
                    if hasattr(os, 'sched_getaffinity') else None
                ),
                'threadpools': pools,
            }
    finally:
        if original_affinity is not None:
            os.sched_setaffinity(0, original_affinity)


def cpu_timer_trials(
    operation, warmup_iterations, iterations_per_trial, trials,
):
    if warmup_iterations < 0 or iterations_per_trial < 1 or trials < 1:
        raise ValueError('warm-up、迭代次数或 trial 数不合法')
    for _ in range(warmup_iterations):
        operation()
    values = []
    for _ in range(trials):
        started = time.perf_counter_ns()
        for _ in range(iterations_per_trial):
            operation()
        elapsed = time.perf_counter_ns() - started
        values.append(float(elapsed / iterations_per_trial / 1_000_000.0))
    return {
        'clock': 'time.perf_counter_ns',
        'warmup_iterations': int(warmup_iterations),
        'iterations_per_trial': int(iterations_per_trial),
        'trials': int(trials),
        'latency_ms': summarize_trials(values),
    }


def svm_storage_counts(classifier):
    required = (
        'support_vectors_', 'dual_coef_', 'intercept_', 'probA_', 'probB_',
    )
    if any(not hasattr(classifier, name) for name in required):
        raise ValueError('SVM 尚未拟合或缺少 probability=True 状态')
    arrays = {
        name: np.asarray(getattr(classifier, name)) for name in required
    }
    return {
        'support_vectors_n': int(len(classifier.support_)),
        'n_support_by_class': np.asarray(
            classifier.n_support_, dtype=np.int64
        ).tolist(),
        'support_vector_feature_values': int(arrays['support_vectors_'].size),
        'dual_coefficient_values': int(arrays['dual_coef_'].size),
        'intercept_values': int(arrays['intercept_'].size),
        'probability_calibration_values': int(
            arrays['probA_'].size + arrays['probB_'].size
        ),
        'stored_numeric_scalars_core': int(
            sum(array.size for array in arrays.values())
        ),
        'note': (
            'sklearn.svm.SVC predict_proba deploys support vectors and dual '
            'coefficients even though a linear kernel admits a 77,029-scalar '
            'primal decision function.'
        ),
    }


def cpu_model_name():
    path = Path('/proc/cpuinfo')
    if path.is_file():
        for line in path.read_text(encoding='utf-8').splitlines():
            if line.lower().startswith('model name'):
                return line.split(':', 1)[1].strip()
    return platform.processor() or 'unknown'


def load_locked_fold():
    fold_root = LINEAR_RESULT / 'features' / f'fold_{FOLD_NUMBER}'
    train_val_path = fold_root / 'train_val.npy'
    test_path = fold_root / 'test.npy'
    train_val = np.load(train_val_path, mmap_mode='r', allow_pickle=False)
    test = np.load(test_path, mmap_mode='r', allow_pickle=False)
    labels = np.asarray(
        np.load(CACHE, allow_pickle=True)['labels'], dtype=np.int64
    )
    if train_val.shape != (688, FEATURE_DIM):
        raise RuntimeError(f'train_val 特征形状异常：{train_val.shape}')
    if test.shape != (172, FEATURE_DIM):
        raise RuntimeError(f'test 特征形状异常：{test.shape}')
    if labels.shape != (860,) or not np.isin(labels, (0, 1)).all():
        raise RuntimeError('标签缓存异常')
    folds = {}
    for key in ('lr', 'svm'):
        path = (
            LINEAR_RESULT / 'models' / key / 'folds'
            / f'fold_{FOLD_NUMBER}.json'
        )
        folds[key] = json.loads(path.read_text(encoding='utf-8'))
    if folds['lr']['train_indices'] != folds['svm']['train_indices']:
        raise RuntimeError('LR/SVM 锁定训练索引不一致')
    if folds['lr']['test_indices'] != folds['svm']['test_indices']:
        raise RuntimeError('LR/SVM 锁定测试索引不一致')
    return train_val, test, labels, folds, train_val_path, test_path


def verify_reproduction(model_key, classifier, fold, val_scores, test_scores):
    coefficient_hash = array_sha256(
        np.asarray(classifier.coef_, dtype=np.float64)
    )
    expected_hash = fold['fit_diagnostics']['coefficient_sha256']
    expected_val = np.asarray(fold['validation']['scores'], dtype=np.float64)
    expected_test = np.asarray(fold['test']['scores'], dtype=np.float64)
    val_error = float(np.max(np.abs(val_scores - expected_val)))
    test_error = float(np.max(np.abs(test_scores - expected_test)))
    intercept_error = float(np.max(np.abs(
        np.asarray(classifier.intercept_, dtype=np.float64)
        - np.asarray(fold['fit_diagnostics']['intercept'], dtype=np.float64)
    )))
    actual_norm = float(np.linalg.norm(classifier.coef_))
    expected_norm = float(fold['fit_diagnostics']['coefficient_l2_norm'])
    coefficient_norm_error = abs(actual_norm - expected_norm)
    tolerance = 1e-12
    actual_support = None
    support_counts_match = None
    if model_key == 'svm':
        actual_support = np.asarray(classifier.n_support_).tolist()
        support_counts_match = (
            actual_support == fold['fit_diagnostics']['n_support_']
        )
    numerical_equivalence = (
        val_error <= tolerance
        and test_error <= tolerance
        and intercept_error <= tolerance
        and coefficient_norm_error <= tolerance
        and (support_counts_match is not False)
    )
    hash_matches = coefficient_hash == expected_hash
    if not hash_matches and not (
        model_key == 'svm' and numerical_equivalence
    ):
        raise RuntimeError(
            f'{model_key} 系数哈希未复现且未满足数值等价条件：'
            f'expected={expected_hash}, actual={coefficient_hash}, '
            f'actual_norm={actual_norm}, expected_norm={expected_norm}, '
            f'norm_error={coefficient_norm_error}, val_error={val_error}, '
            f'test_error={test_error}, intercept_error={intercept_error}, '
            f'actual_n_support={actual_support}'
        )
    if not numerical_equivalence:
        raise RuntimeError(
            f'{model_key} 锁定预测未复现：val={val_error}, '
            f'test={test_error}, intercept={intercept_error}, '
            f'norm={coefficient_norm_error}, '
            f'n_support_match={support_counts_match}'
        )
    return {
        'coefficient_sha256': coefficient_hash,
        'locked_coefficient_sha256': expected_hash,
        'coefficient_hash_matches_locked_fold': hash_matches,
        'numerical_equivalence_matches_locked_fold': True,
        'numerical_equivalence_absolute_tolerance': tolerance,
        'coefficient_l2_norm_absolute_error': coefficient_norm_error,
        'maximum_validation_probability_absolute_error': val_error,
        'maximum_test_probability_absolute_error': test_error,
        'maximum_intercept_absolute_error': intercept_error,
        'support_vector_counts_match_locked_fold': support_counts_match,
        'reproduction_policy': (
            'Exact coefficient SHA-256 is required for LR. For SVC only, a '
            'hash mismatch caused by machine-level reduction order is accepted '
            'when validation/test probabilities, intercept, coefficient L2 '
            'norm, and support-vector counts all match within the recorded '
            'absolute tolerance.'
        ),
    }


def timing_profile(model_key):
    """返回足以估计中位数且不会放大高维 SVC 成本的计时配置。"""
    if model_key == 'lr':
        return {
            'trials': 7,
            'batch_1': {
                'warmup_iterations': 50,
                'iterations_per_trial': 500,
            },
            'batch_64': {
                'warmup_iterations': 5,
                'iterations_per_trial': 20,
            },
        }
    if model_key == 'svm':
        return {
            'trials': 7,
            'batch_1': {
                'warmup_iterations': 3,
                'iterations_per_trial': 10,
            },
            'batch_64': {
                'warmup_iterations': 1,
                'iterations_per_trial': 1,
            },
        }
    raise ValueError(model_key)


def benchmark_model(model_key, scaler, classifier, test, x_test):
    raw_one = np.asarray(test[:1])
    scaled_one = np.asarray(x_test[:1])
    raw_64 = np.asarray(test[:64])
    scaled_64 = np.asarray(x_test[:64])
    profile = timing_profile(model_key)
    single = profile['batch_1']
    batch = profile['batch_64']
    trials = profile['trials']
    classifier_single = cpu_timer_trials(
        lambda: classifier.predict_proba(scaled_one),
        single['warmup_iterations'], single['iterations_per_trial'], trials,
    )
    pipeline_single = cpu_timer_trials(
        lambda: classifier.predict_proba(scaler.transform(raw_one)),
        single['warmup_iterations'], single['iterations_per_trial'], trials,
    )
    classifier_batch = cpu_timer_trials(
        lambda: classifier.predict_proba(scaled_64),
        batch['warmup_iterations'], batch['iterations_per_trial'], trials,
    )
    pipeline_batch = cpu_timer_trials(
        lambda: classifier.predict_proba(scaler.transform(raw_64)),
        batch['warmup_iterations'], batch['iterations_per_trial'], trials,
    )
    batch_median = pipeline_batch['latency_ms']['median']
    return {
        'batch_1_classifier_only': classifier_single,
        'batch_1_scaler_plus_classifier': pipeline_single,
        'batch_64_classifier_only': classifier_batch,
        'batch_64_scaler_plus_classifier': pipeline_batch,
        'batch_64_throughput_samples_per_second_at_median': float(
            64.0 / (batch_median / 1000.0)
        ),
        'batch_64_amortized_ms_per_sample_at_median': float(
            batch_median / 64.0
        ),
        'input_policy': (
            'The first locked fold-5 test sample (batch=1) or first 64 '
            'locked test samples (batch=64) are repeatedly timed.'
        ),
    }


def environment_record():
    affinity = (
        sorted(os.sched_getaffinity(0))
        if hasattr(os, 'sched_getaffinity') else None
    )
    return {
        'host': platform.node(),
        'platform': platform.platform(),
        'python': platform.python_version(),
        'numpy': np.__version__,
        'scipy': scipy.__version__,
        'scikit_learn': sklearn.__version__,
        'cpu': cpu_model_name(),
        'cpu_logical_count': int(os.cpu_count() or 0),
        'process_cpu_affinity': affinity,
        'thread_environment': {
            name: os.environ.get(name)
            for name in (
                'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS',
                'CUDA_VISIBLE_DEVICES',
            )
        },
    }


def build_report():
    progress_event('load_inputs_started')
    (
        train_val, test, labels, folds, train_val_path, test_path,
    ) = load_locked_fold()
    progress_event(
        'load_inputs_completed',
        train_val_shape=list(train_val.shape), test_shape=list(test.shape),
    )
    train_indices = np.asarray(folds['lr']['train_indices'], dtype=np.int64)
    val_indices = np.asarray(folds['lr']['validation_indices'], dtype=np.int64)
    test_indices = np.asarray(folds['lr']['test_indices'], dtype=np.int64)
    if (len(train_indices), len(val_indices), len(test_indices)) != (602, 86, 172):
        raise RuntimeError('第 5 折规模不是 602/86/172')
    progress_event('feature_scaling_started')
    scaler = StandardScaler().fit(np.asarray(train_val[:602]))
    x_train = scaler.transform(np.asarray(train_val[:602]))
    x_val = scaler.transform(np.asarray(train_val[602:]))
    x_test = scaler.transform(np.asarray(test))
    progress_event('feature_scaling_completed')
    available_affinity = (
        sorted(os.sched_getaffinity(0))
        if hasattr(os, 'sched_getaffinity') else [0]
    )
    benchmark_cpu = int(available_affinity[0])
    fitting_threadpools = threadpool_info()
    model_reports = {}
    for model_key in ('lr', 'svm'):
        classifier = make_model(model_key)
        progress_event('model_fit_started', model=model_key)
        started = time.perf_counter()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            classifier.fit(x_train, labels[train_indices])
        fit_seconds = float(time.perf_counter() - started)
        progress_event(
            'model_fit_completed', model=model_key,
            fit_seconds=fit_seconds,
        )
        val_scores = classifier.predict_proba(x_val)[:, 1]
        test_scores = classifier.predict_proba(x_test)[:, 1]
        reproduction = verify_reproduction(
            model_key, classifier, folds[model_key], val_scores, test_scores
        )
        progress_event('model_reproduction_passed', model=model_key)
        progress_event(
            'model_benchmark_started', model=model_key,
            benchmark_cpu=benchmark_cpu,
        )
        with controlled_benchmark_environment(benchmark_cpu) as benchmark_environment:
            latency = benchmark_model(
                model_key, scaler, classifier, test, x_test
            )
        progress_event(
            'model_benchmark_completed', model=model_key,
            batch_1_pipeline_median_ms=latency[
                'batch_1_scaler_plus_classifier'
            ]['latency_ms']['median'],
        )
        serialized = pickle.dumps(
            {'scaler': scaler, 'classifier': classifier}, protocol=5
        )
        report = {
            'model_key': model_key,
            'fold': FOLD_NUMBER,
            'feature_dim': FEATURE_DIM,
            'fit_seconds_reproduction': fit_seconds,
            'fit_warnings': [
                {'category': item.category.__name__, 'message': str(item.message)}
                for item in caught
            ],
            'reproduction': reproduction,
            'effective_parameter_count': (
                effective_parameter_counts(FEATURE_DIM)['lr']
                if model_key == 'lr'
                else effective_parameter_counts(FEATURE_DIM)[
                    'svm_probability_total'
                ]
            ),
            'serialized_pipeline_bytes_pickle_protocol_5': int(len(serialized)),
            'latency': latency,
            'benchmark_environment': benchmark_environment,
            'peak_gpu_memory': {
                'status': 'not_applicable',
                'reason': (
                    'CPU-only StandardScaler and classifier inference; '
                    'CUDA disabled'
                ),
            },
        }
        if model_key == 'svm':
            report['svm_parameter_detail'] = {
                **effective_parameter_counts(FEATURE_DIM),
                **svm_storage_counts(classifier),
            }
        else:
            report['lr_parameter_detail'] = {
                'coefficient_values': int(classifier.coef_.size),
                'intercept_values': int(classifier.intercept_.size),
            }
        model_reports[model_key] = report
        del serialized, classifier
    inputs = [
        CACHE, train_val_path, test_path,
        LINEAR_RESULT / 'features' / f'fold_{FOLD_NUMBER}' / 'train_val.json',
        LINEAR_RESULT / 'features' / f'fold_{FOLD_NUMBER}' / 'test.json',
        LINEAR_RESULT / 'models' / 'lr' / 'folds'
        / f'fold_{FOLD_NUMBER}.json',
        LINEAR_RESULT / 'models' / 'svm' / 'folds'
        / f'fold_{FOLD_NUMBER}.json',
        ROOT / 'revision_srts_v2' / 'run_srts_linear_standard5.py',
        Path(__file__).resolve(),
    ]
    progress_event('report_assembly_started')
    return {
        'status': 'completed',
        'artifact_version': 'efficiency-core-cpu-classifiers-v1',
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'measurement_scope': (
            'fold-5 locked SRTS svec, training-fold StandardScaler, and '
            'reproduced LR/SVC probability inference on CPU'
        ),
        'environment': {
            **environment_record(),
            'fitting_threadpools': fitting_threadpools,
        },
        'parameter_count_definition': {
            'lr': 'feature coefficients + intercept',
            'svm': (
                'linear primal coefficients + intercept + two Platt '
                'probability-calibration scalars'
            ),
        },
        'models': model_reports,
        'provenance': {
            'input_sha256': {
                str(path): sha256_file(path) for path in inputs
            },
        },
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main():
    output = parse_args().output.resolve()
    if output.exists():
        raise FileExistsError(f'拒绝覆盖：{output}')
    output.parent.mkdir(parents=True, exist_ok=True)
    report = build_report()
    with output.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    output_sha256 = sha256_file(output)
    progress_event(
        'output_written', output=str(output), sha256=output_sha256,
    )
    print(json.dumps({
        'status': report['status'],
        'output': str(output),
        'sha256': output_sha256,
        'lr_parameter_count': report['models']['lr']['effective_parameter_count'],
        'svm_parameter_count': report['models']['svm']['effective_parameter_count'],
        'lr_latency_ms_median': report['models']['lr']['latency'][
            'batch_1_scaler_plus_classifier'
        ]['latency_ms']['median'],
        'svm_latency_ms_median': report['models']['svm']['latency'][
            'batch_1_scaler_plus_classifier'
        ]['latency_ms']['median'],
    }, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
