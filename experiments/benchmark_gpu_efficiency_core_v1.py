# -*- coding: utf-8 -*-
"""正文核心效率证据的独立 GPU 测量器。

只测量推理图与 SRTS 表示构建成本，不生成或修改任何预测指标。输出文件使用
独占创建；已存在时拒绝覆盖。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
REVISION_ROOT = ROOT / 'revision_srts_v2'
RESULT_ROOT = ROOT / 'results_revision_v2'
STANDARD_RESULT = RESULT_ROOT / 'standard5_seed42'
LINEAR_RESULT = RESULT_ROOT / 'srts_linear_standard5_seed42_v1'
CACHE = ROOT / 'cc400_cache_t146.npz'
DEFAULT_OUTPUT = (
    RESULT_ROOT / 'efficiency_core_evidence_v1_staging'
    / 'gpu_measurements.json'
)
sys.path.insert(0, str(REVISION_ROOT))
sys.path.insert(0, str(ROOT))

from model_revision import RevisionBrainNetMLP  # noqa: E402
import run_revision as rr  # noqa: E402

MIB = 1024.0 ** 2
GAMMAS = (0.90, 0.90, 0.70, 0.50, 0.90)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def validate_cache_manifest_labels(labels, manifest):
    actual = np.asarray(labels, dtype=np.int64)
    expected = np.asarray(
        [row['label'] for row in manifest], dtype=np.int64
    )
    if (
        actual.shape != expected.shape
        or not np.isin(actual, (0, 1)).all()
        or not np.array_equal(actual, expected)
    ):
        raise RuntimeError('缓存标签与锁定 manifest 逐项不一致')
    counts = np.bincount(actual, minlength=2)
    return {'class_0': int(counts[0]), 'class_1': int(counts[1])}


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


def svec_to_symmetric(vector, n_roi):
    source = np.asarray(vector, dtype=np.float32)
    single = source.ndim == 1
    if single:
        source = source[None, :]
    expected = n_roi + n_roi * (n_roi - 1) // 2
    if source.ndim != 2 or source.shape[1] != expected:
        raise ValueError(f'svec 形状应为 (*, {expected})，实际为 {source.shape}')
    matrices = np.zeros(
        (source.shape[0], n_roi, n_roi), dtype=np.float32
    )
    diagonal = np.arange(n_roi)
    matrices[:, diagonal, diagonal] = source[:, :n_roi]
    rows, cols = np.triu_indices(n_roi, k=1)
    upper = source[:, n_roi:] / np.sqrt(2.0)
    matrices[:, rows, cols] = upper
    matrices[:, cols, rows] = upper
    return matrices[0] if single else matrices


def fit_reference(base_matrices, train_indices):
    indices = torch.as_tensor(
        train_indices, device=base_matrices.device, dtype=torch.long
    )
    reference = base_matrices[indices].double().mean(0)
    return rr.matrix_function(reference, 'rsqrt', floor=1e-12)


def transform_with_reference(
    base_matrices, target_indices, reference_isqrt, batch_size=32,
):
    target = np.asarray(target_indices, dtype=np.int64)
    output = []
    for start in range(0, len(target), batch_size):
        index = torch.as_tensor(
            target[start:start + batch_size],
            device=base_matrices.device,
            dtype=torch.long,
        )
        white = reference_isqrt @ base_matrices[index] @ reference_isqrt
        white = (white + white.transpose(-1, -2)) * 0.5
        eigval, eigvec = torch.linalg.eigh(white.double())
        logval = eigval.clamp_min(1e-12).log()
        tangent = (
            (eigvec * logval.unsqueeze(-2))
            @ eigvec.transpose(-1, -2)
        )
        output.append(tangent.float())
    return torch.cat(output)


def cuda_event_trials(
    operation, warmup_iterations, iterations_per_trial, trials,
):
    if not torch.cuda.is_available():
        raise RuntimeError('GPU 正式效率测量需要 CUDA')
    if warmup_iterations < 0 or iterations_per_trial < 1 or trials < 1:
        raise ValueError('warm-up、迭代次数或 trial 数不合法')
    for _ in range(warmup_iterations):
        operation()
    torch.cuda.synchronize()
    latencies = []
    for _ in range(trials):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iterations_per_trial):
            operation()
        end.record()
        end.synchronize()
        latencies.append(
            float(start.elapsed_time(end) / iterations_per_trial)
        )
    return {
        'clock': 'torch.cuda.Event',
        'warmup_iterations': int(warmup_iterations),
        'iterations_per_trial': int(iterations_per_trial),
        'trials': int(trials),
        'latency_ms': summarize_trials(latencies),
    }


def peak_cuda_memory(operation):
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    baseline_allocated = torch.cuda.memory_allocated()
    baseline_reserved = torch.cuda.memory_reserved()
    result = operation()
    torch.cuda.synchronize()
    peak_allocated = torch.cuda.max_memory_allocated()
    peak_reserved = torch.cuda.max_memory_reserved()
    del result
    return {
        'baseline_allocated_mib': float(baseline_allocated / MIB),
        'baseline_reserved_mib': float(baseline_reserved / MIB),
        'peak_allocated_mib': float(peak_allocated / MIB),
        'peak_reserved_mib': float(peak_reserved / MIB),
        'incremental_peak_allocated_mib': float(
            (peak_allocated - baseline_allocated) / MIB
        ),
        'incremental_peak_reserved_mib': float(
            (peak_reserved - baseline_reserved) / MIB
        ),
    }


def cpu_model_name():
    path = Path('/proc/cpuinfo')
    if path.is_file():
        for line in path.read_text(encoding='utf-8').splitlines():
            if line.lower().startswith('model name'):
                return line.split(':', 1)[1].strip()
    return platform.processor() or 'unknown'


def load_inputs():
    cache = np.load(CACHE, allow_pickle=True)
    ts = np.asarray(cache['ts'], dtype=np.float32)
    labels = np.asarray(cache['labels'], dtype=np.int64)
    split_data = json.loads(
        (STANDARD_RESULT / 'splits_manifest.json').read_text(encoding='utf-8')
    )
    if ts.shape != (860, 392, 146) or labels.shape != (860,):
        raise RuntimeError('真实缓存形状不符合锁定协议')
    if len(split_data['splits']) != 5:
        raise RuntimeError('锁定划分不是五折')
    return ts, labels, split_data


def measure_brainnet_mlp(ts, splits):
    n_roi = ts.shape[1]
    feature_path = LINEAR_RESULT / 'features' / 'fold_1' / 'test.npy'
    features = np.load(feature_path, mmap_mode='r', allow_pickle=False)
    test_indices = np.asarray(splits[0]['test'], dtype=np.int64)
    if features.shape != (172, 77028):
        raise RuntimeError('fold 1 SRTS 测试特征形状异常')

    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    model = RevisionBrainNetMLP(
        n_roi=392, hidden=32, drop=0.0, k=40, num_classes=2
    ).eval().cuda()
    parameters = int(sum(item.numel() for item in model.parameters()))
    if parameters != 2477634:
        raise RuntimeError(f'BrainNetMLP 参数量异常：{parameters}')

    fc_64_np = svec_to_symmetric(np.asarray(features[:64]), n_roi)
    ts_64_np = ts[test_indices[:64]]
    with torch.inference_mode():
        ts_1 = torch.from_numpy(ts_64_np[:1]).cuda()
        fc_1 = torch.from_numpy(fc_64_np[:1]).cuda()
        latency_1 = cuda_event_trials(
            lambda: model(ts_1, fc_1),
            warmup_iterations=100,
            iterations_per_trial=1000,
            trials=10,
        )
        memory_1 = peak_cuda_memory(lambda: model(ts_1, fc_1))
        del ts_1, fc_1
        torch.cuda.empty_cache()

        ts_64 = torch.from_numpy(ts_64_np).cuda()
        fc_64 = torch.from_numpy(fc_64_np).cuda()
        latency_64 = cuda_event_trials(
            lambda: model(ts_64, fc_64),
            warmup_iterations=20,
            iterations_per_trial=100,
            trials=10,
        )
        memory_64 = peak_cuda_memory(lambda: model(ts_64, fc_64))
        batch_median = latency_64['latency_ms']['median']
        throughput = 64.0 / (batch_median / 1000.0)
        amortized = batch_median / 64.0
        del ts_64, fc_64
    del model
    torch.cuda.empty_cache()
    return {
        'architecture': 'RevisionBrainNetMLP',
        'model_source': str(REVISION_ROOT / 'model_revision.py'),
        'input': {
            'time_series_shape_batch1': [1, 392, 146],
            'tangent_matrix_shape_batch1': [1, 392, 392],
            'source_fold': 1,
            'source_subject_indices_batch64': test_indices[:64].tolist(),
            'svec_source': str(feature_path),
        },
        'weights_for_benchmark': (
            'deterministic seed-42 initialization; weights do not alter '
            'the inference graph, parameter count, tensor shapes, or '
            'allocated activation memory'
        ),
        'parameters_trainable': parameters,
        'batch_1': {
            'latency': latency_1,
            'peak_gpu_memory': memory_1,
        },
        'batch_64': {
            'latency': latency_64,
            'peak_gpu_memory': memory_64,
            'throughput_samples_per_second_at_median': float(throughput),
            'amortized_ms_per_sample_at_median': float(amortized),
        },
    }


def measure_srts_preprocessing(ts, splits):
    x = torch.from_numpy(ts).cuda()
    with torch.inference_mode():
        correlation_latency = cuda_event_trials(
            lambda: rr.corr_matrix_all(x, batch_size=64),
            warmup_iterations=2,
            iterations_per_trial=1,
            trials=10,
        )
        correlation_memory = peak_cuda_memory(
            lambda: rr.corr_matrix_all(x, batch_size=64)
        )
        raw_corr = rr.corr_matrix_all(x, batch_size=64)
        torch.cuda.synchronize()

        fold_rows = []
        for fold_number, (split, gamma) in enumerate(
            zip(splits, GAMMAS), start=1
        ):
            train = split['train']
            target = split['test']
            if (len(train), len(target)) != (602, 172):
                raise RuntimeError('SRTS 效率测量划分规模异常')
            base = rr.shrink(raw_corr, gamma)

            reference_latency = cuda_event_trials(
                lambda: fit_reference(base, train),
                warmup_iterations=2,
                iterations_per_trial=1,
                trials=10,
            )
            reference_isqrt = fit_reference(base, train)
            fixed_batch_latency = cuda_event_trials(
                lambda: transform_with_reference(
                    base, target, reference_isqrt, batch_size=1
                ),
                warmup_iterations=1,
                iterations_per_trial=1,
                trials=5,
            )
            per_target_values = [
                value / len(target)
                for value in fixed_batch_latency['latency_ms']['values']
            ]

            end_to_end = cuda_event_trials(
                lambda: rr.tangent_transform(
                    base, train, target, batch_size=32
                )[0],
                warmup_iterations=0,
                iterations_per_trial=1,
                trials=1,
            )
            end_to_end_memory = peak_cuda_memory(
                lambda: rr.tangent_transform(
                    base, train, target, batch_size=32
                )[0]
            )
            fold_rows.append({
                'fold': fold_number,
                'gamma': float(gamma),
                'n_train_reference': len(train),
                'n_test_targets': len(target),
                'reference_fit_latency': reference_latency,
                'fixed_reference_transform_batch_1': {
                    'batch_latency': fixed_batch_latency,
                    'ms_per_target': summarize_trials(per_target_values),
                },
                'end_to_end_reference_plus_172_targets_batch_32': {
                    'latency': end_to_end,
                    'ms_per_target_amortized': float(
                        end_to_end['latency_ms']['mean'] / len(target)
                    ),
                    'peak_gpu_memory': end_to_end_memory,
                },
            })
            del base, reference_isqrt
            torch.cuda.empty_cache()

    correlation_per_subject = [
        value / len(ts)
        for value in correlation_latency['latency_ms']['values']
    ]
    reference_values = [
        row['reference_fit_latency']['latency_ms']['mean']
        for row in fold_rows
    ]
    fixed_values = [
        row['fixed_reference_transform_batch_1']['ms_per_target']['mean']
        for row in fold_rows
    ]
    end_to_end_values = [
        row['end_to_end_reference_plus_172_targets_batch_32'][
            'ms_per_target_amortized'
        ]
        for row in fold_rows
    ]
    del raw_corr, x
    torch.cuda.empty_cache()
    return {
        'implementation_source': str(REVISION_ROOT / 'run_revision.py'),
        'input_shape': [860, 392, 146],
        'correlation_all_860': {
            'latency': correlation_latency,
            'ms_per_subject': summarize_trials(correlation_per_subject),
            'peak_gpu_memory': correlation_memory,
        },
        'folds': fold_rows,
        'across_fold_summary': {
            'reference_fit_ms': summarize_trials(reference_values),
            'fixed_reference_batch1_transform_ms_per_target': (
                summarize_trials(fixed_values)
            ),
            'end_to_end_batch32_amortized_ms_per_target': (
                summarize_trials(end_to_end_values)
            ),
        },
        'scope_note': (
            'SRTS correlation, training-only reference fitting, and tangent '
            'mapping are shared representation costs and are excluded from '
            'the classifier-head latency table.'
        ),
    }


def environment_record():
    properties = torch.cuda.get_device_properties(0)
    return {
        'host': platform.node(),
        'platform': platform.platform(),
        'python': platform.python_version(),
        'numpy': np.__version__,
        'torch': torch.__version__,
        'cuda_runtime': torch.version.cuda,
        'cuda_device': torch.cuda.get_device_name(0),
        'cuda_total_memory_mib': float(properties.total_memory / MIB),
        'cpu': cpu_model_name(),
        'cpu_logical_count': int(__import__('os').cpu_count() or 0),
    }


def build_report():
    if not torch.cuda.is_available():
        raise RuntimeError('未检测到 CUDA，拒绝生成 GPU 正式效率证据')
    ts, labels, split_data = load_inputs()
    validate_cache_manifest_labels(labels, split_data['manifest'])
    brainnet = measure_brainnet_mlp(ts, split_data['splits'])
    srts = measure_srts_preprocessing(ts, split_data['splits'])
    inputs = [
        CACHE,
        STANDARD_RESULT / 'splits_manifest.json',
        LINEAR_RESULT / 'features' / 'fold_1' / 'test.npy',
        REVISION_ROOT / 'model_revision.py',
        REVISION_ROOT / 'run_revision.py',
        Path(__file__).resolve(),
    ]
    return {
        'status': 'completed',
        'artifact_version': 'efficiency-core-gpu-v1',
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'measurement_scope': (
            'actual ABIDE-I CC400 tensor shapes and locked SRTS folds; '
            'no AUC or prediction generation'
        ),
        'environment': environment_record(),
        'brainnet_mlp': brainnet,
        'srts_preprocessing': srts,
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
    args = parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f'拒绝覆盖：{output}')
    output.parent.mkdir(parents=True, exist_ok=True)
    report = build_report()
    with output.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({
        'status': report['status'],
        'output': str(output),
        'sha256': sha256_file(output),
        'brainnet_batch1_latency_ms_median': (
            report['brainnet_mlp']['batch_1']['latency'][
                'latency_ms'
            ]['median']
        ),
        'brainnet_batch1_peak_allocated_mib': (
            report['brainnet_mlp']['batch_1']['peak_gpu_memory'][
                'peak_allocated_mib'
            ]
        ),
        'srts_end_to_end_ms_per_target_mean': (
            report['srts_preprocessing']['across_fold_summary'][
                'end_to_end_batch32_amortized_ms_per_target'
            ]['mean']
        ),
    }, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
