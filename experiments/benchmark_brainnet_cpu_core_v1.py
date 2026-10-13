# -*- coding: utf-8 -*-
"""BrainNetMLP 在与线性分类器一致的单核心 CPU 环境下复测推理效率。

仅测量已驻留内存的分类器前向图；不包含磁盘读取、svec 到对称矩阵转换，也不包含
三种分类器共享的 SRTS 表示构建。输出采用独占创建，拒绝覆盖已有正式结果。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from threadpoolctl import threadpool_info, threadpool_limits

ROOT = Path(__file__).resolve().parent
REVISION_ROOT = ROOT / 'revision_srts_v2'
RESULT_ROOT = ROOT / 'results_revision_v2'
STANDARD_RESULT = RESULT_ROOT / 'standard5_seed42'
LINEAR_RESULT = RESULT_ROOT / 'srts_linear_standard5_seed42_v1'
CACHE = ROOT / 'cc400_cache_t146.npz'
FEATURE_PATH = LINEAR_RESULT / 'features' / 'fold_1' / 'test.npy'
DEFAULT_OUTPUT = (
    RESULT_ROOT / 'efficiency_core_evidence_v1_staging'
    / 'brainnet_cpu_measurements.json'
)
sys.path.insert(0, str(REVISION_ROOT))

from model_revision import RevisionBrainNetMLP  # noqa: E402


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


def summarize_latency(
    batch_size, latency_ms, warmup_iterations, iterations_per_trial,
):
    summary = summarize_trials(latency_ms)
    median_ms = summary['median']
    return {
        'clock': 'time.perf_counter_ns',
        'warmup_iterations': int(warmup_iterations),
        'iterations_per_trial': int(iterations_per_trial),
        'trials': int(len(latency_ms)),
        'latency_ms': summary,
        'throughput_samples_per_second_at_median': float(
            batch_size / (median_ms / 1000.0)
        ),
        'amortized_ms_per_sample_at_median': float(
            median_ms / batch_size
        ),
    }


def cpu_timer_trials(
    operation, batch_size, warmup_iterations, iterations_per_trial, trials,
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
    return summarize_latency(
        batch_size, values, warmup_iterations, iterations_per_trial,
    )


def timing_profile():
    return {
        'trials': 7,
        'batch_1': {
            'warmup_iterations': 20,
            'iterations_per_trial': 100,
        },
        'batch_64': {
            'warmup_iterations': 2,
            'iterations_per_trial': 3,
        },
    }


def build_model():
    torch.manual_seed(42)
    model = RevisionBrainNetMLP(
        n_roi=392, hidden=32, drop=0.0, k=40, num_classes=2,
    ).eval().cpu()
    return model


def parameter_count(model):
    return int(sum(item.numel() for item in model.parameters()))


def svec_to_symmetric(vector, n_roi):
    source = np.asarray(vector, dtype=np.float32)
    single = source.ndim == 1
    if single:
        source = source[None, :]
    expected = n_roi + n_roi * (n_roi - 1) // 2
    if source.ndim != 2 or source.shape[1] != expected:
        raise ValueError(f'svec 形状应为 (*, {expected})，实际为 {source.shape}')
    matrices = np.zeros(
        (source.shape[0], n_roi, n_roi), dtype=np.float32,
    )
    diagonal = np.arange(n_roi)
    matrices[:, diagonal, diagonal] = source[:, :n_roi]
    rows, cols = np.triu_indices(n_roi, k=1)
    upper = source[:, n_roi:] / np.sqrt(2.0)
    matrices[:, rows, cols] = upper
    matrices[:, cols, rows] = upper
    return matrices[0] if single else matrices


def cpu_model_name():
    path = Path('/proc/cpuinfo')
    if path.is_file():
        for line in path.read_text(encoding='utf-8').splitlines():
            if line.lower().startswith('model name'):
                return line.split(':', 1)[1].strip()
    return platform.processor() or 'unknown'


def configure_torch_threads():
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    if torch.get_num_threads() != 1 or torch.get_num_interop_threads() != 1:
        raise RuntimeError('无法将 PyTorch CPU 线程限制为 1')


@contextmanager
def controlled_cpu_environment(cpu):
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
                'torch_num_threads': int(torch.get_num_threads()),
                'torch_num_interop_threads': int(
                    torch.get_num_interop_threads()
                ),
                'threadpools': pools,
            }
    finally:
        if original_affinity is not None:
            os.sched_setaffinity(0, original_affinity)


def load_inputs():
    cache = np.load(CACHE, allow_pickle=True)
    ts = np.asarray(cache['ts'], dtype=np.float32)
    split_data = json.loads(
        (STANDARD_RESULT / 'splits_manifest.json').read_text(encoding='utf-8')
    )
    features = np.load(FEATURE_PATH, mmap_mode='r', allow_pickle=False)
    if ts.shape != (860, 392, 146):
        raise RuntimeError(f'时间序列形状异常：{ts.shape}')
    if len(split_data['splits']) != 5:
        raise RuntimeError('锁定划分不是五折')
    if features.shape != (172, 77028):
        raise RuntimeError(f'fold 1 SRTS 测试特征形状异常：{features.shape}')
    test_indices = np.asarray(split_data['splits'][0]['test'], dtype=np.int64)
    if test_indices.shape != (172,):
        raise RuntimeError('fold 1 测试索引异常')
    ts_64 = np.ascontiguousarray(ts[test_indices[:64]])
    fc_64 = np.ascontiguousarray(
        svec_to_symmetric(np.asarray(features[:64]), 392)
    )
    return ts_64, fc_64, test_indices


def measure_model(model, ts_64, fc_64, benchmark_cpu):
    profile = timing_profile()
    trials = profile['trials']
    ts_1 = torch.from_numpy(ts_64[:1])
    fc_1 = torch.from_numpy(fc_64[:1])
    ts_batch = torch.from_numpy(ts_64)
    fc_batch = torch.from_numpy(fc_64)
    with controlled_cpu_environment(benchmark_cpu) as environment:
        with torch.inference_mode():
            single = profile['batch_1']
            latency_1 = cpu_timer_trials(
                lambda: model(ts_1, fc_1),
                batch_size=1,
                warmup_iterations=single['warmup_iterations'],
                iterations_per_trial=single['iterations_per_trial'],
                trials=trials,
            )
            batch = profile['batch_64']
            latency_64 = cpu_timer_trials(
                lambda: model(ts_batch, fc_batch),
                batch_size=64,
                warmup_iterations=batch['warmup_iterations'],
                iterations_per_trial=batch['iterations_per_trial'],
                trials=trials,
            )
    return latency_1, latency_64, environment


def environment_record():
    return {
        'host': platform.node(),
        'platform': platform.platform(),
        'python': platform.python_version(),
        'numpy': np.__version__,
        'pytorch': torch.__version__,
        'cpu': cpu_model_name(),
        'cpu_logical_count': int(os.cpu_count() or 0),
        'thread_environment': {
            name: os.environ.get(name)
            for name in (
                'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS',
                'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS',
                'CUDA_VISIBLE_DEVICES',
            )
        },
    }


def build_report():
    progress_event('runtime_configuration_started')
    configure_torch_threads()
    progress_event(
        'runtime_configuration_completed',
        torch_num_threads=torch.get_num_threads(),
        torch_num_interop_threads=torch.get_num_interop_threads(),
    )
    progress_event('load_inputs_started')
    ts_64, fc_64, test_indices = load_inputs()
    progress_event('load_inputs_completed')
    model = build_model()
    parameters = parameter_count(model)
    if parameters != 2477634:
        raise RuntimeError(f'BrainNetMLP 参数量异常：{parameters}')
    available_affinity = (
        sorted(os.sched_getaffinity(0))
        if hasattr(os, 'sched_getaffinity') else [0]
    )
    benchmark_cpu = int(available_affinity[0])
    progress_event(
        'brainnet_cpu_benchmark_started', benchmark_cpu=benchmark_cpu,
    )
    latency_1, latency_64, benchmark_environment = measure_model(
        model, ts_64, fc_64, benchmark_cpu,
    )
    progress_event(
        'brainnet_cpu_benchmark_completed',
        batch_1_median_ms=latency_1['latency_ms']['median'],
        batch_64_median_ms=latency_64['latency_ms']['median'],
    )
    inputs = [
        CACHE,
        STANDARD_RESULT / 'splits_manifest.json',
        FEATURE_PATH,
        LINEAR_RESULT / 'features' / 'fold_1' / 'test.json',
        REVISION_ROOT / 'model_revision.py',
        Path(__file__).resolve(),
    ]
    return {
        'status': 'completed',
        'artifact_version': 'efficiency-core-brainnet-cpu-v1',
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'measurement_scope': (
            'RevisionBrainNetMLP inference on one CPU core with inputs '
            'resident in host memory'
        ),
        'architecture': 'RevisionBrainNetMLP',
        'model_source': str(REVISION_ROOT / 'model_revision.py'),
        'weights_for_benchmark': (
            'deterministic seed-42 initialization; weights do not alter the '
            'inference graph, parameter count, tensor shapes, or activation '
            'operations'
        ),
        'parameters_trainable': parameters,
        'input': {
            'time_series_shape_batch1': [1, 392, 146],
            'tangent_matrix_shape_batch1': [1, 392, 392],
            'source_fold': 1,
            'source_subject_indices_batch64': test_indices[:64].tolist(),
            'svec_source': str(FEATURE_PATH),
        },
        'latency_scope': {
            'included': 'RevisionBrainNetMLP forward inference, including rFFT',
            'excluded': (
                'disk I/O, svec-to-symmetric conversion, SRTS representation '
                'construction, and any host-to-device transfer'
            ),
        },
        'batch_1': {'latency': latency_1},
        'batch_64': {'latency': latency_64},
        'environment': {
            **environment_record(),
            'benchmark': benchmark_environment,
        },
        'peak_gpu_memory': {
            'status': 'not_applicable',
            'reason': 'CPU-only BrainNetMLP inference; CUDA disabled',
        },
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
        'parameter_count': report['parameters_trainable'],
        'batch_1_latency_ms_median': report['batch_1']['latency'][
            'latency_ms'
        ]['median'],
        'batch_64_latency_ms_median': report['batch_64']['latency'][
            'latency_ms'
        ]['median'],
    }, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
