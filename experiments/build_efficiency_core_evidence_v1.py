# -*- coding: utf-8 -*-
"""构建正文核心效率证据 bundle。"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import statistics
import tempfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESULT_ROOT = ROOT / 'results_revision_v2'
LINEAR_RESULT = RESULT_ROOT / 'srts_linear_standard5_seed42_v1'
LR_SUMMARY = LINEAR_RESULT / 'models' / 'lr' / 'summary.json'
SVM_SUMMARY = LINEAR_RESULT / 'models' / 'svm' / 'summary.json'
BRAINNET_SUMMARY = (
    RESULT_ROOT / 'standard5_seed42' / 'postprocess' / 'metrics_summary.csv'
)
STAGING = RESULT_ROOT / 'efficiency_core_evidence_v1_staging'
CPU_MEASUREMENTS = STAGING / 'cpu_classifier_measurements.json'
BRAINNET_CPU_MEASUREMENTS = STAGING / 'brainnet_cpu_measurements.json'
GPU_MEASUREMENTS = STAGING / 'gpu_measurements.json'
DEFAULT_OUTPUT = RESULT_ROOT / 'efficiency_core_evidence_v1'


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def _metric_from_summary(path):
    payload = read_json(path)
    metric = payload['metrics']['auc']
    folds = [float(value) for value in metric['fold_values']]
    if len(folds) != 5:
        raise RuntimeError(f'{path} 不含五折 AUC')
    calculated_mean = statistics.fmean(folds)
    calculated_sd = statistics.stdev(folds)
    if (
        abs(calculated_mean - float(metric['mean'])) > 1e-15
        or abs(calculated_sd - float(metric['sample_sd'])) > 1e-15
    ):
        raise RuntimeError(f'{path} 的 AUC 汇总不能由五折值重算')
    return {
        'mean': float(metric['mean']),
        'sample_sd': float(metric['sample_sd']),
        'fold_values': folds,
        'source': str(path),
        'sd_definition': 'sample SD across five folds (ddof=1)',
    }


def _brainnet_auc_from_csv(path):
    with Path(path).open('r', encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    selected = [row for row in rows if row['variant'] == 'srts']
    if len(selected) != 1:
        raise RuntimeError('BrainNetMLP 汇总中完整 SRTS 行缺失或重复')
    row = selected[0]
    folds = [float(row[f'auc_fold_{fold}']) for fold in range(1, 6)]
    mean = float(row['auc_mean'])
    sample_sd = float(row['auc_sample_sd'])
    if (
        abs(statistics.fmean(folds) - mean) > 1e-15
        or abs(statistics.stdev(folds) - sample_sd) > 1e-15
    ):
        raise RuntimeError('BrainNetMLP AUC 汇总不能由五折值重算')
    return {
        'mean': mean,
        'sample_sd': sample_sd,
        'fold_values': folds,
        'source': str(path),
        'source_variant': 'srts',
        'sd_definition': 'sample SD across five folds (ddof=1)',
    }


def load_auc_evidence():
    return {
        'lr': _metric_from_summary(LR_SUMMARY),
        'svm': _metric_from_summary(SVM_SUMMARY),
        'brainnet_mlp': _brainnet_auc_from_csv(BRAINNET_SUMMARY),
    }


def auc_spread(evidence):
    means = [float(evidence[key]['mean']) for key in (
        'lr', 'svm', 'brainnet_mlp'
    )]
    if not all(0.0 <= value <= 1.0 for value in means):
        raise ValueError('AUC 均值必须位于 [0, 1]')
    return max(means) - min(means)


def build_classifier_rows(auc, cpu, brainnet_cpu, gpu):
    cpu_name = cpu['environment']['cpu']
    brainnet_cpu_name = brainnet_cpu['environment']['cpu']
    if brainnet_cpu_name != cpu_name:
        raise RuntimeError('三种分类器的同 CPU 测量设备不一致')
    gpu_name = gpu['environment']['cuda_device']
    gpu_memory = gpu['brainnet_mlp']['batch_1']['peak_gpu_memory']
    brainnet_parameters = int(brainnet_cpu['parameters_trainable'])
    if brainnet_parameters != int(
        gpu['brainnet_mlp']['parameters_trainable']
    ):
        raise RuntimeError('BrainNetMLP CPU/GPU 参数量不一致')
    specifications = (
        {
            'key': 'lr',
            'method': 'SRTS + LR',
            'parameters': cpu['models']['lr']['effective_parameter_count'],
            'cpu_latency': cpu['models']['lr']['latency'][
                'batch_1_scaler_plus_classifier'
            ]['latency_ms']['median'],
            'native_latency': cpu['models']['lr']['latency'][
                'batch_1_scaler_plus_classifier'
            ]['latency_ms']['median'],
            'native_device': cpu_name,
            'allocated': None,
            'reserved': None,
            'cpu_scope': (
                'StandardScaler.transform + LogisticRegression.predict_proba'
            ),
            'native_scope': (
                'StandardScaler.transform + LogisticRegression.predict_proba'
            ),
        },
        {
            'key': 'svm',
            'method': 'SRTS + SVM',
            'parameters': cpu['models']['svm']['effective_parameter_count'],
            'cpu_latency': cpu['models']['svm']['latency'][
                'batch_1_scaler_plus_classifier'
            ]['latency_ms']['median'],
            'native_latency': cpu['models']['svm']['latency'][
                'batch_1_scaler_plus_classifier'
            ]['latency_ms']['median'],
            'native_device': cpu_name,
            'allocated': None,
            'reserved': None,
            'cpu_scope': 'StandardScaler.transform + linear SVC.predict_proba',
            'native_scope': (
                'StandardScaler.transform + linear SVC.predict_proba'
            ),
        },
        {
            'key': 'brainnet_mlp',
            'method': 'SRTS + BrainNetMLP',
            'parameters': brainnet_parameters,
            'cpu_latency': brainnet_cpu['batch_1']['latency'][
                'latency_ms'
            ]['median'],
            'native_latency': gpu['brainnet_mlp']['batch_1']['latency'][
                'latency_ms'
            ]['median'],
            'native_device': gpu_name,
            'allocated': gpu_memory['peak_allocated_mib'],
            'reserved': gpu_memory['peak_reserved_mib'],
            'cpu_scope': 'RevisionBrainNetMLP.forward on CPU, batch=1',
            'native_scope': (
                'RevisionBrainNetMLP.forward on CUDA, batch=1, '
                'device-resident input'
            ),
        },
    )
    rows = []
    for specification in specifications:
        allocated = specification['allocated']
        reserved = specification['reserved']
        if allocated is None:
            memory_display = 'N/A（无需 GPU）'
        else:
            memory_display = (
                f'{float(allocated):.3f} MiB allocated / '
                f'{float(reserved):.3f} MiB reserved'
            )
        key = specification['key']
        rows.append({
            'model_key': key,
            'method': specification['method'],
            'auc_mean': float(auc[key]['mean']),
            'auc_sample_sd': float(auc[key]['sample_sd']),
            'parameter_count': int(specification['parameters']),
            'cpu_latency_ms_median': float(specification['cpu_latency']),
            'native_latency_ms_median': float(
                specification['native_latency']
            ),
            'peak_gpu_memory_mib': (
                None if allocated is None else float(allocated)
            ),
            'peak_gpu_reserved_memory_mib': (
                None if reserved is None else float(reserved)
            ),
            'memory_display': memory_display,
            'cpu_device': str(cpu_name),
            'native_device': str(specification['native_device']),
            'cpu_inference_scope': specification['cpu_scope'],
            'native_inference_scope': specification['native_scope'],
        })
    return rows


def build_srts_preprocessing_rows(gpu):
    preprocessing = gpu['srts_preprocessing']
    across = preprocessing['across_fold_summary']
    correlation = preprocessing['correlation_all_860']['ms_per_subject']
    device = gpu.get('environment', {}).get('cuda_device', 'GPU')
    rows = [
        {
            'stage': '860 人相关矩阵计算',
            'unit': 'ms/人',
            'mean': float(correlation['mean']),
            'sample_sd': float(correlation['sample_sd']),
            'measurement_n': int(correlation.get('n', 1)),
            'device': device,
        },
        {
            'stage': '训练折参考矩阵拟合',
            'unit': 'ms/折',
            'mean': float(across['reference_fit_ms']['mean']),
            'sample_sd': float(across['reference_fit_ms']['sample_sd']),
            'measurement_n': int(across['reference_fit_ms'].get('n', 5)),
            'device': device,
        },
        {
            'stage': '固定参考矩阵下单目标切线映射',
            'unit': 'ms/人',
            'mean': float(
                across['fixed_reference_batch1_transform_ms_per_target']['mean']
            ),
            'sample_sd': float(
                across['fixed_reference_batch1_transform_ms_per_target'][
                    'sample_sd'
                ]
            ),
            'measurement_n': int(
                across['fixed_reference_batch1_transform_ms_per_target'].get(
                    'n', 5
                )
            ),
            'device': device,
        },
        {
            'stage': '参考拟合+172目标切线映射（batch=32，摊销）',
            'unit': 'ms/人',
            'mean': float(
                across['end_to_end_batch32_amortized_ms_per_target']['mean']
            ),
            'sample_sd': float(
                across['end_to_end_batch32_amortized_ms_per_target'][
                    'sample_sd'
                ]
            ),
            'measurement_n': int(
                across['end_to_end_batch32_amortized_ms_per_target'].get(
                    'n', 5
                )
            ),
            'device': device,
        },
    ]
    return rows


def render_markdown(rows, auc_max_spread, cpu, gpu):
    table_lines = [
        '| 方法 | AUC（五折均值±样本标准差） | 参数量 | '
        '同 CPU 推理时延（batch=1，中位数） | '
        '原生部署时延（batch=1，中位数） | 峰值显存 | 原生推理设备 |',
        '|---|---:|---:|---:|---:|---:|---|',
    ]
    for row in rows:
        table_lines.append(
            f'| {row["method"]} | {row["auc_mean"]:.4f} ± '
            f'{row["auc_sample_sd"]:.4f} | {row["parameter_count"]:,} | '
            f'{row["cpu_latency_ms_median"]:.4f} ms | '
            f'{row["native_latency_ms_median"]:.4f} ms | '
            f'{row["memory_display"]} | {row["native_device"]} |'
        )
    preprocessing_rows = build_srts_preprocessing_rows(gpu)
    preprocessing_lines = [
        '| 阶段 | 时间（均值±样本标准差） | 测量数 | 设备 |',
        '|---|---:|---:|---|',
    ]
    for row in preprocessing_rows:
        preprocessing_lines.append(
            f'| {row["stage"]} | {row["mean"]:.4f} ± '
            f'{row["sample_sd"]:.4f} {row["unit"]} | '
            f'{row["measurement_n"]} | {row["device"]} |'
        )
    svm = cpu['models']['svm']['svm_parameter_detail']
    return '\n'.join([
        '# 正文核心效率证据',
        '',
        '三种分类器采用同一标准五折划分。AUC 统一报告为五折均值±'
        '样本标准差（ddof=1）；最大五折均值差为 '
        f'**{auc_max_spread:.4f} < 0.004**。该比较说明三者判别性能在'
        '描述性层面接近，不构成统计等效性检验。',
        '',
        *table_lines,
        '',
        '“同 CPU”列在同一 CPU、单核心和单线程环境下比较三种分类器：'
        'LR 与 SVM 的时延包含 `StandardScaler.transform + predict_proba`，'
        'BrainNetMLP 为包含 rFFT 的完整前向传播。“原生部署”列保留原始实现：'
        'LR/SVM 使用 scikit-learn CPU 推理，BrainNetMLP 使用 CUDA '
        'device-resident 前向传播（输入已在 GPU，不含主机到设备传输）。'
        '两列均不包含三种方法共享的 SRTS 表示预处理。由于原生部署列使用异构'
        '设备，其毫秒数仅用于呈现各自部署开销，不作硬件归一化速度倍数比较。'
        'LR 与 SVM 无需 GPU，因此峰值显存记为 `N/A（无需 GPU）`，而不是'
        '未经测量的 0 MB。',
        '',
        f'LR 参数量按 77,028 个特征系数加 1 个截距计算。线性 SVM 的 '
        f'77,031 个概率推理有效参数包括 77,028 个 primal 系数、1 个截距'
        f'和 2 个 Platt calibration 标量；本次锁定第 5 折的实际 '
        f'`SVC.predict_proba` 部署仍保留 {svm["support_vectors_n"]} 个支持'
        f'向量，核心数值状态共 {svm["stored_numeric_scalars_core"]:,} 个标量。',
        '',
        '## SRTS 表示预处理成本',
        '',
        '以下成本在正文中单独报告，避免把共享表示构建成本隐藏在分类器时延中。',
        '',
        *preprocessing_lines,
        '',
        '端到端 SRTS 行按每折训练集拟合参考矩阵，并对该折 172 个测试目标'
        '执行 batch=32 切线映射，再摊销至每个目标；参考矩阵未使用测试折。',
        '',
    ])


def _write_json(path, payload):
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def _write_classifier_csv(path, rows):
    fields = [
        'model_key', 'method', 'auc_mean', 'auc_sample_sd',
        'parameter_count', 'cpu_latency_ms_median',
        'native_latency_ms_median', 'peak_gpu_memory_mib',
        'peak_gpu_reserved_memory_mib', 'memory_display', 'cpu_device',
        'native_device', 'cpu_inference_scope', 'native_inference_scope',
    ]
    with Path(path).open('x', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_preprocessing_csv(path, rows):
    fields = ['stage', 'unit', 'mean', 'sample_sd', 'measurement_n', 'device']
    with Path(path).open('x', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def build_bundle(output):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f'拒绝覆盖：{output}')
    output.parent.mkdir(parents=True, exist_ok=True)
    auc = load_auc_evidence()
    spread = auc_spread(auc)
    if spread >= 0.004:
        raise RuntimeError(f'三种分类器 AUC 最大差异不小于 0.004：{spread}')
    cpu = read_json(CPU_MEASUREMENTS)
    brainnet_cpu = read_json(BRAINNET_CPU_MEASUREMENTS)
    gpu = read_json(GPU_MEASUREMENTS)
    if (
        cpu.get('status') != 'completed'
        or brainnet_cpu.get('status') != 'completed'
        or gpu.get('status') != 'completed'
    ):
        raise RuntimeError('CPU、BrainNetMLP CPU 或 GPU 正式测量未完成')
    rows = build_classifier_rows(auc, cpu, brainnet_cpu, gpu)
    preprocessing_rows = build_srts_preprocessing_rows(gpu)
    markdown = render_markdown(rows, spread, cpu, gpu)
    created_at = datetime.now(timezone.utc).isoformat()
    core = {
        'status': 'completed',
        'artifact_version': 'efficiency-core-evidence-v1',
        'created_at_utc': created_at,
        'auc_reporting': {
            'definition': 'five-fold mean and sample SD (ddof=1)',
            'evidence': auc,
            'maximum_mean_spread': spread,
            'maximum_mean_spread_below_0_004': spread < 0.004,
            'interpretation': (
                'descriptively close; not a statistical equivalence test'
            ),
        },
        'classifier_rows': rows,
        'srts_preprocessing_rows': preprocessing_rows,
        'latency_boundary': {
            'shared_exclusion': (
                'classifier latency excludes shared SRTS representation '
                'preprocessing'
            ),
            'same_cpu': (
                'all classifiers measured on one CPU core with one thread'
            ),
            'native_device': (
                'LR/SVM use CPU; BrainNetMLP uses device-resident CUDA; '
                'heterogeneous-device milliseconds are not normalized speedups'
            ),
        },
    }
    build_dir = Path(tempfile.mkdtemp(
        prefix=f'.{output.name}_building_', dir=output.parent
    ))
    core_path = build_dir / 'core_efficiency.json'
    classifier_path = build_dir / 'classifier_efficiency.csv'
    preprocessing_path = build_dir / 'srts_preprocessing.csv'
    report_path = build_dir / 'report.md'
    provenance_path = build_dir / 'provenance.json'
    _write_json(core_path, core)
    _write_classifier_csv(classifier_path, rows)
    _write_preprocessing_csv(preprocessing_path, preprocessing_rows)
    report_path.write_text(markdown, encoding='utf-8', newline='\n')
    inputs = [
        LR_SUMMARY, SVM_SUMMARY, BRAINNET_SUMMARY,
        CPU_MEASUREMENTS, BRAINNET_CPU_MEASUREMENTS, GPU_MEASUREMENTS,
        Path(__file__).resolve(),
    ]
    provenance = {
        'artifact_version': 'efficiency-core-evidence-v1-provenance',
        'created_at_utc': created_at,
        'input_sha256': {str(path): sha256_file(path) for path in inputs},
        'output_sha256': {
            path.name: sha256_file(path) for path in (
                core_path, classifier_path, preprocessing_path, report_path
            )
        },
        'build_policy': 'non-overwriting temporary directory then atomic rename',
    }
    _write_json(provenance_path, provenance)
    os.replace(build_dir, output)
    return output, core


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main():
    output, core = build_bundle(parse_args().output)
    print(json.dumps({
        'status': core['status'],
        'output': str(output),
        'auc_maximum_mean_spread': core['auc_reporting'][
            'maximum_mean_spread'
        ],
        'output_sha256': {
            path.name: sha256_file(path)
            for path in sorted(output.iterdir()) if path.is_file()
        },
    }, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
