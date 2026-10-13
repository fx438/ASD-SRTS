"""全队列有效秩与功率谱诊断。

三项内容：
1. 860 名被试相关矩阵的有效秩，按站点与 TR 分组；
2. 各 TR 组的平均功率谱，用于判定源数据是否经过带通滤波
   （不依赖对预处理管线的假设，直接由数据判定）；
3. 秩与名义时间点数、TR 的关系。

纯描述性统计，不涉及任何模型、任何训练与任何测试预测。
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault('OMP_NUM_THREADS', '8')
os.environ.setdefault('MKL_NUM_THREADS', '8')

import numpy as np
import torch

torch.set_num_threads(8)

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'results_revision_v2/standard5_seed42'
TR_MAP = ROOT / 'tr_acquisition_map_abide1_v1.json'
OUTPUT = ROOT / 'results_revision_v2/effective_rank_all_subjects_v1'
RANK_TOL = 1e-8
BATCH = 8


def summarize(values) -> dict:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return {}
    return {
        'n': int(values.size),
        'mean': float(values.mean()),
        'sample_sd': float(values.std(ddof=1)) if values.size > 1 else 0.0,
        'min': float(values.min()),
        'median': float(np.median(values)),
        'max': float(values.max()),
    }


def match_tr(manifest_row, rules):
    """按 site_id 匹配，file_id_prefix 非空时再作细化；须唯一命中。"""
    hits = []
    for rule in rules:
        if rule['site_id'] != manifest_row['site_id']:
            continue
        prefix = rule.get('file_id_prefix')
        if prefix and not str(manifest_row['file_id']).startswith(prefix):
            continue
        hits.append(rule)
    if len(hits) != 1:
        raise ValueError(
            'file_id=%s site=%s 匹配到 %d 条 TR 规则'
            % (manifest_row['file_id'], manifest_row['site_id'], len(hits))
        )
    return float(hits[0]['tr_seconds'])


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(ROOT))
    import dataset_cc400 as dataset

    ts_all, labels, sites = dataset.load_cc400(t_use=146, verbose=False)
    manifest = json.loads((BASE / 'splits_manifest.json').read_text())['manifest']
    tr_map = json.loads(TR_MAP.read_text())
    tr_values = np.asarray([match_tr(row, tr_map['rules']) for row in manifest])
    sites = np.asarray(sites, dtype=object)

    n, n_roi, t_use = ts_all.shape
    assert len(tr_values) == n, (len(tr_values), n)

    ranks = np.zeros(n, dtype=np.int64)
    spectra = np.zeros((n, t_use // 2 + 1), dtype=np.float64)

    for start in range(0, n, BATCH):
        block = torch.as_tensor(ts_all[start:start + BATCH]).double()
        c = block @ block.transpose(-1, -2) / block.shape[-1]
        c = ((c + c.transpose(-1, -2)) * 0.5).clamp(-1.0, 1.0)
        eigval = torch.linalg.eigvalsh(c)
        ranks[start:start + block.shape[0]] = (
            (eigval > RANK_TOL).sum(dim=1).cpu().numpy()
        )
        # 每名被试的平均功率谱（先对 ROI 取平均，再归一到总功率为 1）
        amp = torch.abs(torch.fft.rfft(block, dim=2)) ** 2
        power = amp.mean(dim=1)
        power = power / power.sum(dim=1, keepdim=True).clamp_min(1e-30)
        spectra[start:start + block.shape[0]] = power.cpu().numpy()
        del block, c, eigval, amp, power

    def group_rows(keys, label):
        rows = []
        for key in sorted(set(keys.tolist()), key=lambda v: (float(v) if label == 'TR' else str(v))):
            idx = np.where(keys == key)[0]
            tr_here = float(np.unique(tr_values[idx])[0]) if label == 'site' else float(key)
            duration = tr_here * t_use
            rows.append({
                'group': f'{float(key):.3f}' if label == 'TR' else str(key),
                'n': int(len(idx)),
                'tr_seconds': tr_here,
                'scan_duration_s': duration,
                'nyquist_hz': 0.5 / tr_here,
                'frequency_resolution_hz': 1.0 / duration,
                'rank': summarize(ranks[idx]),
            })
        return rows

    # 功率谱：判定是否存在阻带。按 TR 组给出归一化谱的累计分布。
    spectral_rows = []
    for tr in sorted(set(tr_values.tolist())):
        idx = np.where(tr_values == tr)[0]
        mean_power = spectra[idx].mean(axis=0)
        freqs = np.fft.rfftfreq(t_use, d=tr)
        nonzero = mean_power[1:]                       # 去直流
        freq_nz = freqs[1:]
        cumulative = np.cumsum(nonzero) / nonzero.sum()
        # 携带 99% 功率的最高频率，以及高频段残余功率占比
        f99 = float(freq_nz[int(np.searchsorted(cumulative, 0.99))])
        above_01 = float(nonzero[freq_nz > 0.10].sum() / nonzero.sum())
        above_015 = float(nonzero[freq_nz > 0.15].sum() / nonzero.sum())
        below_001 = float(nonzero[freq_nz < 0.01].sum() / nonzero.sum())
        spectral_rows.append({
            'tr_seconds': float(tr),
            'n': int(len(idx)),
            'nyquist_hz': float(0.5 / tr),
            'frequency_bins_excluding_dc': int(len(freq_nz)),
            'frequency_at_99pct_cumulative_power_hz': f99,
            'power_fraction_below_0.01hz': below_001,
            'power_fraction_above_0.10hz': above_01,
            'power_fraction_above_0.15hz': above_015,
            'mean_normalized_power_first_10_bins': [
                float(v) for v in nonzero[:10]
            ],
            'mean_normalized_power_last_10_bins': [
                float(v) for v in nonzero[-10:]
            ],
        })

    report = {
        'status': 'completed',
        'artifact_version': 'effective-rank-all-subjects-v1',
        'analysis_role': (
            'descriptive statistics of correlation-matrix effective rank and '
            'temporal power spectrum for the full cohort; determines empirically '
            'whether the source derivative was band-pass filtered'
        ),
        'uses_test_predictions': False,
        'models_involved': False,
        'n_subjects': int(n),
        'n_roi': int(n_roi),
        't_use': int(t_use),
        'rank_tolerance': RANK_TOL,
        'theoretical_rank_upper_bound': int(t_use - 1),
        'tr_map_source': tr_map.get('official_index_url'),
        'overall_rank': summarize(ranks),
        'overall_null_dimension': summarize(n_roi - ranks),
        'rank_by_tr': group_rows(tr_values, 'TR'),
        'rank_by_site': group_rows(sites, 'site'),
        'power_spectrum_by_tr': spectral_rows,
        'interpretation_constraint': (
            'Purely descriptive; no model fitting, no test predictions, and no '
            'influence on any locked selection.'
        ),
    }

    out_file = OUTPUT / 'report.json'
    with out_file.open('w', encoding='utf-8') as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write('\n')

    print('全队列秩 :', json.dumps(report['overall_rank'], ensure_ascii=False))
    print('零空间维 :', json.dumps(report['overall_null_dimension'], ensure_ascii=False))
    print()
    print('--- 按 TR 分组的秩 ---')
    print('%-8s %4s %8s %10s %9s %9s' % ('TR/s', 'n', '时长/s', '频率分辨/Hz', '秩均值', '秩SD'))
    for row in report['rank_by_tr']:
        print('%-8s %4d %8.0f %10.5f %9.2f %9.2f' % (
            row['group'], row['n'], row['scan_duration_s'],
            row['frequency_resolution_hz'], row['rank']['mean'],
            row['rank']['sample_sd']))
    print()
    print('--- 功率谱（判定是否带通滤波）---')
    print('%-8s %4s %10s %12s %12s %12s' % (
        'TR/s', 'n', 'Nyquist', '<0.01Hz占比', '>0.10Hz占比', '99%功率频率'))
    for row in spectral_rows:
        print('%-8.3f %4d %10.4f %12.5f %12.5f %12.5f' % (
            row['tr_seconds'], row['n'], row['nyquist_hz'],
            row['power_fraction_below_0.01hz'],
            row['power_fraction_above_0.10hz'],
            row['frequency_at_99pct_cumulative_power_hz']))
    print()
    print('--- 按站点的秩 ---')
    for row in report['rank_by_site']:
        print('%-10s n=%4d TR=%.3f 秩=%6.2f±%5.2f' % (
            row['group'], row['n'], row['tr_seconds'],
            row['rank']['mean'], row['rank']['sample_sd']))
    print()
    print('written:', out_file)


if __name__ == '__main__':
    main()
