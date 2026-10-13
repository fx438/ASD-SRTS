"""按站点统计恒定 ROI 的分布。

只读 cc400_cache_t146.npz，不加载任何模型，不触及已锁定结果。
恒定判据与 dataset_cc400.py 的标准化实现一致：ROI 在截取窗口内
总体标准差为 0 时，(a-mean)/(std+1e-6) 结果为零向量。
"""

from __future__ import annotations

import json
import hashlib
from collections import Counter
from pathlib import Path

import numpy as np

CACHE = Path(__file__).resolve().parents[1] / 'cc400_cache_t146.npz'
OUTPUT = Path(__file__).resolve().parents[1] / 'results_revision_v2' / 'constant_roi_site_distribution_v1'
EPSILON = 1e-6


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    data = np.load(CACHE, allow_pickle=True)
    ts = data['ts']                      # (n, n_roi, t) 已标准化
    labels = data['labels']
    sites = np.asarray(data['sites'], dtype=object)
    n, n_roi, t_use = ts.shape

    # 已标准化序列中，恒定 ROI 表现为整行全零。
    row_absmax = np.abs(ts).max(axis=2)          # (n, n_roi)
    constant_mask = row_absmax == 0.0

    per_subject = constant_mask.sum(axis=1)
    per_roi = constant_mask.sum(axis=0)

    site_rows = []
    for site in sorted(set(sites.tolist())):
        idx = np.where(sites == site)[0]
        mask = constant_mask[idx]
        affected = int((mask.sum(axis=1) > 0).sum())
        site_rows.append({
            'site': site,
            'n_subjects': int(len(idx)),
            'constant_pairs': int(mask.sum()),
            'affected_subjects': affected,
            'affected_subject_fraction': float(affected / len(idx)),
            'distinct_rois_affected': int((mask.sum(axis=0) > 0).sum()),
            'mean_constant_rois_per_subject': float(mask.sum(axis=1).mean()),
            'max_constant_rois_in_one_subject': int(mask.sum(axis=1).max()),
        })

    affected_idx = np.where(per_subject > 0)[0]
    asd_affected = int(labels[affected_idx].sum())
    report = {
        'status': 'completed',
        'artifact_version': 'constant-roi-site-distribution-v1',
        'analysis_role': 'descriptive data-quality statistic; no model involved',
        'uses_test_predictions': False,
        'source_cache': str(CACHE),
        'source_cache_sha256': sha256_file(CACHE),
        'epsilon_in_standardization': EPSILON,
        'constant_criterion': 'standardized ROI row is exactly all-zero',
        'shape': {'n_subjects': int(n), 'n_roi': int(n_roi), 't_use': int(t_use)},
        'overall': {
            'constant_subject_roi_pairs': int(constant_mask.sum()),
            'affected_subjects': int(len(affected_idx)),
            'affected_subject_fraction': float(len(affected_idx) / n),
            'distinct_rois_affected': int((per_roi > 0).sum()),
            'affected_asd': asd_affected,
            'affected_tc': int(len(affected_idx) - asd_affected),
            'mean_constant_rois_per_affected_subject':
                float(per_subject[affected_idx].mean()) if len(affected_idx) else 0.0,
            'max_constant_rois_in_one_subject': int(per_subject.max()),
        },
        'by_site': site_rows,
        'top_rois': [
            {'roi_index': int(r), 'n_subjects_constant': int(per_roi[r])}
            for r in np.argsort(-per_roi)[:15] if per_roi[r] > 0
        ],
        'site_concentration': {
            'sites_with_any_affected_subject':
                int(sum(1 for r in site_rows if r['affected_subjects'] > 0)),
            'n_sites': len(site_rows),
        },
    }

    out_file = OUTPUT / 'report.json'
    with out_file.open('w', encoding='utf-8') as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write('\n')
    print(json.dumps(report['overall'], ensure_ascii=False, indent=2))
    print()
    print('site concentration:', json.dumps(report['site_concentration']))
    print()
    for row in site_rows:
        print('%-10s n=%4d affected=%4d (%5.1f%%) pairs=%5d rois=%4d max=%3d' % (
            row['site'], row['n_subjects'], row['affected_subjects'],
            100 * row['affected_subject_fraction'], row['constant_pairs'],
            row['distinct_rois_affected'], row['max_constant_rois_in_one_subject'],
        ))
    print()
    print('written:', out_file)


if __name__ == '__main__':
    main()
