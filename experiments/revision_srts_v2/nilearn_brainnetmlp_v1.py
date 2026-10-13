# -*- coding: utf-8 -*-
"""nilearn 默认切线空间表示 + BrainNetMLP（与表 3 表示消融严格对齐）。

表示：nilearn 0.13.1 ConnectivityMeasure(kind="tangent")，其余参数全部默认
（LedoitWolf(store_precision=False) 协方差、训练折几何均值参考点），fit 只用训练折，
输出 392×392 切线矩阵，按与其他表示相同的方式（上三角含对角）输入 BrainNetMLP。
训练：直接调用 run_revision.train_model / test_trained，超参、种子（42+fold）、
划分、早停、阈值规则与 standard5_seed42 完全相同。这是基线，不做任何选择。
复现核对：同时重跑 raw 表示，与归档 standard5_seed42 的 raw 测试概率比对。
"""
import os
for _k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_k] = '10'

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

SOURCE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE_ROOT / 'revision_srts_v2'))
sys.path.insert(0, str(SOURCE_ROOT))

import dataset_cc400 as dataset  # noqa: E402
import run_revision as rr  # noqa: E402

OUT = SOURCE_ROOT / 'results_revision_v2' / 'nilearn_brainnetmlp_v1'
ARCHIVE = SOURCE_ROOT / 'results_revision_v2' / 'standard5_seed42' / 'summary.json'
N_BOOT, BOOT_SEED = 20000, 20260824


def paired_bootstrap(y, fold, a, b):
    rng = np.random.default_rng(BOOT_SEED)
    groups = [np.where((fold == k) & (y == c))[0] for k in np.unique(fold) for c in (0, 1)]
    point = roc_auc_score(y, a) - roc_auc_score(y, b)
    ds = np.empty(N_BOOT)
    for i in range(N_BOOT):
        idx = np.concatenate([g[rng.integers(0, len(g), len(g))] for g in groups])
        ds[i] = roc_auc_score(y[idx], a[idx]) - roc_auc_score(y[idx], b[idx])
    fd = [roc_auc_score(y[fold == k], a[fold == k]) - roc_auc_score(y[fold == k], b[fold == k])
          for k in np.unique(fold)]
    return {'point': float(point), 'ci95': np.percentile(ds, [2.5, 97.5]).tolist(),
            'p_le_0': float((ds <= 0).mean()), 'fold_deltas': [float(v) for v in fd],
            'folds_positive': int(sum(v > 0 for v in fd))}


def main():
    from nilearn.connectome import ConnectivityMeasure
    import nilearn
    smoke = '--smoke' in sys.argv
    sys.argv = [sys.argv[0]]
    args = rr.parse_args()  # 全部取 standard5_seed42 的默认超参
    if smoke:
        args.epochs, args.min_epochs = 3, 1
    out = OUT.with_name(OUT.name + '_smoke') if smoke else OUT
    out.mkdir(parents=True, exist_ok=False)

    ts, labels, sites = dataset.load_cc400(t_use=146, verbose=False)
    splits = rr.build_splits(labels, seed=42)
    archive = json.loads(ARCHIVE.read_text(encoding='utf-8'))
    with (ARCHIVE.parent / 'splits_manifest.json').open(encoding='utf-8') as fh:
        assert json.load(fh)['splits'] == splits, '划分与归档不一致'
    (out / 'config.json').write_text(json.dumps({
        'representation': 'nilearn ConnectivityMeasure(kind="tangent"), all other defaults; fit on training fold',
        'nilearn_version': nilearn.__version__, 'readout': 'RevisionBrainNetMLP via run_revision.train_model',
        'args': {k: (list(v) if isinstance(v, tuple) else v) for k, v in vars(args).items() if k != 'output'},
        'smoke': smoke}, indent=1, default=str), encoding='utf-8')

    device = torch.device('cuda')
    x = torch.from_numpy(ts).to(device)
    y = torch.from_numpy(labels).to(device)
    raw_corr = rr.corr_matrix_all(x, args.eval_batch_size)
    series = [np.ascontiguousarray(ts[i].T, dtype=np.float64) for i in range(len(ts))]

    records = []
    for sp in splits:
        k = sp['fold']
        tr, va, te = (np.asarray(sp[s], dtype=np.int64) for s in ('train', 'val', 'test'))
        t0 = time.time()
        cm = ConnectivityMeasure(kind='tangent')
        cm.fit([series[i] for i in tr])
        mats = torch.from_numpy(cm.transform(series).astype(np.float32)).to(device)
        t_feat = time.time() - t0
        res = {'fold': k, 'test_indices': te.tolist(), 'feature_seconds': t_feat}
        for name, fc in (('nilearn_tangent', mats), ('raw_recheck', raw_corr)):
            trained = rr.train_model(x[tr], fc[tr], y[tr], x[va], fc[va], y[va],
                                     args, seed=args.seed + k, mixup=False)
            tested = rr.test_trained(trained, x[te], fc[te], y[te], args.eval_batch_size)
            res[name] = {'best_val_auc': trained['best_val_auc'], 'epochs_run': trained['epochs_run'],
                         'test': tested}
            print(f'fold {k + 1} {name}: val {trained["best_val_auc"]:.4f} test AUC {tested["auc"]:.4f} '
                  f'ACC {tested["accuracy"]:.4f} ep {trained["epochs_run"]}', flush=True)
        records.append(res)
        del mats
        torch.cuda.empty_cache()
    (out / 'folds.json').write_text(json.dumps(rr.json_ready(records)), encoding='utf-8')

    oof = archive['oof_predictions']
    n = len(oof)
    yv = np.array([r['label'] for r in oof])
    fv = np.array([r['fold'] for r in oof])
    arch = {v: np.array([r[f'{v}_prob'] for r in oof]) for v in rr.CORE_VARIANTS}
    new = {name: np.full(n, np.nan) for name in ('nilearn_tangent', 'raw_recheck')}
    for res in records:
        for name in new:
            new[name][res['test_indices']] = res[name]['test']['probs']
    assert not any(np.isnan(v).any() for v in new.values())

    def fold_stats(name, metric):
        vals = np.array([r[name]['test'][metric] for r in records])
        return {'mean': float(vals.mean()), 'sample_sd': float(vals.std(ddof=1)), 'fold_values': vals.tolist()}

    summary = {
        'nilearn_tangent': {m: fold_stats('nilearn_tangent', m) for m in rr.METRICS},
        'raw_recheck': {m: fold_stats('raw_recheck', m) for m in ('auc', 'accuracy')},
        'raw_recheck_max_abs_prob_diff_vs_archive': float(np.abs(new['raw_recheck'] - arch['raw']).max()),
        'pooled_oof_auc': {**{f'archive_{v}': float(roc_auc_score(yv, arch[v])) for v in rr.CORE_VARIANTS},
                           'nilearn_tangent': float(roc_auc_score(yv, new['nilearn_tangent']))},
    }
    if not smoke:
        summary['paired'] = {
            'srts_minus_nilearn': paired_bootstrap(yv, fv, arch['srts'], new['nilearn_tangent']),
            'nilearn_minus_raw': paired_bootstrap(yv, fv, new['nilearn_tangent'], arch['raw']),
            'nilearn_minus_lw_tangent': paired_bootstrap(yv, fv, new['nilearn_tangent'], arch['lw_tangent']),
        }
    (out / 'summary.json').write_text(json.dumps(rr.json_ready(summary), indent=1), encoding='utf-8')
    print(json.dumps({k: v for k, v in summary.items() if k != 'nilearn_tangent'}, indent=1), flush=True)
    a = summary['nilearn_tangent']
    print(f"nilearn_tangent AUC {a['auc']['mean']:.4f}±{a['auc']['sample_sd']:.4f} "
          f"ACC {a['accuracy']['mean']:.4f}±{a['accuracy']['sample_sd']:.4f}", flush=True)


if __name__ == '__main__':
    main()
