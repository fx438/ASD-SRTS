# -*- coding: utf-8 -*-
"""nilearn 默认切线空间基线（Dadi 等 2019 推荐流程），与 SRTS 严格对齐比较。

nilearn 0.13.1 ConnectivityMeasure(kind="tangent", vectorize=True)，其余参数全部取默认：
- 协方差：LedoitWolf(store_precision=False)（tangent 模式下不另做信号标准化，输入时间序列已按式(1)逐 ROI 标准化）；
- 参考点：训练折协方差的几何均值（_geometric_mean，max_iter=30，tol=1e-7），fit 只用训练折；
- 向量化：sym_matrix_to_vec，对角元除以 √2（StandardScaler 之后与 svec 的 √2 缩放等价）。
读出与 SRTS 完全相同：StandardScaler（只在训练折拟合）+ LR（l2，C=1，liblinear，dual=True），阈值在验证集锁定。
这是基线而非候选模块：不做任何选择，直接报告测试集；SRTS 使用此前各折已按验证 AUC 选定的 γ 重现测试预测，
以便在合并 OOF 上做与论文相同的配对自助法（每折内按类别成对重采样 20000 次，seed=20260824）。
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '16')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '16')
os.environ.setdefault('MKL_NUM_THREADS', '16')

import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

SOURCE_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = SOURCE_ROOT / 'revision_srts_v2'
sys.path.insert(0, str(SCRIPT_ROOT))
sys.path.insert(0, str(SOURCE_ROOT))

import dataset_cc400 as dataset  # noqa: E402
import run_revision as rr  # noqa: E402
import nt_control_v1 as ntc  # noqa: E402

OUT = SOURCE_ROOT / 'results_revision_v2' / 'nilearn_baseline_v1'
SRTS_GAMMA = {0: 0.9, 1: 0.9, 2: 0.5, 3: 0.3, 4: 0.9}  # 各折按 LR 验证 AUC 选定（nullfill_v1 / sitecenter_v1 日志一致）
N_BOOT, BOOT_SEED = 20000, 20260824

TS = LABELS = SPLITS = None


def scored_eval(feats, labels, split):
    clf, scaler, vs, vauc = ntc.fit_val(feats, labels, split)
    te = np.asarray(split['test'])
    scores = clf.predict_proba(scaler.transform(feats[te]))[:, 1]
    thr = rr.best_threshold(vs, labels[np.asarray(split['val'])])
    return vauc, ntc.test_eval(clf, scaler, vs, feats, labels, split), scores, (scores >= thr).astype(int)


def nilearn_fold(k):
    from nilearn.connectome import ConnectivityMeasure
    t0 = time.time()
    sp = SPLITS[k]
    series = [np.ascontiguousarray(TS[i].T, dtype=np.float64) for i in range(len(TS))]
    cm = ConnectivityMeasure(kind='tangent', vectorize=True)
    cm.fit([series[i] for i in sp['train']])
    feats = cm.transform(series).astype(np.float32)
    vauc, test, scores, pred = scored_eval(feats, LABELS, sp)
    return {'fold': k, 'val_auc': vauc, 'test': test, 'scores': scores.tolist(), 'pred': pred.tolist(),
            'n_features': int(feats.shape[1]), 'seconds': time.time() - t0}


def auc(y, s):
    from sklearn.metrics import roc_auc_score
    return roc_auc_score(y, s)


def paired_bootstrap(groups, y, s_a, p_a, s_b, p_b):
    rng = np.random.default_rng(BOOT_SEED)
    d_auc, d_acc = np.empty(N_BOOT), np.empty(N_BOOT)
    for b in range(N_BOOT):
        idx = np.concatenate([rng.choice(g, size=len(g), replace=True) for g in groups])
        d_auc[b] = auc(y[idx], s_a[idx]) - auc(y[idx], s_b[idx])
        d_acc[b] = (p_a[idx] == y[idx]).mean() - (p_b[idx] == y[idx]).mean()
    ci = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
    return {'delta_auc_ci95': ci(d_auc), 'delta_acc_ci95': ci(d_acc),
            'p_delta_auc_le_0': float((d_auc <= 0).mean())}


def main():
    global TS, LABELS, SPLITS
    import nilearn
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'config.json').write_text(json.dumps({
        'baseline': 'nilearn ConnectivityMeasure(kind="tangent", vectorize=True), all other defaults',
        'nilearn_version': nilearn.__version__, 'readout': 'StandardScaler + LR(l2, C=1, liblinear, dual=True)',
        'srts_gamma_per_fold': SRTS_GAMMA, 'bootstrap': {'n': N_BOOT, 'seed': BOOT_SEED,
                                                         'scheme': 'within-fold, within-class paired resampling, pooled OOF'},
        'n_roi': 392, 't_use': 146, 'split_seed': 42}, indent=2), encoding='utf-8')
    TS, LABELS, _ = dataset.load_cc400(t_use=146, verbose=True)
    LABELS = np.asarray(LABELS)
    SPLITS = rr.build_splits(LABELS, seed=42)
    t0 = time.time()

    with ProcessPoolExecutor(max_workers=5) as pool:
        futures = [pool.submit(nilearn_fold, k) for k in range(5)]

        # 主进程同时在 GPU 上重现 SRTS 的测试预测
        import torch
        dev = torch.device('cuda')
        x = torch.from_numpy(np.ascontiguousarray(TS, dtype=np.float32)).to(dev)
        raw = rr.corr_matrix_all(x, 64)
        del x
        srts = []
        for k, sp in enumerate(SPLITS):
            feats = ntc.tangent_features(rr.shrink(raw, SRTS_GAMMA[k]), sp['train'], 32)
            vauc, test, scores, pred = scored_eval(feats, LABELS, sp)
            srts.append({'fold': k, 'gamma': SRTS_GAMMA[k], 'val_auc': vauc, 'test': test,
                         'scores': scores.tolist(), 'pred': pred.tolist()})
            print(f'SRTS fold {k + 1}: val {vauc:.4f} test AUC {test["auc"]:.4f}  {time.time() - t0:.0f}s', flush=True)
            del feats
        nil = []
        for f in futures:
            r = f.result()
            nil.append(r)
            print(f'nilearn fold {r["fold"] + 1}: val {r["val_auc"]:.4f} test AUC {r["test"]["auc"]:.4f}  '
                  f'({r["seconds"]:.0f}s)', flush=True)
    nil.sort(key=lambda r: r['fold'])

    summ = {name: ntc.summarize(v) for name, v in (('srts', srts), ('nilearn_default', nil))}
    groups, start = [], 0
    y = np.concatenate([LABELS[np.asarray(sp['test'])] for sp in SPLITS])
    for sp in SPLITS:
        yt = LABELS[np.asarray(sp['test'])]
        for c in (0, 1):
            groups.append(start + np.where(yt == c)[0])
        start += len(yt)
    cat = lambda v, key: np.concatenate([np.asarray(r[key]) for r in v])
    s_a, p_a, s_b, p_b = cat(srts, 'scores'), cat(srts, 'pred'), cat(nil, 'scores'), cat(nil, 'pred')
    pooled = {'srts_oof_auc': float(auc(y, s_a)), 'nilearn_oof_auc': float(auc(y, s_b)),
              'srts_oof_acc': float((p_a == y).mean()), 'nilearn_oof_acc': float((p_b == y).mean())}
    boot = paired_bootstrap(groups, y, s_a, p_a, s_b, p_b)
    (OUT / 'folds.json').write_text(json.dumps({'srts': srts, 'nilearn_default': nil}, indent=1), encoding='utf-8')
    (OUT / 'summary.json').write_text(json.dumps({'five_fold': summ, 'pooled_oof': pooled, 'paired_bootstrap': boot},
                                                 indent=1), encoding='utf-8')
    print('SUMMARY', json.dumps({n: s['auc']['mean'] for n, s in summ.items()}), json.dumps(pooled), json.dumps(boot),
          flush=True)
    print(f'done {time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
