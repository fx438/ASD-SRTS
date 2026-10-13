# -*- coding: utf-8 -*-
"""CC400 图谱数据加载器：ABIDE_CC400_Complete/（1035 例、20 站点、392 ROI）。
与 dataset_fix.py 接口一致：返回 (ts, labels, sites)，ts 形状 (N, n_roi, t_use)，逐 ROI z-score。
原 dataset_fix.py 不动，继续作为 AAL116/504 例系列的数据源。
"""
import os, glob
import numpy as np
import pandas as pd

ROOT = os.environ.get('ABIDE_CC400_DIR', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'ABIDE_CC400_Complete'))
CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cc400_cache_t%d.npz')
PHENO_COLS = ['AGE_AT_SCAN', 'SEX', 'FIQ', 'EYE_STATUS_AT_SCAN', 'HANDEDNESS_SCORES']


def load_cc400(t_use=176, verbose=True, with_pheno=False):
    cache = CACHE % t_use
    if os.path.exists(cache):
        d = np.load(cache, allow_pickle=True)
        if verbose:
            print('cache hit', d['ts'].shape)
        if with_pheno:
            return d['ts'], d['labels'], d['sites'], d['pheno']
        return d['ts'], d['labels'], d['sites']

    ph = pd.read_csv(os.path.join(ROOT, 'Phenotypic.csv'))
    ph = ph.dropna(subset=['SUB_ID'])
    for c in PHENO_COLS:
        if c not in ph.columns:
            ph[c] = np.nan
    lut = {int(r.SUB_ID): (int(r.DX_GROUP), str(r.SITE_ID)) for r in ph.itertuples()}
    plut = {int(row.SUB_ID): [row[c] for c in PHENO_COLS]
            for _, row in ph.set_index('SUB_ID', drop=False).iterrows()}

    ts_list, lab, site, phe = [], [], [], []
    skipped = {'no_label': 0, 'too_short': 0, 'bad': 0}
    for f in sorted(glob.glob(os.path.join(ROOT, 'CSV', '*.csv'))):
        sid = int(os.path.basename(f).rsplit('_', 1)[1].split('.')[0])
        if sid not in lut:
            skipped['no_label'] += 1
            continue
        try:
            a = pd.read_csv(f, header=None).values.astype(np.float32)
        except Exception:
            skipped['bad'] += 1
            continue
        if a.ndim != 2 or a.shape[0] < t_use:
            skipped['too_short'] += 1
            continue
        a = a[:t_use].T                                   # (n_roi, t_use)
        m = a.mean(1, keepdims=True)
        s = a.std(1, keepdims=True) + 1e-6
        ts_list.append((a - m) / s)
        dx, st = lut[sid]
        lab.append(0 if dx == 2 else 1)                   # DX_GROUP: 1=ASD, 2=TC
        site.append(st)
        phe.append(plut.get(sid, [np.nan] * len(PHENO_COLS)))

    ts = np.stack(ts_list).astype(np.float32)
    labels = np.array(lab, dtype=np.int64)
    sites = np.array(site, dtype=object)
    P = np.array(phe, dtype=np.float64)
    P[np.abs(P) > 900] = np.nan                           # ABIDE 缺失值哨兵 -9999
    col_med = np.nanmedian(P, axis=0)
    idx = np.where(np.isnan(P))
    P[idx] = np.take(col_med, idx[1])
    P = ((P - P.mean(0)) / (P.std(0) + 1e-6)).astype(np.float32)
    np.savez_compressed(cache, ts=ts, labels=labels, sites=sites, pheno=P)
    if verbose:
        print('loaded', ts.shape, 'ASD', int(labels.sum()), 'TC', int((labels == 0).sum()),
              'sites', len(set(site)), 'skipped', skipped)
    if with_pheno:
        return ts, labels, sites, P
    return ts, labels, sites


if __name__ == '__main__':
    load_cc400()
