# -*- coding: utf-8 -*-
"""留站点验证（leave-one-site-out, LOSO）：检验正定化方式的优劣排序在跨站点条件下是否保持。

预先确定的协议（结果产生前写入 config.json，不再改动）：
- 外层：17 个站点轮流作为测试站点；其余 16 个站点按（站点, 类别）分层随机划出 12.5% 作为验证集
  （与主实验 86/688 的比例相同，seed=42），其余为训练集；测试站点的任何信息都不参与拟合与选择；
- 表示：原始 FC（svec，无切线映射）、特征值截断（下限 1e-6）+ 切线、Ledoit-Wolf + 切线、
  线性收缩 + 切线（γ 在论文同一网格上按 LR 验证 AUC 选择）；切线参考矩阵只用训练集；
- 读出：与论文相同的 LR（l2, C=1, liblinear dual），StandardScaler 只在训练集拟合，阈值按验证集准确率锁定；
- 主要终点：合并 LOSO 预测（860 人）上“线性收缩 − 特征值截断”的 AUC 差及配对自助法 95% CI；
  次要终点：线性收缩 − 原始 FC、线性收缩 − Ledoit-Wolf；各站点 AUC 的均值差（描述性）；
- 自助法：每个站点内按类别成对重采样 20000 次（seed=20260824），同时给出合并 AUC 差与站点平均 AUC 差。
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

SOURCE_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = SOURCE_ROOT / 'revision_srts_v2'
sys.path.insert(0, str(SCRIPT_ROOT))
sys.path.insert(0, str(SOURCE_ROOT))

import dataset_cc400 as dataset  # noqa: E402
import run_revision as rr  # noqa: E402
import nt_control_v1 as ntc  # noqa: E402

GAMMAS = rr.GAMMAS
EIGCLIP_FLOOR = 1e-6
VAL_FRAC = 86 / 688
SPLIT_SEED = 42
N_BOOT, BOOT_SEED = 20000, 20260824
SCHEMES = ('raw_fc', 'eigclip', 'ledoit_wolf', 'linear_shrinkage')
COMPARISONS = (('linear_shrinkage', 'eigclip'), ('linear_shrinkage', 'raw_fc'), ('linear_shrinkage', 'ledoit_wolf'))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--smoke', action='store_true', help='只跑最小站点、γ=0.9、自助 200 次，输出到 loso_v1_smoke')
    p.add_argument('--eig-batch-size', type=int, default=32)
    p.add_argument('--device', default='cuda', help='cuda 或 cpu（2026-10-13 容器重启后 GPU 上 cusolver 特征分解间歇失败，改用 cpu）')
    return p.parse_args()


def evaluate(clf, scaler, val_scores, feats, labels, split):
    """测试站点上的打分与按验证集锁定阈值后的预测。"""
    va, te = split['val'], split['test']
    thr = rr.best_threshold(val_scores, labels[va])
    scores = clf.predict_proba(scaler.transform(feats[te]))[:, 1]
    y = labels[te]
    pred = (scores >= thr).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
    return {'threshold': float(thr), 'scores': scores.tolist(), 'pred': pred.tolist(),
            'test': {'auc': float(roc_auc_score(y, scores)), 'accuracy': float((pred == y).mean()),
                     'sensitivity': tp / (tp + fn) if tp + fn else 0.0,
                     'specificity': tn / (tn + fp) if tn + fp else 0.0}}


def run_fixed(feats, labels, split):
    clf, scaler, vs, vauc = ntc.fit_val(feats, labels, split)
    out = evaluate(clf, scaler, vs, feats, labels, split)
    out['val_auc'] = vauc
    return out


def fast_auc(y, s):
    r = rankdata(s)
    n1 = y.sum(); n0 = len(y) - n1
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def summarize(out, sites, labels, n_boot):
    recs = [json.loads((out / 'sites' / f'{s}.json').read_text(encoding='utf-8')) for s in sites]
    y = np.concatenate([labels[np.asarray(r['test_idx'])] for r in recs])
    S = {k: np.concatenate([np.asarray(r[k]['scores']) for r in recs]) for k in SCHEMES}
    P = {k: np.concatenate([np.asarray(r[k]['pred']) for r in recs]) for k in SCHEMES}
    site_groups, start = [], 0
    for r in recs:
        yt = labels[np.asarray(r['test_idx'])]
        site_groups.append([start + np.where(yt == c)[0] for c in (0, 1)])
        start += len(yt)

    def metrics(k):
        s, p = S[k], P[k]
        per = [r[k]['test']['auc'] for r in recs]
        w = np.array([r['n_test'] for r in recs], dtype=float)
        return {'pooled_auc': float(roc_auc_score(y, s)), 'pooled_accuracy': float((p == y).mean()),
                'pooled_sensitivity': float(p[y == 1].mean()), 'pooled_specificity': float(1 - p[y == 0].mean()),
                'site_auc_mean': float(np.mean(per)), 'site_auc_sd': float(np.std(per, ddof=1)) if len(per) > 1 else None,
                'site_auc_weighted_mean': float(np.sum(w * per) / w.sum())}

    summ = {k: metrics(k) for k in SCHEMES}
    rng = np.random.default_rng(BOOT_SEED)
    boot = {f'{a}-{c}': {'d_pooled_auc': np.empty(n_boot), 'd_pooled_acc': np.empty(n_boot),
                         'd_site_mean_auc': np.empty(n_boot)} for a, c in COMPARISONS}
    for b in range(n_boot):
        per_site = [np.concatenate([rng.choice(g, size=len(g), replace=True) for g in gs]) for gs in site_groups]
        idx = np.concatenate(per_site)
        yb = y[idx]
        for a, c in COMPARISONS:
            d = boot[f'{a}-{c}']
            d['d_pooled_auc'][b] = fast_auc(yb, S[a][idx]) - fast_auc(yb, S[c][idx])
            d['d_pooled_acc'][b] = (P[a][idx] == yb).mean() - (P[c][idx] == yb).mean()
            d['d_site_mean_auc'][b] = np.mean([fast_auc(y[i], S[a][i]) - fast_auc(y[i], S[c][i]) for i in per_site])
    comps = {}
    for a, c in COMPARISONS:
        key = f'{a}-{c}'
        point = {'d_pooled_auc': summ[a]['pooled_auc'] - summ[c]['pooled_auc'],
                 'd_pooled_acc': summ[a]['pooled_accuracy'] - summ[c]['pooled_accuracy'],
                 'd_site_mean_auc': summ[a]['site_auc_mean'] - summ[c]['site_auc_mean']}
        comps[key] = {m: {'point': float(point[m]),
                          'ci95': [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))],
                          'p_le_0': float((v <= 0).mean())} for m, v in boot[key].items()}
        comps[key]['sites_improved'] = int(sum(r[a]['test']['auc'] > r[c]['test']['auc'] for r in recs))
    per_site = [{'site': r['site'], 'n_test': r['n_test'], 'gamma': r['linear_shrinkage']['gamma'],
                 **{k: r[k]['test']['auc'] for k in SCHEMES}} for r in recs]
    return {'n_sites': len(recs), 'n_participants': int(len(y)), 'schemes': summ,
            'comparisons': comps, 'per_site_auc': per_site, 'bootstrap': {'n': n_boot, 'seed': BOOT_SEED}}


def main():
    args = parse_args()
    out = SOURCE_ROOT / 'results_revision_v2' / ('loso_v1_smoke' if args.smoke else 'loso_v1')
    (out / 'sites').mkdir(parents=True, exist_ok=True)
    gammas = (0.9,) if args.smoke else GAMMAS
    n_boot = 200 if args.smoke else N_BOOT
    bs = args.eig_batch_size
    cfg = out / 'config.json'
    if not cfg.exists():
        cfg.write_text(json.dumps({
            'design': 'leave-one-site-out; inner validation = stratified (site, label) random split of remaining sites',
            'val_frac': VAL_FRAC, 'split_seed': SPLIT_SEED, 'gammas': list(gammas), 'eigclip_floor': EIGCLIP_FLOOR,
            'schemes': list(SCHEMES), 'readout': 'StandardScaler + LR(l2, C=1, liblinear, dual=True), threshold by val acc',
            'primary_endpoint': 'pooled LOSO AUC, linear_shrinkage - eigclip, paired bootstrap 95% CI',
            'secondary': ['linear_shrinkage - raw_fc', 'linear_shrinkage - ledoit_wolf', 'mean per-site AUC difference'],
            'bootstrap': {'n': n_boot, 'seed': BOOT_SEED, 'scheme': 'within-site, within-class paired resampling'},
            'smoke': args.smoke, 'n_roi': 392, 't_use': 146}, indent=2), encoding='utf-8')

    dev = torch.device(args.device)
    ts, labels, sites = dataset.load_cc400(t_use=146, verbose=True)
    labels = np.asarray(labels); sites = np.asarray(sites)
    site_names, site_id = np.unique(sites, return_inverse=True)
    sizes = {s: int((sites == s).sum()) for s in site_names}
    order = sorted(site_names, key=lambda s: sizes[s])
    if args.smoke:
        order = order[:1]
    print(f'{len(site_names)} sites; run order {order}', flush=True)

    x = torch.from_numpy(np.ascontiguousarray(ts, dtype=np.float32)).to(dev)
    with torch.no_grad():
        raw = rr.corr_matrix_all(x, 64)
        eigclip = rr.matrix_function(raw, 'clip', floor=EIGCLIP_FLOOR, batch_size=bs)
    del x
    lw_np, _ = rr.ledoit_wolf_correlations(np.ascontiguousarray(ts, dtype=np.float32))
    lw = torch.from_numpy(lw_np).to(dev)
    del lw_np
    raw_feats = ntc.svec(raw)
    dev.type == 'cuda' and torch.cuda.empty_cache()

    t0 = time.time()
    for site in order:
        path = out / 'sites' / f'{site}.json'
        if path.exists():
            print(f'[skip] {site}', flush=True)
            continue
        te = np.where(sites == site)[0]
        rest = np.where(sites != site)[0]
        tr, va = train_test_split(rest, test_size=VAL_FRAC, stratify=site_id[rest] * 2 + labels[rest],
                                  random_state=SPLIT_SEED)
        split = {'train': np.sort(tr), 'val': np.sort(va), 'test': te}
        rec = {'site': str(site), 'n_test': int(len(te)), 'n_train': int(len(tr)), 'n_val': int(len(va)),
               'test_idx': te.tolist(), 'raw_fc': run_fixed(raw_feats, labels, split)}
        for name, base in (('eigclip', eigclip), ('ledoit_wolf', lw)):
            feats = ntc.tangent_features(base, split['train'], bs)
            rec[name] = run_fixed(feats, labels, split)
            del feats
        per, best = [], None
        for g in gammas:
            feats = ntc.tangent_features(rr.shrink(raw, g), split['train'], bs)
            clf, scaler, vs, vauc = ntc.fit_val(feats, labels, split)
            per.append({'gamma': g, 'val_auc': vauc})
            if best is None or vauc > best['val_auc'] + 1e-12:
                best = evaluate(clf, scaler, vs, feats, labels, split)
                best.update({'val_auc': vauc, 'gamma': g})
            del feats, clf, scaler
            dev.type == 'cuda' and torch.cuda.empty_cache()
        best['per_gamma_val_auc'] = per
        rec['linear_shrinkage'] = best
        tmp = path.with_suffix('.tmp')
        tmp.write_text(json.dumps(rec), encoding='utf-8')
        tmp.rename(path)
        print(f'{site} (n={len(te)}): ' + ', '.join(f'{k}={rec[k]["test"]["auc"]:.4f}' for k in SCHEMES)
              + f'  g={best["gamma"]}  {time.time() - t0:.0f}s', flush=True)

    summ = summarize(out, order, labels, n_boot)
    (out / 'summary.json').write_text(json.dumps(summ, indent=1), encoding='utf-8')
    print('SUMMARY', json.dumps({k: round(v['pooled_auc'], 4) for k, v in summ['schemes'].items()}),
          json.dumps({k: v['d_pooled_auc'] for k, v in summ['comparisons'].items()}), flush=True)
    print(f'done {time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
