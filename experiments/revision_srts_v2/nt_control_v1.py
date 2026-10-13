# -*- coding: utf-8 -*-
"""N/T 对照实验：改变脑区数 N 与时间点数 T，检验正定化方式的影响是否随 N/T 增大而放大。

与论文主实验同口径：
- 数据：CC400 t146 缓存（860 名被试），划分沿用 rr.build_splits(seed=42)，即 602/86/172；
- 三种正定化：特征值截断（下限 1e-6）、Ledoit-Wolf、线性收缩（γ 在每折验证集上按 LR 验证 AUC 选择）；
- 切线映射：rr.tangent_transform，参考矩阵只用训练折；svec 向量化；
- 分类器：与 run_srts_linear_standard5.py 相同的 LR，StandardScaler 只在训练集拟合，阈值在验证集上锁定。

变量：
- T：取前 T 个时间点后逐 ROI 重新 z-score（T=146 时不重做，保持与主实验逐位一致）；
- N：392 为全部 ROI；其余档位按固定种子随机抽取一次 ROI。
设计：十字形（N=392 时 T∈{146,100,75,50}；T=146 时 N∈{200,100,50}），共 7 个配置。
"""
import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

SOURCE_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = SOURCE_ROOT / 'revision_srts_v2'
sys.path.insert(0, str(SCRIPT_ROOT))
sys.path.insert(0, str(SOURCE_ROOT))

import dataset_cc400 as dataset  # noqa: E402
import run_revision as rr  # noqa: E402

DEFAULT_OUTPUT = SOURCE_ROOT / 'results_revision_v2' / 'nt_control_v1'
GAMMAS = rr.GAMMAS
T_LEVELS = (146, 100, 75, 50)
N_LEVELS = (392, 200, 100, 50)
EIGCLIP_FLOOR = 1e-6


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    p.add_argument('--eval-batch-size', type=int, default=64)
    p.add_argument('--eig-batch-size', type=int, default=32)
    p.add_argument('--only', default=None, help='只跑某个配置，如 N392_T146_r0')
    p.add_argument('--skip', nargs='*', default=[], help='跳过的配置（例如正由另一进程运行）')
    return p.parse_args()


def make_lr():
    return LogisticRegression(penalty='l2', C=1.0, solver='liblinear', dual=True,
                              max_iter=5000, random_state=42)


def prepare_ts(ts_full, roi_idx, t_use):
    x = ts_full[:, roi_idx, :t_use]
    if t_use != ts_full.shape[2]:
        m = x.mean(2, keepdims=True)
        s = x.std(2, keepdims=True) + 1e-6
        x = (x - m) / s
    return np.ascontiguousarray(x, dtype=np.float32)


def svec(tangent):
    n = tangent.shape[-1]
    rows, cols = np.triu_indices(n, k=1)
    rows = torch.as_tensor(rows, device=tangent.device)
    cols = torch.as_tensor(cols, device=tangent.device)
    diag = torch.diagonal(tangent, dim1=-2, dim2=-1)
    upper = tangent[:, rows, cols] * np.sqrt(2.0)
    return torch.cat([diag, upper], dim=1).cpu().numpy().astype(np.float32, copy=False)


@torch.no_grad()
def tangent_features(base, train_idx, eig_batch):
    n = base.shape[0]
    tangent, _ = rr.tangent_transform(base, train_idx, np.arange(n), batch_size=eig_batch)
    feats = svec(tangent)
    del tangent
    torch.cuda.empty_cache()
    return feats


def fit_val(feats, labels, split):
    tr, va = np.asarray(split['train']), np.asarray(split['val'])
    scaler = StandardScaler().fit(feats[tr])
    clf = make_lr()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        clf.fit(scaler.transform(feats[tr]), labels[tr])
    val_scores = clf.predict_proba(scaler.transform(feats[va]))[:, 1]
    return clf, scaler, val_scores, float(roc_auc_score(labels[va], val_scores))


def test_eval(clf, scaler, val_scores, feats, labels, split):
    va, te = np.asarray(split['val']), np.asarray(split['test'])
    thr = rr.best_threshold(val_scores, labels[va])
    scores = clf.predict_proba(scaler.transform(feats[te]))[:, 1]
    y = labels[te]
    pred = (scores >= thr).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
    return {
        'auc': float(roc_auc_score(y, scores)),
        'accuracy': float((pred == y).mean()),
        'sensitivity': tp / (tp + fn) if tp + fn else 0.0,
        'specificity': tn / (tn + fp) if tn + fp else 0.0,
        'threshold': float(thr),
    }


def summarize(fold_results):
    out = {}
    for key in ('auc', 'accuracy', 'sensitivity', 'specificity'):
        vals = [f['test'][key] for f in fold_results]
        out[key] = {'mean': float(np.mean(vals)), 'sample_sd': float(np.std(vals, ddof=1)),
                    'fold_values': vals}
    return out


def run_config(ts_full, labels, splits, n_roi, t_use, rep, args, device):
    if n_roi == ts_full.shape[1]:
        roi_idx = np.arange(n_roi)
    else:
        rng = np.random.default_rng(1000 * n_roi + rep)
        roi_idx = np.sort(rng.choice(ts_full.shape[1], size=n_roi, replace=False))
    ts = prepare_ts(ts_full, roi_idx, t_use)
    x = torch.from_numpy(ts).to(device)
    with torch.no_grad():
        raw = rr.corr_matrix_all(x, args.eval_batch_size)
        # 原始相关矩阵谱：数值秩与被截断特征值的占比
        eig = torch.cat([torch.linalg.eigvalsh(raw[i:i + args.eig_batch_size].double())
                         for i in range(0, len(raw), args.eig_batch_size)])
        n_clipped = (eig < EIGCLIP_FLOOR).sum(1).float()
        eigclip = rr.matrix_function(raw, 'clip', floor=EIGCLIP_FLOOR,
                                     batch_size=args.eig_batch_size)
    lw_np, lw_coef = rr.ledoit_wolf_correlations(ts)
    lw = torch.from_numpy(lw_np).to(device)
    del lw_np

    schemes = {'eigclip': [], 'ledoit_wolf': [], 'linear_shrinkage': []}
    for k, split in enumerate(splits):
        tr = split['train']
        for name, base in (('eigclip', eigclip), ('ledoit_wolf', lw)):
            feats = tangent_features(base, tr, args.eig_batch_size)
            clf, scaler, vs, vauc = fit_val(feats, labels, split)
            schemes[name].append({'fold': k, 'val_auc': vauc,
                                  'test': test_eval(clf, scaler, vs, feats, labels, split)})
            del feats
        # 线性收缩：γ 只按验证 AUC 选择，选定后才评估测试集
        per_gamma, best = [], None
        for g in GAMMAS:
            feats = tangent_features(rr.shrink(raw, g), tr, args.eig_batch_size)
            clf, scaler, vs, vauc = fit_val(feats, labels, split)
            per_gamma.append({'gamma': g, 'val_auc': vauc})
            if best is None or vauc > best[0] + 1e-12:
                best = (vauc, g, clf, scaler, vs, feats)
            else:
                del feats
        vauc, g, clf, scaler, vs, feats = best
        schemes['linear_shrinkage'].append({
            'fold': k, 'gamma': g, 'val_auc': vauc, 'per_gamma_val_auc': per_gamma,
            'test': test_eval(clf, scaler, vs, feats, labels, split)})
        del best, feats
        print(f'  fold {k + 1}/5 done: ' + ', '.join(
            f'{s}={v[-1]["test"]["auc"]:.4f}' for s, v in schemes.items()), flush=True)

    result = {
        'n_roi': int(n_roi), 't_use': int(t_use), 'repeat': int(rep),
        'n_over_t': n_roi / t_use, 'theoretical_rank': int(min(n_roi, t_use - 1)),
        'roi_indices': roi_idx.tolist(),
        'raw_spectrum': {
            'mean_clipped_eigenvalues': float(n_clipped.mean()),
            'mean_clipped_fraction': float((n_clipped / n_roi).mean()),
        },
        'ledoit_wolf_shrinkage': {'mean': float(lw_coef.mean()), 'sd': float(lw_coef.std(ddof=1))},
        'schemes': {name: {'folds': folds, 'metrics': summarize(folds)}
                    for name, folds in schemes.items()},
    }
    del x, raw, eigclip, lw
    torch.cuda.empty_cache()
    return result


def main():
    args = parse_args()
    out = args.output.resolve()
    (out / 'configs').mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda')
    ts_full, labels, _ = dataset.load_cc400(t_use=146, verbose=True)
    assert ts_full.shape == (860, 392, 146)
    splits = rr.build_splits(labels, seed=42)

    # 十字形设计：固定 N=392 改变 T；固定 T=146 改变 N；每档一次
    plan = [(392, t, 0) for t in T_LEVELS] + [(n, 146, 0) for n in N_LEVELS if n != 392]
    meta = {'gammas': list(GAMMAS),
            'eigclip_floor': EIGCLIP_FLOOR, 't_levels': list(T_LEVELS),
            'n_levels': list(N_LEVELS), 'design': 'cross', 'split_seed': 42,
            'roi_subset_seed_rule': 'default_rng(1000*N + 0)',
            'torch': torch.__version__, 'numpy': np.__version__}
    meta_path = out / 'config.json'
    if not meta_path.exists():
        meta_path.write_text(json.dumps(meta, indent=2), encoding='utf-8')

    for n_roi, t_use, rep in plan:
        tag = f'N{n_roi}_T{t_use}_r{rep}'
        if (args.only and tag != args.only) or tag in args.skip:
            continue
        path = out / 'configs' / f'{tag}.json'
        if path.exists():
            print(f'[skip] {tag} 已完成', flush=True)
            continue
        t0 = time.time()
        print(f'[run] {tag}  N/T={n_roi / t_use:.2f}', flush=True)
        res = run_config(ts_full, labels, splits, n_roi, t_use, rep, args, device)
        res['runtime_seconds'] = time.time() - t0
        tmp = path.with_suffix('.tmp')
        tmp.write_text(json.dumps(res, indent=1), encoding='utf-8')
        tmp.rename(path)
        m = {s: v['metrics']['auc']['mean'] for s, v in res['schemes'].items()}
        print(f'[done] {tag} {time.time() - t0:.0f}s  ' +
              ', '.join(f'{s}={a:.4f}' for s, a in m.items()), flush=True)


if __name__ == '__main__':
    main()
