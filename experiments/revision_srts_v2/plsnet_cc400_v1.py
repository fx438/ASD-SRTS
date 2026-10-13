# -*- coding: utf-8 -*-
"""PLSNet 对照实验：CC400（N=392）、T=146，与 SRTS 主实验相同的 860 人和五折划分。

预先确定的协议（结果产生前写入 config.json，不再改动）：
- 模型与训练步骤：官方 PLSNet 代码（CodeGoat24/PLSNet, commit 734442d），超参数取官方 setting/abide_PLSNet.yaml：
  embedding_size=8，Adam lr=1e-4、weight_decay=1e-4，batch=16，最多 500 轮，mixup(α=1)，2×CE + group loss；
  早停：至少 60 轮，验证集 AUC 连续 15 轮未提高超过 1e-4 即停（官方无早停，固定 500 轮；主实验深度模型为 40 轮）；
  输入为时间序列（按训练折全局均值/标准差标准化）、Pearson 相关矩阵（节点特征）和单位阵（位置嵌入输入）。
- 与官方代码的唯一差异：官方按测试集准确率挑选轮次；这里按验证集 AUC 挑选（并列取最早），测试集只在锁定后评估一次。
- 划分：rr.build_splits(LABELS, seed=42)，即主实验的 602/86/172 五折。
- 阈值：与 SRTS 相同，按验证集准确率锁定（rr.best_threshold）；另报 0.5 阈值（官方 argmax）准确率。
- 种子：0、1、2 三个初始化种子；主要结果为三种子测试指标的均值；每个种子分别与 SRTS（LR 选 γ，
  nilearn_baseline_v1/folds.json 中的 srts 分数）做合并 OOF 配对自助法（折内、类内，20000 次，seed=20260824）。
"""
import os
import argparse
import json
import random
import sys
import time
import types
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

SOURCE_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = SOURCE_ROOT / 'revision_srts_v2'
PLSNET_ROOT = Path(os.environ.get('PLSNET_DIR', Path(__file__).resolve().parents[2] / 'external' / 'PLSNet'))
for p in (SOURCE_ROOT, SCRIPT_ROOT):
    sys.path.insert(0, str(p))

import dataset_cc400 as dataset  # noqa: E402
import run_revision as rr  # noqa: E402

# 官方 model/__init__.py 引用了仓库中不存在的 model.GSL，util/__init__.py 也有无关依赖；
# 用空包占位，只加载 PLSNet 实际用到的模块，不修改官方文件。
for _name in ('model', 'util'):
    _pkg = types.ModuleType(_name)
    _pkg.__path__ = [str(PLSNET_ROOT / _name)]
    sys.modules[_name] = _pkg
sys.path.insert(0, str(PLSNET_ROOT))
from model.model import PLSNet  # noqa: E402
from util.prepossess import mixup_criterion, mixup_data  # noqa: E402
from util.loss import mixup_cluster_loss  # noqa: E402

MODEL_CFG = {'type': 'PLSNet', 'extractor_type': 'attention', 'embedding_size': 8, 'window_size': 4,
             'cnn_pool_size': 16, 'num_gru_layers': 4, 'dropout': 0.5}
LR, WD, BATCH, EPOCHS = 1e-4, 1e-4, 16, 500
MIN_EPOCHS, PATIENCE, MIN_DELTA = 60, 15, 1e-4
SEEDS = (0, 1, 2)
N_BOOT, BOOT_SEED = 20000, 20260824
SRTS_FOLDS = SOURCE_ROOT / 'results_revision_v2' / 'nilearn_baseline_v1' / 'folds.json'


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--smoke', action='store_true', help='第 1 折、种子 0、3 轮，只测流程与耗时')
    p.add_argument('--seeds', type=int, nargs='*', help='只跑这些种子（便于按种子并行）；全部完成后不带此参数再运行一次生成汇总')
    return p.parse_args()


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


@torch.no_grad()
def predict(model, ts, corr, eye, idx, dev):
    model.eval()
    out = []
    for i in range(0, len(idx), 64):
        b = idx[i:i + 64]
        x = ts[b].to(dev); c = corr[b].to(dev); e = eye.expand(len(b), -1, -1)
        [logits, _], _, _ = model(x, c, e)
        out.append(F.softmax(logits, dim=1)[:, 1].float().cpu())
    return torch.cat(out).numpy()


def train_fold(ts, corr, labels, split, seed, epochs, dev):
    set_seed(seed)
    tr, va = np.asarray(split['train']), np.asarray(split['val'])
    n_roi, t_len = ts.shape[1], ts.shape[2]
    mu, sd = ts[tr].mean(), ts[tr].std()          # 官方为全数据集全局标准化，这里只用训练折统计量
    x_all = (ts - mu) / sd
    eye = torch.eye(n_roi, device=dev).unsqueeze(0)
    model = PLSNet(MODEL_CFG, n_roi, n_roi, t_len).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')
    y_t = torch.from_numpy(labels).long()
    best = {'val_auc': -1.0, 'epoch': -1, 'state': None}
    hist = []
    for ep in range(epochs):
        model.train()
        perm = tr[np.random.permutation(len(tr))]
        tot, nb = 0.0, 0
        for i in range(0, len(perm), BATCH):
            b = perm[i:i + BATCH]
            x, c, y = x_all[b].to(dev), corr[b].to(dev), y_t[b].to(dev)
            e = eye.expand(len(b), -1, -1)
            xi, ci, ya, yb, lam = mixup_data(x, c, y, 1, dev)
            [out, _], learn_m, _ = model(xi, ci, e)
            loss = 2 * mixup_criterion(loss_fn, out, ya, yb, lam) + mixup_cluster_loss(learn_m, ya, yb, lam)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss); nb += 1
        if not np.isfinite(tot):
            raise RuntimeError(f'non-finite loss at epoch {ep}')
        vs = predict(model, x_all, corr, eye, va, dev)
        vauc = float(roc_auc_score(labels[va], vs))
        hist.append([ep, tot / nb, vauc])
        if vauc > best['val_auc'] + MIN_DELTA:
            best = {'val_auc': vauc, 'epoch': ep,
                    'state': {k: v.detach().clone() for k, v in model.state_dict().items()}}
        elif ep >= MIN_EPOCHS and ep - best['epoch'] >= PATIENCE:
            break
    model.load_state_dict(best['state'])
    return model, x_all, eye, best, hist


def evaluate(model, x_all, corr, eye, labels, split, dev):
    va, te = np.asarray(split['val']), np.asarray(split['test'])
    vs = predict(model, x_all, corr, eye, va, dev)
    thr = rr.best_threshold(vs, labels[va])
    s = predict(model, x_all, corr, eye, te, dev)
    y = labels[te]
    p = (s >= thr).astype(int)
    tp = int(((p == 1) & (y == 1)).sum()); tn = int(((p == 0) & (y == 0)).sum())
    fp = int(((p == 1) & (y == 0)).sum()); fn = int(((p == 0) & (y == 1)).sum())
    return {'threshold': float(thr), 'scores': s.tolist(), 'pred': p.tolist(),
            'test': {'auc': float(roc_auc_score(y, s)), 'accuracy': float((p == y).mean()),
                     'accuracy_at_0.5': float(((s > 0.5).astype(int) == y).mean()),
                     'sensitivity': tp / (tp + fn), 'specificity': tn / (tn + fp)}}


def fast_auc(y, s):
    from scipy.stats import rankdata
    r = rankdata(s); n1 = y.sum(); n0 = len(y) - n1
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def paired_boot(labels, splits, s_a, p_a, s_b, p_b):
    """a − b 的合并 OOF AUC/ACC 差，折内、类内配对重采样。"""
    y = np.concatenate([labels[np.asarray(sp['test'])] for sp in splits])
    groups, start = [], 0
    for sp in splits:
        yt = labels[np.asarray(sp['test'])]
        groups += [start + np.where(yt == c)[0] for c in (0, 1)]
        start += len(yt)
    rng = np.random.default_rng(BOOT_SEED)
    da, dc = np.empty(N_BOOT), np.empty(N_BOOT)
    for b in range(N_BOOT):
        idx = np.concatenate([rng.choice(g, size=len(g), replace=True) for g in groups])
        yb = y[idx]
        da[b] = fast_auc(yb, s_a[idx]) - fast_auc(yb, s_b[idx])
        dc[b] = (p_a[idx] == yb).mean() - (p_b[idx] == yb).mean()
    pa = float(roc_auc_score(y, s_a) - roc_auc_score(y, s_b))
    pc = float((p_a == y).mean() - (p_b == y).mean())
    f = lambda v, pt: {'point': pt, 'ci95': [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))],
                       'p_le_0': float((v <= 0).mean())}
    return {'d_auc': f(da, pa), 'd_acc': f(dc, pc)}


def main():
    args = parse_args()
    out = SOURCE_ROOT / 'results_revision_v2' / ('plsnet_cc400_v1_smoke' if args.smoke else 'plsnet_cc400_v1')
    (out / 'runs').mkdir(parents=True, exist_ok=True)
    cfg = out / 'config.json'
    if not cfg.exists():
        cfg.write_text(json.dumps({'model_cfg': MODEL_CFG, 'lr': LR, 'weight_decay': WD, 'batch': BATCH,
                                   'epochs': EPOCHS, 'min_epochs': MIN_EPOCHS, 'patience': PATIENCE, 'min_delta': MIN_DELTA,
                                   'seeds': list(SEEDS), 'selection': 'best val AUC epoch (improvement > min_delta)',
                                   'threshold': 'val accuracy (rr.best_threshold)', 'splits': 'rr.build_splits(seed=42)',
                                   'plsnet_commit': '734442dd57a7374072c8a891df8c0a05776f4eda',
                                   'n_roi': 392, 't_use': 146, 'smoke': args.smoke}, indent=2), encoding='utf-8')
    dev = torch.device('cuda')
    torch.set_num_threads(2)
    ts_np, labels, _ = dataset.load_cc400(t_use=146, verbose=True)
    labels = np.asarray(labels)
    ts = torch.from_numpy(np.ascontiguousarray(ts_np, dtype=np.float32))
    xc = ts - ts.mean(-1, keepdim=True)
    xc = xc / xc.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    corr = (xc @ xc.transpose(1, 2)).clamp(-1, 1)            # Pearson 相关（PLSNet 节点特征）
    splits = rr.build_splits(labels, seed=42)
    folds = splits[:1] if args.smoke else splits
    seeds = SEEDS[:1] if args.smoke else (tuple(args.seeds) if args.seeds else SEEDS)
    epochs = 3 if args.smoke else EPOCHS
    print(f'n={len(labels)} ts={tuple(ts.shape)} folds={len(folds)} seeds={seeds} epochs={epochs}', flush=True)

    t0 = time.time()
    for seed in seeds:
        for sp in folds:
            path = out / 'runs' / f'seed{seed}_fold{sp["fold"]}.json'
            if path.exists():
                print(f'[skip] {path.name}', flush=True)
                continue
            model, x_all, eye, best, hist = train_fold(ts, corr, labels, sp, seed, epochs, dev)
            rec = evaluate(model, x_all, corr, eye, labels, sp, dev)
            rec.update({'seed': seed, 'fold': sp['fold'], 'best_epoch': best['epoch'], 'val_auc': best['val_auc'],
                        'epochs_run': len(hist),
                        'history': hist})
            tmp = path.with_suffix('.tmp'); tmp.write_text(json.dumps(rec), encoding='utf-8'); tmp.rename(path)
            print(f'seed{seed} fold{sp["fold"]}: val={best["val_auc"]:.4f}@{best["epoch"]} '
                  f'ran={len(hist)} test auc={rec["test"]["auc"]:.4f} acc={rec["test"]["accuracy"]:.4f}  {time.time() - t0:.0f}s', flush=True)
            del model
            torch.cuda.empty_cache()
    if args.smoke or args.seeds:
        print(f'done {time.time() - t0:.0f}s', flush=True)
        return

    srts = json.loads(SRTS_FOLDS.read_text(encoding='utf-8'))['srts']
    assert [f['fold'] for f in srts] == [sp['fold'] for sp in splits]
    s_srts = np.concatenate([np.asarray(f['scores']) for f in srts])
    p_srts = np.concatenate([np.asarray(f['pred']) for f in srts])
    summary = {'per_seed': {}, 'srts_reference': 'nilearn_baseline_v1 folds.json srts (LR-selected gamma)'}
    for seed in SEEDS:
        recs = [json.loads((out / 'runs' / f'seed{seed}_fold{sp["fold"]}.json').read_text(encoding='utf-8'))
                for sp in splits]
        s = np.concatenate([np.asarray(r['scores']) for r in recs])
        p = np.concatenate([np.asarray(r['pred']) for r in recs])
        fold_metrics = {m: [r['test'][m] for r in recs] for m in recs[0]['test']}
        summary['per_seed'][seed] = {
            'fold_mean': {m: float(np.mean(v)) for m, v in fold_metrics.items()},
            'fold_sd': {m: float(np.std(v, ddof=1)) for m, v in fold_metrics.items()},
            'best_epochs': [r['best_epoch'] for r in recs],
            'srts_minus_plsnet': paired_boot(labels, splits, s_srts, p_srts, s, p)}
    ms = list(summary['per_seed'][SEEDS[0]]['fold_mean'])
    summary['seed_mean_of_fold_means'] = {m: float(np.mean([summary['per_seed'][s]['fold_mean'][m] for s in SEEDS]))
                                          for m in ms}
    (out / 'summary.json').write_text(json.dumps(summary, indent=1), encoding='utf-8')
    print('SUMMARY', json.dumps(summary['seed_mean_of_fold_means']), flush=True)
    print(f'done {time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
