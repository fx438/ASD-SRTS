# -*- coding: utf-8 -*-
"""方案 1：零空间补全模块（null-space completion, NSC）。

对每名被试的原始相关矩阵 C = U diag(λ) Uᵀ，以容差 τ 划分值域 U_r 与零空间 U_0，
用训练折原始相关矩阵的算术平均 R0 补全零空间：
    C̃ = U_r Λ_r U_rᵀ + U_0 (U_0ᵀ R0 U_0) U_0ᵀ
在 [U_r, U_0] 基下 C̃ 为块对角，且 U_0ᵀ R0 U_0 正定，故 C̃ 严格正定。
随后与 SRTS 相同：线性收缩 → 训练折参考矩阵 → 白化 + 矩阵对数 → svec → LR。

预先确定的协议（结果产生前写入 config.json，不再改动）：
- τ = 1e-8（与论文谱诊断相同），R0 只用训练折；γ 网格与 SRTS 相同，按 LR 验证 AUC 逐折选择；
- 对照为同一流程下的 SRTS（不做补全）；
- 接受标准与论文增强模块相同：验证集平均 AUC 提升 ≥ 0.01 且五折中至少四折改进；
- 第一阶段只算验证集；未通过接受标准时不计算模块的测试集指标。
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

SOURCE_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = SOURCE_ROOT / 'revision_srts_v2'
sys.path.insert(0, str(SCRIPT_ROOT))
sys.path.insert(0, str(SOURCE_ROOT))

import dataset_cc400 as dataset  # noqa: E402
import run_revision as rr  # noqa: E402
import nt_control_v1 as ntc  # noqa: E402

OUT = SOURCE_ROOT / 'results_revision_v2' / 'nullfill_v1'
GAMMAS = rr.GAMMAS
TAU = 1e-8
ACCEPT_MEAN, ACCEPT_FOLDS = 0.01, 4


def raw_eig(raw, bs):
    vals, vecs = [], []
    for s in range(0, len(raw), bs):
        b = raw[s:s + bs].double()
        ev, eu = torch.linalg.eigh((b + b.transpose(-1, -2)) * 0.5)
        vals.append(ev); vecs.append(eu)
    return torch.cat(vals), torch.cat(vecs)


@torch.no_grad()
def complete(eigval, eigvec, r0, bs):
    out = []
    for s in range(0, len(eigval), bs):
        lam, u = eigval[s:s + bs], eigvec[s:s + bs]
        null = lam <= TAU
        b = u.transpose(-1, -2) @ r0 @ u
        m = b * (null.unsqueeze(-1) & null.unsqueeze(-2))
        m = m + torch.diag_embed(torch.where(null, torch.zeros_like(lam), lam))
        c = u @ m @ u.transpose(-1, -2)
        out.append(((c + c.transpose(-1, -2)) * 0.5).float())
    return torch.cat(out)


def select(base_fn, labels, split, eig_bs):
    """按 LR 验证 AUC 选 γ，只返回验证信息与选中的 γ。"""
    per, best = [], None
    for g in GAMMAS:
        feats = ntc.tangent_features(base_fn(g), split['train'], eig_bs)
        _, _, _, vauc = ntc.fit_val(feats, labels, split)
        per.append({'gamma': g, 'val_auc': vauc})
        if best is None or vauc > best[0] + 1e-12:
            best = (vauc, g)
        del feats
    return {'gamma': best[1], 'val_auc': best[0], 'per_gamma_val_auc': per}


def test_at(base_fn, g, labels, split, eig_bs):
    feats = ntc.tangent_features(base_fn(g), split['train'], eig_bs)
    clf, scaler, vs, _ = ntc.fit_val(feats, labels, split)
    return ntc.test_eval(clf, scaler, vs, feats, labels, split)


@torch.no_grad()
def null_energy(base, split, eigval, eigvec, eig_bs):
    """第 1 折验证被试：切线矩阵落在原始零空间块/值域块的 Frobenius 能量占比。"""
    val = np.asarray(split['val'])
    tangent, _ = rr.tangent_transform(base, split['train'], val, batch_size=eig_bs)
    fr_null, fr_range = [], []
    for k, i in enumerate(val.tolist()):
        s = tangent[k].double()
        keep = eigval[i] > TAU
        u0, ur = eigvec[i][:, ~keep], eigvec[i][:, keep]
        tot = float((s * s).sum())
        bn = u0.T @ s @ u0; br = ur.T @ s @ ur
        fr_null.append(float((bn * bn).sum()) / tot); fr_range.append(float((br * br).sum()) / tot)
    return {'null_block_fraction_mean': float(np.mean(fr_null)), 'range_block_fraction_mean': float(np.mean(fr_range))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--eig-batch-size', type=int, default=32)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = OUT / 'config.json'
    if not cfg.exists():
        cfg.write_text(json.dumps({
            'module': 'null-space completion with training-fold arithmetic mean of raw correlations',
            'tau': TAU, 'gammas': list(GAMMAS), 'selection': 'LR validation AUC per fold',
            'acceptance': {'mean_val_auc_gain': ACCEPT_MEAN, 'min_improved_folds': ACCEPT_FOLDS},
            'test_policy': 'module test metrics computed only if accepted',
            'n_roi': 392, 't_use': 146, 'split_seed': 42}, indent=2), encoding='utf-8')
    dev = torch.device('cuda')
    ts, labels, _ = dataset.load_cc400(t_use=146, verbose=True)
    splits = rr.build_splits(labels, seed=42)
    x = torch.from_numpy(np.ascontiguousarray(ts, dtype=np.float32)).to(dev)
    raw = rr.corr_matrix_all(x, 64)
    # 秩与零空间须在双精度相关矩阵上判定：float32 舍入噪声会把零特征值抬到 1e-8 以上
    raw64 = torch.cat([(lambda b: (b @ b.transpose(-1, -2)) / b.shape[-1])(x[s:s + 32].double())
                       for s in range(0, len(x), 32)])
    del x
    eigval, eigvec = raw_eig(raw64, args.eig_batch_size)
    del raw64; torch.cuda.empty_cache()
    rank = (eigval > TAU).sum(1).float()
    print(f'raw rank (tau={TAU}): mean {rank.mean():.2f}', flush=True)

    t0 = time.time()
    folds = []
    for k, sp in enumerate(splits):
        r0 = raw[torch.as_tensor(sp['train'], device=dev)].double().mean(0)
        r0_min = float(torch.linalg.eigvalsh(r0)[0])
        comp = complete(eigval, eigvec, r0, args.eig_batch_size)
        base = select(lambda g: rr.shrink(raw, g), labels, sp, args.eig_batch_size)
        mod = select(lambda g: rr.shrink(comp, g), labels, sp, args.eig_batch_size)
        folds.append({'fold': k, 'r0_min_eigenvalue': r0_min, 'srts': base, 'nsc': mod})
        print(f'fold {k + 1}: SRTS val {base["val_auc"]:.4f} (g={base["gamma"]})  '
              f'NSC val {mod["val_auc"]:.4f} (g={mod["gamma"]})  {time.time() - t0:.0f}s', flush=True)
        if k == 0:
            folds[0]['diag_srts'] = null_energy(rr.shrink(raw, base['gamma']), sp, eigval, eigvec, args.eig_batch_size)
            folds[0]['diag_nsc'] = null_energy(rr.shrink(comp, mod['gamma']), sp, eigval, eigvec, args.eig_batch_size)
            print('  fold1 energy SRTS', folds[0]['diag_srts'], ' NSC', folds[0]['diag_nsc'], flush=True)
        del comp
        torch.cuda.empty_cache()

    gains = [f['nsc']['val_auc'] - f['srts']['val_auc'] for f in folds]
    accepted = bool(np.mean(gains) >= ACCEPT_MEAN and sum(g > 0 for g in gains) >= ACCEPT_FOLDS)
    decision = {'val_gain_per_fold': gains, 'val_gain_mean': float(np.mean(gains)),
                'improved_folds': int(sum(g > 0 for g in gains)), 'accepted': accepted}
    (OUT / 'phase1_validation.json').write_text(json.dumps({'folds': folds, 'decision': decision}, indent=1), encoding='utf-8')
    print('DECISION', json.dumps(decision), flush=True)

    # 第二阶段：SRTS 测试；模块仅在通过接受标准后才计算测试
    res = {'srts': [], 'nsc': []}
    for k, sp in enumerate(splits):
        res['srts'].append(test_at(lambda g: rr.shrink(raw, g), folds[k]['srts']['gamma'], labels, sp, args.eig_batch_size))
        if accepted:
            r0 = raw[torch.as_tensor(sp['train'], device=dev)].double().mean(0)
            comp = complete(eigval, eigvec, r0, args.eig_batch_size)
            res['nsc'].append(test_at(lambda g: rr.shrink(comp, g), folds[k]['nsc']['gamma'], labels, sp, args.eig_batch_size))
            del comp; torch.cuda.empty_cache()
    summ = {name: ({m: {'mean': float(np.mean([f[m] for f in v])), 'sample_sd': float(np.std([f[m] for f in v], ddof=1)),
                        'fold_values': [f[m] for f in v]} for m in ('auc', 'accuracy', 'sensitivity', 'specificity')}
                   if v else None) for name, v in res.items()}
    (OUT / 'phase2_test.json').write_text(json.dumps({'accepted': accepted, 'test': summ}, indent=1), encoding='utf-8')
    print('TEST', json.dumps({n: (s['auc']['mean'] if s else None) for n, s in summ.items()}), flush=True)
    print(f'done {time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
