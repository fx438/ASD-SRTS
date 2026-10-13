# -*- coding: utf-8 -*-
"""ABIDE-I CC400 论文修订实验。

实现标准分层五折、SRTS 折内 gamma 选择、五种连接表示，以及替代旧 SSJR 的
固定双分支潜表征 Mixup。Mixup 先只看验证集，达到预注册门槛后才评估测试集。
"""
from __future__ import annotations

import os
import argparse
import glob
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim
from sklearn.covariance import LedoitWolf
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

from model_revision import RevisionBrainNetMLP

DATA_ROOT = Path(os.environ.get('ABIDE_CC400_DIR', Path(__file__).resolve().parents[2] / 'data' / 'ABIDE_CC400_Complete'))
SOURCE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = SOURCE_ROOT / 'results_revision_v2' / 'standard5_seed42'
GAMMAS = (0.05, 0.10, 0.20, 0.30, 0.50, 0.70, 0.80, 0.90)
CORE_VARIANTS = ('raw', 'shrink_fisher', 'eigclip_tangent', 'lw_tangent', 'srts')
METRICS = ('auc', 'accuracy', 'sensitivity', 'specificity', 'precision', 'f1', 'acc_at_half')


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def json_ready(value):
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


def write_json_new(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(json_ready(data), ensure_ascii=False, indent=2, allow_nan=False)
    with path.open('x', encoding='utf-8', newline='\n') as fh:
        fh.write(text + '\n')


def torch_save_new(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as fh:
        torch.save(obj, fh)


def build_splits(labels, seed=42):
    """五折均为 602/86/172，并确保每名被试恰好测试一次。"""
    outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    splits = []
    test_count = np.zeros(len(labels), dtype=np.int64)
    for fold, (train_val, test) in enumerate(outer.split(np.zeros(len(labels)), labels)):
        train, val = train_test_split(
            train_val, test_size=86, random_state=seed + fold,
            stratify=labels[train_val],
        )
        train, val, test = map(np.asarray, (train, val, test))
        assert (len(train), len(val), len(test)) == (602, 86, 172)
        assert not set(train) & set(val)
        assert not set(train) & set(test)
        assert not set(val) & set(test)
        test_count[test] += 1
        splits.append({
            'fold': fold,
            'train': train.tolist(),
            'val': val.tolist(),
            'test': test.tolist(),
            'class_counts': {
                'train': np.bincount(labels[train], minlength=2).tolist(),
                'val': np.bincount(labels[val], minlength=2).tolist(),
                'test': np.bincount(labels[test], minlength=2).tolist(),
            },
        })
    assert np.all(test_count == 1)
    return splits


def load_manifest(t_use, labels, sites):
    """按 loader 的 sorted CSV 规则重建身份，绝不按 phenotype 行序连接。"""
    ph = pd.read_csv(DATA_ROOT / 'Phenotypic.csv').dropna(subset=['SUB_ID'])
    by_sid = {int(r.SUB_ID): r for r in ph.itertuples()}
    rows = []
    for text in sorted(glob.glob(str(DATA_ROOT / 'CSV' / '*.csv'))):
        path = Path(text)
        sid = int(path.name.rsplit('_', 1)[1].split('.')[0])
        if sid not in by_sid:
            continue
        with path.open('rb') as fh:
            n_timepoints = sum(1 for _ in fh)
        if n_timepoints < t_use:
            continue
        r = by_sid[sid]
        rows.append({
            'index': len(rows),
            'sub_id': sid,
            'file_id': str(r.FILE_ID),
            'site_id': str(r.SITE_ID),
            'dx_group': int(r.DX_GROUP),
            'label': 0 if int(r.DX_GROUP) == 2 else 1,
            'n_timepoints_original': n_timepoints,
            'csv_basename': path.stem,
        })
    assert len(rows) == len(labels) == 860
    assert np.array_equal(np.asarray([r['label'] for r in rows]), labels)
    assert np.array_equal(np.asarray([r['site_id'] for r in rows], dtype=object), sites)
    assert len({r['sub_id'] for r in rows}) == 860
    assert all(r['csv_basename'] == r['file_id'] for r in rows)
    return rows


def corr_matrix_all(x, batch_size=64):
    result = []
    for start in range(0, len(x), batch_size):
        xb = x[start:start + batch_size]
        c = xb @ xb.transpose(-1, -2) / xb.shape[-1]
        result.append(((c + c.transpose(-1, -2)) * 0.5).clamp(-1.0, 1.0))
    return torch.cat(result)


def matrix_function(m, kind, floor=1e-12, batch_size=32):
    """float64 特征分解实现对称矩阵函数。"""
    single = m.ndim == 2
    source = m.unsqueeze(0) if single else m
    output = []
    for start in range(0, len(source), batch_size):
        block = source[start:start + batch_size].double()
        block = (block + block.transpose(-1, -2)) * 0.5
        eigval, eigvec = torch.linalg.eigh(block)
        if kind == 'clip':
            transformed = eigval.clamp_min(floor)
        elif kind == 'rsqrt':
            transformed = eigval.clamp_min(floor).rsqrt()
        elif kind == 'log':
            transformed = eigval.clamp_min(floor).log()
        else:
            raise ValueError(kind)
        rebuilt = (eigvec * transformed.unsqueeze(-2)) @ eigvec.transpose(-1, -2)
        output.append(rebuilt.float())
    result = torch.cat(output)
    return result[0] if single else result


def tangent_transform(base_matrices, train_indices, target_indices,
                      diag_positions=None, batch_size=32):
    """参考矩阵只取 train_indices；变换 target_indices 并返回可选数值诊断。"""
    device = base_matrices.device
    train_t = torch.as_tensor(train_indices, device=device, dtype=torch.long)
    target = np.asarray(target_indices, dtype=np.int64)
    reference = base_matrices[train_t].double().mean(0)
    reference_isqrt = matrix_function(reference, 'rsqrt', floor=1e-12)
    diag_positions = set() if diag_positions is None else {int(v) for v in diag_positions}
    output, diagnostics = [], []
    for start in range(0, len(target), batch_size):
        idx_np = target[start:start + batch_size]
        idx = torch.as_tensor(idx_np, device=device, dtype=torch.long)
        white = reference_isqrt @ base_matrices[idx] @ reference_isqrt
        white = (white + white.transpose(-1, -2)) * 0.5
        eigval, eigvec = torch.linalg.eigh(white.double())
        logval = eigval.clamp_min(1e-12).log()
        tangent = (eigvec * logval.unsqueeze(-2)) @ eigvec.transpose(-1, -2)
        output.append(tangent.float())
        for local, global_idx in enumerate(idx_np.tolist()):
            if global_idx in diag_positions:
                ev = eigval[local]
                min_eig = float(ev[0].item())
                max_eig = float(ev[-1].item())
                cond = max_eig / min_eig if min_eig > 0 else None
                diagnostics.append({
                    'subject_index': global_idx,
                    'min_eigenvalue': min_eig,
                    'max_eigenvalue': max_eig,
                    'log10_condition_number': (
                        float(np.log10(cond)) if cond is not None and np.isfinite(cond) else None
                    ),
                })
    return torch.cat(output), diagnostics


def fisher_z_offdiag(c, delta=1e-6):
    """对角线先清零，从而绝不对对角线 1 执行 atanh。"""
    z = c.clone()
    z.diagonal(dim1=-2, dim2=-1).zero_()
    return torch.atanh(z.clamp(-1.0 + delta, 1.0 - delta))


def shrink(c, gamma):
    eye = torch.eye(c.shape[-1], device=c.device, dtype=c.dtype)
    return (1.0 - gamma) * c + gamma * eye


def ledoit_wolf_correlations(x_cpu):
    matrices, coefficients = [], []
    for i, subject in enumerate(x_cpu):
        estimator = LedoitWolf(assume_centered=True, store_precision=False).fit(subject.T)
        covariance = estimator.covariance_.astype(np.float64, copy=False)
        scale = np.sqrt(np.clip(np.diag(covariance), 1e-12, None))
        correlation = covariance / np.outer(scale, scale)
        correlation = (correlation + correlation.T) * 0.5
        np.fill_diagonal(correlation, 1.0)
        matrices.append(correlation.astype(np.float32))
        coefficients.append(float(estimator.shrinkage_))
        if (i + 1) % 100 == 0 or i + 1 == len(x_cpu):
            print(f'[Ledoit-Wolf] {i + 1}/{len(x_cpu)}', flush=True)
    return np.stack(matrices), np.asarray(coefficients, dtype=np.float64)


def best_threshold(probabilities, labels):
    candidates = np.unique(np.concatenate([
        [0.5], np.quantile(probabilities, np.linspace(0.02, 0.98, 97))
    ]))
    accuracies = [((probabilities >= t).astype(int) == labels).mean() for t in candidates]
    best = max(accuracies)
    tied = [t for t, a in zip(candidates, accuracies) if a >= best - 1e-12]
    return float(min(tied, key=lambda t: abs(t - 0.5)))


@torch.no_grad()
def evaluate(model, x, fc, y, batch_size=64, threshold=0.5):
    model.eval()
    probabilities = []
    for start in range(0, len(x), batch_size):
        logits = model(x[start:start + batch_size], fc[start:start + batch_size])
        probabilities.append(torch.softmax(logits, 1)[:, 1])
    p = torch.cat(probabilities).cpu().numpy()
    yy = y.detach().cpu().numpy()
    pred = (p >= threshold).astype(np.int64)
    tp = int(((pred == 1) & (yy == 1)).sum())
    tn = int(((pred == 0) & (yy == 0)).sum())
    fp = int(((pred == 1) & (yy == 0)).sum())
    fn = int(((pred == 0) & (yy == 1)).sum())
    sen = tp / (tp + fn) if tp + fn else 0.0
    spe = tn / (tn + fp) if tn + fp else 0.0
    pre = tp / (tp + fp) if tp + fp else 0.0
    return {
        'auc': float(roc_auc_score(yy, p)),
        'accuracy': float((pred == yy).mean()),
        'sensitivity': float(sen),
        'specificity': float(spe),
        'precision': float(pre),
        'f1': float(2 * pre * sen / (pre + sen)) if pre + sen else 0.0,
        'acc_at_half': float(((p >= 0.5).astype(np.int64) == yy).mean()),
        'threshold': float(threshold),
        'probs': p.tolist(),
        'labels': yy.tolist(),
    }


def train_model(x_train, fc_train, y_train, x_val, fc_val, y_val,
                args, seed, mixup=False):
    set_seed(seed)
    model = RevisionBrainNetMLP(
        n_roi=x_train.shape[1], hidden=args.hidden, drop=args.drop, k=args.k
    ).to(x_train.device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5
    )
    best_auc, best_state, wait, epoch = -np.inf, None, 0, 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = torch.randperm(len(x_train), device=x_train.device)
        for start in range(0, len(order), args.batch_size):
            idx = order[start:start + args.batch_size]
            xb = x_train[idx]
            if args.noise > 0:
                xb = xb + torch.randn_like(xb) * args.noise
            fcb, yb = fc_train[idx], y_train[idx]
            optimizer.zero_grad(set_to_none=True)
            if mixup:
                spatial, spectral = model.encode_branches(fcb, xb)
                pair = torch.randperm(len(idx), device=x_train.device)
                values = np.random.beta(args.mixup_alpha, args.mixup_alpha, size=len(idx))
                lam = torch.as_tensor(values, device=x_train.device,
                                      dtype=spatial.dtype).unsqueeze(1)
                spatial = lam * spatial + (1.0 - lam) * spatial[pair]
                spectral = lam * spectral + (1.0 - lam) * spectral[pair]
                soft_y = (
                    lam * F.one_hot(yb, 2).float()
                    + (1.0 - lam) * F.one_hot(yb[pair], 2).float()
                )
                logits = model.classify_embeddings(spatial, spectral)
                loss = -(soft_y * F.log_softmax(logits, dim=1)).sum(1).mean()
            else:
                loss = F.cross_entropy(model(xb, fcb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        val = evaluate(model, x_val, fc_val, y_val, args.eval_batch_size)
        scheduler.step(val['auc'])
        if val['auc'] > best_auc + 1e-4:
            best_auc, wait = val['auc'], 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if epoch >= args.min_epochs and wait >= args.patience:
                break
    if best_state is None:
        raise RuntimeError('训练未产生有效 checkpoint')
    model.load_state_dict(best_state)
    val = evaluate(model, x_val, fc_val, y_val, args.eval_batch_size)
    return {
        'model': model,
        'state_dict': best_state,
        'best_val_auc': float(best_auc),
        'epochs_run': int(epoch),
        'val': val,
        'n_parameters': int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
    }


def test_trained(trained, x_test, fc_test, y_test, batch_size):
    val_p = np.asarray(trained['val']['probs'])
    val_y = np.asarray(trained['val']['labels'])
    threshold = best_threshold(val_p, val_y)
    return evaluate(trained['model'], x_test, fc_test, y_test, batch_size, threshold)


def strip_training(result):
    return {
        'best_val_auc': result['best_val_auc'],
        'epochs_run': result['epochs_run'],
        'n_parameters': result['n_parameters'],
        'val': result['val'],
    }


def run_fold(fold_item, x, y, raw_corr, eigclip_corr, lw_corr, args, output):
    fold = int(fold_item['fold'])
    tr = np.asarray(fold_item['train'], dtype=np.int64)
    va = np.asarray(fold_item['val'], dtype=np.int64)
    te = np.asarray(fold_item['test'], dtype=np.int64)
    train_val = np.concatenate([tr, va])
    fold_seed = args.seed + fold
    print(f'\n=== fold {fold + 1}/5: 602/86/172 ===', flush=True)

    gamma_trials, selected = [], None
    for gamma in args.gammas:
        fc_train_val, numerics = tangent_transform(
            shrink(raw_corr, gamma), tr, train_val,
            diag_positions=set(va.tolist()), batch_size=args.eig_batch_size,
        )
        trained = train_model(
            x[tr], fc_train_val[:len(tr)], y[tr],
            x[va], fc_train_val[len(tr):], y[va],
            args, seed=fold_seed, mixup=False,
        )
        gamma_trials.append({
            'gamma': gamma,
            'best_val_auc': trained['best_val_auc'],
            'epochs_run': trained['epochs_run'],
            'numerics': numerics,
        })
        print(f"[fold {fold + 1}] gamma={gamma:.2f} val_auc="
              f"{trained['best_val_auc']:.4f} ep={trained['epochs_run']}", flush=True)
        if selected is None or trained['best_val_auc'] > selected['best_val_auc'] + 1e-12:
            selected = {
                'gamma': gamma,
                'best_val_auc': trained['best_val_auc'],
                'state_dict': trained['state_dict'],
                'val': trained['val'],
                'epochs_run': trained['epochs_run'],
                'n_parameters': trained['n_parameters'],
            }
        del trained, fc_train_val
        torch.cuda.empty_cache()

    selected_gamma = float(selected['gamma'])
    print(f'[fold {fold + 1}] selected gamma={selected_gamma:.2f} by validation only', flush=True)
    target_all = np.concatenate([tr, va, te])
    srts_all, selected_numerics = tangent_transform(
        shrink(raw_corr, selected_gamma), tr, target_all,
        diag_positions=set(va.tolist()), batch_size=args.eig_batch_size,
    )
    ntr, nva = len(tr), len(va)
    fc_srts_tr = srts_all[:ntr]
    fc_srts_va = srts_all[ntr:ntr + nva]
    fc_srts_te = srts_all[ntr + nva:]

    srts_model = RevisionBrainNetMLP(
        n_roi=x.shape[1], hidden=args.hidden, drop=args.drop, k=args.k
    ).to(x.device)
    srts_model.load_state_dict(selected['state_dict'])
    srts_test = test_trained(
        {'model': srts_model, 'val': selected['val']},
        x[te], fc_srts_te, y[te], args.eval_batch_size,
    )
    variants = {
        'srts': {
            'best_val_auc': selected['best_val_auc'],
            'epochs_run': selected['epochs_run'],
            'n_parameters': selected['n_parameters'],
            'selected_gamma': selected_gamma,
            'test': srts_test,
        }
    }

    def fit_fixed(name, fc_train, fc_val, fc_test):
        trained = train_model(
            x[tr], fc_train, y[tr], x[va], fc_val, y[va],
            args, seed=fold_seed, mixup=False,
        )
        tested = test_trained(trained, x[te], fc_test, y[te], args.eval_batch_size)
        variants[name] = {
            'best_val_auc': trained['best_val_auc'],
            'epochs_run': trained['epochs_run'],
            'n_parameters': trained['n_parameters'],
            'test': tested,
        }
        print(f"[fold {fold + 1}] {name} val={trained['best_val_auc']:.4f} "
              f"test={tested['auc']:.4f}", flush=True)

    fit_fixed('raw', raw_corr[tr], raw_corr[va], raw_corr[te])
    fisher = fisher_z_offdiag(shrink(raw_corr, selected_gamma))
    fit_fixed('shrink_fisher', fisher[tr], fisher[va], fisher[te])
    del fisher
    torch.cuda.empty_cache()

    eig_tangent, _ = tangent_transform(
        eigclip_corr, tr, target_all, batch_size=args.eig_batch_size
    )
    fit_fixed('eigclip_tangent', eig_tangent[:ntr],
              eig_tangent[ntr:ntr + nva], eig_tangent[ntr + nva:])
    del eig_tangent
    torch.cuda.empty_cache()

    lw_tangent, _ = tangent_transform(
        lw_corr, tr, target_all, batch_size=args.eig_batch_size
    )
    fit_fixed('lw_tangent', lw_tangent[:ntr],
              lw_tangent[ntr:ntr + nva], lw_tangent[ntr + nva:])
    del lw_tangent
    torch.cuda.empty_cache()

    # 新模块二先只做验证；脚本在五折验证门槛通过前绝不读取其测试表现。
    mixup_validation, checkpoints = {}, {}
    for name, fc_train, fc_val in (
        ('raw_mixup', raw_corr[tr], raw_corr[va]),
        ('srts_mixup', fc_srts_tr, fc_srts_va),
    ):
        trained = train_model(
            x[tr], fc_train, y[tr], x[va], fc_val, y[va],
            args, seed=fold_seed, mixup=True,
        )
        mixup_validation[name] = strip_training(trained)
        checkpoints[name] = {
            'state_dict': trained['state_dict'],
            'val': trained['val'],
            'selected_gamma': selected_gamma if name == 'srts_mixup' else None,
        }
        print(f"[fold {fold + 1}] {name} validation_only="
              f"{trained['best_val_auc']:.4f}", flush=True)

    checkpoint_path = output / 'checkpoints' / f'fold_{fold + 1}_mixup.pt'
    torch_save_new(checkpoint_path, checkpoints)
    fold_result = {
        'fold': fold,
        'sizes': [len(tr), len(va), len(te)],
        'class_counts': fold_item['class_counts'],
        'test_indices': te.tolist(),
        'selected_gamma': selected_gamma,
        'gamma_trials': gamma_trials,
        'selected_gamma_numerics': selected_numerics,
        'variants': variants,
        'mixup_validation_only': mixup_validation,
        'mixup_checkpoint': str(checkpoint_path),
    }
    write_json_new(output / 'folds' / f'fold_{fold + 1}.json', fold_result)
    del srts_all, fc_srts_tr, fc_srts_va, fc_srts_te
    torch.cuda.empty_cache()
    return fold_result


def evaluate_mixup_after_gate(fold_results, splits, x, y, raw_corr, args, output):
    deltas = []
    improving = 0
    for result in fold_results:
        base = result['variants']['srts']['best_val_auc']
        mixed = result['mixup_validation_only']['srts_mixup']['best_val_auc']
        delta = float(mixed - base)
        deltas.append(delta)
        improving += int(delta > 0.0)
    mean_delta = float(np.mean(deltas))
    accepted = bool(
        mean_delta >= args.mixup_min_delta
        and improving >= args.mixup_min_improved_folds
    )
    decision = {
        'module': 'dual_branch_latent_mixup',
        'alpha': args.mixup_alpha,
        'selection_uses_test_data': False,
        'rule': {
            'minimum_mean_validation_auc_delta': args.mixup_min_delta,
            'minimum_improved_folds': args.mixup_min_improved_folds,
        },
        'validation_auc_deltas_srts_mixup_minus_srts': deltas,
        'mean_validation_auc_delta': mean_delta,
        'improved_folds': improving,
        'accepted': accepted,
        'test_evaluated': accepted,
    }
    write_json_new(output / 'mixup_decision.json', decision)
    print(f"\n[Mixup gate] mean val delta={mean_delta:+.4f}, "
          f"improved={improving}/5, accepted={accepted}", flush=True)
    if not accepted:
        return decision, None

    test_results = {'raw_mixup': [], 'srts_mixup': []}
    for result, split in zip(fold_results, splits):
        fold = int(result['fold'])
        tr = np.asarray(split['train'], dtype=np.int64)
        te = np.asarray(split['test'], dtype=np.int64)
        checkpoints = torch.load(
            output / 'checkpoints' / f'fold_{fold + 1}_mixup.pt',
            map_location=x.device, weights_only=False,
        )
        for name in ('raw_mixup', 'srts_mixup'):
            model = RevisionBrainNetMLP(
                n_roi=x.shape[1], hidden=args.hidden, drop=args.drop, k=args.k
            ).to(x.device)
            model.load_state_dict(checkpoints[name]['state_dict'])
            if name == 'raw_mixup':
                fc_test = raw_corr[te]
            else:
                gamma = float(checkpoints[name]['selected_gamma'])
                fc_test, _ = tangent_transform(
                    shrink(raw_corr, gamma), tr, te,
                    batch_size=args.eig_batch_size,
                )
            trained = {'model': model, 'val': checkpoints[name]['val']}
            tested = test_trained(
                trained, x[te], fc_test, y[te], args.eval_batch_size
            )
            test_results[name].append({
                'fold': fold,
                'test_indices': te.tolist(),
                'test': tested,
            })
            print(f"[Mixup unlocked] fold={fold + 1} {name} "
                  f"test_auc={tested['auc']:.4f}", flush=True)
            del model, fc_test
            torch.cuda.empty_cache()
    write_json_new(output / 'mixup_test_results.json', test_results)
    return decision, test_results


def summarize_ledoit(coefficients):
    return {
        'n': len(coefficients),
        'mean': float(coefficients.mean()),
        'sample_sd': float(coefficients.std(ddof=1)),
        'min': float(coefficients.min()),
        'q25': float(np.quantile(coefficients, 0.25)),
        'median': float(np.quantile(coefficients, 0.50)),
        'q75': float(np.quantile(coefficients, 0.75)),
        'max': float(coefficients.max()),
    }


def summarize(fold_results, decision, mixup_test, manifest, lw_coefficients, elapsed):
    summary = {
        'protocol': {
            'folds': 5,
            'per_fold_sizes': [602, 86, 172],
            'test_union_n': 860,
            'test_count_per_subject': 1,
            'seed': 42,
            'sd_definition': 'sample SD across five folds (ddof=1)',
        },
        'selected_gammas': [float(f['selected_gamma']) for f in fold_results],
        'variants': {},
        'mixup_decision': decision,
        'ledoit_wolf_shrinkage': summarize_ledoit(lw_coefficients),
        'elapsed_minutes': float(elapsed),
    }
    for variant in CORE_VARIANTS:
        summary['variants'][variant] = {}
        for metric in METRICS:
            values = np.asarray([
                f['variants'][variant]['test'][metric] for f in fold_results
            ], dtype=float)
            summary['variants'][variant][metric] = {
                'mean': float(values.mean()),
                'sample_sd': float(values.std(ddof=1)),
                'fold_values': values.tolist(),
            }
        summary['variants'][variant]['mean_validation_auc'] = float(np.mean([
            f['variants'][variant]['best_val_auc'] for f in fold_results
        ]))

    if mixup_test is not None:
        for variant in ('raw_mixup', 'srts_mixup'):
            summary['variants'][variant] = {}
            for metric in METRICS:
                values = np.asarray([
                    r['test'][metric] for r in mixup_test[variant]
                ], dtype=float)
                summary['variants'][variant][metric] = {
                    'mean': float(values.mean()),
                    'sample_sd': float(values.std(ddof=1)),
                    'fold_values': values.tolist(),
                }
            summary['variants'][variant]['mean_validation_auc'] = float(np.mean([
                f['mixup_validation_only'][variant]['best_val_auc']
                for f in fold_results
            ]))

    seen = np.zeros(len(manifest), dtype=np.int64)
    oof = []
    mixup_lookup = {}
    if mixup_test is not None:
        for variant, records in mixup_test.items():
            for record in records:
                for index, prob in zip(record['test_indices'], record['test']['probs']):
                    mixup_lookup[(variant, index)] = prob
    for result in fold_results:
        indices = result['test_indices']
        seen[indices] += 1
        for position, index in enumerate(indices):
            row = {
                'subject_index': index,
                'sub_id': manifest[index]['sub_id'],
                'file_id': manifest[index]['file_id'],
                'site_id': manifest[index]['site_id'],
                'label': manifest[index]['label'],
                'fold': result['fold'],
            }
            for variant in CORE_VARIANTS:
                row[f'{variant}_prob'] = result['variants'][variant]['test']['probs'][position]
            if mixup_test is not None:
                row['raw_mixup_prob'] = mixup_lookup[('raw_mixup', index)]
                row['srts_mixup_prob'] = mixup_lookup[('srts_mixup', index)]
            oof.append(row)
    assert np.all(seen == 1)
    oof.sort(key=lambda r: r['subject_index'])
    summary['oof_predictions'] = oof
    return summary


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--t-use', type=int, default=146)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--min-epochs', type=int, default=60)
    parser.add_argument('--patience', type=int, default=40)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--eval-batch-size', type=int, default=64)
    parser.add_argument('--eig-batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--noise', type=float, default=0.02)
    parser.add_argument('--hidden', type=int, default=32)
    parser.add_argument('--drop', type=float, default=0.0)
    parser.add_argument('--k', type=int, default=40)
    parser.add_argument('--mixup-alpha', type=float, default=0.20)
    parser.add_argument('--mixup-min-delta', type=float, default=0.01)
    parser.add_argument('--mixup-min-improved-folds', type=int, default=4)
    parser.add_argument('--gammas', type=float, nargs='+', default=list(GAMMAS))
    return parser.parse_args()


def main():
    args = parse_args()
    if args.seed != 42 or args.t_use != 146:
        raise ValueError('正式修订实验固定 seed=42、t_use=146')
    if tuple(args.gammas) != GAMMAS:
        raise ValueError(f'正式实验 gamma 候选必须固定为 {GAMMAS}')
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    final_path = output / 'summary.json'
    if final_path.exists():
        raise FileExistsError(f'拒绝覆盖已完成结果：{final_path}')

    sys.path.insert(0, str(SOURCE_ROOT))
    import dataset_cc400 as dataset
    started = time.time()
    ts, labels, sites = dataset.load_cc400(t_use=args.t_use, verbose=True)
    manifest = load_manifest(args.t_use, labels, sites)
    splits = build_splits(labels, seed=args.seed)

    config = vars(args).copy()
    config['output'] = str(output)
    config['gammas'] = list(args.gammas)
    config.update({
        'old_ssjr_enabled': False,
        'model_module': 'model_revision.py',
        'representation_floor_eigclip': 1e-6,
        'tangent_numeric_floor': 1e-12,
        'time_series_standardization_ddof': 0,
    })
    config_path = output / 'config.json'
    if not config_path.exists():
        write_json_new(config_path, config)
    elif not args.resume:
        raise FileExistsError(f'目录已有配置；中断续跑请加 --resume：{config_path}')

    split_payload = {'manifest': manifest, 'splits': splits}
    split_path = output / 'splits_manifest.json'
    if not split_path.exists():
        write_json_new(split_path, split_payload)
    else:
        with split_path.open(encoding='utf-8') as fh:
            if json.load(fh) != split_payload:
                raise RuntimeError('已有划分与当前重建结果不一致')

    provenance_path = output / 'provenance.json'
    if not provenance_path.exists():
        source_files = [
            SOURCE_ROOT / 'dataset_cc400.py',
            Path(__file__).resolve(),
            Path(__file__).resolve().with_name('model_revision.py'),
        ]
        write_json_new(provenance_path, {
            'source_sha256': {str(p): sha256_file(p) for p in source_files},
            'data_cache': str(SOURCE_ROOT / 'cc400_cache_t146.npz'),
            'data_cache_sha256': sha256_file(SOURCE_ROOT / 'cc400_cache_t146.npz'),
            'torch_version': torch.__version__,
            'numpy_version': np.__version__,
        })

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type != 'cuda':
        raise RuntimeError('正式实验要求 CUDA')
    x = torch.from_numpy(ts).to(device)
    y = torch.from_numpy(labels).to(device)
    print(f'device={device}, X={tuple(x.shape)}, ASD={int(labels.sum())}, '
          f'TC={int((labels == 0).sum())}', flush=True)

    with torch.no_grad():
        raw_corr = corr_matrix_all(x, args.eval_batch_size)
        print('raw correlation matrices ready', flush=True)
        eigclip_corr = matrix_function(
            raw_corr, 'clip', floor=1e-6, batch_size=args.eig_batch_size
        )
        print('fixed-eigenvalue-clipped matrices ready', flush=True)
    lw_np, lw_coefficients = ledoit_wolf_correlations(ts)
    lw_corr = torch.from_numpy(lw_np).to(device)
    del lw_np
    print('Ledoit-Wolf correlation matrices ready', flush=True)

    fold_results = []
    for split in splits:
        fold_path = output / 'folds' / f"fold_{split['fold'] + 1}.json"
        if fold_path.exists():
            if not args.resume:
                raise FileExistsError(f'折结果已存在：{fold_path}')
            with fold_path.open(encoding='utf-8') as fh:
                fold_results.append(json.load(fh))
            print(f"resume: fold {split['fold'] + 1} loaded", flush=True)
        else:
            fold_results.append(run_fold(
                split, x, y, raw_corr, eigclip_corr, lw_corr, args, output
            ))

    decision_path = output / 'mixup_decision.json'
    if decision_path.exists():
        if not args.resume:
            raise FileExistsError(f'模块二决策已存在：{decision_path}')
        with decision_path.open(encoding='utf-8') as fh:
            decision = json.load(fh)
        mixup_path = output / 'mixup_test_results.json'
        if mixup_path.exists():
            with mixup_path.open(encoding='utf-8') as fh:
                mixup_test = json.load(fh)
        else:
            mixup_test = None
    else:
        decision, mixup_test = evaluate_mixup_after_gate(
            fold_results, splits, x, y, raw_corr, args, output
        )

    elapsed = (time.time() - started) / 60.0
    summary = summarize(
        fold_results, decision, mixup_test, manifest, lw_coefficients, elapsed
    )
    write_json_new(final_path, summary)
    print(f'\nDONE in {elapsed:.2f} min -> {final_path}', flush=True)
    for variant, metrics in summary['variants'].items():
        auc, acc = metrics['auc'], metrics['accuracy']
        print(f"{variant:20s} AUC={auc['mean']:.4f}±{auc['sample_sd']:.4f} "
              f"ACC={acc['mean']:.4f}±{acc['sample_sd']:.4f}", flush=True)


if __name__ == '__main__':
    main()
