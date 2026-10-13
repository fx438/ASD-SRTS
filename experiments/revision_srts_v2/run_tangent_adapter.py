#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""固定候选“轻量切线潜表征残差适配器”的验证门与条件测试。

本脚本复用已经锁定的标准五折、每折 gamma 和 SRTS 表示。候选模型先仅在五折
训练/验证集上运行；只有平均验证 AUC 增量不少于 0.01 且至少 4/5 折提高，才会
计算候选模型的测试预测。任何测试指标都不参与结构或超参数选择。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

SOURCE_ROOT = Path(__file__).resolve().parents[1]
REVISION_ROOT = SOURCE_ROOT / 'revision_srts_v2'
BASE_RESULTS = SOURCE_ROOT / 'results_revision_v2' / 'standard5_seed42'
DEFAULT_OUTPUT = SOURCE_ROOT / 'results_revision_v2' / 'tangent_adapter_standard5_seed42_v1'
sys.path.insert(0, str(SOURCE_ROOT))
sys.path.insert(0, str(REVISION_ROOT))

import dataset_cc400 as dataset
from model_tangent_adapter import TangentAdapterBrainNetMLP
from run_revision import (
    build_splits,
    corr_matrix_all,
    evaluate,
    json_ready,
    set_seed,
    shrink,
    tangent_transform,
    test_trained,
)

METRICS = ('auc', 'accuracy', 'sensitivity', 'specificity', 'precision', 'f1', 'acc_at_half')
MODULE_NAME = 'lightweight_tangent_latent_adapter'


def read_json(path: Path):
    with path.open(encoding='utf-8') as handle:
        return json.load(handle)


def write_json_new(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8', newline='\n') as handle:
        json.dump(json_ready(data), handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')


def torch_save_new(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as handle:
        torch.save(value, handle)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def train_candidate(x_train, fc_train, y_train, x_val, fc_val, y_val, args, seed):
    set_seed(seed)
    model = TangentAdapterBrainNetMLP(
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
            index = order[start:start + args.batch_size]
            xb = x_train[index]
            if args.noise > 0:
                xb = xb + torch.randn_like(xb) * args.noise
            logits = model(xb, fc_train[index])
            loss = F.cross_entropy(logits, y_train[index])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        val = evaluate(model, x_val, fc_val, y_val, args.eval_batch_size)
        scheduler.step(val['auc'])
        if val['auc'] > best_auc + 1e-4:
            best_auc, wait = val['auc'], 0
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
        else:
            wait += 1
            if epoch >= args.min_epochs and wait >= args.patience:
                break
    if best_state is None:
        raise RuntimeError('候选模型训练未产生有效 checkpoint')
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


def metric_summary(test_rows):
    result = {}
    for metric in METRICS:
        values = np.asarray([row['test'][metric] for row in test_rows], dtype=np.float64)
        result[metric] = {
            'fold_values': values.tolist(),
            'mean': float(values.mean()),
            'sample_sd': float(values.std(ddof=1)),
        }
    return result


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-results', type=Path, default=BASE_RESULTS)
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
    parser.add_argument('--minimum-mean-val-delta', type=float, default=0.01)
    parser.add_argument('--minimum-improved-folds', type=int, default=4)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.seed != 42 or args.t_use != 146:
        raise ValueError('候选实验固定 seed=42、t_use=146')
    if (args.minimum_mean_val_delta, args.minimum_improved_folds) != (0.01, 4):
        raise ValueError('候选模块验证门固定为平均增量 0.01 且至少 4/5 折提高')

    started = time.time()
    base = args.base_results.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'summary.json').exists():
        raise FileExistsError(f'拒绝覆盖已完成结果：{output / "summary.json"}')

    base_config = read_json(base / 'config.json')
    split_document = read_json(base / 'splits_manifest.json')
    locked_splits = split_document['splits']
    base_folds = [
        read_json(base / 'folds' / f'fold_{fold}.json')
        for fold in range(1, 6)
    ]
    if int(base_config['seed']) != args.seed or int(base_config['t_use']) != args.t_use:
        raise RuntimeError('基础实验配置与候选实验不一致')

    config = vars(args).copy()
    config.update({
        'base_results': str(base),
        'output': str(output),
        'module': MODULE_NAME,
        'candidate_status': 'fourth_quick_validation_candidate',
        'primary_outer_test_results_known_before_module_design': True,
        'candidate_specific_test_predictions_unseen_until_gate': True,
        'test_selection_rule': 'test only after fixed validation gate passes',
        'architecture_tuned': False,
        'hyperparameters_reused_from_primary_experiment': True,
        'selected_gammas_reused_without_reselection': [
            float(row['selected_gamma']) for row in base_folds
        ],
    })
    config_path = output / 'config.json'
    if not config_path.exists():
        write_json_new(config_path, config)
    elif not args.resume:
        raise FileExistsError(f'目录已有配置；中断续跑请加 --resume：{config_path}')
    elif read_json(config_path) != json_ready(config):
        raise RuntimeError('续跑参数与已有配置不一致')

    provenance_path = output / 'provenance.json'
    if not provenance_path.exists():
        source_paths = [
            Path(__file__).resolve(),
            REVISION_ROOT / 'model_tangent_adapter.py',
            REVISION_ROOT / 'run_revision.py',
            SOURCE_ROOT / 'dataset_cc400.py',
        ]
        write_json_new(provenance_path, {
            'source_sha256': {str(path): sha256(path) for path in source_paths},
            'base_config_sha256': sha256(base / 'config.json'),
            'base_splits_sha256': sha256(base / 'splits_manifest.json'),
            'base_fold_sha256': {
                f'fold_{fold}': sha256(base / 'folds' / f'fold_{fold}.json')
                for fold in range(1, 6)
            },
            'data_cache': str(SOURCE_ROOT / 'cc400_cache_t146.npz'),
            'data_cache_sha256': sha256(SOURCE_ROOT / 'cc400_cache_t146.npz'),
            'torch_version': torch.__version__,
            'numpy_version': np.__version__,
        })

    ts, labels, _ = dataset.load_cc400(t_use=args.t_use, verbose=True)
    rebuilt_splits = build_splits(labels, seed=args.seed)
    if rebuilt_splits != locked_splits:
        raise RuntimeError('重建划分与基础实验锁定划分不一致')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type != 'cuda':
        raise RuntimeError('正式候选实验要求 CUDA')
    x = torch.from_numpy(ts).to(device)
    y = torch.from_numpy(labels).to(device)
    with torch.no_grad():
        raw_corr = corr_matrix_all(x, args.eval_batch_size)
    print(f'device={device}, X={tuple(x.shape)}; locked SRTS gamma reused', flush=True)

    validation_rows = []
    for split, base_fold in zip(locked_splits, base_folds):
        fold = int(split['fold'])
        fold_path = output / 'folds' / f'fold_{fold + 1}_validation.json'
        checkpoint_path = output / 'checkpoints' / f'fold_{fold + 1}.pt'
        if fold_path.exists() and checkpoint_path.exists():
            if not args.resume:
                raise FileExistsError(f'候选折结果已存在：{fold_path}')
            row = read_json(fold_path)
            validation_rows.append(row)
            print(f'resume: fold {fold + 1} validation loaded', flush=True)
            continue

        train = np.asarray(split['train'], dtype=np.int64)
        val = np.asarray(split['val'], dtype=np.int64)
        gamma = float(base_fold['selected_gamma'])
        target = np.concatenate((train, val))
        srts, _ = tangent_transform(
            shrink(raw_corr, gamma), train, target,
            diag_positions=set(val.tolist()), batch_size=args.eig_batch_size,
        )
        trained = train_candidate(
            x[train], srts[:len(train)], y[train],
            x[val], srts[len(train):], y[val],
            args, seed=args.seed + fold,
        )
        base_val_auc = float(base_fold['variants']['srts']['best_val_auc'])
        row = {
            'fold': fold,
            'sizes': [len(train), len(val), len(split['test'])],
            'selected_gamma_reused': gamma,
            'base_srts_val_auc': base_val_auc,
            'candidate_val_auc': trained['best_val_auc'],
            'validation_auc_delta': float(trained['best_val_auc'] - base_val_auc),
            'epochs_run': trained['epochs_run'],
            'n_parameters': trained['n_parameters'],
            'val': trained['val'],
            'test_fields_present': False,
        }
        torch_save_new(checkpoint_path, {
            'state_dict': trained['state_dict'],
            'val': trained['val'],
            'selected_gamma': gamma,
        })
        write_json_new(fold_path, row)
        validation_rows.append(row)
        print(
            f"[fold {fold + 1}] base_val={base_val_auc:.4f} "
            f"candidate_val={trained['best_val_auc']:.4f} "
            f"delta={row['validation_auc_delta']:+.4f}",
            flush=True,
        )
        del srts, trained
        torch.cuda.empty_cache()

    deltas = [float(row['validation_auc_delta']) for row in validation_rows]
    mean_delta = float(np.mean(deltas))
    improved = int(sum(delta > 0.0 for delta in deltas))
    accepted = bool(
        mean_delta >= args.minimum_mean_val_delta
        and improved >= args.minimum_improved_folds
    )
    decision = {
        'module': MODULE_NAME,
        'exploratory_module': True,
        'selection_uses_candidate_test_data': False,
        'rule': {
            'minimum_mean_validation_auc_delta': args.minimum_mean_val_delta,
            'minimum_improved_folds': args.minimum_improved_folds,
        },
        'validation_auc_deltas_candidate_minus_srts': deltas,
        'mean_validation_auc_delta': mean_delta,
        'improved_folds': improved,
        'accepted': accepted,
        'test_evaluated': accepted,
    }
    decision_path = output / 'decision.json'
    if not decision_path.exists():
        write_json_new(decision_path, decision)
    elif read_json(decision_path) != decision:
        raise RuntimeError('已有验证门决策与重算结果不一致')
    print(
        f'\n[Tangent-adapter gate] mean val delta={mean_delta:+.4f}, '
        f'improved={improved}/5, accepted={accepted}',
        flush=True,
    )

    test_rows = None
    if accepted:
        test_path = output / 'test_results.json'
        if test_path.exists():
            if not args.resume:
                raise FileExistsError(f'候选测试结果已存在：{test_path}')
            test_rows = read_json(test_path)
        else:
            test_rows = []
            for split, validation in zip(locked_splits, validation_rows):
                fold = int(split['fold'])
                train = np.asarray(split['train'], dtype=np.int64)
                test = np.asarray(split['test'], dtype=np.int64)
                gamma = float(validation['selected_gamma_reused'])
                srts_test, _ = tangent_transform(
                    shrink(raw_corr, gamma), train, test,
                    batch_size=args.eig_batch_size,
                )
                checkpoint = torch.load(
                    output / 'checkpoints' / f'fold_{fold + 1}.pt',
                    map_location=device, weights_only=False,
                )
                model = TangentAdapterBrainNetMLP(
                    n_roi=x.shape[1], hidden=args.hidden, drop=args.drop, k=args.k
                ).to(device)
                model.load_state_dict(checkpoint['state_dict'])
                tested = test_trained(
                    {'model': model, 'val': checkpoint['val']},
                    x[test], srts_test, y[test], args.eval_batch_size,
                )
                test_rows.append({
                    'fold': fold,
                    'test_indices': test.tolist(),
                    'selected_gamma_reused': gamma,
                    'test': tested,
                })
                print(
                    f"[fold {fold + 1}] candidate test_auc={tested['auc']:.4f} "
                    f"test_acc={tested['accuracy']:.4f}",
                    flush=True,
                )
                del srts_test, model
                torch.cuda.empty_cache()
            write_json_new(test_path, test_rows)

    summary = {
        'module': MODULE_NAME,
        'decision': decision,
        'validation': validation_rows,
        'test_metrics': metric_summary(test_rows) if test_rows is not None else None,
        'elapsed_minutes': float((time.time() - started) / 60.0),
        'interpretation_constraint': (
            'This module was proposed after the primary SRTS analysis and is exploratory. '
            'Its candidate test predictions were computed only if the fixed validation gate passed.'
        ),
    }
    write_json_new(output / 'summary.json', summary)
    if test_rows is None:
        print('候选模块未通过验证门；未计算任何候选测试指标。', flush=True)
    else:
        auc = summary['test_metrics']['auc']
        acc = summary['test_metrics']['accuracy']
        print(
            f"candidate AUC={auc['mean']:.4f}±{auc['sample_sd']:.4f} "
            f"ACC={acc['mean']:.4f}±{acc['sample_sd']:.4f}",
            flush=True,
        )
    print(f"DONE -> {output / 'summary.json'}", flush=True)


if __name__ == '__main__':
    main()
