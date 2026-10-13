# -*- coding: utf-8 -*-
"""Score a retrained published model with the protocol of the paper.

Input: a CSV with one row per (fold, split, participant) and columns
    fold   0-4, as in folds_seed42.json
    split  'val' or 'test'
    index  row index into inputs_cc400_t146.npz
    prob   predicted probability of ASD
Validation rows are used only to fix the decision threshold (run_revision.best_threshold: the
accuracy-maximizing cut-off, ties broken toward 0.5); test rows are then scored once.

Optionally, --reference gives a CSV with columns index,prob holding pooled out-of-fold test
probabilities of another method (e.g. SRTS + LR) for a paired bootstrap on the AUC difference.

    python published_models/evaluate_predictions.py preds_gbt.csv [--reference srts_oof.csv]
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score, precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from srts.stats import paired_bootstrap_auc  # noqa: E402


def best_threshold(probabilities, labels):
    """Copied from experiments/revision_srts_v2/run_revision.py (kept torch-free here)."""
    candidates = np.unique(np.concatenate([[0.5], np.quantile(probabilities, np.linspace(0.02, 0.98, 97))]))
    accuracies = [((probabilities >= t).astype(int) == labels).mean() for t in candidates]
    best = max(accuracies)
    tied = [t for t, a in zip(candidates, accuracies) if a >= best - 1e-12]
    return float(min(tied, key=lambda t: abs(t - 0.5)))


def metrics(y, p, threshold):
    pred = (p >= threshold).astype(int)
    tp, tn = int(((pred == 1) & (y == 1)).sum()), int(((pred == 0) & (y == 0)).sum())
    return {'auc': roc_auc_score(y, p), 'accuracy': float((pred == y).mean()),
            'sensitivity': tp / max(int((y == 1).sum()), 1), 'specificity': tn / max(int((y == 0).sum()), 1),
            'precision': float(precision_score(y, pred, zero_division=0)), 'f1': float(f1_score(y, pred))}


def read_rows(path):
    with open(path, newline='', encoding='utf-8') as fh:
        return list(csv.DictReader(fh))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('predictions', type=Path)
    ap.add_argument('--reference', type=Path)
    args = ap.parse_args()

    labels = np.load(Path(__file__).resolve().parent / 'inputs_cc400_t146.npz')['labels']
    rows = read_rows(args.predictions)
    per_fold, oof = [], np.full(len(labels), np.nan)
    oof_fold = np.full(len(labels), -1)
    for k in range(5):
        val = [r for r in rows if int(r['fold']) == k and r['split'] == 'val']
        test = [r for r in rows if int(r['fold']) == k and r['split'] == 'test']
        vi, vp = np.array([int(r['index']) for r in val]), np.array([float(r['prob']) for r in val])
        ti, tp = np.array([int(r['index']) for r in test]), np.array([float(r['prob']) for r in test])
        threshold = best_threshold(vp, labels[vi])
        per_fold.append(metrics(labels[ti], tp, threshold))
        oof[ti], oof_fold[ti] = tp, k
    assert not np.isnan(oof).any(), 'every participant must appear once in a test split'

    summary = {m: {'mean': float(np.mean([f[m] for f in per_fold])),
                   'sd': float(np.std([f[m] for f in per_fold], ddof=1))} for m in per_fold[0]}
    summary['pooled_oof_auc'] = float(roc_auc_score(labels, oof))
    if args.reference:
        ref = np.full(len(labels), np.nan)
        for r in read_rows(args.reference):
            ref[int(r['index'])] = float(r['prob'])
        summary['reference_minus_model'] = paired_bootstrap_auc(labels, oof_fold, ref, oof)
    print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
