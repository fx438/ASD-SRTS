# -*- coding: utf-8 -*-
"""Export the exact inputs and folds of the paper for retraining published models.

Writes published_models/inputs_cc400_t146.npz with
    time_series  (860, 392, 146) float32, each ROI z-scored over time (same loader as the paper)
    pearson_fc   (860, 392, 392) float32, Pearson correlation computed from time_series
    labels       (860,) int64, 1 = ASD, 0 = typically developing control
    sites        (860,) str, ABIDE-I SITE_ID
    file_ids     (860,) str, PCP FILE_ID (row order of the paper)
and published_models/folds_seed42.json: five folds, each with 'train' (602), 'val' (86) and
'test' (172) row indices into the arrays above. Every participant is tested exactly once.

Protocol used in the paper for every published model (Section 2.5):
  * official implementation and default hyperparameters;
  * checkpoint / epoch chosen on the validation split only, never on the test split;
  * decision threshold fixed on the validation split (evaluate_predictions.py does this);
  * the test split is evaluated once, after selection.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'experiments'))

import dataset_cc400  # noqa: E402


def main():
    ts, labels, sites = dataset_cc400.load_cc400(t_use=146, verbose=True)
    subjects = (ROOT / 'data' / 'subjects_860.csv').read_text(encoding='utf-8').splitlines()[1:]
    file_ids = np.array([row.split(',')[2] for row in subjects])
    ref_labels = np.array([int(row.split(',')[4]) for row in subjects])
    assert len(ts) == 860 and np.array_equal(labels, ref_labels), 'data differ from the paper (see data/README)'

    x = ts.astype(np.float64)
    fc = np.clip(x @ np.swapaxes(x, 1, 2) / x.shape[-1], -1.0, 1.0).astype(np.float32)
    out = Path(__file__).resolve().parent
    np.savez_compressed(out / 'inputs_cc400_t146.npz', time_series=ts, pearson_fc=fc, labels=labels,
                        sites=sites.astype(str), file_ids=file_ids)
    splits = json.loads((ROOT / 'data' / 'splits_seed42.json').read_text(encoding='utf-8'))
    folds = [{k: s[k] for k in ('fold', 'train', 'val', 'test')} for s in splits]
    (out / 'folds_seed42.json').write_text(json.dumps(folds), encoding='utf-8')
    print('written', out / 'inputs_cc400_t146.npz', out / 'folds_seed42.json')


if __name__ == '__main__':
    main()
