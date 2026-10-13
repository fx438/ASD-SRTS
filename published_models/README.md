# Published deep models (Table 5)

Table 5 of the paper compares SRTS + LR with five published models retrained on the same 860
participants, CC400 time series and five folds. Their code is **not** redistributed here; use
the official implementations:

| Model | Reference | Official code |
|---|---|---|
| BrainGNN | Li et al. 2021, Med Image Anal 74:102233 | see the paper's code link |
| Brain Network Transformer (BrainNetTF) | Kan et al. 2022, NeurIPS 35 | https://github.com/Wayfear/BrainNetworkTransformer |
| PLSNet | Wang et al. 2023, Comput Biol Med 163:107184 | https://github.com/CodeGoat24/PLSNet (commit 734442d) |
| RGTNet | Wang et al. 2024, Comput Methods Programs Biomed 247:108065 | see the paper's code link |
| GBT | Peng et al. 2024, MICCAI 2024 | https://github.com/CUHK-AIM-Group/GBT |

## Shared protocol

1. `python published_models/export_inputs.py` writes `inputs_cc400_t146.npz` (time series,
   Pearson FC, labels, sites, FILE_IDs) and `folds_seed42.json` (602 / 86 / 172 per fold).
2. Train each official model with its default hyperparameters on `train`, select the
   checkpoint on `val` only, and predict `val` and `test` once.
3. Save the predictions as CSV (`fold,split,index,prob`) and run
   `python published_models/evaluate_predictions.py preds.csv [--reference srts_oof.csv]`.
   The threshold is fixed on the validation split, exactly as for SRTS and BrainNetMLP.

## PLSNet

`experiments/revision_srts_v2/plsnet_cc400_v1.py` is the adapter that produced the PLSNet row
(three initialization seeds; Online Resource 1, S9). The official code selects the epoch on the
test set; the adapter selects it by validation AUC with early stopping (at least 60 epochs,
patience 15). Clone the official repository and point `PLSNET_DIR` to it:

```bash
git clone https://github.com/CodeGoat24/PLSNet external/PLSNet
git -C external/PLSNet checkout 734442d
python experiments/revision_srts_v2/plsnet_cc400_v1.py
```

## BrainGNN, BrainNetTF, RGTNet and GBT

To retrain these models on the same participants and folds, feed the official implementations
with `export_inputs.py` and score them with `evaluate_predictions.py` as described above. Results
can differ slightly from Table 5 because of nondeterminism in GPU training.
