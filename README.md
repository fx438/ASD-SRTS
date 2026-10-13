# SRTS: shrinkage-regularized tangent-space connectivity

Code for the paper

> T. Tan, X. Fang. *Shrinkage-regularized tangent-space connectivity removes an input bottleneck of
> deep models for autism classification from short resting-state fMRI.* Submitted to Medical &
> Biological Engineering & Computing.

With fine atlases and short scans (here N = 392 CC400 regions, T = 146 volumes), every
connectivity matrix is rank-deficient and must be made positive definite before the tangent-space
map. SRTS shrinks each correlation matrix toward the identity at one validation-selected
intensity, whitens it by the training-fold mean and takes the matrix logarithm. On 860 ABIDE-I
participants it improved BrainNetMLP over raw correlation (AUC 0.769 vs 0.747), whereas
eigenvalue clipping (0.632) and Ledoit–Wolf (0.734; nilearn default 0.759) did not. On SRTS input,
logistic regression with 32 times fewer parameters matched BrainNetMLP (AUC 0.773).

## Layout

| Path | Content |
|---|---|
| `srts/` | Reference NumPy implementation: `SRTS` transformer, spectral-energy diagnostic, paired bootstrap |
| `tests/` | Unit tests, including a check that `srts/` matches the implementation used in the experiments |
| `data/` | Download script, the 1035 PCP FILE_IDs, the 860 participants used and the five folds |
| `experiments/` | The scripts that produced the results in the paper (see below) |
| `results/` | Result summaries of those runs (JSON/CSV/Markdown; features and weights omitted) |
| `published_models/` | Inputs, folds and scoring for retraining the published models of Table 5 |
| `figures/` | Scripts that draw Figs. 1–4 and the graphical abstract |

## Installation

Python 3.12; the experiments ran with numpy 2.1.3, scipy 1.16.3, scikit-learn 1.8.0, pandas 3.0.1,
nilearn 0.13.1 and PyTorch 2.5.1 (CUDA 12.4).

```bash
pip install -r requirements.txt
python -m pytest tests
```

## Using SRTS on your own data

```python
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from srts import SRTS, correlation_matrices

C = correlation_matrices(time_series)            # (n_subjects, n_roi, n_timepoints) -> (n, N, N)
model = make_pipeline(SRTS(gamma=0.9), StandardScaler(),
                      LogisticRegression(C=1.0, solver='liblinear', dual=True, max_iter=5000))
model.fit(C[train], y[train])                    # reference and scaler use training data only
scores = model.predict_proba(C[test])[:, 1]
```

Choose `gamma` on a validation split from `srts.GAMMA_GRID` (0.05–0.90), as in the paper.

## Data

The paper uses the ABIDE-I CC400 ROI time series of the Preprocessed Connectomes Project
(C-PAC, band-pass filtering and global signal regression, "filt_global").

```bash
python data/download_cc400.py          # about 1035 files from the public fcp-indi bucket
```

This writes `data/ABIDE_CC400_Complete/{Phenotypic.csv, CSV/*.csv}`. Set `ABIDE_CC400_DIR` to use
another location. The loader keeps participants with at least 146 volumes (860; listed in
`data/subjects_860.csv`) and truncates them to the first 146. On first use it writes a cache
`experiments/cc400_cache_t146.npz`.

## Reproducing the paper

Run the scripts from `experiments/revision_srts_v2/` in the order below; later steps read the
per-fold shrinkage coefficients and splits saved by `run_revision.py`. Outputs go to
`experiments/results_revision_v2/<name>/`; the corresponding results of our runs are in
`results/<name>/`. Deep models need a CUDA GPU; the other scripts run on CPU or GPU.

| Paper | Script | Output (`results/…`) |
|---|---|---|
| Table 2 rows 1–5, Table 3 (BrainNetMLP column), hidden-layer mixup screen (Table S11) | `run_revision.py` | `standard5_seed42` |
| Table 2 row 6 (nilearn default on BrainNetMLP) | `nilearn_brainnetmlp_v1.py` | `nilearn_brainnetmlp_v1` |
| Table 3 (LR, linear SVM) | `run_srts_linear_standard5.py --model lr` / `--model svm` | `srts_linear_standard5_seed42_v1` |
| Table 4 (aligned LR comparisons) | `run_raw_fc_full_lr_standard5.py`, `nilearn_baseline_v1.py`, `compare_aligned_lr_oof_v1.py` | `raw_fc_full_lr_…`, `nilearn_baseline_v1`, `aligned_lr_representation_comparison_v1` |
| Table S4 (conventional pipelines) | `run_traditional_ml_standard5.py --model {lr,knn,mlp,rf,svm}` | `traditional_ml_standard5_seed42_v1` |
| Fig. 2, Table S5 (spectral-energy diagnostic) | `analyze_tangent_spectrum_v2.py`, `analyze_tangent_spectrum_lw_v1.py` | `tangent_spectrum_diagnostic_v1` |
| Fig. 3, Sect. 3.3 (shrinkage grid) | `analyze_gamma_airm_v1.py`, `audit_gamma_boundary.py`, `run_srts_lr_fixed_gamma070_standard5.py` | `gamma_airm_validation_…`, `gamma_boundary_audit_…`, `srts_lr_fixed_gamma070_…` |
| Fig. 4, Table S6 (controlled N/T) | `nt_control_v1.py` | `nt_control_v1` |
| Sect. 3.6, Tables S7–S9 (sites, TR) | `../analyze_site_tr_oof_v1.py` | `site_tr_oof_stratified_standard5_seed42_v1` |
| Table S10 (cost) | `../benchmark_*.py`, `../build_efficiency_core_evidence_v1.py` | `efficiency_core_evidence_v1` |
| Table S11 (rejected modules) | `run_tangent_adapter.py`, `run_cross_fusion.py`, `run_roi_interaction.py`, `run_spectral_sequence.py`, `nullfill_v1.py` | `*_standard5_seed42_v1`, `nullfill_v1` |
| Tables S12–S14 (leave-one-site-out) | `loso_v1.py` | `loso_v1` |
| Table 5, Table S15 (PLSNet) | `plsnet_cc400_v1.py` (see `published_models/`) | `plsnet_cc400_v1` |
| Supplementary rank diagnostics (S3) | `analyze_effective_rank_v1.py`, `analyze_rank_spectrum_all_v1.py`, `analyze_constant_roi_by_site_v1.py`, `diagnose_raw_fc_diagonal_v1.py` | `effective_rank_*`, `constant_roi_site_distribution_v1` |

Figures: `python figures/make_figs_en.py` (Figs. 1–4) and `python figures/make_graphical_abstract.py`;
both write to `figures/output/`.

Each run directory records its configuration (`config.json`) before any result is produced, and
candidate modules compute test metrics only after passing the pre-declared validation criterion
(mean validation AUC gain ≥ 0.01 and improvement in at least four of five folds).

## Notes

* The experiment scripts are the versions that were run; only the absolute server paths were
  replaced by paths relative to the repository. Their comments are in Chinese. Provenance files in
  `results/` still show the original server paths.
* Splits: stratified five-fold cross-validation (seed 42), 602 / 86 / 172 participants for
  training / validation / test in every fold; `data/splits_seed42.json`.
* Paired comparisons use a bootstrap within fold and class (20,000 resamples, seed 20260824) on the
  pooled out-of-fold predictions (`srts.paired_bootstrap_auc`).
* GPU results can differ in the last digits between hardware and library versions.

## License

MIT, see `LICENSE`. ABIDE data are subject to the ABIDE data usage terms.
