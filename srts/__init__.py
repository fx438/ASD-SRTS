"""SRTS: shrinkage-regularized tangent-space connectivity for short, rank-deficient fMRI."""
from .core import GAMMA_GRID, SRTS, correlation_matrices, shrink, svec
from .diagnostics import energy_shares, mean_energy_shares
from .stats import paired_bootstrap_auc

__all__ = ['GAMMA_GRID', 'SRTS', 'correlation_matrices', 'shrink', 'svec',
           'energy_shares', 'mean_energy_shares', 'paired_bootstrap_auc']
__version__ = '1.0.0'
