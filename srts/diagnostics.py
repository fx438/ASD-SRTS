# -*- coding: utf-8 -*-
"""Spectral-energy diagnostic (Section 2.6, Eq. 12).

For participant i, P_r and P_0 project onto the eigenvectors of the original correlation matrix
C_i with eigenvalues above and below ``tol``. The tangent matrix S_i is split into

    e_0 = ||P_0 S P_0||_F^2 / ||S||_F^2   (null space)
    e_r = ||P_r S P_r||_F^2 / ||S||_F^2   (range space)
    e_x = 1 - e_0 - e_r                   (cross blocks)
"""
from __future__ import annotations

import numpy as np


def energy_shares(correlation, tangent, tol=1e-8):
    """Return (e_null, e_range, e_cross) for one participant."""
    c = 0.5 * (correlation + correlation.T)
    w, v = np.linalg.eigh(c)
    v0, vr = v[:, w <= tol], v[:, w > tol]
    total = float(np.sum(tangent ** 2))
    if total == 0.0:
        return 0.0, 0.0, 0.0
    e0 = float(np.sum((v0.T @ tangent @ v0) ** 2)) / total
    er = float(np.sum((vr.T @ tangent @ vr) ** 2)) / total
    return e0, er, 1.0 - e0 - er


def mean_energy_shares(correlations, tangents, tol=1e-8):
    shares = np.array([energy_shares(c, t, tol) for c, t in zip(correlations, tangents)])
    return dict(zip(('null', 'range', 'cross'), shares.mean(0).tolist()))
