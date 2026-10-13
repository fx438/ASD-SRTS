# -*- coding: utf-8 -*-
"""MBEC English figures: Arial lettering, no in-figure titles, 174 mm / 84 mm widths, TIFF 600 dpi + PDF.

Fig1 workflow, Fig2 spectral energy, Fig3 shrinkage grid, Fig4 N/T controlled experiment,
Fig5 strictly aligned comparisons (needs data/aligned_summary.json). The graphical abstract is drawn by
make_graphical_abstract.py.
"""
import json
import os
import sys

import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from PIL import Image

SP = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(SP, 'output')
os.makedirs(OUT, exist_ok=True)
mpl.rcParams.update({
    'font.family': 'Arial', 'font.size': 8, 'mathtext.fontset': 'custom',
    'mathtext.rm': 'Arial', 'mathtext.it': 'Arial:italic', 'mathtext.bf': 'Arial:bold',
    'axes.linewidth': 0.6, 'axes.edgecolor': '#333333', 'axes.labelcolor': '#222222',
    'xtick.major.width': 0.6, 'ytick.major.width': 0.6, 'xtick.major.size': 2.5, 'ytick.major.size': 0,
    'xtick.color': '#333333', 'ytick.color': '#222222', 'legend.frameon': False,
    'pdf.fonttype': 42, 'svg.fonttype': 'none', 'axes.unicode_minus': True,
})
MM = 1 / 25.4
C_CLIP, C_LW, C_SHR, C_NIL = '#E64B35', '#4DBBD5', '#3C5488', '#00A087'
INK, SEC, LINE = '#18212B', '#52606D', '#415466'


def clean(ax, left=False):
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    if not left:
        ax.spines['left'].set_visible(False)


def panel(ax, s, x=-0.02, y=1.03):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=10, fontweight='bold', va='bottom', ha='right')


def save(fig, name, previews=True):
    fig.savefig(os.path.join(OUT, f'{name}.pdf'), bbox_inches='tight', pad_inches=0.02)
    png = os.path.join(OUT, f'{name}.png')
    fig.savefig(png, dpi=600, bbox_inches='tight', pad_inches=0.02)
    Image.open(png).convert('RGB').save(os.path.join(OUT, f'{name}.tif'), dpi=(600, 600), compression='tiff_lzw')
    plt.close(fig)


# ---------------- Fig 1: workflow ----------------
def box(ax, x, y, w, h, text, face, edge=LINE, lw=0.9, weight='normal', fs=6.8):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.006,rounding_size=0.015',
                                facecolor=face, edgecolor=edge, linewidth=lw))
    ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=fs, color=INK,
            weight=weight, linespacing=1.3)


def arrow(ax, a, b, color=LINE):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle='-|>', mutation_scale=8, linewidth=0.9, color=color))


def fig1():
    fig, ax = plt.subplots(figsize=(174 * MM, 74 * MM))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')
    xs = np.linspace(0.005, 0.845, 6); w, y0, h = 0.15, 0.60, 0.30
    labels = ['ROI time series\n$X_i\\in\\mathbb{R}^{392\\times146}$',
              'Row-wise z-scoring\n(constant ROI\n→ zero row)',
              'Pearson FC\n$C_i=X_iX_i^{\\top}/T$\nrank $\\leq$ 145',
              'Linear shrinkage\n$(1-\\gamma)C_i+\\gamma I$\n$\\lambda_{\\min}=\\gamma$',
              'Training-fold\nreference $R$\n(602 subjects)',
              'Whitening +\nmatrix logarithm\n$\\log(R^{-1/2}\\tilde C_iR^{-1/2})$']
    faces = ['#E7F0FB', '#F2F4F6', '#E3F2EA', '#F0E9F7', '#F0E9F7', '#FFF0D9']
    for i, (x, lab, f) in enumerate(zip(xs, labels, faces)):
        box(ax, x, y0, w, h, lab, f, weight='bold' if i in (3, 4) else 'normal')
        if i < 5:
            arrow(ax, (x + w, y0 + h / 2), (xs[i + 1], y0 + h / 2))
    # readouts
    cx = xs[5] + w / 2
    ax.plot([cx, cx, 0.17, 0.83], [y0, 0.47, 0.47, 0.47], color=LINE, lw=0.9)
    outs = [(0.03, 'Isometric svec (77,028-D)\nStandardScaler → L2 logistic\nregression (main readout)', '#F9E4E4', '#B5444F', 1.4),
            (0.36, 'Isometric svec (77,028-D)\nStandardScaler → linear SVM\n(capacity control)', '#F2F4F6', LINE, 0.9),
            (0.69, 'Upper-triangle input\nBrainNetMLP\n(capacity control)', '#F2F4F6', LINE, 0.9)]
    for x, t, f, e, lw in outs:
        box(ax, x, 0.20, 0.28, 0.20, t, f, edge=e, lw=lw)
        arrow(ax, (x + 0.14, 0.47), (x + 0.14, 0.40))
    box(ax, 0.0, 0.015, 1.0, 0.11,
        'Information boundary: reference matrix, scaler and classifier weights are fitted on the training fold only; '
        '$\\gamma$ and the decision\nthreshold are selected on the validation split; test subjects are transformed '
        'only after all quantities are locked', '#F2F4F6', edge='#8996A3', fs=6.5)
    save(fig, 'Fig1')


# ---------------- Fig 2: spectral energy ----------------
def fig2():
    names = ['Eigenvalue clipping', 'Ledoit–Wolf', 'Linear shrinkage']
    null = np.array([92.75, 71.27, 17.56]); rng_ = np.array([2.18, 11.85, 60.91]); cross = 100 - null - rng_
    auc = [0.6319, 0.7341, 0.7687]; aucsd = [0.0373, 0.0469, 0.0465]
    fro = [241.402, 33.106, 4.052]; mxl = [15.86, 4.09, 1.39]; mne = [1.32e-7, 0.0170, 0.4211]
    cols = [C_CLIP, C_LW, C_SHR]
    fig = plt.figure(figsize=(174 * MM, 60 * MM))
    gs = fig.add_gridspec(1, 5, width_ratios=[3.0, 1.2, 1.05, 1.05, 1.05], wspace=0.35)
    ax = fig.add_subplot(gs[0]); yy = np.arange(3); left = np.zeros(3)
    for vals, lab, c in ((null, 'Null-space block', '#C0392B'), (cross, 'Cross block', '#D9D9D9'),
                         (rng_, 'Range-space block', '#2E4A7D')):
        ax.barh(yy, vals, left=left, height=0.58, color=c, edgecolor='white', lw=0.6, label=lab)
        for i, (l, v) in enumerate(zip(left, vals)):
            if v >= 15:
                ax.text(l + v / 2, i, f'{v:.2f}%', ha='center', va='center', fontsize=6.3,
                        color='white' if c != '#D9D9D9' else '#333333')
        left += vals
    ax.text(100, -0.36, f'cross {cross[0]:.2f}%, range {rng_[0]:.2f}%', ha='right', va='bottom', fontsize=6.5, color='#555555')
    ax.text(100, 0.64, f'range {rng_[1]:.2f}%', ha='right', va='bottom', fontsize=6.5, color='#555555')
    ax.set_yticks(yy); ax.set_yticklabels(names); ax.set_ylim(2.6, -0.6)
    ax.set_xlim(0, 100); ax.set_xticks([0, 25, 50, 75, 100]); ax.set_xlabel('Share of tangent-matrix Frobenius energy (%)')
    ax.legend(ncol=3, loc='upper center', bbox_to_anchor=(0.45, -0.22), fontsize=7, handlelength=1.0, columnspacing=0.9)
    clean(ax); panel(ax, 'a', x=-0.28)

    def dotcol(axx, vals, xlabel, log=False, err=None, fmt='{:.3g}', letter=None, xl=None, txt=None):
        for i, (v, c) in enumerate(zip(vals, cols)):
            if err is not None:
                axx.errorbar(v, i, xerr=err[i], fmt='none', ecolor=c, elinewidth=0.9, capsize=1.8, alpha=0.8)
            axx.plot([xl[0], v], [i, i], color='#E3E3E3', lw=1.0, zorder=0)
            axx.scatter(v, i, s=28, color=c, edgecolor='white', lw=0.6, zorder=3)
            axx.text(v, i - 0.30, txt[i] if txt else fmt.format(v), ha='center', va='bottom', fontsize=6.5, color='#333333')
        if log:
            axx.set_xscale('log')
        axx.set_xlim(*xl); axx.set_ylim(2.6, -0.6); axx.set_yticks([])
        axx.set_xlabel(xlabel, fontsize=7); axx.tick_params(axis='x', labelsize=6.5)
        axx.xaxis.set_minor_locator(mpl.ticker.NullLocator()); clean(axx)
        if letter:
            panel(axx, letter, x=-0.08)

    dotcol(fig.add_subplot(gs[1]), auc, 'Test AUC\n(mean ± SD)', err=aucsd, fmt='{:.4f}', letter='b', xl=(0.55, 0.85))
    dotcol(fig.add_subplot(gs[2]), fro, 'Frobenius norm', log=True, fmt='{:.1f}', letter='c', xl=(1, 1000))
    dotcol(fig.add_subplot(gs[3]), mxl, 'max $|\\ln\\,\\eta|$', log=True, fmt='{:.2f}', letter='d', xl=(0.5, 50))
    a5 = fig.add_subplot(gs[4])
    dotcol(a5, mne, 'Min. whitened\neigenvalue', log=True, xl=(1e-8, 3), letter='e',
           txt=['1.32×10$^{-7}$', '0.0170', '0.4211'])
    a5.set_xticks([1e-7, 1e-4, 1e-1])
    save(fig, 'Fig2')


# ---------------- Fig 3: shrinkage grid ----------------
def fig3():
    gam = ['0.05', '0.10', '0.20', '0.30', '0.50', '0.70', '0.80', '0.90', '0.95']
    vauc = [0.7392, 0.7461, 0.7566, 0.7637, 0.7810, 0.7910, 0.7911, 0.7978, 0.7882]
    vsd = [0.0686, 0.0658, 0.0629, 0.0631, 0.0519, 0.0465, 0.0397, 0.0389, 0.0301]
    mine = [0.0063, 0.0131, 0.0281, 0.0459, 0.0951, 0.1829, 0.2636, 0.4191, 0.5816]
    lgc = [3.670, 3.314, 2.902, 2.612, 2.137, 1.661, 1.371, 0.973, 0.661]
    airm = [47.63, 36.58, 26.31, 20.67, 13.76, 8.88, 6.59, 4.05, 2.46]
    rows = [('Validation AUC', vauc, +1, lambda i: f'{vauc[i]:.4f}\n±{vsd[i]:.4f}'),
            ('Min. whitened\neigenvalue', mine, +1, lambda i: f'{mine[i]:.4f}'),
            ('log$_{10}$ condition\nnumber', lgc, -1, lambda i: f'{lgc[i]:.3f}'),
            ('AIRM distance', airm, -1, lambda i: f'{airm[i]:.2f}')]
    Z = np.array([((np.array(v) * s) - min(np.array(v) * s)) / (max(np.array(v) * s) - min(np.array(v) * s))
                  for _, v, s, _ in rows])
    cmap = mpl.colors.LinearSegmentedColormap.from_list('b', ['#F7F9FC', '#9FB3D6', '#2E4A7D'])
    fig, ax = plt.subplots(figsize=(174 * MM, 62 * MM))
    im = ax.imshow(Z, cmap=cmap, aspect='auto', vmin=0, vmax=1)
    for r_, (_, v, _, f) in enumerate(rows):
        for c_ in range(len(gam)):
            ax.text(c_, r_, f(c_), ha='center', va='center', fontsize=6.8, linespacing=1.05,
                    color='white' if Z[r_, c_] > 0.62 else '#222222')
    ax.add_patch(Rectangle((7.5, -0.5), 1, 4, fill=False, hatch='////', ec='#9A9A9A', lw=0))
    ax.set_xticks(range(len(gam))); ax.set_xticklabels(gam)
    ax.set_yticks(range(4)); ax.set_yticklabels([r[0] for r in rows])
    ax.set_xlabel('Shrinkage coefficient $\\gamma$')
    ax.tick_params(axis='both', length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks(np.arange(-0.5, len(gam), 1), minor=True); ax.set_yticks(np.arange(-0.5, 4, 1), minor=True)
    ax.grid(which='minor', color='white', lw=1.2); ax.tick_params(which='minor', length=0)
    for c_, t in {4: 'fold 4', 5: 'fold 3', 7: 'folds 1, 2, 5', 8: 'audit only'}.items():
        col = '#9A9A9A' if c_ == 8 else C_SHR
        ax.annotate(t, xy=(c_, -0.52), xytext=(c_, -0.95), ha='center', va='bottom', fontsize=6.8, color=col,
                    arrowprops=dict(arrowstyle='-|>', lw=0.6, color=col, mutation_scale=5))
    ax.set_ylim(3.5, -1.15)
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.015, ticks=[0, 1])
    cb.ax.set_yticklabels(['worse', 'better'], fontsize=6.8); cb.outline.set_visible(False)
    cb.set_label('Row-normalized', fontsize=6.8, labelpad=-2)
    save(fig, 'Fig3')


# ---------------- Fig 4: N/T ----------------
def fig4():
    nt = json.load(open(os.path.join(SP, 'data', 'nt_folds.json'), encoding='utf-8'))
    order = ['N392_T146_r0', 'N392_T100_r0', 'N392_T75_r0', 'N392_T50_r0', 'N50_T146_r0', 'N100_T146_r0', 'N200_T146_r0']
    ys = [0, 1, 2, 3, 5.2, 6.2, 7.2]
    fig = plt.figure(figsize=(174 * MM, 76 * MM))
    gs = fig.add_gridspec(1, 2, width_ratios=[3.3, 1], wspace=0.10)
    ax, axb = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
    rng = np.random.default_rng(7)
    for key, y in zip(order, ys):
        r = nt[key]; m = {s: np.mean(v) for s, v in r['auc'].items()}
        ax.plot([m['eigclip'], m['linear_shrinkage']], [y, y], color='#E3E3E3', lw=5.5, solid_capstyle='round', zorder=1)
        for s, c, off in (('eigclip', C_CLIP, -0.2), ('ledoit_wolf', C_LW, 0.0), ('linear_shrinkage', C_SHR, 0.2)):
            v = np.array(r['auc'][s])
            ax.scatter(v, y + off + rng.uniform(-0.06, 0.06, len(v)), s=5, color=c, alpha=0.35, lw=0, zorder=2)
        ax.scatter(m['ledoit_wolf'], y, s=26, marker='D', facecolor='white', edgecolor=C_LW, lw=1.0, zorder=3)
        ax.scatter(m['eigclip'], y, s=34, color=C_CLIP, edgecolor='white', lw=0.6, zorder=4)
        ax.scatter(m['linear_shrinkage'], y, s=34, color=C_SHR, edgecolor='white', lw=0.6, zorder=4)
        gap = round(m['linear_shrinkage'], 4) - round(m['eigclip'], 4)
        ax.text(0.849, y, f'+{gap:.3f}', va='center', ha='left', fontsize=7, color='#444444')
        ax.text(0.524, y, f"$N$={r['n']}, $T$={r['t']}\n$N/T$={r['n'] / r['t']:.2f}", va='center', ha='right',
                fontsize=7, linespacing=1.15)
    ax.set_yticks([]); ax.set_ylim(7.75, -1.2); ax.set_xlim(0.53, 0.875)
    ax.set_xlabel('Five-fold test AUC')
    ax.text(0.849, -0.85, 'Δ AUC', ha='left', va='center', fontsize=7, color='#444444')
    ax.text(0.533, -0.85, 'Fixed $N$=392, shorter $T$', fontsize=7, color='#666666', va='center')
    ax.text(0.533, 4.35, 'Fixed $T$=146, fewer $N$', fontsize=7, color='#666666', va='center')
    ax.axhline(3.85, color='#DDDDDD', lw=0.6, ls=(0, (3, 2)))
    ax.grid(axis='x', color='#F0F0F0', lw=0.5); ax.set_axisbelow(True); clean(ax)
    handles = [Line2D([], [], marker='o', ls='', color=C_CLIP, ms=5, label='Eigenvalue clipping'),
               Line2D([], [], marker='D', ls='', mfc='white', mec=C_LW, ms=4.5, label='Ledoit–Wolf'),
               Line2D([], [], marker='o', ls='', color=C_SHR, ms=5, label='Linear shrinkage (SRTS)'),
               Line2D([], [], marker='o', ls='', color='#999999', alpha=0.4, ms=2.5, label='Single fold')]
    fig.legend(handles=handles, ncol=4, loc='upper center', bbox_to_anchor=(0.43, 1.0), fontsize=7, handletextpad=0.3, columnspacing=0.9, frameon=False)
    panel(ax, 'a', x=-0.20)
    clip = [nt[k]['clip'] * 100 for k in order]
    cmap = mpl.colors.LinearSegmentedColormap.from_list('r', ['#FCE6DF', '#E64B35', '#8E1C10'])
    axb.barh(ys, clip, height=0.62, color=[cmap(v / 100) for v in clip], edgecolor='none')
    for y, v in zip(ys, clip):
        axb.text(v + 2, y, f'{v:.1f}', va='center', fontsize=7, color='#333333')
    axb.set_ylim(7.75, -1.2); axb.set_xlim(0, 112); axb.set_yticks([])
    axb.set_xticks([0, 50, 100]); axb.set_xlabel('Clipped eigenvalues (%)')
    axb.axhline(3.85, color='#DDDDDD', lw=0.6, ls=(0, (3, 2))); clean(axb); panel(axb, 'b', x=0.0)
    save(fig, 'Fig4')


# ---------------- Fig 5: aligned comparisons (forest) ----------------
def fig5():
    s = json.load(open(os.path.join(SP, 'data', 'aligned_summary.json'), encoding='utf-8'))
    rows = s['rows']  # list of {label, metric, d, lo, hi}
    fig, ax = plt.subplots(figsize=(84 * MM, 0.42 * len(rows) * 25.4 * MM + 18 * MM))
    for i, r in enumerate(rows):
        sig = r['lo'] > 0 or r['hi'] < 0
        c = r.get('color', C_SHR) if sig else '#9A9A9A'
        ax.plot([r['lo'], r['hi']], [i, i], color=c, lw=1.4, solid_capstyle='round')
        ax.scatter(r['d'], i, s=26, color=c if sig else 'white', edgecolor=c, lw=1.0, zorder=3, marker='s')
    ax.axvline(0, color='#555555', lw=0.7, ls=(0, (3, 2)))
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([r['label'] for r in rows], fontsize=7)
    ax.set_ylim(len(rows) - 0.4, -0.6)
    ax.set_xlabel('Difference, SRTS − baseline (pooled OOF, 95% CI)', fontsize=7)
    ax.tick_params(axis='x', labelsize=7); clean(ax)
    save(fig, 'Fig5')


if __name__ == '__main__':
    which = sys.argv[1:] or ['1', '2', '3', '4']
    for w in which:
        globals()[f'fig{w}']()
    print('done', which)
