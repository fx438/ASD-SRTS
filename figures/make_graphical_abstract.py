# -*- coding: utf-8 -*-
"""MBEC 图形摘要：同一 BrainNetMLP 骨干、不同连接输入的测试 AUC 与零空间能量；SRTS 上 LR 追平 BrainNetMLP。

MBEC 投稿指南：成品 32.93 mm 宽 × 37.63 mm 高（100% 比例），含文字的 TIFF 600 dpi，
文件名 Graphical_Abstract_<作者姓>。数值取自稿件表 2、表 3 与第 3.3 节零空间能量占比。
输出：figures/output/Graphical_Abstract_Tan.{pdf,png,tif}，600 dpi。
"""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUT = Path(__file__).resolve().parent / 'output' / 'Graphical_Abstract_Tan'
MM = 1 / 25.4
W, H = 32.93, 37.63

INK, INK2, MUTED, GRID = '#1f2328', '#4b5563', '#8a9099', '#e5e7eb'
CLIP, SRTS, OTHER = '#b8322a', '#1f4e8c', '#a9b0b9'

# 表 2：BrainNetMLP 读出，5 折平均 AUC；第 3.3 节：切线能量落在零空间的比例（raw 与 nilearn 未计算）
ROWS = [('Raw FC', 0.7474, None, OTHER),
        ('Clipping', 0.6319, 93, CLIP),
        ('Ledoit–Wolf', 0.7341, 71, OTHER),
        ('nilearn', 0.7586, None, OTHER),
        ('SRTS', 0.7687, 18, SRTS)]
X0, X1 = 0.60, 0.80          # AUC 轴范围


def main():
    plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans'],
                         'pdf.fonttype': 42, 'svg.fonttype': 'none'})
    fig = plt.figure(figsize=(W * MM, H * MM))
    fig.patch.set_facecolor('white')
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis('off')

    ax.text(W / 2, 35.9, 'Same BrainNetMLP,', ha='center', va='center', fontsize=5.4, weight='bold', color=INK)
    ax.text(W / 2, 33.6, 'different input', ha='center', va='center', fontsize=5.4, weight='bold', color=INK)
    ax.text(W / 2, 31.5, 'N = 392 ROIs > T = 146', ha='center', va='center', fontsize=4.3, color=INK2)

    # 点图：左列方法名，中间 AUC 轴（0.60–0.80），右列零空间能量
    lx, ax0, ax1 = 10.6, 11.6, 25.4                  # 名称右端、轴左右端（mm）

    def xpos(v):
        return ax0 + (v - X0) / (X1 - X0) * (ax1 - ax0)

    ys = [27.8, 25.1, 22.4, 19.7, 17.0]
    raw_x = xpos(ROWS[0][1])
    ax.plot([raw_x, raw_x], [15.9, 28.9], color=MUTED, lw=0.4, ls=(0, (2, 1.5)), zorder=1)
    for y, (name, auc, null, c) in zip(ys, ROWS):
        ax.text(lx, y, name, ha='right', va='center', fontsize=4.5, color=INK,
                weight='bold' if name == 'SRTS' else 'normal')
        ax.plot([raw_x, xpos(auc)], [y, y], color=c, lw=1.0, solid_capstyle='butt', zorder=2)
        ax.scatter([xpos(auc)], [y], s=7, color=c, edgecolor='white', linewidth=0.5, zorder=3)
        ax.text(xpos(auc) + 0.6, y + 1.05, f'{auc:.3f}', fontsize=3.8,
                color=INK, ha='left' if auc >= ROWS[0][1] else 'right', va='center')
        if null is not None:
            ax.text(31.9, y, f'{null}%', ha='right', va='center', fontsize=4.3, color=INK,
                    weight='bold' if name == 'SRTS' else 'normal')
    ax.text((ax0 + ax1) / 2, 13.6, 'test AUC', ha='center', va='center', fontsize=3.9, color=INK2)
    ax.text(31.9, 29.9, 'null-space', ha='right', va='center', fontsize=3.6, color=INK2)
    for v in (0.6, 0.7, 0.8):
        ax.text(xpos(v), 15.9, f'{v:.1f}', ha='center', va='top', fontsize=3.5, color=MUTED)

    # 底部：SRTS 输入下 LR 与 BrainNetMLP
    ax.add_patch(FancyBboxPatch((1.2, 1.2), W - 2.4, 10.6, boxstyle='round,pad=0,rounding_size=0.8',
                                fc='#f3f6fa', ec=GRID, lw=0.4))
    ax.add_patch(FancyBboxPatch((1.2, 1.2), 0.7, 10.6, boxstyle='square,pad=0', fc=SRTS, ec='none'))
    ax.text(W / 2 + 0.3, 9.6, 'SRTS + logistic regression', ha='center', va='center', fontsize=4.4, color=INK2)
    ax.text(W / 2 + 0.3, 6.5, 'AUC 0.773', ha='center', va='center', fontsize=6.6, weight='bold', color=INK)
    ax.text(W / 2 + 0.3, 3.4, '= BrainNetMLP, 32× fewer params', ha='center', va='center', fontsize=3.9,
            color=INK2)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    for ext in ('pdf', 'png', 'tif'):
        kw = {'dpi': 600} if ext != 'pdf' else {}
        if ext == 'tif':
            kw['pil_kwargs'] = {'compression': 'tiff_lzw'}
        fig.savefig(OUT.with_suffix('.' + ext), facecolor='white', **kw)
    print(OUT)


if __name__ == '__main__':
    main()
