#!/usr/bin/env python3
"""Render the system / agent-environment schematic for the project report.

Two-panel figure:
  Left:  axial slice of the brain (MNI152 t1_crop) with agent position,
         target landmark, 7x7x7 neighborhood box, and direction arrow.
  Right: MDP interaction loop (state -> policy -> action -> env -> reward).
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.patches import Rectangle, FancyArrowPatch, FancyBboxPatch
import nibabel as nib

TEXT_OUTLINE = [pe.Stroke(linewidth=2.5, foreground='black'), pe.Normal()]

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VOLUME_PATH = os.path.join(REPO, 'public', 't1_crop.nii.gz')
OUT_PATH = os.path.join(REPO, 'docs', 'figures', 'system_diagram.png')


def load_axial_slice():
    img = nib.load(VOLUME_PATH)
    data = np.asarray(img.dataobj).astype(np.float32)
    # Central axial slice along the third axis.
    z = data.shape[2] // 2
    sl = data[:, :, z]
    # Simple min-max normalize for display.
    lo, hi = np.percentile(sl, [2, 98])
    sl = np.clip((sl - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    return sl


def panel_volume(ax):
    try:
        sl = load_axial_slice()
        ax.imshow(sl.T, origin='lower', cmap='gray', aspect='equal')
        H, W = sl.T.shape
    except Exception as err:
        print(f"[warn] could not load volume: {err}; using placeholder")
        W, H = 200, 200
        ax.set_xlim(0, W); ax.set_ylim(0, H); ax.set_facecolor('#202020')

    # Target landmark (Thalamus-ish location for illustration) and agent.
    # Positioned so labels can fan out without crossing the arrow.
    tgt = np.array([W * 0.55, H * 0.58])
    agt = np.array([W * 0.28, H * 0.30])
    d = tgt - agt

    # 7x7x7 neighborhood box around the agent (display scale: 12 px edge).
    nbr_edge = 14
    ax.add_patch(Rectangle(
        (agt[0] - nbr_edge/2, agt[1] - nbr_edge/2), nbr_edge, nbr_edge,
        fill=False, edgecolor='#4CAF50', linewidth=1.8,
    ))

    # Direction arrow (agent -> target).
    ax.add_patch(FancyArrowPatch(
        tuple(agt), tuple(tgt), arrowstyle='-|>', mutation_scale=14,
        color='#FFC107', linewidth=1.8,
    ))

    # Agent and target markers.
    ax.plot(*agt, marker='o', markersize=10, markerfacecolor='#4CAF50',
            markeredgecolor='white', markeredgewidth=1.5, zorder=5)
    ax.plot(*tgt, marker='*', markersize=18, markerfacecolor='#F44336',
            markeredgecolor='white', markeredgewidth=1.2, zorder=5)

    # Text labels (with a black outline so they stay readable on brain tissue).
    def lbl(xy, xytext, text, color):
        ann = ax.annotate(text, xy=xy, xytext=xytext, color=color, fontsize=9,
                          arrowprops=dict(arrowstyle='-', color=color, lw=0.6))
        ann.set_path_effects(TEXT_OUTLINE)

    # Fan labels out: agent -> upper-left; neighborhood -> lower-left;
    # target -> upper-right; direction -> below the arrow.
    lbl(agt, (agt[0] - 38, agt[1] + 22), 'agent', 'white')
    lbl((agt[0] - nbr_edge/2, agt[1] - nbr_edge/2),
        (agt[0] - 48, agt[1] - 28), '7$^3$ neighborhood', '#8BC34A')
    lbl(tgt, (tgt[0] + 12, tgt[1] + 18), 'target\nlandmark', 'white')
    # Direction label: perpendicular offset below the arrow so it doesn't sit on it.
    mid = (agt + tgt) / 2
    perp = np.array([d[1], -d[0]]) / max(np.linalg.norm(d), 1e-6)
    lp = mid + perp * 16
    t = ax.text(lp[0], lp[1], 'direction $\\hat{d}$',
                color='#FFC107', fontsize=9, ha='center', va='center')
    t.set_path_effects(TEXT_OUTLINE)

    ax.set_title('(a) Agent in MNI152 volume (axial slice)', fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])


def panel_mdp(ax):
    ax.set_xlim(0, 10); ax.set_ylim(0, 10)
    ax.set_aspect('equal'); ax.axis('off')

    def box(xy, w, h, text, fc):
        x, y = xy
        ax.add_patch(FancyBboxPatch(
            (x, y), w, h,
            boxstyle='round,pad=0.08,rounding_size=0.18',
            fc=fc, ec='#333', linewidth=1.2,
        ))
        ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=9)

    def arrow(xy_from, xy_to, label=None, label_dy=0.25, label_color='#333'):
        ax.add_patch(FancyArrowPatch(
            xy_from, xy_to, arrowstyle='-|>', mutation_scale=12,
            color='#333', linewidth=1.2,
        ))
        if label:
            mx = (xy_from[0] + xy_to[0]) / 2
            my = (xy_from[1] + xy_to[1]) / 2 + label_dy
            ax.text(mx, my, label, ha='center', va='center',
                    fontsize=8, color=label_color)

    # State box (top-left).
    box((0.4, 6.8), 3.2, 2.0,
        'state $s_t$\n$(\\mathbf{n}_t,\\ k\\!\\cdot\\!\\hat{\\mathbf{d}}_t)$',
        '#E8F5E9')
    # Policy/value box (top-right).
    box((6.4, 6.8), 3.2, 2.0,
        'policy $\\pi_\\theta$\nvalue $V_\\phi$',
        '#E3F2FD')
    # Environment box (bottom-right).
    box((6.4, 1.2), 3.2, 2.0,
        'environment\n(step action, clamp)',
        '#FFF3E0')
    # Reward box (bottom-left).
    box((0.4, 1.2), 3.2, 2.0,
        'reward $r_t$\n$-(d_{t+1}\\!-\\!d_t) - 0.1\\ (+10\\text{ on success})$',
        '#FCE4EC')

    # Arrows between boxes.
    arrow((3.7, 7.8), (6.3, 7.8), label='346-d state')
    # Right-hand vertical arrow: action. Put both labels to the right of the arrow.
    arrow((8.0, 6.7), (8.0, 3.3))
    ax.text(8.25, 5.4, 'action $a_t$', fontsize=8, ha='left', va='center', color='#333')
    ax.text(8.25, 5.0, '$\\{\\pm x,\\pm y,\\pm z\\}$', fontsize=8,
            ha='left', va='center', color='#333')
    arrow((6.3, 2.2), (3.7, 2.2), label='$r_t,\\ s_{t+1}$')
    # Left-hand vertical arrow: next-step feedback. Label to the left.
    arrow((2.0, 3.4), (2.0, 6.6))
    ax.text(1.75, 5.0, 'next\nstep', fontsize=8, ha='right', va='center', color='#333')

    ax.set_title('(b) MDP interaction loop', fontsize=10)


def main():
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0),
                             gridspec_kw={'width_ratios': [1.0, 1.25]})
    panel_volume(axes[0])
    panel_mdp(axes[1])
    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=160, bbox_inches='tight')
    print(f'Saved {OUT_PATH}')


if __name__ == '__main__':
    main()
