#!/usr/bin/env python3
"""Per-landmark difficulty features for the 15 subcortical targets.

Computes for each landmark:
  - PPO last-50 success rate (from docs/all_results.json)
  - Patch std: intensity stdev in a 7^3 cube centered on the target voxel
  - Decoy count: number of voxel positions within 30 voxels of the target whose
    7^3 patch has mean intensity within 5% of the target patch mean (lower is better)
  - Axis dominance: |max(|d_i|)| / sum(|d_i|) for the unit direction vector from
    the volume centroid to the target (analytic geometry proxy; higher = the
    target lies along one dominant axis from typical starts)
  - Anatomical class: ventricle / gray-nucleus / brain-stem / cerebellum

Saves a scatter to docs/figures/landmark_difficulty.png and prints a sorted table.
"""
import json
import os
from pathlib import Path
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent
VOL_PATH = REPO / 'public' / 't1_crop.nii.gz'
PPO_PATH = REPO / 'docs' / 'all_results.json'
OUT_PATH = REPO / 'docs' / 'figures' / 'landmark_difficulty.png'

# From src/lib/landmarks.ts
MNI_COORDS = {
    'Thalamus':                   (99, 134, 82),
    'Caudate':                    (108, 155, 90),
    'Putamen':                    (115, 145, 82),
    'Pallidum':                   (110, 140, 80),
    'Hippocampus':                (110, 115, 68),
    'Amygdala':                   (115, 115, 62),
    'Brain-Stem':                 (90, 118, 58),
    'Cerebellum-White-Matter':    (100, 100, 45),
    'Cerebellum-Cortex':          (108, 90, 40),
    'Lateral-Ventricle':          (96, 145, 92),
    'Inferior-Lateral-Ventricle': (108, 120, 62),
    '3rd-Ventricle':              (90, 140, 82),
    '4th-Ventricle':              (90, 108, 50),
    'Accumbens-area':             (112, 155, 60),
    'VentralDC':                  (102, 130, 72),
}

CLASS = {
    'Lateral-Ventricle': 'CSF', 'Inferior-Lateral-Ventricle': 'CSF',
    '3rd-Ventricle': 'CSF', '4th-Ventricle': 'CSF',
    'Thalamus': 'gray-nucleus', 'Caudate': 'gray-nucleus',
    'Putamen': 'gray-nucleus', 'Pallidum': 'gray-nucleus',
    'Hippocampus': 'gray-nucleus', 'Amygdala': 'gray-nucleus',
    'Accumbens-area': 'gray-nucleus', 'VentralDC': 'gray-nucleus',
    'Brain-Stem': 'brain-stem',
    'Cerebellum-White-Matter': 'cerebellum', 'Cerebellum-Cortex': 'cerebellum',
}

CLASS_COLOR = {
    'CSF': '#1976D2',
    'gray-nucleus': '#E64A19',
    'brain-stem': '#7B1FA2',
    'cerebellum': '#388E3C',
}

PATCH_RADIUS = 3   # 7^3 cube
DECOY_RADIUS = 30  # voxel ball
DECOY_TOL = 0.05   # within 5% of target patch mean intensity

def patch_at(vol, x, y, z, r=PATCH_RADIUS):
    return vol[x-r:x+r+1, y-r:y+r+1, z-r:z+r+1]

def normalize(vol):
    lo, hi = float(vol.min()), float(vol.max())
    return (vol.astype(np.float32) - lo) / max(hi - lo, 1e-8)

def features():
    vol = normalize(nib.load(str(VOL_PATH)).get_fdata())
    cx, cy, cz = (np.array(vol.shape) // 2).tolist()

    out = []
    for name, (x, y, z) in MNI_COORDS.items():
        # patch std at landmark
        p = patch_at(vol, x, y, z)
        patch_mean = float(p.mean())
        patch_std = float(p.std())

        # decoy count: voxels within DECOY_RADIUS whose patch mean is within DECOY_TOL of target
        x0, x1 = max(PATCH_RADIUS, x-DECOY_RADIUS), min(vol.shape[0]-PATCH_RADIUS-1, x+DECOY_RADIUS)
        y0, y1 = max(PATCH_RADIUS, y-DECOY_RADIUS), min(vol.shape[1]-PATCH_RADIUS-1, y+DECOY_RADIUS)
        z0, z1 = max(PATCH_RADIUS, z-DECOY_RADIUS), min(vol.shape[2]-PATCH_RADIUS-1, z+DECOY_RADIUS)
        decoys = 0
        sample_step = 2  # subsample for speed
        for xi in range(x0, x1+1, sample_step):
            for yi in range(y0, y1+1, sample_step):
                for zi in range(z0, z1+1, sample_step):
                    if abs(xi-x) < PATCH_RADIUS and abs(yi-y) < PATCH_RADIUS and abs(zi-z) < PATCH_RADIUS:
                        continue  # skip the target itself
                    if (xi-x)**2 + (yi-y)**2 + (zi-z)**2 > DECOY_RADIUS**2:
                        continue
                    pm = float(vol[xi-PATCH_RADIUS:xi+PATCH_RADIUS+1,
                                   yi-PATCH_RADIUS:yi+PATCH_RADIUS+1,
                                   zi-PATCH_RADIUS:zi+PATCH_RADIUS+1].mean())
                    if abs(pm - patch_mean) < DECOY_TOL:
                        decoys += 1

        # axis dominance: |d| max / sum, from volume centroid to target
        d = np.array([x - cx, y - cy, z - cz], dtype=float)
        absd = np.abs(d)
        axis_dom = float(absd.max() / max(absd.sum(), 1e-8))

        out.append({
            'name': name,
            'class': CLASS[name],
            'patch_std': patch_std,
            'patch_mean': patch_mean,
            'decoys': decoys,
            'axis_dom': axis_dom,
            'centroid_dist': float(np.linalg.norm(d)),
        })
    return out

def ppo_outcomes():
    """PPO last-50 success and final distance per landmark from docs/all_results.json."""
    runs = json.load(open(PPO_PATH))
    succ, dist = {}, {}
    for r in runs:
        c = r['config']
        if c.get('agentType') != 'ppo':
            continue
        eps = r['episodes'][-50:]
        succ.setdefault(c['landmark'], []).append(float(np.mean([e['success'] for e in eps])))
        dist.setdefault(c['landmark'], []).append(float(np.mean([e['finalDistance'] for e in eps])))
    return ({k: float(np.mean(v)) for k, v in succ.items()},
            {k: float(np.mean(v)) for k, v in dist.items()})

def residuals(y, x):
    """Residuals of y after linear regression on x."""
    X = np.column_stack([np.asarray(x, dtype=float), np.ones_like(x, dtype=float)])
    from numpy.linalg import lstsq
    beta, *_ = lstsq(X, y, rcond=None)
    return y - X @ beta

def main():
    feats = features()
    succ, dist = ppo_outcomes()

    rows = []
    for f in feats:
        rows.append({**f,
                     'ppo_succ': succ.get(f['name'], np.nan),
                     'ppo_dist': dist.get(f['name'], np.nan)})

    rows.sort(key=lambda r: r['ppo_dist'] if not np.isnan(r['ppo_dist']) else 1e9)

    print(f"{'Landmark':>28} | {'class':>13} | {'PPO succ':>9} | {'PPO dist':>9} | {'patch_std':>9} | {'decoys':>6} | {'axis_dom':>8}")
    print('-' * 110)
    for r in rows:
        print(f"{r['name']:>28} | {r['class']:>13} | {r['ppo_succ']*100:>8.1f}% | "
              f"{r['ppo_dist']:>9.1f} | {r['patch_std']:>9.3f} | {r['decoys']:>6d} | {r['axis_dom']:>8.3f}")

    # Pearson correlations (ignore NaN)
    arr = np.array([(r['ppo_succ'], r['ppo_dist'], r['patch_std'], r['decoys'], r['axis_dom'], r['centroid_dist'])
                    for r in rows if not np.isnan(r['ppo_succ'])])
    if len(arr) >= 3:
        from itertools import combinations
        names = ['PPO succ', 'PPO dist', 'patch_std', 'decoys', 'axis_dom', 'centr_dist']
        print('\nPearson r (n={}):'.format(len(arr)))
        for i, j in combinations(range(arr.shape[1]), 2):
            r = float(np.corrcoef(arr[:, i], arr[:, j])[0, 1])
            print(f'  {names[i]:>10} ~ {names[j]:>10}: r = {r:+.3f}')

    # Partial correlation of patch_std with PPO dist, controlling for centroid_dist
    # (does appearance still predict difficulty after we account for "is the target near the volume center"?)
    from numpy.linalg import lstsq
    if len(arr) >= 5:
        ppo_dist = arr[:, 1]
        patch_std = arr[:, 2]
        decoys = arr[:, 3]
        centr = arr[:, 5]
        for name, feat in [('patch_std', patch_std), ('decoys', decoys)]:
            ry = residuals(ppo_dist, centr)
            rx = residuals(feat, centr)
            pr = float(np.corrcoef(ry, rx)[0, 1])
            print(f'  partial r(PPO dist, {name} | centroid_dist) = {pr:+.3f}')

    # Two-panel: (left) raw geometry effect, (right) appearance after geometry partialled out
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    valid = [r for r in rows if not np.isnan(r['ppo_dist'])]
    centr = np.array([r['centroid_dist'] for r in valid])
    pdist = np.array([r['ppo_dist'] for r in valid])
    pstd  = np.array([r['patch_std'] for r in valid])
    pdist_resid = residuals(pdist, centr)
    pstd_resid  = residuals(pstd, centr)

    # Left: PPO dist vs centroid_dist (the geometry confound)
    ax = axes[0]
    for cls, color in CLASS_COLOR.items():
        xs = [r['centroid_dist'] for r in valid if r['class'] == cls]
        ys = [r['ppo_dist']      for r in valid if r['class'] == cls]
        ax.scatter(xs, ys, s=80, color=color, alpha=0.85, edgecolor='black',
                   linewidth=0.5, label=cls)
    for r in valid:
        ax.annotate(r['name'].replace('Cerebellum-', 'Cb-').replace('Inferior-Lateral', 'I-Lat')[:14],
                    (r['centroid_dist'], r['ppo_dist']), fontsize=7, alpha=0.8,
                    xytext=(4, 2), textcoords='offset points')
    ax.set_xlabel('Centroid distance: target $\\to$ volume center (vox)', fontsize=9)
    ax.set_ylabel('PPO last-50 final distance (vox)', fontsize=9)
    ax.set_title(f'Geometry confound: r = {float(np.corrcoef(centr, pdist)[0,1]):+.3f}', fontsize=10)
    ax.grid(True, alpha=0.3); ax.tick_params(labelsize=8); ax.legend(fontsize=8, loc='best')

    # Right: PPO dist residual vs patch_std residual (geometry partialled out)
    ax = axes[1]
    for cls, color in CLASS_COLOR.items():
        idx = [i for i, r in enumerate(valid) if r['class'] == cls]
        ax.scatter(pstd_resid[idx], pdist_resid[idx], s=80, color=color, alpha=0.85,
                   edgecolor='black', linewidth=0.5, label=cls)
    for i, r in enumerate(valid):
        ax.annotate(r['name'].replace('Cerebellum-', 'Cb-').replace('Inferior-Lateral', 'I-Lat')[:14],
                    (pstd_resid[i], pdist_resid[i]), fontsize=7, alpha=0.8,
                    xytext=(4, 2), textcoords='offset points')
    pr = float(np.corrcoef(pstd_resid, pdist_resid)[0, 1])
    # add a regression line
    xline = np.linspace(pstd_resid.min(), pstd_resid.max(), 100)
    slope, intercept = np.polyfit(pstd_resid, pdist_resid, 1)
    ax.plot(xline, slope * xline + intercept, color='gray', linestyle='--', linewidth=1, alpha=0.7)
    ax.set_xlabel('Patch std residual (after partialing out centroid distance)', fontsize=9)
    ax.set_ylabel('PPO dist residual (vox)', fontsize=9)
    ax.set_title(f'Appearance, geometry-controlled: partial r = {pr:+.3f}', fontsize=10)
    ax.grid(True, alpha=0.3); ax.tick_params(labelsize=8); ax.legend(fontsize=8, loc='best')

    fig.suptitle('What predicts per-landmark difficulty? PPO at $k=1$, 15 subcortical landmarks',
                 fontsize=11, fontweight='bold')
    fig.tight_layout()
    fig.savefig(str(OUT_PATH), dpi=150, bbox_inches='tight')
    print(f'\nSaved {OUT_PATH}')

if __name__ == '__main__':
    main()
