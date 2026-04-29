"""MNI152 sampler: load the deployment volume + landmark centroids and emit
training batches in the same shape as train_goal_net.sample_batch().

Why this exists. The submitted goal-vector network was trained on AOMIC
FreeSurfer brain.mgz volumes, which have a very different intensity
distribution from the MNI152 T1 the browser env actually loads. Cosine
similarity drops from >=0.99 (in-domain) to ~0.6 (cross-domain). To close that
gap without giving up the AOMIC training signal, the trainer mixes:
    (a) AOMIC patches with intensities histogram-matched to MNI152's
        brain-interior distribution (handled in train_goal_net.py), and
    (b) synthetic samples drawn from MNI152 directly using the ground-truth
        landmark voxel coordinates parsed out of src/lib/landmarks.ts (this
        module).

Public API:
    load_mni(volume_path, landmarks_json_path)
        -> (brain, aseg_synth, centroids)
        Mirrors the (brain, aseg, centroids) tuple shape that
        train_goal_net.train_subjects entries already use. The aseg here is a
        synthetic int32 array with a single voxel labeled per landmark (just
        enough so any caller that walks aseg gets a non-empty mask). centroids
        is a (N_LM, 3) float32 array indexed in the same order as
        train_goal_net.LM_NAMES.

    sample_batch_mni(mni_subject, batch_size, patch_radius, strides,
                     sample_radius, rng)
        -> (X_patch, X_pos, X_target, y_dir)
        Same return shape as train_goal_net.sample_batch(); accepts a single
        MNI152 "subject" tuple as produced by load_mni().

    mni_intensity_percentiles(brain, lo=2.0, hi=98.0)
        -> (p_lo, p_hi)
        Helper used by train_goal_net.py to compute the histogram-match target.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import nibabel as nib
import numpy as np


# Same 15-landmark order as train_goal_net.LM_NAMES. Kept inline so importing
# this module never requires importing tensorflow via train_goal_net.
LM_NAMES: list[str] = [
    "Lateral-Ventricle",
    "Inf-Lat-Vent",
    "Cerebellum-White-Matter",
    "Cerebellum-Cortex",
    "Thalamus",
    "Caudate",
    "Putamen",
    "Pallidum",
    "3rd-Ventricle",
    "4th-Ventricle",
    "Brain-Stem",
    "Hippocampus",
    "Amygdala",
    "Accumbens-area",
    "VentralDC",
]
N_LM = len(LM_NAMES)


def load_mni(
    volume_path: Path | str,
    landmarks_json_path: Path | str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load the MNI152 volume + centroids JSON as a (brain, aseg, centroids)
    tuple compatible with train_goal_net.train_subjects entries.

    The brain volume is normalized to [0, 1] the same way train_goal_net does
    for AOMIC (divide by max(brain.max(), 1.0)). The aseg returned here is a
    synthetic int32 stub with a unique label at each landmark voxel; it's
    only there so callers that walk aseg find a non-empty mask. The actual
    centroids array is what sample_batch_mni() consumes.

    Centroids JSON format: {landmark_name: [x, y, z]} for every name in
    LM_NAMES. Missing names get NaN centroids (so they're skipped at sample
    time, matching train_goal_net's behaviour for absent FreeSurfer labels).
    """
    volume_path = Path(volume_path)
    landmarks_json_path = Path(landmarks_json_path)

    img = nib.load(str(volume_path))
    brain = np.asarray(img.dataobj).astype(np.float32)
    brain /= max(float(brain.max()), 1.0)

    centroids_raw = json.loads(Path(landmarks_json_path).read_text())
    centroids = np.full((N_LM, 3), np.nan, dtype=np.float32)
    for i, name in enumerate(LM_NAMES):
        if name in centroids_raw:
            c = centroids_raw[name]
            centroids[i] = [float(c[0]), float(c[1]), float(c[2])]

    # Synthetic aseg: zeros everywhere except one voxel per landmark (label
    # = lm_index + 1 so 0 stays "unknown"). This is purely defensive; the
    # sampler does not consume aseg.
    aseg = np.zeros(brain.shape, dtype=np.int32)
    H, W, D = brain.shape
    for i, c in enumerate(centroids):
        if not np.isfinite(c[0]):
            continue
        x = int(np.clip(round(float(c[0])), 0, H - 1))
        y = int(np.clip(round(float(c[1])), 0, W - 1))
        z = int(np.clip(round(float(c[2])), 0, D - 1))
        aseg[x, y, z] = i + 1
    return brain, aseg, centroids


def _extract_multiscale_patch(
    brain: np.ndarray,
    p: np.ndarray,
    pr: int,
    strides: Sequence[int],
) -> np.ndarray:
    """Same multiscale patch extraction as train_goal_net.extract_multiscale_patch.

    Duplicated here to avoid a circular import (train_goal_net imports
    tensorflow at module load, and we want mni_sampler.py importable
    standalone for the smoke test). Keep this byte-equivalent to the original.
    """
    pd = 2 * pr + 1
    out = np.zeros((pd, pd, pd, len(strides)), dtype=np.float32)
    offsets = np.arange(-pr, pr + 1)
    H, W, D = brain.shape
    for ci, s in enumerate(strides):
        ii = np.clip(p[0] + s * offsets, 0, H - 1)
        jj = np.clip(p[1] + s * offsets, 0, W - 1)
        kk = np.clip(p[2] + s * offsets, 0, D - 1)
        out[..., ci] = brain[np.ix_(ii, jj, kk)]
    return out


def sample_batch_mni(
    mni_subject: tuple[np.ndarray, np.ndarray, np.ndarray],
    batch_size: int,
    patch_radius: int,
    strides: Sequence[int],
    sample_radius: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Same return signature as train_goal_net.sample_batch().

    Always samples from the single MNI152 subject. Picks one of the present
    landmarks uniformly, draws an offset within +/- sample_radius, clamps to
    a margin large enough that the multiscale patch never reads outside the
    volume, and rejects zero-intensity start voxels (up to 20 attempts) so
    training samples are inside the head.
    """
    brain, _aseg, centroids = mni_subject
    pr = patch_radius
    pd = 2 * pr + 1
    n_scales = len(strides)
    max_stride = max(strides)
    X_patch = np.zeros((batch_size, pd, pd, pd, n_scales), dtype=np.float32)
    X_pos = np.zeros((batch_size, 3), dtype=np.float32)
    X_target = np.zeros((batch_size, N_LM), dtype=np.float32)
    y_dir = np.zeros((batch_size, 3), dtype=np.float32)

    valid = np.where(np.isfinite(centroids[:, 0]))[0]
    if len(valid) == 0:
        raise ValueError("MNI sampler: no valid landmark centroids")

    H, W, D = brain.shape
    dims = np.array([H, W, D], dtype=np.float32)
    margin = max_stride * pr

    for b in range(batch_size):
        lm = int(rng.choice(valid))
        c = centroids[lm]
        for _attempt in range(20):
            offset = rng.uniform(-sample_radius, sample_radius, size=3)
            p = np.clip(c + offset, margin, dims - margin - 1).astype(np.int32)
            if brain[p[0], p[1], p[2]] > 0:
                break
        X_patch[b] = _extract_multiscale_patch(brain, p, pr, strides)
        X_pos[b] = 2.0 * (p.astype(np.float32) / dims) - 1.0
        X_target[b, lm] = 1.0
        d = c - p.astype(np.float32)
        n = np.linalg.norm(d) + 1e-8
        y_dir[b] = d / n
    return X_patch, X_pos, X_target, y_dir


def mni_intensity_percentiles(
    brain: np.ndarray, lo: float = 2.0, hi: float = 98.0
) -> tuple[float, float]:
    """Return (p_lo, p_hi) of the brain-interior intensity distribution.

    "Brain interior" = strictly positive voxels. Used by train_goal_net.py to
    histogram-match AOMIC patches into MNI152's intensity range when --mni-frac
    > 0. Operates on the [0, 1]-normalized array that load_mni() returns, so
    the percentile values are also in [0, 1].
    """
    interior = brain[brain > 0]
    if interior.size == 0:
        return 0.0, 1.0
    p_lo = float(np.percentile(interior, lo))
    p_hi = float(np.percentile(interior, hi))
    if p_hi <= p_lo:
        p_hi = p_lo + 1e-6
    return p_lo, p_hi


def histogram_match_to(
    patch: np.ndarray,
    src_lo: float,
    src_hi: float,
    dst_lo: float,
    dst_hi: float,
) -> np.ndarray:
    """Simple percentile-based intensity remap.

    Clamp `patch` to [src_lo, src_hi], then linearly map onto [dst_lo, dst_hi].
    Voxels at intensity 0 stay at 0 (we only want to remap brain-interior
    voxels; the trainer keeps a "background" mask of patch == 0 and reapplies
    it after the remap so air doesn't get pulled into the brain range).
    """
    bg = patch <= 0.0
    src_span = max(src_hi - src_lo, 1e-6)
    dst_span = dst_hi - dst_lo
    out = np.clip(patch, src_lo, src_hi)
    out = (out - src_lo) / src_span
    out = out * dst_span + dst_lo
    out[bg] = 0.0
    return out.astype(np.float32, copy=False)


__all__ = [
    "LM_NAMES",
    "N_LM",
    "load_mni",
    "sample_batch_mni",
    "mni_intensity_percentiles",
    "histogram_match_to",
]
