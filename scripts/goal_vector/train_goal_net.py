#!/usr/bin/env python3
"""Prototype goal-vector network on AOMIC ID1000.

Given a 7^3 T1 patch around an agent voxel and a one-hot target landmark,
predict the unit direction vector toward the target's centroid in that
subject's brain. Training samples are generated on the fly: pick a subject,
target, and a random voxel inside the brain (within R voxels of the target),
extract the patch, compute the unit direction.

This is the supervised pretraining task for an RL goal vector network. We
just want to see the loss go down on held-out subjects -- if it does, the
signal is learnable from intensity context alone.
"""
import argparse
import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import tensorflow as tf

# Use Keras 2 (tf_keras) rather than the bundled Keras 3 so that
# tensorflowjs's converter emits a model.json that tfjs-layers 4.x can load
# without per-layer schema patching. The model topology is identical.
import tf_keras as keras
from tf_keras import layers, Model

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "data" / "aomic_id1000"

# Make the sibling mni_sampler module importable whether this script is run as
# `python scripts/goal_vector/train_goal_net.py` or `python -m
# scripts.goal_vector.train_goal_net`.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from mni_sampler import (  # noqa: E402
    load_mni as _mni_load,
    sample_batch_mni as _mni_sample_batch,
    mni_intensity_percentiles as _mni_percentiles,
    histogram_match_to as _mni_histogram_match,
)

# Same 15 subcortical structures as src/lib/landmarks.ts
LANDMARKS: dict[str, tuple[int, ...]] = {
    "Lateral-Ventricle":       (4, 43),
    "Inf-Lat-Vent":            (5, 44),
    "Cerebellum-White-Matter": (7, 46),
    "Cerebellum-Cortex":       (8, 47),
    "Thalamus":                (10, 49),
    "Caudate":                 (11, 50),
    "Putamen":                 (12, 51),
    "Pallidum":                (13, 52),
    "3rd-Ventricle":           (14,),
    "4th-Ventricle":           (15,),
    "Brain-Stem":              (16,),
    "Hippocampus":             (17, 53),
    "Amygdala":                (18, 54),
    "Accumbens-area":          (26, 58),
    "VentralDC":               (28, 60),
}
LM_NAMES = list(LANDMARKS.keys())
N_LM = len(LM_NAMES)


def load_subject(sub_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    brain = np.asarray(nib.load(sub_dir / "brain.mgz").dataobj).astype(np.float32)
    aseg = np.asarray(nib.load(sub_dir / "aseg.mgz").dataobj).astype(np.int32)
    # FreeSurfer brain.mgz is roughly [0, 110]; normalize to [0, 1].
    brain /= max(brain.max(), 1.0)
    return brain, aseg


def landmark_centroids(aseg: np.ndarray) -> np.ndarray:
    """Returns (N_LM, 3) array of centroid voxel indices, NaN where absent."""
    out = np.full((N_LM, 3), np.nan, dtype=np.float32)
    for i, name in enumerate(LM_NAMES):
        mask = np.isin(aseg, LANDMARKS[name])
        if mask.any():
            out[i] = np.argwhere(mask).mean(axis=0)
    return out


def extract_multiscale_patch(
    brain: np.ndarray, p: np.ndarray, pr: int, strides: tuple[int, ...]
) -> np.ndarray:
    """Returns a (pd, pd, pd, len(strides)) patch sampled at each stride
    around center p. Sampling clamps to volume bounds."""
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


def sample_batch(
    subjects: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    batch_size: int,
    patch_radius: int,
    strides: tuple[int, ...],
    sample_radius: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Pick (subject, landmark, position) tuples; return X_patch, X_pos, X_target, y_dir."""
    pr = patch_radius
    pd = 2 * pr + 1
    n_scales = len(strides)
    max_stride = max(strides)
    X_patch = np.zeros((batch_size, pd, pd, pd, n_scales), dtype=np.float32)
    X_pos = np.zeros((batch_size, 3), dtype=np.float32)
    X_target = np.zeros((batch_size, N_LM), dtype=np.float32)
    y_dir = np.zeros((batch_size, 3), dtype=np.float32)

    margin = max_stride * pr
    for b in range(batch_size):
        s = rng.integers(len(subjects))
        brain, _aseg, centroids = subjects[s]
        valid = np.where(np.isfinite(centroids[:, 0]))[0]
        lm = int(rng.choice(valid))
        c = centroids[lm]
        H, W, D = brain.shape
        dims = np.array([H, W, D], dtype=np.float32)
        for _attempt in range(20):
            offset = rng.uniform(-sample_radius, sample_radius, size=3)
            p = np.clip(c + offset, margin, dims - margin - 1).astype(np.int32)
            if brain[p[0], p[1], p[2]] > 0:
                break
        X_patch[b] = extract_multiscale_patch(brain, p, pr, strides)
        # Normalize position to [-1, 1].
        X_pos[b] = 2.0 * (p.astype(np.float32) / dims) - 1.0
        X_target[b, lm] = 1.0
        d = c - p.astype(np.float32)
        n = np.linalg.norm(d) + 1e-8
        y_dir[b] = d / n
    return X_patch, X_pos, X_target, y_dir


def _scale_branch(x, patch_size: int):
    """One conv branch: 16 -> 32 -> 32, ReLU, padding='same', then a global
    average via AveragePooling3D(pool_size=patch_size) + Flatten. We avoid
    GlobalAveragePooling3D because tfjs-layers 4.x doesn't ship it; the
    pool+flatten combination is mathematically identical and supported."""
    x = layers.Conv3D(16, 3, activation="relu", padding="same")(x)
    x = layers.Conv3D(32, 3, activation="relu", padding="same")(x)
    x = layers.Conv3D(32, 3, activation="relu", padding="same")(x)
    x = layers.AveragePooling3D(pool_size=patch_size)(x)
    return layers.Flatten()(x)


def _wide_branch(x, patch_size: int):
    """Wider per-scale branch with two downsample stages.

    The default _scale_branch keeps spatial dims at patch_size throughout and
    averages everything at the end, which means almost all spatial information
    is collapsed into the channel statistics. _wide_branch instead actually
    downsamples (Conv -> MaxPool -> Conv -> MaxPool -> Conv -> AvgPool) so
    deeper layers see broader receptive fields with more channels. Works for
    any patch_size >= 8 (15, 17, 19, ...). Uses only Conv3D, MaxPooling3D,
    AveragePooling3D, ReLU -- all supported by tfjs-layers 4.x.
    """
    x = layers.Conv3D(32, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling3D(pool_size=2, padding="valid")(x)  # patch_size // 2
    s1 = patch_size // 2
    x = layers.Conv3D(64, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling3D(pool_size=2, padding="valid")(x)  # patch_size // 4
    s2 = s1 // 2
    x = layers.Conv3D(96, 3, padding="same", activation="relu")(x)
    # Collapse the remaining s2^3 cube to 1^3 via average pool over the full
    # extent, then flatten to 96 features. tfjs-layers requires a static
    # pool_size, which is fine because patch_size is fixed at construction.
    x = layers.AveragePooling3D(pool_size=s2)(x)
    return layers.Flatten()(x)


def build_model(
    patch_size: int,
    n_scales: int,
    use_patch: bool = True,
    hierarchical: bool = False,
    wide_trunk: bool = False,
) -> Model:
    """Goal-vector model.

    - hierarchical=False: scales enter as channels of a single conv trunk.
      Filters at every layer mix all scales together.
    - hierarchical=True: each scale runs through its own conv branch and the
      branch features are concatenated before the dense head (Ghesu-style).
      The branches do not share weights.
    - wide_trunk=True: each branch uses _wide_branch (Conv->MaxPool x2 ->
      Conv -> AvgPool) instead of the default _scale_branch. Requires
      patch_size >= 8. Pairs naturally with hierarchical=True for a
      multi-scale wide-FOV model.
    """
    inp_patch = layers.Input((patch_size, patch_size, patch_size, n_scales), name="patch")
    inp_pos = layers.Input((3,), name="pos")
    inp_target = layers.Input((N_LM,), name="target")
    branch_fn = _wide_branch if wide_trunk else _scale_branch
    head_units = 128 if wide_trunk else 64
    if use_patch:
        if hierarchical:
            branches = []
            for s in range(n_scales):
                x_s = layers.Lambda(
                    lambda t, idx=s: t[..., idx : idx + 1],
                    output_shape=(patch_size, patch_size, patch_size, 1),
                )(inp_patch)
                branches.append(branch_fn(x_s, patch_size))
            feat = layers.Concatenate()(branches) if len(branches) > 1 else branches[0]
        else:
            feat = branch_fn(inp_patch, patch_size)
        merged = layers.Concatenate()([feat, inp_pos, inp_target])
    else:
        # Keep inp_patch in the graph but with zero contribution so Keras
        # accepts it as a model input (the data generator always supplies it).
        zero_patch = layers.Lambda(
            lambda t: tf.zeros_like(t[:, :1, 0, 0, 0]),
            output_shape=(1,),
        )(inp_patch)
        merged = layers.Concatenate()([zero_patch, inp_pos, inp_target])
    x = layers.Dense(head_units, activation="relu")(merged)
    x = layers.Dense(head_units // 2, activation="relu")(x)
    x = layers.Dense(3)(x)
    out = layers.Lambda(
        lambda t: tf.math.l2_normalize(t, axis=-1),
        output_shape=(3,),
        name="unit_dir",
    )(x)
    return Model([inp_patch, inp_pos, inp_target], out)


def cosine_loss(y_true, y_pred):
    # y_true and y_pred are unit vectors; loss = 1 - cos.
    return 1.0 - tf.reduce_sum(y_true * y_pred, axis=-1)


def cosine_similarity_metric(y_true, y_pred):
    return tf.reduce_mean(tf.reduce_sum(y_true * y_pred, axis=-1))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--patch-radius", type=int, default=3, help="half-width (3 -> 7^3)")
    ap.add_argument("--sample-radius", type=int, default=50, help="max start distance from target")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--steps-per-epoch", type=int, default=200)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-patch", action="store_true",
                    help="ablate the T1 patch input (pos + target only)")
    ap.add_argument("--strides", type=int, nargs="+", default=[1],
                    help="patch sampling strides (e.g. 1 4 for multi-scale)")
    ap.add_argument("--hierarchical", action="store_true",
                    help="separate conv branch per scale (Ghesu-style); "
                         "default mixes scales as channels of a shared trunk")
    ap.add_argument("--wide-trunk", action="store_true",
                    help="use the deeper downsampling branch (Conv->MaxPool x2 "
                         "-> Conv -> AvgPool); requires patch_size >= 8")
    ap.add_argument("--save-dir", type=str, default=None,
                    help="if set, save trained model + metadata.json here for browser deploy")
    ap.add_argument("--mni-frac", type=float, default=0.0,
                    help="probability per sample of drawing from the MNI152 source "
                         "instead of AOMIC. 0 disables MNI mixing entirely. "
                         "When >0 also enables histogram-matching of AOMIC patches "
                         "to the MNI152 brain-interior intensity distribution.")
    ap.add_argument("--mni-volume", type=str, default=None,
                    help="path to the MNI152 NIfTI volume (e.g. data/mni152.nii.gz). "
                         "Required when --mni-frac > 0.")
    ap.add_argument("--mni-landmarks-json", type=str, default=None,
                    help="path to the MNI152 landmark centroids JSON written by "
                         "scripts/goal_vector/dump_mni_landmarks.py. "
                         "Required when --mni-frac > 0.")
    args = ap.parse_args()
    strides = tuple(args.strides)
    if args.mni_frac > 0.0:
        if not args.mni_volume or not args.mni_landmarks_json:
            raise SystemExit(
                "--mni-frac > 0 requires --mni-volume and --mni-landmarks-json"
            )

    rng = np.random.default_rng(args.seed)
    sub_dirs = sorted(p for p in DATA.glob("sub-*") if (p / "aseg.mgz").exists())
    use_aomic = args.mni_frac < 1.0
    if use_aomic and not sub_dirs:
        raise SystemExit(f"no subjects at {DATA}; run scripts/goal_vector/download.py first")
    if use_aomic:
        rng.shuffle(sub_dirs)
        n_val = max(1, int(args.val_frac * len(sub_dirs)))
        val_dirs, train_dirs = sub_dirs[:n_val], sub_dirs[n_val:]
        print(f"loading {len(train_dirs)} train + {len(val_dirs)} val subjects from {DATA}")
        train_subjects = [(*load_subject(d), landmark_centroids(load_subject(d)[1])) for d in train_dirs]
        val_subjects = [(*load_subject(d), landmark_centroids(load_subject(d)[1])) for d in val_dirs]
    else:
        train_dirs, val_dirs = [], []
        train_subjects, val_subjects = [], []

    # Optional MNI152 source. When --mni-frac > 0, draw a fraction of samples
    # from the deployment volume directly using ground-truth landmark voxel
    # coords (parsed from src/lib/landmarks.ts via dump_mni_landmarks.py). The
    # MNI subject is shared between train and val splits since there's only one
    # volume; it serves both as additional training signal and as the
    # MNI-domain validation slice we actually care about.
    mni_subject = None
    mni_p_lo = mni_p_hi = aomic_p_lo = aomic_p_hi = None
    if args.mni_frac > 0.0:
        print(f"[mni] loading {args.mni_volume} + {args.mni_landmarks_json}")
        mni_subject = _mni_load(args.mni_volume, args.mni_landmarks_json)
        mni_brain = mni_subject[0]
        mni_p_lo, mni_p_hi = _mni_percentiles(mni_brain, 2.0, 98.0)
        print(f"[mni] brain shape={mni_brain.shape} interior p2={mni_p_lo:.4f} p98={mni_p_hi:.4f}")
        # Compute AOMIC's matching percentiles once so histogram-matching is a
        # constant-time linear remap per patch. Train + val share one set of
        # statistics; the goal is to bridge the two domains, not to model
        # subject-specific intensity.
        if use_aomic:
            aomic_pool = np.concatenate([s[0][s[0] > 0].ravel() for s in train_subjects])
            aomic_p_lo = float(np.percentile(aomic_pool, 2.0))
            aomic_p_hi = float(np.percentile(aomic_pool, 98.0))
            print(f"[mni] AOMIC pooled p2={aomic_p_lo:.4f} p98={aomic_p_hi:.4f} "
                  f"-> histogram-match into [{mni_p_lo:.4f},{mni_p_hi:.4f}]")
        # When a MNI subject is mixed in, also append it to train/val so the
        # metadata's split summary reflects the extra "subject" we trained on.
        # The actual sampling probability is controlled by args.mni_frac, not
        # by the position in the list.
        train_subjects = list(train_subjects) + [mni_subject]
        val_subjects = list(val_subjects) + [mni_subject]

    pd = 2 * args.patch_radius + 1
    n_scales = len(strides)
    if args.wide_trunk and pd < 8:
        raise SystemExit(f"--wide-trunk needs patch_size >= 8 (got {pd}); "
                         f"raise --patch-radius to >= 4")
    print(f"strides = {strides}  patch shape = ({pd},{pd},{pd},{n_scales})  "
          f"use_patch={not args.no_patch}  hierarchical={args.hierarchical}  "
          f"wide_trunk={args.wide_trunk}")
    model = build_model(
        pd,
        n_scales,
        use_patch=not args.no_patch,
        hierarchical=args.hierarchical,
        wide_trunk=args.wide_trunk,
    )
    model.compile(optimizer=keras.optimizers.Adam(3e-4), loss=cosine_loss, metrics=[cosine_similarity_metric])
    model.summary(print_fn=lambda s: print("  " + s))

    def _maybe_histogram_match(X_p: np.ndarray) -> np.ndarray:
        """When mixing AOMIC + MNI, remap AOMIC patch intensities into the MNI
        brain-interior range so the network sees one intensity scale. The remap
        keeps zero-valued voxels at zero (background mask preserved)."""
        if mni_p_lo is None or aomic_p_lo is None:
            return X_p
        return _mni_histogram_match(X_p, aomic_p_lo, aomic_p_hi, mni_p_lo, mni_p_hi)

    def gen(subjects, *, mni_only_subjects=None):
        # mni_only_subjects: list of MNI subject tuples to sample from when the
        # mni_frac coin flip succeeds. When None we fall back to any element of
        # `subjects`, but in practice main() always supplies it explicitly.
        while True:
            if mni_only_subjects and rng.random() < args.mni_frac:
                X_p, X_pos, X_t, y = _mni_sample_batch(
                    mni_only_subjects[0],
                    args.batch_size,
                    args.patch_radius,
                    strides,
                    args.sample_radius,
                    rng,
                )
            else:
                # AOMIC source. If mni_frac > 0 we histogram-match the patches
                # so the network gets a single intensity distribution.
                aomic_pool = [s for s in subjects if s is not mni_subject] if mni_subject is not None else subjects
                if not aomic_pool:
                    # mni_frac == 1.0 path; the coin flip above always picks MNI.
                    aomic_pool = subjects
                X_p, X_pos, X_t, y = sample_batch(
                    aomic_pool,
                    args.batch_size,
                    args.patch_radius,
                    strides,
                    args.sample_radius,
                    rng,
                )
                if mni_subject is not None:
                    X_p = _maybe_histogram_match(X_p)
            yield (X_p, X_pos, X_t), y

    sig = (
        (
            tf.TensorSpec((None, pd, pd, pd, n_scales), tf.float32),
            tf.TensorSpec((None, 3), tf.float32),
            tf.TensorSpec((None, N_LM), tf.float32),
        ),
        tf.TensorSpec((None, 3), tf.float32),
    )
    mni_pool = [mni_subject] if mni_subject is not None else None
    train_ds = tf.data.Dataset.from_generator(
        lambda: gen(train_subjects, mni_only_subjects=mni_pool),
        output_signature=sig,
    ).prefetch(2)
    val_ds = tf.data.Dataset.from_generator(
        lambda: gen(val_subjects, mni_only_subjects=mni_pool),
        output_signature=sig,
    ).prefetch(2)

    history = model.fit(
        train_ds,
        steps_per_epoch=args.steps_per_epoch,
        epochs=args.epochs,
        validation_data=val_ds,
        validation_steps=max(20, args.steps_per_epoch // 5),
    )
    best_val_cos = max(history.history["val_cosine_similarity_metric"])
    print(f"\n[summary] strides={list(strides)} use_patch={not args.no_patch} "
          f"hierarchical={args.hierarchical} epochs={args.epochs} "
          f"best_val_cos={best_val_cos:.4f}")

    if args.save_dir:
        save_dir = Path(args.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        # Save in native Keras format. The browser side will load via tfjs after
        # running scripts/goal_vector/convert_to_tfjs.py against this directory.
        model.save(save_dir / "model.keras")
        meta = {
            "patch_radius": args.patch_radius,
            "patch_size": pd,
            "strides": list(strides),
            "n_scales": n_scales,
            "use_patch": not args.no_patch,
            "hierarchical": args.hierarchical,
            "wide_trunk": args.wide_trunk,
            "sample_radius": args.sample_radius,
            "landmark_names": LM_NAMES,
            "n_landmarks": N_LM,
            "input_order": ["patch", "pos", "target"],
            # Python builds patches as brain[ii, jj, kk] -> shape (x, y, z).
            # The TS env loops dz outer / dx inner, so its flat array reshapes
            # to (z, y, x). The browser must transpose (z,y,x) -> (x,y,z) before
            # feeding the model so axis 0 = x at both train and inference time.
            "patch_axes": ["x", "y", "z"],
            "position_normalization": "2 * voxel / dims - 1",
            "best_val_cos": float(best_val_cos),
            "epochs": args.epochs,
            "n_train_subjects": len(train_dirs),
            "n_val_subjects": len(val_dirs),
            "mni_frac": float(args.mni_frac),
            "mni_volume_used": str(args.mni_volume) if args.mni_volume else None,
            "mni_landmarks_json_used": (
                str(args.mni_landmarks_json) if args.mni_landmarks_json else None
            ),
            "mni_intensity_p2_p98": (
                [mni_p_lo, mni_p_hi] if mni_p_lo is not None else None
            ),
            "aomic_intensity_p2_p98": (
                [aomic_p_lo, aomic_p_hi] if aomic_p_lo is not None else None
            ),
            "domain": "mixed" if args.mni_frac > 0.0 else "aomic",
        }
        (save_dir / "metadata.json").write_text(json.dumps(meta, indent=2))
        print(f"[save] wrote {save_dir / 'model.keras'} and metadata.json")


if __name__ == "__main__":
    main()
