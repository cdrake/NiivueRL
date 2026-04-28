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
from pathlib import Path

import nibabel as nib
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "data" / "aomic_id1000"

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


def build_model(patch_size: int, n_scales: int, use_patch: bool = True) -> Model:
    inp_patch = layers.Input((patch_size, patch_size, patch_size, n_scales), name="patch")
    inp_pos = layers.Input((3,), name="pos")
    inp_target = layers.Input((N_LM,), name="target")
    if use_patch:
        x = layers.Conv3D(16, 3, activation="relu", padding="same")(inp_patch)
        x = layers.Conv3D(32, 3, activation="relu", padding="same")(x)
        x = layers.Conv3D(32, 3, activation="relu", padding="same")(x)
        feat = layers.GlobalAveragePooling3D()(x)
        merged = layers.Concatenate()([feat, inp_pos, inp_target])
    else:
        # Keep inp_patch in the graph but with zero contribution so Keras
        # accepts it as a model input (the data generator always supplies it).
        zero_patch = layers.Lambda(lambda t: tf.zeros_like(t[:, :1, 0, 0, 0]))(inp_patch)
        merged = layers.Concatenate()([zero_patch, inp_pos, inp_target])
    x = layers.Dense(64, activation="relu")(merged)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dense(3)(x)
    out = layers.Lambda(lambda t: tf.math.l2_normalize(t, axis=-1), name="unit_dir")(x)
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
    args = ap.parse_args()
    strides = tuple(args.strides)

    sub_dirs = sorted(p for p in DATA.glob("sub-*") if (p / "aseg.mgz").exists())
    if not sub_dirs:
        raise SystemExit(f"no subjects at {DATA}; run scripts/goal_vector/download.py first")
    rng = np.random.default_rng(args.seed)
    rng.shuffle(sub_dirs)
    n_val = max(1, int(args.val_frac * len(sub_dirs)))
    val_dirs, train_dirs = sub_dirs[:n_val], sub_dirs[n_val:]

    print(f"loading {len(train_dirs)} train + {len(val_dirs)} val subjects from {DATA}")
    train_subjects = [(*load_subject(d), landmark_centroids(load_subject(d)[1])) for d in train_dirs]
    val_subjects = [(*load_subject(d), landmark_centroids(load_subject(d)[1])) for d in val_dirs]

    pd = 2 * args.patch_radius + 1
    n_scales = len(strides)
    print(f"strides = {strides}  patch shape = ({pd},{pd},{pd},{n_scales})  use_patch={not args.no_patch}")
    model = build_model(pd, n_scales, use_patch=not args.no_patch)
    model.compile(optimizer=tf.keras.optimizers.Adam(3e-4), loss=cosine_loss, metrics=[cosine_similarity_metric])
    model.summary(print_fn=lambda s: print("  " + s))

    def gen(subjects):
        while True:
            X_p, X_pos, X_t, y = sample_batch(
                subjects, args.batch_size, args.patch_radius, strides, args.sample_radius, rng
            )
            yield (X_p, X_pos, X_t), y

    sig = (
        (
            tf.TensorSpec((None, pd, pd, pd, n_scales), tf.float32),
            tf.TensorSpec((None, 3), tf.float32),
            tf.TensorSpec((None, N_LM), tf.float32),
        ),
        tf.TensorSpec((None, 3), tf.float32),
    )
    train_ds = tf.data.Dataset.from_generator(lambda: gen(train_subjects), output_signature=sig).prefetch(2)
    val_ds = tf.data.Dataset.from_generator(lambda: gen(val_subjects), output_signature=sig).prefetch(2)

    history = model.fit(
        train_ds,
        steps_per_epoch=args.steps_per_epoch,
        epochs=args.epochs,
        validation_data=val_ds,
        validation_steps=max(20, args.steps_per_epoch // 5),
    )
    best_val_cos = max(history.history["val_cosine_similarity_metric"])
    print(f"\n[summary] strides={list(strides)} use_patch={not args.no_patch} "
          f"epochs={args.epochs} best_val_cos={best_val_cos:.4f}")


if __name__ == "__main__":
    main()
