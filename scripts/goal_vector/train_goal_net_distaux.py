#!/usr/bin/env python3
"""Goal-vector network with auxiliary distance head and hard-example mining.

Forked from train_goal_net.py to test the report's "shape vs. mean" hypothesis:
that better mean cosine on the wide-FOV variant doesn't translate to better
policy success because the *distribution* of per-step error has heavy tails at
decision points.

Two changes target the per-step error distribution rather than its mean:

1. Auxiliary distance regression head. A second Dense(1) head -- with its own
   small MLP off the merged feature -- predicts ||centroid - position|| in voxel
   units. Sharing the trunk with the direction head pushes the encoder toward
   features that capture spatial structure (distance from a landmark requires
   knowing where the landmark is, even when no patch voxel is on it).

2. Online hard-example mining. After a warmup epoch, samples with large
   per-step direction error are upweighted: weight = 1 + alpha * stop_grad(err),
   where err = 1 - cos(y_dir, pred_dir) in [0, 2]. Stop-gradient is required
   so the weighting itself doesn't backprop -- we want to push the model toward
   reducing tail errors, not toward gaming the weight.

The training script saves *two* models:
    * model.keras           multi-output, used to resume training / inspect
    * model_deploy.keras    single-output (unit_dir only), what convert_to_tfjs
                            consumes; the browser only needs unit_dir.
"""
import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import tensorflow as tf

# Keras 2 (tf_keras) for tfjs converter compatibility.
import tf_keras as keras
from tf_keras import layers, Model

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "data" / "aomic_id1000"

# Same 15 subcortical structures as src/lib/landmarks.ts.
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
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Pick (subject, landmark, position) tuples; return X_patch, X_pos,
    X_target, y_dir, y_dist. y_dist is ||centroid - position|| in voxel
    units."""
    pr = patch_radius
    pd = 2 * pr + 1
    n_scales = len(strides)
    max_stride = max(strides)
    X_patch = np.zeros((batch_size, pd, pd, pd, n_scales), dtype=np.float32)
    X_pos = np.zeros((batch_size, 3), dtype=np.float32)
    X_target = np.zeros((batch_size, N_LM), dtype=np.float32)
    y_dir = np.zeros((batch_size, 3), dtype=np.float32)
    y_dist = np.zeros((batch_size, 1), dtype=np.float32)

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
        X_pos[b] = 2.0 * (p.astype(np.float32) / dims) - 1.0
        X_target[b, lm] = 1.0
        d = c - p.astype(np.float32)
        n = float(np.linalg.norm(d) + 1e-8)
        y_dir[b] = d / n
        y_dist[b, 0] = n
    return X_patch, X_pos, X_target, y_dir, y_dist


def _scale_branch(x, patch_size: int):
    x = layers.Conv3D(16, 3, activation="relu", padding="same")(x)
    x = layers.Conv3D(32, 3, activation="relu", padding="same")(x)
    x = layers.Conv3D(32, 3, activation="relu", padding="same")(x)
    x = layers.AveragePooling3D(pool_size=patch_size)(x)
    return layers.Flatten()(x)


def _wide_branch(x, patch_size: int):
    x = layers.Conv3D(32, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling3D(pool_size=2, padding="valid")(x)
    s1 = patch_size // 2
    x = layers.Conv3D(64, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling3D(pool_size=2, padding="valid")(x)
    s2 = s1 // 2
    x = layers.Conv3D(96, 3, padding="same", activation="relu")(x)
    x = layers.AveragePooling3D(pool_size=s2)(x)
    return layers.Flatten()(x)


def build_model(
    patch_size: int,
    n_scales: int,
    use_patch: bool = True,
    hierarchical: bool = False,
    wide_trunk: bool = False,
    distance_head: bool = True,
) -> Model:
    """Goal-vector model with optional auxiliary distance head.

    When distance_head=True, the model has two outputs:
        * unit_dir: (B, 3), L2-normalized direction from position to centroid.
        * distance: (B, 1), predicted ||centroid - position|| in voxel units.
    Both heads share the conv trunk and the merged-feature concat, but each
    has its own 2-layer MLP. When distance_head=False, the model is a drop-in
    replacement for train_goal_net.build_model() (single unit_dir output).
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
        zero_patch = layers.Lambda(
            lambda t: tf.zeros_like(t[:, :1, 0, 0, 0]),
            output_shape=(1,),
        )(inp_patch)
        merged = layers.Concatenate()([zero_patch, inp_pos, inp_target])

    # Direction head.
    xd = layers.Dense(head_units, activation="relu", name="dir_dense_1")(merged)
    xd = layers.Dense(head_units // 2, activation="relu", name="dir_dense_2")(xd)
    xd = layers.Dense(3, name="dir_logits")(xd)
    out_dir = layers.Lambda(
        lambda t: tf.math.l2_normalize(t, axis=-1),
        output_shape=(3,),
        name="unit_dir",
    )(xd)

    if not distance_head:
        return Model([inp_patch, inp_pos, inp_target], out_dir)

    # Distance head: separate small MLP off the same merged feature. Linear
    # output (no activation) predicts distance in voxel units.
    xs = layers.Dense(head_units // 2, activation="relu", name="dist_dense_1")(merged)
    xs = layers.Dense(head_units // 4, activation="relu", name="dist_dense_2")(xs)
    out_dist = layers.Dense(1, name="distance")(xs)

    return Model([inp_patch, inp_pos, inp_target], [out_dir, out_dist])


class GoalVectorTrainer(keras.Model):
    """Wraps a 2-output (unit_dir, distance) model with a custom train_step
    that does cosine loss + auxiliary MSE distance loss + per-sample hard-
    example reweighting.

    During warmup_epochs the per-sample weight is 1.0; afterward, each sample's
    cosine loss is scaled by 1 + alpha * stop_grad(per_sample_cos_err). The
    distance loss is *not* hard-example-weighted -- it's a regularizer on the
    encoder, and over-weighting it would warp the trunk away from the direction
    objective.
    """

    def __init__(
        self,
        inner: Model,
        dist_loss_weight: float = 0.1,
        hard_example_alpha: float = 1.0,
        warmup_epochs: int = 1,
    ):
        super().__init__()
        self.inner = inner
        self.dist_loss_weight = float(dist_loss_weight)
        self.hard_example_alpha = float(hard_example_alpha)
        self.warmup_epochs = int(warmup_epochs)
        # Track epoch via a tf.Variable so train_step can read it under tf.function.
        self.epoch_var = tf.Variable(0, trainable=False, dtype=tf.int32, name="epoch")

        self.cos_metric = keras.metrics.Mean(name="cos_sim")
        self.dist_metric = keras.metrics.Mean(name="dist_err")
        self.weight_metric = keras.metrics.Mean(name="sample_weight")
        self.loss_metric = keras.metrics.Mean(name="loss")

    def call(self, inputs, training=False):
        return self.inner(inputs, training=training)

    @property
    def metrics(self):
        return [self.loss_metric, self.cos_metric, self.dist_metric, self.weight_metric]

    def _split_targets(self, y):
        """Accept either dict {'unit_dir':..., 'distance':...} or tuple
        (y_dir, y_dist). The data pipeline yields tuples so this handles
        both for safety."""
        if isinstance(y, dict):
            return y["unit_dir"], y["distance"]
        if isinstance(y, (tuple, list)):
            return y[0], y[1]
        raise ValueError(f"unexpected target structure {type(y)}")

    def _step_metrics(self, y_dir, pred_dir, y_dist, pred_dist, sample_weight, total_loss):
        cos_sim = tf.reduce_sum(y_dir * pred_dir, axis=-1)
        self.cos_metric.update_state(cos_sim)
        self.dist_metric.update_state(tf.abs(y_dist - pred_dist))
        self.weight_metric.update_state(sample_weight)
        self.loss_metric.update_state(total_loss)

    def train_step(self, data):
        x, y = data
        y_dir, y_dist = self._split_targets(y)

        with tf.GradientTape() as tape:
            pred_dir, pred_dist = self.inner(x, training=True)
            # Per-sample cosine error in [0, 2]. Stop gradient on the weight so
            # the model doesn't try to game its own loss multiplier.
            per_sample_err = 1.0 - tf.reduce_sum(y_dir * pred_dir, axis=-1)
            err_sg = tf.stop_gradient(per_sample_err)
            in_warmup = tf.cast(self.epoch_var < self.warmup_epochs, tf.float32)
            sample_weight = (
                in_warmup * tf.ones_like(err_sg)
                + (1.0 - in_warmup) * (1.0 + self.hard_example_alpha * err_sg)
            )
            cos_loss = tf.reduce_mean(sample_weight * per_sample_err)
            dist_loss = tf.reduce_mean(tf.square(y_dist - pred_dist))
            total_loss = cos_loss + self.dist_loss_weight * dist_loss

        grads = tape.gradient(total_loss, self.inner.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.inner.trainable_variables))
        self._step_metrics(y_dir, pred_dir, y_dist, pred_dist, sample_weight, total_loss)
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        x, y = data
        y_dir, y_dist = self._split_targets(y)
        pred_dir, pred_dist = self.inner(x, training=False)
        per_sample_err = 1.0 - tf.reduce_sum(y_dir * pred_dir, axis=-1)
        # Validation reports unweighted loss so the curves are comparable across
        # warmup/post-warmup boundaries.
        cos_loss = tf.reduce_mean(per_sample_err)
        dist_loss = tf.reduce_mean(tf.square(y_dist - pred_dist))
        total_loss = cos_loss + self.dist_loss_weight * dist_loss
        sample_weight = tf.ones_like(per_sample_err)
        self._step_metrics(y_dir, pred_dir, y_dist, pred_dist, sample_weight, total_loss)
        return {m.name: m.result() for m in self.metrics}


class EpochTickCallback(keras.callbacks.Callback):
    """Bumps trainer.epoch_var so train_step knows whether warmup is done."""

    def __init__(self, trainer: GoalVectorTrainer):
        super().__init__()
        self.trainer = trainer

    def on_epoch_begin(self, epoch, logs=None):
        self.trainer.epoch_var.assign(int(epoch))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--patch-radius", type=int, default=3)
    ap.add_argument("--sample-radius", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--steps-per-epoch", type=int, default=200)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-patch", action="store_true",
                    help="ablate the T1 patch input")
    ap.add_argument("--strides", type=int, nargs="+", default=[1])
    ap.add_argument("--hierarchical", action="store_true")
    ap.add_argument("--wide-trunk", action="store_true")
    # Distance-head + hard-example mining flags.
    ap.add_argument("--distance-head", dest="distance_head", action="store_true",
                    default=True, help="enable auxiliary distance head (default)")
    ap.add_argument("--no-distance-head", dest="distance_head", action="store_false",
                    help="disable auxiliary distance head")
    ap.add_argument("--dist-loss-weight", type=float, default=0.1,
                    help="lambda multiplying the distance MSE term")
    ap.add_argument("--hard-example-alpha", type=float, default=1.0,
                    help="hard-example mining strength; 0 disables")
    ap.add_argument("--warmup-epochs", type=int, default=1,
                    help="epochs of uniform-weight training before HEM kicks in")
    ap.add_argument("--save-dir", type=str, default=None)
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
    if args.wide_trunk and pd < 8:
        raise SystemExit(f"--wide-trunk needs patch_size >= 8 (got {pd})")
    print(f"strides = {strides}  patch shape = ({pd},{pd},{pd},{n_scales})  "
          f"use_patch={not args.no_patch}  hierarchical={args.hierarchical}  "
          f"wide_trunk={args.wide_trunk}  distance_head={args.distance_head}")

    inner = build_model(
        pd,
        n_scales,
        use_patch=not args.no_patch,
        hierarchical=args.hierarchical,
        wide_trunk=args.wide_trunk,
        distance_head=args.distance_head,
    )
    inner.summary(print_fn=lambda s: print("  " + s))

    if not args.distance_head:
        # Fall back to plain compile/fit for the no-aux-head ablation.
        inner.compile(
            optimizer=keras.optimizers.Adam(3e-4),
            loss=lambda yt, yp: 1.0 - tf.reduce_sum(yt * yp, axis=-1),
            metrics=[lambda yt, yp: tf.reduce_mean(tf.reduce_sum(yt * yp, axis=-1))],
        )
        # Keep the rest of the script (data pipeline + save) consistent by
        # building tuple-y datasets that just drop the dist target.

    trainer = GoalVectorTrainer(
        inner,
        dist_loss_weight=args.dist_loss_weight,
        hard_example_alpha=args.hard_example_alpha,
        warmup_epochs=args.warmup_epochs,
    ) if args.distance_head else None

    if trainer is not None:
        trainer.compile(optimizer=keras.optimizers.Adam(3e-4))

    def gen(subjects):
        while True:
            X_p, X_pos, X_t, y_d, y_n = sample_batch(
                subjects, args.batch_size, args.patch_radius, strides, args.sample_radius, rng
            )
            if args.distance_head:
                yield (X_p, X_pos, X_t), (y_d, y_n)
            else:
                yield (X_p, X_pos, X_t), y_d

    if args.distance_head:
        sig = (
            (
                tf.TensorSpec((None, pd, pd, pd, n_scales), tf.float32),
                tf.TensorSpec((None, 3), tf.float32),
                tf.TensorSpec((None, N_LM), tf.float32),
            ),
            (
                tf.TensorSpec((None, 3), tf.float32),
                tf.TensorSpec((None, 1), tf.float32),
            ),
        )
    else:
        sig = (
            (
                tf.TensorSpec((None, pd, pd, pd, n_scales), tf.float32),
                tf.TensorSpec((None, 3), tf.float32),
                tf.TensorSpec((None, N_LM), tf.float32),
            ),
            tf.TensorSpec((None, 3), tf.float32),
        )

    train_ds = tf.data.Dataset.from_generator(
        lambda: gen(train_subjects), output_signature=sig
    ).prefetch(2)
    val_ds = tf.data.Dataset.from_generator(
        lambda: gen(val_subjects), output_signature=sig
    ).prefetch(2)

    if args.distance_head:
        callbacks = [EpochTickCallback(trainer)]
        history = trainer.fit(
            train_ds,
            steps_per_epoch=args.steps_per_epoch,
            epochs=args.epochs,
            validation_data=val_ds,
            validation_steps=max(20, args.steps_per_epoch // 5),
            callbacks=callbacks,
        )
        best_val_cos = float(max(history.history.get("val_cos_sim", [0.0])))
    else:
        history = inner.fit(
            train_ds,
            steps_per_epoch=args.steps_per_epoch,
            epochs=args.epochs,
            validation_data=val_ds,
            validation_steps=max(20, args.steps_per_epoch // 5),
        )
        # The lambda metric in Keras 2 is named after its position.
        val_keys = [k for k in history.history if k.startswith("val_") and "lambda" in k]
        best_val_cos = float(max(history.history[val_keys[0]])) if val_keys else 0.0

    print(f"\n[summary] strides={list(strides)} use_patch={not args.no_patch} "
          f"hierarchical={args.hierarchical} wide_trunk={args.wide_trunk} "
          f"distance_head={args.distance_head} epochs={args.epochs} "
          f"best_val_cos={best_val_cos:.4f}")

    if args.save_dir:
        save_dir = Path(args.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        # Full multi-output model (training-resumable, full architecture).
        inner.save(save_dir / "model.keras")
        # Deploy-only model: same inputs, only the unit_dir output. The browser
        # never needs the distance prediction; convert_to_tfjs.py reads this
        # one. Building it as a fresh Model instead of slicing weights keeps
        # the TFJS converter happy (no Lambda graph surgery).
        if args.distance_head:
            deploy = Model(inner.inputs, inner.get_layer("unit_dir").output)
            deploy.save(save_dir / "model_deploy.keras")
        else:
            inner.save(save_dir / "model_deploy.keras")
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
            "patch_axes": ["x", "y", "z"],
            "position_normalization": "2 * voxel / dims - 1",
            "best_val_cos": best_val_cos,
            "epochs": args.epochs,
            "n_train_subjects": len(train_dirs),
            "n_val_subjects": len(val_dirs),
            # New schema fields for distance-head + hard-example mining.
            "has_distance_head": bool(args.distance_head),
            "dist_loss_weight": float(args.dist_loss_weight),
            "hard_example_alpha": float(args.hard_example_alpha),
            "warmup_epochs": int(args.warmup_epochs),
            "deploy_model_file": "model_deploy.keras",
        }
        (save_dir / "metadata.json").write_text(json.dumps(meta, indent=2))
        print(f"[save] wrote {save_dir / 'model.keras'}, "
              f"{save_dir / 'model_deploy.keras'}, and metadata.json")


if __name__ == "__main__":
    main()
