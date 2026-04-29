#!/usr/bin/env python3
"""Quantify the per-step angular-error distribution of a trained goal-vector
model on a held-out subject.

For N sampled (subject, landmark, position) tuples, runs the model and writes
a JSON of per-sample angular errors (degrees), per-landmark means, mean cosine,
and mean error. The CSCE 775 report claims that "per-step error *shape*
matters more than mean cosine"; this is the data backing that claim.

Run inside the .venv-tfjs venv (tensorflow 2.19 + tf_keras), e.g.:

    .venv-tfjs/bin/python scripts/goal_vector/eval_per_step_error.py \
        --model-dir models/goal_vector_theia_20260428_180017 \
        --subject-dir data/aomic_id1000/sub-0083 \
        --n-samples 2000 \
        --output-json docs/per_step_error_baseline.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import nibabel as nib  # noqa: F401  (used transitively via train_goal_net)
import numpy as np
import tensorflow as tf  # noqa: F401  (used by tf_keras)
import tf_keras as keras

# Reuse the training pipeline's loaders and patch extractor verbatim so that
# any change there propagates to evaluation automatically.
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
from train_goal_net import (  # type: ignore  noqa: E402
    LM_NAMES,
    extract_multiscale_patch,
    landmark_centroids,
    load_subject,
)


def _build_inputs(
    X_patch: np.ndarray,
    X_pos: np.ndarray,
    X_target: np.ndarray,
    *,
    per_scale_inputs: bool,
    n_scales: int,
) -> list[np.ndarray]:
    """Format model inputs.

    Training-time models take a single (B, ps, ps, ps, n_scales) patch tensor.
    Hierarchical inference models exported via convert_to_tfjs.py instead take
    n_scales separate (B, ps, ps, ps, 1) tensors. We support both so that the
    eval works against either flavor.
    """
    if per_scale_inputs:
        patches = [X_patch[..., s : s + 1] for s in range(n_scales)]
        return [*patches, X_pos, X_target]
    return [X_patch, X_pos, X_target]


def _select_unit_dir(pred) -> np.ndarray:
    """Pull out the unit-direction tensor from the model's output.

    Some variants emit a single (B, 3) tensor; others (planned distance-aux
    head, future multi-output models) emit a list/dict where the first or
    `unit_dir`-named entry is the direction. We accept all three shapes.
    """
    if isinstance(pred, (list, tuple)):
        return np.asarray(pred[0])
    if isinstance(pred, dict):
        if "unit_dir" in pred:
            return np.asarray(pred["unit_dir"])
        # Fallback: take whichever tensor has trailing dim 3.
        for v in pred.values():
            arr = np.asarray(v)
            if arr.ndim >= 2 and arr.shape[-1] == 3:
                return arr
        raise SystemExit(f"could not find a unit-dir output in dict keys {list(pred)}")
    return np.asarray(pred)


def _l2_normalize(v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, eps)


def evaluate(
    model_dir: Path,
    subject_dir: Path,
    n_samples: int,
    sample_radius: int,
    seed: int,
) -> dict:
    meta = json.loads((model_dir / "metadata.json").read_text())
    patch_radius = int(meta["patch_radius"])
    patch_size = int(meta.get("patch_size", 2 * patch_radius + 1))
    strides = tuple(meta.get("strides", [1]))
    n_scales = int(meta.get("n_scales", len(strides)))
    per_scale_inputs = bool(meta.get("per_scale_inputs", False))
    output_post_normalize = bool(meta.get("output_post_normalize", False))

    print(f"[eval] loading model from {model_dir / 'model.keras'}")
    model = keras.models.load_model(
        model_dir / "model.keras", compile=False, safe_mode=False
    )
    print(f"[eval] model has {len(model.inputs)} inputs and "
          f"{len(model.outputs)} outputs")

    print(f"[eval] loading subject from {subject_dir}")
    brain, aseg = load_subject(subject_dir)
    centroids = landmark_centroids(aseg)
    valid_lm = [int(i) for i in np.where(np.isfinite(centroids[:, 0]))[0]]
    if not valid_lm:
        raise SystemExit(f"no valid landmarks found in {subject_dir}/aseg.mgz")
    print(f"[eval] {len(valid_lm)}/{len(LM_NAMES)} landmarks present")

    rng = np.random.default_rng(seed)
    pr = patch_radius
    pd = 2 * pr + 1
    if pd != patch_size:
        # Trust metadata over patch_radius (defensive: future variants may
        # set patch_size != 2*patch_radius+1 if anyone introduces asymmetric
        # patches). We follow patch_radius for sampling and compare.
        print(f"[eval] note: metadata patch_size={patch_size} disagrees with "
              f"2*patch_radius+1={pd}; using patch_radius")
    max_stride = max(strides)
    H, W, D = brain.shape
    dims = np.array([H, W, D], dtype=np.float32)
    margin = max_stride * pr

    # Distribute samples uniformly across the *present* landmarks.
    # Round-robin gives exactly balanced coverage even for small N.
    lm_per_sample = np.array(
        [valid_lm[i % len(valid_lm)] for i in range(n_samples)], dtype=np.int32
    )
    rng.shuffle(lm_per_sample)

    X_patch = np.zeros((n_samples, pd, pd, pd, n_scales), dtype=np.float32)
    X_pos = np.zeros((n_samples, 3), dtype=np.float32)
    X_target = np.zeros((n_samples, len(LM_NAMES)), dtype=np.float32)
    y_dir = np.zeros((n_samples, 3), dtype=np.float32)

    for b in range(n_samples):
        lm = int(lm_per_sample[b])
        c = centroids[lm]
        for _attempt in range(20):
            offset = rng.uniform(-sample_radius, sample_radius, size=3)
            p = np.clip(c + offset, margin, dims - margin - 1).astype(np.int32)
            if brain[p[0], p[1], p[2]] > 0:
                break
        X_patch[b] = extract_multiscale_patch(brain, p, pr, strides)
        X_pos[b] = 2.0 * (p.astype(np.float32) / dims) - 1.0
        X_target[b, lm] = 1.0
        d = c - p.astype(np.float32)
        n = np.linalg.norm(d) + 1e-8
        y_dir[b] = d / n

    inputs = _build_inputs(
        X_patch, X_pos, X_target,
        per_scale_inputs=per_scale_inputs,
        n_scales=n_scales,
    )
    print(f"[eval] running inference on {n_samples} samples ...")
    pred_raw = model.predict(inputs, batch_size=512, verbose=0)
    pred = _select_unit_dir(pred_raw)

    # If the saved model is a "deployed inference" model (post-norm Lambda
    # stripped), the outputs are unnormalized logits. The training-time
    # checkpoints ARE normalized (l2_normalize Lambda is the final layer),
    # but normalizing twice is a no-op so this is safe in both cases.
    if output_post_normalize or not np.allclose(
        np.linalg.norm(pred[: min(16, n_samples)], axis=-1), 1.0, atol=1e-3
    ):
        pred = _l2_normalize(pred)

    cos = np.sum(pred * y_dir, axis=-1).astype(np.float64)
    cos_clipped = np.clip(cos, -1.0, 1.0)
    err_deg = np.degrees(np.arccos(cos_clipped))

    per_lm: dict[str, float] = {}
    per_lm_count: dict[str, int] = {}
    for lm in valid_lm:
        mask = lm_per_sample == lm
        if mask.any():
            per_lm[LM_NAMES[lm]] = float(err_deg[mask].mean())
            per_lm_count[LM_NAMES[lm]] = int(mask.sum())

    result: dict = {
        "model_dir": str(model_dir),
        "subject_dir": str(subject_dir),
        "n_samples": int(n_samples),
        "sample_radius": int(sample_radius),
        "seed": int(seed),
        "patch_radius": patch_radius,
        "patch_size": patch_size,
        "strides": list(strides),
        "n_scales": n_scales,
        "hierarchical": bool(meta.get("hierarchical", False)),
        "wide_trunk": bool(meta.get("wide_trunk", False)),
        "best_val_cos": meta.get("best_val_cos"),
        "mean_cosine": float(cos.mean()),
        "mean_err_deg": float(err_deg.mean()),
        "median_err_deg": float(np.median(err_deg)),
        "p75_err_deg": float(np.percentile(err_deg, 75)),
        "p90_err_deg": float(np.percentile(err_deg, 90)),
        "p95_err_deg": float(np.percentile(err_deg, 95)),
        "frac_err_gt_30": float((err_deg > 30.0).mean()),
        "frac_err_gt_60": float((err_deg > 60.0).mean()),
        "errors_deg": err_deg.astype(float).tolist(),
        "landmarks": [LM_NAMES[int(l)] for l in lm_per_sample.tolist()],
        "per_landmark_mean_err": per_lm,
        "per_landmark_count": per_lm_count,
    }
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-dir", type=Path, required=True,
                    help="dir containing model.keras + metadata.json")
    ap.add_argument("--subject-dir", type=Path, required=True,
                    help="AOMIC sub-XXXX dir with brain.mgz + aseg.mgz")
    ap.add_argument("--n-samples", type=int, default=20000)
    ap.add_argument("--sample-radius", type=int, default=50,
                    help="max start distance from each landmark centroid")
    ap.add_argument("--output-json", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if not (args.model_dir / "model.keras").exists():
        raise SystemExit(f"missing {args.model_dir / 'model.keras'}")
    if not (args.subject_dir / "brain.mgz").exists():
        raise SystemExit(f"missing {args.subject_dir / 'brain.mgz'}")
    if not (args.subject_dir / "aseg.mgz").exists():
        raise SystemExit(f"missing {args.subject_dir / 'aseg.mgz'}")

    result = evaluate(
        model_dir=args.model_dir,
        subject_dir=args.subject_dir,
        n_samples=args.n_samples,
        sample_radius=args.sample_radius,
        seed=args.seed,
    )

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2))
    print(f"[eval] mean cosine = {result['mean_cosine']:.4f}  "
          f"mean err = {result['mean_err_deg']:.2f} deg  "
          f"P90 = {result['p90_err_deg']:.2f}  "
          f"P95 = {result['p95_err_deg']:.2f}  "
          f">30 deg = {100*result['frac_err_gt_30']:.1f}%  "
          f">60 deg = {100*result['frac_err_gt_60']:.1f}%")
    print(f"[eval] wrote {args.output_json}")


if __name__ == "__main__":
    main()
