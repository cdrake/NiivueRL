#!/usr/bin/env python3
"""Convert a trained goal-vector Keras model to TensorFlow.js Layers format.

Reads `<src>/model.keras` and `<src>/metadata.json`, writes the tfjs model
shards plus a copy of metadata.json to `<dst>/`. The browser loads from
`<dst>/model.json` via tf.loadLayersModel() and reads metadata.json to know
the patch shape, strides, and landmark order.

Run inside the `.venv-tfjs` venv (has tensorflowjs installed alongside
tensorflow 2.19), since tfjs's converter pulls in tensorflow_decision_forests
and conflicts with the main env's tensorflow 2.20.
"""
import argparse
import json
import shutil
from pathlib import Path

import tensorflow as tf
# Match the training script: load via tf_keras (Keras 2) since the .keras file
# was written with the tf_keras registry. tfjs-layers 4.x also expects the
# Keras 2 schema, so this avoids per-layer schema patching.
import tf_keras as keras
from tf_keras import layers, Model
from tensorflowjs.converters import save_keras_model


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", required=True, help="dir produced by train_goal_net.py --save-dir")
    ap.add_argument("--dst", required=True, help="output dir for the tfjs model (e.g. public/goal_vector_model)")
    args = ap.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)
    src_model = src / "model.keras"
    src_meta = src / "metadata.json"
    if not src_model.exists():
        raise SystemExit(f"missing {src_model}")
    if not src_meta.exists():
        raise SystemExit(f"missing {src_meta}")

    dst.mkdir(parents=True, exist_ok=True)
    print(f"loading {src_model}")
    model = keras.models.load_model(src_model, compile=False, safe_mode=False)
    meta = json.loads(src_meta.read_text())

    # tfjs-layers can't deserialize Python Lambda layers (they hold pickled
    # function bytecode). The trained graph has up to two kinds of Lambdas:
    #   * a trailing L2-normalize Lambda after the last Dense
    #   * for hierarchical models, n_scales interior slice-Lambdas that split
    #     the patch tensor into per-scale single-channel branches.
    # We strip the trailing one by truncating at the last Dense, and for
    # hierarchical models we rebuild the head with explicit per-scale Input
    # layers (one [ps,ps,ps,1] input per branch), reusing the trained branch
    # weights. The TS wrapper splits the patch into n_scales tensors and
    # L2-normalizes the output.
    if meta.get("hierarchical") and meta.get("n_scales", 1) > 1:
        inference_model = _rebuild_hierarchical(model, meta)
    else:
        pre_norm = None
        for layer in reversed(model.layers):
            if isinstance(layer, layers.Dense):
                pre_norm = layer.output
                break
        if pre_norm is None:
            raise SystemExit("no Dense layer found to use as inference output")
        inference_model = Model(model.inputs, pre_norm, name=model.name + "_no_norm")
    print(f"inference inputs: {[(i.name, tuple(i.shape)) for i in inference_model.inputs]}")
    print(f"inference output shape: {inference_model.outputs[0].shape}")

    print(f"writing tfjs shards to {dst}")
    save_keras_model(inference_model, str(dst))

    # tfjs-layers 4.x expects the Keras-2 model.json schema. Keras-3 introduced
    # `batch_shape` (was `batch_input_shape`) and turned `dtype` into a
    # DTypePolicy object. Rewrite both back to the older shape so that
    # `tf.loadLayersModel` accepts the file. (No layer ops change; only the
    # config metadata around them.)
    _patch_for_tfjs_layers(dst / "model.json")

    # Tag the metadata so the TS wrapper knows to renormalize and (for
    # hierarchical models) to split the patch into per-scale inputs.
    meta["output_post_normalize"] = True
    if meta.get("hierarchical") and meta.get("n_scales", 1) > 1:
        meta["input_order"] = [
            *(f"patch_s{s}" for s in range(meta["n_scales"]))
        ] + ["pos", "target"]
        meta["per_scale_inputs"] = True
    (dst / "metadata.json").write_text(json.dumps(meta, indent=2))
    print("done")
    print("  ->", dst / "model.json")
    print("  ->", dst / "metadata.json")


def _rebuild_hierarchical(model, meta) -> Model:
    """For hierarchical models, replace the slice-Lambdas with per-scale Inputs.

    The training graph splits a single [ps,ps,ps,n_scales] patch input into
    n_scales single-channel branches via Lambda layers (`t[..., s:s+1]`).
    tfjs-layers can't deserialize those, so we rebuild an inference graph
    with n_scales separate [ps,ps,ps,1] Inputs feeding the same per-branch
    conv stacks and the same head, reusing all trained weights.
    """
    ps = meta["patch_size"]
    n_scales = meta["n_scales"]
    n_lm = meta["n_landmarks"]

    # Identify the slice-Lambdas: Lambda layers whose only inbound is the
    # `patch` Input. There should be exactly n_scales of them.
    patch_layer = model.get_layer("patch")
    slice_lambdas = []
    for layer in model.layers:
        if not isinstance(layer, layers.Lambda):
            continue
        # Inbound layers from the first node.
        inbound = layer._inbound_nodes[0].inbound_layers
        if not isinstance(inbound, list):
            inbound = [inbound]
        if len(inbound) == 1 and inbound[0] is patch_layer:
            slice_lambdas.append(layer)
    if len(slice_lambdas) != n_scales:
        raise SystemExit(
            f"expected {n_scales} slice-Lambdas after patch input, found {len(slice_lambdas)}"
        )

    # For each slice-Lambda, walk the chain of layers it feeds (skipping
    # Lambdas) until we hit the merge Concatenate (the one that combines
    # all branch features). Replay each chain on a fresh per-scale Input,
    # so weights are reused but the input topology is rewired.
    new_inputs = [
        keras.Input((ps, ps, ps, 1), name=f"patch_s{s}") for s in range(n_scales)
    ]
    branch_outs = []
    merge_concat = None
    for s, lamb in enumerate(slice_lambdas):
        x = new_inputs[s]
        cur_layer = lamb
        while True:
            outbound = cur_layer._outbound_nodes
            if not outbound:
                raise SystemExit(f"branch {s} dead-ends without reaching the merge concat")
            next_layer = outbound[0].outbound_layer
            if isinstance(next_layer, layers.Concatenate):
                merge_concat = next_layer
                break
            x = next_layer(x)
            cur_layer = next_layer
        branch_outs.append(x)

    if merge_concat is None:
        raise SystemExit("did not find a Concatenate merging branch features")

    feat = layers.Concatenate(name=merge_concat.name + "_re")(branch_outs)

    # Walk from the merge Concatenate forward through the head (Concatenate
    # with pos+target inputs, then the Dense stack), replacing the original
    # patch-side input to the post-merge Concatenate with our `feat`. Stop
    # at the last Dense (drops the trailing L2-normalize Lambda).
    pos_in = keras.Input((3,), name="pos")
    target_in = keras.Input((n_lm,), name="target")

    # Find the post-merge Concatenate that joins (feat, pos, target).
    post_merge = None
    for layer in model.layers:
        if not isinstance(layer, layers.Concatenate) or layer is merge_concat:
            continue
        inbound = layer._inbound_nodes[0].inbound_layers
        if not isinstance(inbound, list):
            inbound = [inbound]
        names = {l.name for l in inbound}
        if "pos" in names and "target" in names:
            post_merge = layer
            break
    if post_merge is None:
        raise SystemExit("did not find the (feat, pos, target) Concatenate")

    head_in = layers.Concatenate(name=post_merge.name + "_re")([feat, pos_in, target_in])

    # Walk from post_merge through the head, stopping at the last Dense.
    cur_layer = post_merge
    x = head_in
    last_dense_out = None
    while True:
        outbound = cur_layer._outbound_nodes
        if not outbound:
            break
        next_layer = outbound[0].outbound_layer
        if isinstance(next_layer, layers.Lambda):
            break  # trailing L2-normalize
        x = next_layer(x)
        if isinstance(next_layer, layers.Dense):
            last_dense_out = x
        cur_layer = next_layer
    if last_dense_out is None:
        raise SystemExit("no Dense layer reachable from the head merge")

    return Model(new_inputs + [pos_in, target_in], last_dense_out, name=model.name + "_inf")


def _patch_for_tfjs_layers(model_json_path: Path) -> None:
    """Translate Keras-3 model.json to the Keras-2 schema tfjs-layers expects."""
    spec = json.loads(model_json_path.read_text())
    layers = spec["modelTopology"]["model_config"]["config"]["layers"]
    for layer in layers:
        cfg = layer.get("config", {})
        # batch_shape -> batchInputShape (only on InputLayer)
        if layer.get("class_name") == "InputLayer" and "batch_shape" in cfg:
            cfg["batchInputShape"] = cfg.pop("batch_shape")
        # dtype objects -> plain string
        if isinstance(cfg.get("dtype"), dict):
            cfg["dtype"] = cfg["dtype"].get("config", {}).get("name", "float32")
        # Drop a couple of fields tfjs-layers warns on but doesn't need.
        cfg.pop("ragged", None)
        cfg.pop("sparse", None)
        cfg.pop("optional", None)
        # Drop kwargs only present in the Keras-3 schema.
        for k in ("synchronized", "registered_name"):
            cfg.pop(k, None)
        # Some initializers also became DTypePolicy-like objects; flatten.
        for ikey in ("kernel_initializer", "bias_initializer"):
            v = cfg.get(ikey)
            if isinstance(v, dict) and "module" in v and "class_name" in v:
                cfg[ikey] = {"class_name": v["class_name"], "config": v.get("config", {})}
    model_json_path.write_text(json.dumps(spec, indent=2))


if __name__ == "__main__":
    main()
