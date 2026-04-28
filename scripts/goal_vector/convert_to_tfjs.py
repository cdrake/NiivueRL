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

    # tfjs-layers can't deserialize Python Lambda layers (they hold pickled
    # function bytecode). Build an inference-only model that ends at the last
    # Dense (the pre-normalization 3-vec); the TS wrapper normalizes itself.
    pre_norm = None
    for layer in reversed(model.layers):
        if isinstance(layer, layers.Dense):
            pre_norm = layer.output
            break
    if pre_norm is None:
        raise SystemExit("no Dense layer found to use as inference output")
    inference_model = Model(model.inputs, pre_norm, name=model.name + "_no_norm")
    print(f"stripped trailing Lambda; inference output: {pre_norm.shape}")

    print(f"writing tfjs shards to {dst}")
    save_keras_model(inference_model, str(dst))

    # tfjs-layers 4.x expects the Keras-2 model.json schema. Keras-3 introduced
    # `batch_shape` (was `batch_input_shape`) and turned `dtype` into a
    # DTypePolicy object. Rewrite both back to the older shape so that
    # `tf.loadLayersModel` accepts the file. (No layer ops change; only the
    # config metadata around them.)
    _patch_for_tfjs_layers(dst / "model.json")

    # Tag the metadata so the TS wrapper knows to renormalize.
    meta = json.loads(src_meta.read_text())
    meta["output_post_normalize"] = True
    (dst / "metadata.json").write_text(json.dumps(meta, indent=2))
    print("done")
    print("  ->", dst / "model.json")
    print("  ->", dst / "metadata.json")


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
