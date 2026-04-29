import * as tf from '@tensorflow/tfjs';
import type { Vec3 } from '../env/types';

export interface GoalVectorMetadata {
  patch_radius: number;
  patch_size: number;
  strides: number[];
  n_scales: number;
  use_patch: boolean;
  hierarchical: boolean;
  landmark_names: string[];
  n_landmarks: number;
  input_order: string[];
  patch_axes: ['x', 'y', 'z'];
  position_normalization: string;
  best_val_cos: number;
  epochs: number;
  n_train_subjects: number;
  n_val_subjects: number;
  /**
   * If true, the converted tfjs model has had its trailing L2-normalize
   * Lambda stripped (tfjs-layers can't deserialize Python Lambda functions),
   * so this wrapper must L2-normalize the output itself before returning.
   */
  output_post_normalize?: boolean;
  /**
   * If true, the converted model takes n_scales separate single-channel
   * patch inputs (`patch_s0`, `patch_s1`, ..., `patch_s{n_scales-1}`)
   * instead of one packed `[patch_size, patch_size, patch_size, n_scales]`
   * tensor. Hierarchical models go this route because their interior
   * slice-Lambdas can't be deserialized by tfjs-layers.
   */
  per_scale_inputs?: boolean;
}

/**
 * Wraps the TF.js-converted goal-vector network. Loads weights from a
 * tfjs Layers model directory (model.json + shards) plus a metadata.json
 * describing patch shape, strides, and landmark order.
 *
 * The Python model was trained with patch tensors indexed brain[ix, iy, iz]
 * (axis 0 = x). The TS BrainEnv loops dz/dy/dx, so we cannot reuse its flat
 * neighborhood; this class extracts its own patch in (x, y, z) order from the
 * raw volume.
 */
export class GoalVectorModel {
  private readonly model: tf.LayersModel;
  readonly metadata: GoalVectorMetadata;
  private readonly nameToIndex: Map<string, number>;

  private constructor(
    model: tf.LayersModel,
    metadata: GoalVectorMetadata,
    nameToIndex: Map<string, number>,
  ) {
    this.model = model;
    this.metadata = metadata;
    this.nameToIndex = nameToIndex;
  }

  static async load(baseUrl: string): Promise<GoalVectorModel> {
    const metaResp = await fetch(`${baseUrl}/metadata.json`);
    if (!metaResp.ok) {
      throw new Error(`failed to fetch ${baseUrl}/metadata.json: ${metaResp.statusText}`);
    }
    const metadata = (await metaResp.json()) as GoalVectorMetadata;
    const model = await tf.loadLayersModel(`${baseUrl}/model.json`);
    const nameToIndex = new Map(metadata.landmark_names.map((n, i) => [n, i] as const));
    return new GoalVectorModel(model, metadata, nameToIndex);
  }

  /** Look up a landmark's one-hot index by name; returns -1 if unknown. */
  landmarkIndex(name: string): number {
    return this.nameToIndex.get(name) ?? -1;
  }

  /**
   * Predict a unit direction vector toward the target landmark from the
   * patch at `position` in `volumeData`. Caller normalizes voxel intensities
   * so that brain interior maps to roughly [0, 1] (matches training).
   */
  predict(
    volumeData: ArrayLike<number>,
    dims: [number, number, number],
    voxelMin: number,
    voxelMax: number,
    position: Vec3,
    targetIndex: number,
  ): [number, number, number] {
    const { patch_radius, patch_size, strides, n_scales, n_landmarks, per_scale_inputs } = this.metadata;
    const r = patch_radius;
    const ps = patch_size;
    const denom = voxelMax === voxelMin ? 1 : voxelMax - voxelMin;
    const [W, H, D] = dims;

    // Build per-scale patches in (x, y, z) order — axis 0 = x. Hierarchical
    // models consume them as separate single-channel inputs; non-hierarchical
    // models consume them packed into a single n_scales-channel tensor.
    const perScaleData: Float32Array[] = strides.map(() => new Float32Array(ps * ps * ps));
    for (let c = 0; c < n_scales; c++) {
      const s = strides[c];
      let i = 0;
      const buf = perScaleData[c];
      for (let ix = -r; ix <= r; ix++) {
        for (let iy = -r; iy <= r; iy++) {
          for (let iz = -r; iz <= r; iz++) {
            const x = Math.min(Math.max(0, position.x + ix * s), W - 1);
            const y = Math.min(Math.max(0, position.y + iy * s), H - 1);
            const z = Math.min(Math.max(0, position.z + iz * s), D - 1);
            const flat = x + y * W + z * W * H;
            buf[i++] = (volumeData[flat] - voxelMin) / denom;
          }
        }
      }
    }

    const posData = new Float32Array([
      (2 * position.x) / W - 1,
      (2 * position.y) / H - 1,
      (2 * position.z) / D - 1,
    ]);
    const targetData = new Float32Array(n_landmarks);
    if (targetIndex >= 0 && targetIndex < n_landmarks) {
      targetData[targetIndex] = 1;
    }

    const result = tf.tidy(() => {
      const posT = tf.tensor(posData, [1, 3]);
      const targetT = tf.tensor(targetData, [1, n_landmarks]);
      let inputs: tf.Tensor[];
      if (per_scale_inputs) {
        const patches = perScaleData.map((d) => tf.tensor(d, [1, ps, ps, ps, 1]));
        inputs = [...patches, posT, targetT];
      } else {
        // Pack the per-scale buffers into one [1, ps, ps, ps, n_scales] tensor
        // in (x, y, z, c) layout — interleave the channels per voxel.
        const packed = new Float32Array(ps * ps * ps * n_scales);
        for (let i = 0; i < ps * ps * ps; i++) {
          for (let c = 0; c < n_scales; c++) {
            packed[i * n_scales + c] = perScaleData[c][i];
          }
        }
        inputs = [tf.tensor(packed, [1, ps, ps, ps, n_scales]), posT, targetT];
      }
      const out = this.model.predict(inputs) as tf.Tensor;
      return out.dataSync();
    });

    let dx = result[0], dy = result[1], dz = result[2];
    if (this.metadata.output_post_normalize) {
      const n = Math.hypot(dx, dy, dz) || 1;
      dx /= n;
      dy /= n;
      dz /= n;
    }
    return [dx, dy, dz];
  }

  dispose(): void {
    this.model.dispose();
  }
}
