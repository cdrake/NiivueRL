import * as tf from '@tensorflow/tfjs';
import { NIFTI1 } from 'nifti-reader-js';
import { loadBcmodel, transposeConvKernel } from './loadBcmodel';
import type { BcmodelFile } from './loadBcmodel';

type TypedArray = Float32Array | Int16Array | Uint8Array | Float64Array | Int32Array | Uint16Array;

/**
 * Quantile-normalize a volume: clamp to [2nd, 98th] percentile, scale to [0, 1].
 */
function quantileNormalize(data: TypedArray): Float32Array {
  // Sample ~100k random voxels for percentile estimation
  const sampleSize = Math.min(100_000, data.length);
  const sample = new Float32Array(sampleSize);
  for (let i = 0; i < sampleSize; i++) {
    sample[i] = data[Math.floor(Math.random() * data.length)];
  }
  sample.sort();

  const qLow = sample[Math.floor(sampleSize * 0.02)];
  const qHigh = sample[Math.floor(sampleSize * 0.98)];
  const range = qHigh - qLow || 1;

  const result = new Float32Array(data.length);
  for (let i = 0; i < data.length; i++) {
    result[i] = Math.min(1, Math.max(0, (data[i] - qLow) / range));
  }
  return result;
}

/**
 * Build a TF.js MeshNet model from bcmodel weights.
 * Architecture: 10 Conv3D layers, dilations [1,2,4,8,16,8,4,2,1,1],
 * first 9 have 30 filters + ELU, last has 18 filters + linear (1x1x1 conv).
 */
function buildMeshNet(bcmodel: BcmodelFile): tf.LayersModel {
  const dilations: number[] = [1, 2, 4, 8, 16, 8, 4, 2, 1];
  const input = tf.input({ shape: [256, 256, 256, 1] });

  let x: tf.SymbolicTensor = input;

  for (let i = 0; i < 10; i++) {
    const isLast = i === 9;
    const filters = isLast ? 18 : 30;
    const kernelSize = isLast ? 1 : 3;
    const dilation = isLast ? 1 : dilations[i];
    const activation = isLast ? 'linear' : 'elu';

    const wKey = `conv${i}.weight`;
    const bKey = `conv${i}.bias`;
    const wTensor = bcmodel.tensors[wKey];
    const bTensor = bcmodel.tensors[bKey];

    const transposed = transposeConvKernel(wTensor.data, wTensor.shape);

    const layer = tf.layers.conv3d({
      filters,
      kernelSize,
      dilationRate: dilation,
      padding: 'same',
      activation,
      name: `conv${i}`,
    });

    x = layer.apply(x) as tf.SymbolicTensor;

    const wTf = tf.tensor(transposed.data, transposed.shape as [number, number, number, number, number]);
    const bTf = tf.tensor(bTensor.data, [bTensor.shape[0]]);
    layer.setWeights([wTf, bTf]);
  }

  return tf.model({ inputs: input, outputs: x });
}

/**
 * Run MeshNet parcellation on a brain volume.
 * Returns a Uint8Array of label indices (argmax of 18-class output).
 */
export async function runParcellation(
  volumeImg: TypedArray,
  dims: [number, number, number],
): Promise<Uint8Array> {
  const bcmodel = await loadBcmodel(`${import.meta.env.BASE_URL}subcortical.bcmodel`);
  const normalized = quantileNormalize(volumeImg);

  const model = buildMeshNet(bcmodel);

  const inputTensor = tf.tensor5d(normalized, [1, dims[0], dims[1], dims[2], 1]);
  const output = model.predict(inputTensor) as tf.Tensor;
  const labels = tf.tidy(() => output.argMax(-1).squeeze().cast('int32'));

  const labelData = await labels.data();
  const result = new Uint8Array(labelData.length);
  for (let i = 0; i < labelData.length; i++) {
    result[i] = labelData[i];
  }

  // Cleanup
  inputTensor.dispose();
  output.dispose();
  labels.dispose();
  model.dispose();

  return result;
}

/**
 * Create a NIfTI-1 ArrayBuffer from label data, using a reference header for spatial info.
 */
export function createParcellationNifti(
  labels: Uint8Array,
  referenceHdr: NIFTI1,
): ArrayBuffer {
  const hdr = new NIFTI1();

  // Copy spatial metadata from reference
  hdr.dims = [...referenceHdr.dims];
  hdr.pixDims = [...referenceHdr.pixDims];
  hdr.affine = referenceHdr.affine.map((row) => [...row]);
  hdr.sform_code = referenceHdr.sform_code;
  hdr.qform_code = referenceHdr.qform_code;
  hdr.quatern_b = referenceHdr.quatern_b;
  hdr.quatern_c = referenceHdr.quatern_c;
  hdr.quatern_d = referenceHdr.quatern_d;
  hdr.qoffset_x = referenceHdr.qoffset_x;
  hdr.qoffset_y = referenceHdr.qoffset_y;
  hdr.qoffset_z = referenceHdr.qoffset_z;
  hdr.qfac = referenceHdr.qfac;
  hdr.xyzt_units = referenceHdr.xyzt_units;

  // Set label-specific fields
  hdr.datatypeCode = NIFTI1.TYPE_UINT8;
  hdr.numBitsPerVoxel = 8;
  hdr.scl_slope = 1;
  hdr.scl_inter = 0;
  hdr.vox_offset = 352; // Standard NIfTI-1 header (348) + 4 extension bytes
  hdr.magic = 'n+1';
  hdr.littleEndian = true;
  hdr.description = 'MeshNet parcellation';

  const headerBuf = hdr.toArrayBuffer();

  // Build final buffer: header + 4 extension bytes + label data
  const totalSize = 352 + labels.byteLength;
  const result = new ArrayBuffer(totalSize);
  const resultView = new Uint8Array(result);

  // Copy header (348 bytes)
  resultView.set(new Uint8Array(headerBuf).subarray(0, 348), 0);

  // Extension flag bytes (4 bytes, all zero = no extensions)
  // Already zero from ArrayBuffer initialization

  // Copy label data
  resultView.set(labels, 352);

  return result;
}
