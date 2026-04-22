import * as tf from '@tensorflow/tfjs';
import { loadBcmodel, transposeConvKernel } from '../lib/loadBcmodel';
import { NUM_ACTIONS } from '../env/types';
import { STATE_DIM } from '../env/BrainEnv';

export interface RLNetworkOutputs {
  policy: tf.Tensor;
  value: tf.Tensor;
}

/**
 * A2C network split into two parts:
 * - featureExtractor: frozen pretrained conv layers (run outside gradient tape)
 * - headModel: trainable dense + policy/value heads
 */
export interface SplitRLNetwork {
  featureExtractor: tf.LayersModel;
  headModel: tf.LayersModel;
  featureDim: number;
  numScales: number;
}

/** Flat A2C network — no conv backbone, same 346-dim input as DQN. */
export interface FlatRLNetwork {
  model: tf.LayersModel;
}

/**
 * Trainable end-to-end 3D conv actor-critic. Two inputs: neighborhood
 * cube (n,n,n,num_scales) and direction (3). Outputs: [policy(6), value(1)].
 */
export interface ConvRLNetwork {
  model: tf.LayersModel;
  neighborhoodSize: number;
  numScales: number;
}

export async function createSplitRLNetwork(numScales: number = 1): Promise<SplitRLNetwork> {
  // Load bcmodel and extract conv weights
  const bcmodel = await loadBcmodel('/subcortical.bcmodel');

  const conv0Weight = bcmodel.tensors['conv0.weight'];
  const conv0Bias = bcmodel.tensors['conv0.bias'];
  const conv1Weight = bcmodel.tensors['conv1.weight'];
  const conv1Bias = bcmodel.tensors['conv1.bias'];

  if (!conv0Weight || !conv0Bias || !conv1Weight || !conv1Bias) {
    throw new Error('Missing conv0/conv1 tensors in bcmodel');
  }

  // Transpose from [O,I,D,H,W] to [D,H,W,I,O] for TF.js channels_last
  const conv0WeightT = transposeConvKernel(conv0Weight.data, conv0Weight.shape);
  const conv1WeightT = transposeConvKernel(conv1Weight.data, conv1Weight.shape);

  // --- Feature extractor (frozen) ---
  const neighborhoodInput = tf.input({ shape: [7, 7, 7, numScales], name: 'neighborhood' });

  const conv0 = tf.layers.conv3d({
    filters: conv0WeightT.shape[4],
    kernelSize: [conv0WeightT.shape[0], conv0WeightT.shape[1], conv0WeightT.shape[2]],
    padding: 'same',
    dilationRate: 1,
    activation: 'elu',
    trainable: false,
    name: 'conv0',
  }).apply(neighborhoodInput) as tf.SymbolicTensor;

  const conv1 = tf.layers.conv3d({
    filters: conv1WeightT.shape[4],
    kernelSize: [conv1WeightT.shape[0], conv1WeightT.shape[1], conv1WeightT.shape[2]],
    padding: 'same',
    dilationRate: 2,
    activation: 'elu',
    trainable: false,
    name: 'conv1',
  }).apply(conv0) as tf.SymbolicTensor;

  const flat = tf.layers.flatten().apply(conv1) as tf.SymbolicTensor;

  const featureExtractor = tf.model({
    inputs: neighborhoodInput,
    outputs: flat,
    name: 'feature_extractor',
  });

  // Set pretrained weights for conv0
  // If numScales > 1, first channel gets weights, others are zeroed.
  const c0w = tf.tensor(conv0WeightT.data, conv0WeightT.shape);
  const c0b = tf.tensor(conv0Bias.data, conv0Bias.shape);

  let effectiveC0w = c0w;
  if (numScales > 1) {
    const shape = [...conv0WeightT.shape];
    shape[3] = numScales;
    const zeros = tf.zeros(shape);
    // Slice first channel
    const slice = zeros.slice([0, 0, 0, 1, 0], [3, 3, 3, numScales - 1, 16]);
    effectiveC0w = tf.concat([c0w, slice], 3);
  }

  featureExtractor.getLayer('conv0').setWeights([effectiveC0w, c0b]);
  featureExtractor.getLayer('conv1').setWeights([
    tf.tensor(conv1WeightT.data, conv1WeightT.shape),
    tf.tensor(conv1Bias.data, conv1Bias.shape),
  ]);

  // Compute feature dim: 7*7*7 * filters
  const featureDim = 7 * 7 * 7 * conv1WeightT.shape[4];

  // --- Trainable head ---
  const featInput = tf.input({ shape: [featureDim], name: 'features' });
  const dirInput = tf.input({ shape: [3], name: 'direction' });

  const dense1 = tf.layers.dense({
    units: 128,
    activation: 'relu',
    name: 'dense1',
  }).apply(featInput) as tf.SymbolicTensor;

  const merged = tf.layers.concatenate().apply([dense1, dirInput]) as tf.SymbolicTensor;
  const dense2 = tf.layers.dense({
    units: 64,
    activation: 'relu',
    name: 'dense2',
  }).apply(merged) as tf.SymbolicTensor;

  const policyOutput = tf.layers.dense({
    units: NUM_ACTIONS,
    activation: 'softmax',
    name: 'policy',
  }).apply(dense2) as tf.SymbolicTensor;

  const valueOutput = tf.layers.dense({
    units: 1,
    activation: 'linear',
    name: 'value',
  }).apply(dense2) as tf.SymbolicTensor;

  const headModel = tf.model({
    inputs: [featInput, dirInput],
    outputs: [policyOutput, valueOutput],
    name: 'head_model',
  });

  return { featureExtractor, headModel, featureDim, numScales };
}

export function createConvRLNetwork(size: number, numScales: number = 1): ConvRLNetwork {
  const nbrInput = tf.input({ shape: [size, size, size, numScales], name: 'neighborhood' });
  const dirInput = tf.input({ shape: [3], name: 'direction' });

  let x = tf.layers.conv3d({
    filters: 16, kernelSize: 3, padding: 'same', activation: 'relu', name: 'tconv0',
  }).apply(nbrInput) as tf.SymbolicTensor;
  x = tf.layers.conv3d({
    filters: 16, kernelSize: 3, padding: 'same', activation: 'relu', name: 'tconv1',
  }).apply(x) as tf.SymbolicTensor;
  // Downsample once if there's room; for size<=5 keep full resolution.
  if (size >= 6) {
    x = tf.layers.maxPooling3d({ poolSize: 2, strides: 2 }).apply(x) as tf.SymbolicTensor;
  }
  x = tf.layers.flatten().apply(x) as tf.SymbolicTensor;

  const merged = tf.layers.concatenate().apply([x, dirInput]) as tf.SymbolicTensor;
  let h = tf.layers.dense({ units: 128, activation: 'relu', name: 'tdense1' }).apply(merged) as tf.SymbolicTensor;
  h = tf.layers.dense({ units: 64, activation: 'relu', name: 'tdense2' }).apply(h) as tf.SymbolicTensor;

  const policyOutput = tf.layers.dense({
    units: NUM_ACTIONS, activation: 'softmax', name: 'policy',
  }).apply(h) as tf.SymbolicTensor;
  const valueOutput = tf.layers.dense({
    units: 1, activation: 'linear', name: 'value',
  }).apply(h) as tf.SymbolicTensor;

  const model = tf.model({
    inputs: [nbrInput, dirInput],
    outputs: [policyOutput, valueOutput],
    name: 'conv_ac',
  });
  return { model, neighborhoodSize: size, numScales };
}

export function createFlatRLNetwork(): FlatRLNetwork {
  const stateInput = tf.input({ shape: [STATE_DIM], name: 'state' });

  let x = tf.layers.dense({ units: 256, activation: 'relu', name: 'dense0' })
    .apply(stateInput) as tf.SymbolicTensor;
  x = tf.layers.dense({ units: 128, activation: 'relu', name: 'dense1' })
    .apply(x) as tf.SymbolicTensor;
  x = tf.layers.dense({ units: 64, activation: 'relu', name: 'dense2' })
    .apply(x) as tf.SymbolicTensor;

  const policyOutput = tf.layers.dense({
    units: NUM_ACTIONS,
    activation: 'softmax',
    name: 'policy',
  }).apply(x) as tf.SymbolicTensor;

  const valueOutput = tf.layers.dense({
    units: 1,
    activation: 'linear',
    name: 'value',
  }).apply(x) as tf.SymbolicTensor;

  const model = tf.model({
    inputs: stateInput,
    outputs: [policyOutput, valueOutput],
    name: 'flat_ac',
  });

  return { model };
}
