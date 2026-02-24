import * as tf from '@tensorflow/tfjs';
import { STATE_DIM } from '../env/BrainEnv';
import { NUM_ACTIONS } from '../env/types';

export function createQNetwork(): tf.Sequential {
  const model = tf.sequential();
  model.add(tf.layers.dense({ inputShape: [STATE_DIM], units: 256, activation: 'relu' }));
  model.add(tf.layers.dense({ units: 128, activation: 'relu' }));
  model.add(tf.layers.dense({ units: 64, activation: 'relu' }));
  model.add(tf.layers.dense({ units: NUM_ACTIONS, activation: 'linear' }));

  model.compile({
    optimizer: tf.train.adam(0.001),
    loss: 'meanSquaredError',
  });

  return model;
}

export function copyWeights(source: tf.Sequential, target: tf.Sequential): void {
  const sourceWeights = source.getWeights();
  target.setWeights(sourceWeights);
}
