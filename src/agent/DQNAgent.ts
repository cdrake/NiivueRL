import * as tf from '@tensorflow/tfjs';
import { createQNetwork, copyWeights } from './network';
import { ReplayBuffer } from './ReplayBuffer';
import type { Experience } from './ReplayBuffer';
import { NUM_ACTIONS } from '../env/types';
import type { State } from '../env/types';
import { STATE_DIM } from '../env/BrainEnv';
import type { Agent, AgentActionResult } from './types';

export class DQNAgent implements Agent {
  private qNetwork: tf.Sequential;
  private targetNetwork: tf.Sequential;
  private replayBuffer: ReplayBuffer;
  private gamma: number;
  private batchSize: number;
  private targetUpdateFreq: number;
  private stepCount: number;

  epsilon: number;
  private epsilonMin: number;
  private epsilonDecay: number;

  constructor() {
    this.qNetwork = createQNetwork();
    this.targetNetwork = createQNetwork();
    copyWeights(this.qNetwork, this.targetNetwork);

    this.replayBuffer = new ReplayBuffer(10000);
    this.gamma = 0.99;
    this.batchSize = 64;
    this.targetUpdateFreq = 100;
    this.stepCount = 0;

    this.epsilon = 1.0;
    this.epsilonMin = 0.05;
    this.epsilonDecay = 0.995;
  }

  selectAction(state: State): AgentActionResult {
    const stateArray = this.stateToArray(state);
    if (Math.random() < this.epsilon) {
      return { action: Math.floor(Math.random() * NUM_ACTIONS) };
    }
    const action = tf.tidy(() => {
      const input = tf.tensor2d([Array.from(stateArray)], [1, STATE_DIM]);
      const qValues = this.qNetwork.predict(input) as tf.Tensor;
      return qValues.argMax(1).dataSync()[0];
    });
    return { action };
  }

  private stateToArray(state: State): Float32Array {
    const arr = new Float32Array(STATE_DIM);
    arr.set(state.neighborhood);
    arr[state.neighborhood.length] = state.direction[0];
    arr[state.neighborhood.length + 1] = state.direction[1];
    arr[state.neighborhood.length + 2] = state.direction[2];
    return arr;
  }

  store(experience: Experience): void {
    this.replayBuffer.push(experience);
  }

  async train(_data?: unknown): Promise<number> {
    if (this.replayBuffer.size < this.batchSize) return 0;

    const batch = this.replayBuffer.sample(this.batchSize);

    // Prepare training data inside tidy, but run trainOnBatch outside
    const { inputTensor, targetTensor } = tf.tidy(() => {
      const states = tf.tensor2d(batch.map(e => Array.from(e.state)), [this.batchSize, STATE_DIM]);
      const nextStates = tf.tensor2d(batch.map(e => Array.from(e.nextState)), [this.batchSize, STATE_DIM]);
      const actions = batch.map(e => e.action);
      const rewards = batch.map(e => e.reward);
      const dones = batch.map(e => e.done ? 1 : 0);

      // Compute target Q-values
      const nextQ = this.targetNetwork.predict(nextStates) as tf.Tensor;
      const maxNextQ = nextQ.max(1);
      const targetValues = tf.tensor1d(rewards).add(
        tf.tensor1d(dones).mul(-1).add(1).mul(maxNextQ).mul(this.gamma)
      );

      // Current Q-values
      const currentQ = this.qNetwork.predict(states) as tf.Tensor2D;
      const currentQArray = currentQ.arraySync() as number[][];

      // Update only the taken action's Q-value
      const targetArray = targetValues.arraySync() as number[];
      for (let i = 0; i < this.batchSize; i++) {
        currentQArray[i][actions[i]] = targetArray[i];
      }

      // Keep these tensors alive outside tidy
      const inputTensor = tf.tensor2d(batch.map(e => Array.from(e.state)), [this.batchSize, STATE_DIM]);
      const targetTensor = tf.tensor2d(currentQArray);
      return { inputTensor, targetTensor };
    });

    const lossVal = await this.qNetwork.trainOnBatch(inputTensor, targetTensor);
    inputTensor.dispose();
    targetTensor.dispose();

    this.stepCount++;
    if (this.stepCount % this.targetUpdateFreq === 0) {
      copyWeights(this.qNetwork, this.targetNetwork);
    }

    return typeof lossVal === 'number' ? lossVal : lossVal[0];
  }

  decayEpsilon(): void {
    this.epsilon = Math.max(this.epsilonMin, this.epsilon * this.epsilonDecay);
  }

  dispose(): void {
    this.qNetwork.dispose();
    this.targetNetwork.dispose();
  }
}
