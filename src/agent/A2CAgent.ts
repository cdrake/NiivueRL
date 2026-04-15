import * as tf from '@tensorflow/tfjs';
import { NUM_ACTIONS } from '../env/types';
import type { State } from '../env/types';
import { STATE_DIM } from '../env/BrainEnv';
import type { Agent, AgentActionResult } from './types';
import { Trajectory } from './Trajectory';
import { createSplitRLNetwork, createFlatRLNetwork } from './RLNetwork';
import type { SplitRLNetwork, FlatRLNetwork } from './RLNetwork';

export interface A2CConfig {
  lr: number;
  entropyCoeff: number;
  valueLossCoeff: number;
  maxGradNorm: number;
  useConvBackbone: boolean;
}

export const DEFAULT_A2C_CONFIG: A2CConfig = {
  lr: 0.0003,
  entropyCoeff: 0.01,
  valueLossCoeff: 0.5,
  maxGradNorm: 0.5,
  useConvBackbone: false,
};

export class A2CAgent implements Agent {
  private splitNet: SplitRLNetwork | null = null;
  private flatNet: FlatRLNetwork | null = null;
  private config: A2CConfig;
  private optimizer: tf.Optimizer;
  private gamma: number;
  private lambda: number;

  epsilon: number;
  private epsilonMin: number;
  private epsilonDecay: number;

  trajectory: Trajectory;

  private constructor(config: A2CConfig) {
    this.config = config;
    this.optimizer = tf.train.adam(config.lr);
    this.gamma = 0.99;
    this.lambda = 0.95;

    this.epsilon = 0;
    this.epsilonMin = 0;
    this.epsilonDecay = 1;

    this.trajectory = new Trajectory();
  }

  static async create(config?: Partial<A2CConfig>): Promise<A2CAgent> {
    const fullConfig = { ...DEFAULT_A2C_CONFIG, ...config };
    const agent = new A2CAgent(fullConfig);

    if (fullConfig.useConvBackbone) {
      agent.splitNet = await createSplitRLNetwork();
    } else {
      agent.flatNet = createFlatRLNetwork();
    }

    return agent;
  }

  /** Run frozen conv feature extractor (outside gradient tape). */
  private extractFeatures(neighborhood: tf.Tensor): tf.Tensor {
    return this.splitNet!.featureExtractor.predict(neighborhood) as tf.Tensor;
  }

  selectAction(state: State): AgentActionResult {
    if (this.flatNet) {
      return this.selectActionFlat(state);
    }
    return this.selectActionConv(state);
  }

  private selectActionFlat(state: State): AgentActionResult {
    return tf.tidy(() => {
      const stateArray = new Float32Array(STATE_DIM);
      stateArray.set(state.neighborhood);
      stateArray[state.neighborhood.length] = state.direction[0];
      stateArray[state.neighborhood.length + 1] = state.direction[1];
      stateArray[state.neighborhood.length + 2] = state.direction[2];

      const input = tf.tensor2d([Array.from(stateArray)], [1, STATE_DIM]);
      const [policyTensor, valueTensor] = this.flatNet!.model.predict(input) as tf.Tensor[];

      const probs = policyTensor.dataSync() as Float32Array;
      const value = (valueTensor.dataSync() as Float32Array)[0];
      const action = this.sampleCategorical(probs);
      const logProb = Math.log(probs[action] + 1e-8);

      return { action, extra: { value, logProb } };
    });
  }

  private selectActionConv(state: State): AgentActionResult {
    const feats = tf.tidy(() => {
      const neighborhood = tf.tensor(
        Array.from(state.neighborhood),
        [1, 7, 7, 7, 1],
      );
      return this.extractFeatures(neighborhood);
    });

    const result = tf.tidy(() => {
      const direction = tf.tensor2d([state.direction], [1, 3]);
      const [policyTensor, valueTensor] = this.splitNet!.headModel.predict(
        [feats, direction],
      ) as tf.Tensor[];

      const probs = policyTensor.dataSync() as Float32Array;
      const value = (valueTensor.dataSync() as Float32Array)[0];

      const action = this.sampleCategorical(probs);
      const logProb = Math.log(probs[action] + 1e-8);

      return { action, extra: { value, logProb } };
    });

    feats.dispose();
    return result;
  }

  async train(data: unknown): Promise<number> {
    const trajectory = data as Trajectory;
    if (trajectory.length === 0) return 0;

    if (this.flatNet) {
      return this.trainFlat(trajectory);
    }
    return this.trainConv(trajectory);
  }

  private trainFlat(trajectory: Trajectory): number {
    const advantages = trajectory.computeAdvantages(this.gamma, this.lambda);
    const returns = trajectory.computeReturns(this.gamma);
    this.normalizeAdvantages(advantages);

    const n = trajectory.length;

    const loss = this.optimizer.minimize(() => {
      const states = tf.tensor2d(
        Array.from({ length: n }, (_, i) => {
          const step = trajectory.steps[i];
          const arr = new Float32Array(STATE_DIM);
          arr.set(step.neighborhood);
          arr[step.neighborhood.length] = step.direction[0];
          arr[step.neighborhood.length + 1] = step.direction[1];
          arr[step.neighborhood.length + 2] = step.direction[2];
          return Array.from(arr);
        }),
        [n, STATE_DIM],
      );

      const [policyTensor, valueTensor] = this.flatNet!.model.predict(states) as tf.Tensor[];
      return this.computeLoss(policyTensor, valueTensor, trajectory, advantages, returns);
    }, true) as tf.Scalar | null;

    const lossValue = loss ? loss.dataSync()[0] : 0;
    loss?.dispose();
    trajectory.clear();
    return lossValue;
  }

  private trainConv(trajectory: Trajectory): number {
    const advantages = trajectory.computeAdvantages(this.gamma, this.lambda);
    const returns = trajectory.computeReturns(this.gamma);
    this.normalizeAdvantages(advantages);

    const n = trajectory.length;

    const featuresTensor = tf.tidy(() => {
      const neighborhoods = tf.tensor(
        Array.from({ length: n }, (_, i) =>
          Array.from(trajectory.steps[i].neighborhood),
        ),
        [n, 7, 7, 7, 1],
      );
      return this.extractFeatures(neighborhoods);
    });

    const loss = this.optimizer.minimize(() => {
      const directions = tf.tensor2d(
        trajectory.steps.map((s) => s.direction),
        [n, 3],
      );

      const [policyTensor, valueTensor] = this.splitNet!.headModel.predict(
        [featuresTensor, directions],
      ) as tf.Tensor[];

      return this.computeLoss(policyTensor, valueTensor, trajectory, advantages, returns);
    }, true) as tf.Scalar | null;

    featuresTensor.dispose();

    const lossValue = loss ? loss.dataSync()[0] : 0;
    loss?.dispose();
    trajectory.clear();
    return lossValue;
  }

  private computeLoss(
    policyTensor: tf.Tensor,
    valueTensor: tf.Tensor,
    trajectory: Trajectory,
    advantages: Float32Array,
    returns: Float32Array,
  ): tf.Scalar {
    const values = valueTensor.squeeze();
    const returnsTensor = tf.tensor1d(Array.from(returns));

    // Value loss
    const valueLoss = returnsTensor.sub(values).square().mean();

    // Policy loss
    const advantagesTensor = tf.tensor1d(Array.from(advantages));
    const actions = trajectory.steps.map((s) => s.action);
    const actionMask = tf.oneHot(tf.tensor1d(actions, 'int32'), NUM_ACTIONS);
    const selectedProbs = policyTensor.mul(actionMask).sum(1);
    const logProbs = selectedProbs.add(1e-8).log();
    const policyLoss = logProbs.mul(advantagesTensor).mean().neg();

    // Entropy bonus
    const entropy = policyTensor.add(1e-8).log().mul(policyTensor).sum(1).mean().neg();

    return policyLoss
      .add(valueLoss.mul(this.config.valueLossCoeff))
      .sub(entropy.mul(this.config.entropyCoeff)) as tf.Scalar;
  }

  private normalizeAdvantages(advantages: Float32Array): void {
    let mean = 0;
    for (let i = 0; i < advantages.length; i++) mean += advantages[i];
    mean /= advantages.length;
    let std = 0;
    for (let i = 0; i < advantages.length; i++) std += (advantages[i] - mean) ** 2;
    std = Math.sqrt(std / advantages.length + 1e-8);
    for (let i = 0; i < advantages.length; i++) {
      advantages[i] = (advantages[i] - mean) / std;
    }
  }

  decayEpsilon(): void {
    this.epsilon = Math.max(this.epsilonMin, this.epsilon * this.epsilonDecay);
  }

  dispose(): void {
    this.splitNet?.featureExtractor.dispose();
    this.splitNet?.headModel.dispose();
    this.flatNet?.model.dispose();
    this.optimizer.dispose();
  }

  private sampleCategorical(probs: Float32Array | number[]): number {
    const r = Math.random();
    let cumulative = 0;
    for (let i = 0; i < probs.length; i++) {
      cumulative += probs[i];
      if (r < cumulative) return i;
    }
    return probs.length - 1;
  }
}
