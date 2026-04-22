import * as tf from '@tensorflow/tfjs';
import { NUM_ACTIONS } from '../env/types';
import type { State } from '../env/types';
import { STATE_DIM, NEIGHBORHOOD_SIZE, STRIDES } from '../env/BrainEnv';
import type { Agent, AgentActionResult } from './types';
import { Trajectory } from './Trajectory';
import type { TrajectoryStep } from './Trajectory';
import { createSplitRLNetwork, createFlatRLNetwork, createConvRLNetwork } from './RLNetwork';
import type { SplitRLNetwork, FlatRLNetwork, ConvRLNetwork } from './RLNetwork';

export type PPOTrunk = 'flat' | 'meshnet' | 'conv';

export interface PPOConfig {
  lr: number;
  entropyCoeff: number;
  valueLossCoeff: number;
  maxGradNorm: number;
  useConvBackbone: boolean;
  /** Network trunk: 'flat' = dense MLP on 346/3378-dim state;
   *  'meshnet' = frozen MeshNet conv backbone (needs 7^3);
   *  'conv' = trainable 3D conv trunk sized to the current neighborhood. */
  trunk?: PPOTrunk;
  clipEpsilon: number;
  numEpochs: number;
  minibatchSize: number;
  /** Number of episodes to accumulate before running a PPO update. */
  rolloutSize: number;
}

export const DEFAULT_PPO_CONFIG: PPOConfig = {
  lr: 0.0003,
  entropyCoeff: 0.01,
  valueLossCoeff: 0.5,
  maxGradNorm: 0.5,
  useConvBackbone: false,
  trunk: 'flat',
  clipEpsilon: 0.2,
  numEpochs: 4,
  minibatchSize: 64,
  rolloutSize: 4,
};

export class PPOAgent implements Agent {
  private splitNet: SplitRLNetwork | null = null;
  private flatNet: FlatRLNetwork | null = null;
  private convNet: ConvRLNetwork | null = null;
  private config: PPOConfig;
  private optimizer: tf.Optimizer;
  private gamma: number;
  private lambda: number;

  epsilon: number;
  private epsilonMin: number;
  private epsilonDecay: number;

  trajectory: Trajectory;

  /** Trajectories collected since last PPO update. Flushed every rolloutSize episodes. */
  private rolloutBuffer: Trajectory[] = [];

  private constructor(config: PPOConfig) {
    this.config = config;
    this.optimizer = tf.train.adam(config.lr);
    this.gamma = 0.99;
    this.lambda = 0.95;

    this.epsilon = 0;
    this.epsilonMin = 0;
    this.epsilonDecay = 1;

    this.trajectory = new Trajectory();
  }

  static async create(config?: Partial<PPOConfig>): Promise<PPOAgent> {
    const fullConfig = { ...DEFAULT_PPO_CONFIG, ...config };
    // Resolve trunk with back-compat for legacy useConvBackbone flag.
    const trunk: PPOTrunk =
      fullConfig.trunk ?? (fullConfig.useConvBackbone ? 'meshnet' : 'flat');
    fullConfig.trunk = trunk;
    const agent = new PPOAgent(fullConfig);

    const numScales = STRIDES.length;

    if (trunk === 'meshnet') {
      agent.splitNet = await createSplitRLNetwork(numScales);
    } else if (trunk === 'conv') {
      agent.convNet = createConvRLNetwork(NEIGHBORHOOD_SIZE, numScales);
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
    if (this.flatNet) return this.selectActionFlat(state);
    if (this.convNet) return this.selectActionTrainableConv(state);
    return this.selectActionConv(state);
  }

  private selectActionTrainableConv(state: State): AgentActionResult {
    return tf.tidy(() => {
      const size = this.convNet!.neighborhoodSize;
      const ns = this.convNet!.numScales;
      const nbr = tf.tensor(Array.from(state.neighborhood), [1, size, size, size, ns]);
      const dir = tf.tensor2d([state.direction], [1, 3]);
      const [policyTensor, valueTensor] = this.convNet!.model.predict([nbr, dir]) as tf.Tensor[];
      const probs = policyTensor.dataSync() as Float32Array;
      const value = (valueTensor.dataSync() as Float32Array)[0];
      const action = this.sampleCategorical(probs);
      const logProb = Math.log(probs[action] + 1e-8);
      return { action, extra: { value, logProb } };
    });
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
      const ns = this.splitNet!.numScales;
      const neighborhood = tf.tensor(
        Array.from(state.neighborhood),
        [1, 7, 7, 7, ns],
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
    const incoming = data as Trajectory;
    if (incoming.length === 0) return 0;

    // Accumulate trajectories until the rollout is full. The caller creates a
    // fresh Trajectory each episode, so it's safe to hold references to its steps.
    this.rolloutBuffer.push(incoming);
    if (this.rolloutBuffer.length < this.config.rolloutSize) {
      return 0;
    }

    // Flatten the rollout into pooled arrays. Advantages and returns are computed
    // per-trajectory so GAE doesn't cross episode boundaries, then concatenated.
    const pooledSteps: TrajectoryStep[] = [];
    let totalLen = 0;
    for (const traj of this.rolloutBuffer) totalLen += traj.length;

    const pooledAdvantages = new Float32Array(totalLen);
    const pooledReturns = new Float32Array(totalLen);
    const pooledOldLogProbs = new Float32Array(totalLen);

    let offset = 0;
    for (const traj of this.rolloutBuffer) {
      const adv = traj.computeAdvantages(this.gamma, this.lambda);
      const ret = traj.computeReturns(this.gamma);
      for (let i = 0; i < traj.length; i++) {
        pooledSteps.push(traj.steps[i]);
        pooledAdvantages[offset + i] = adv[i];
        pooledReturns[offset + i] = ret[i];
        pooledOldLogProbs[offset + i] = traj.steps[i].logProb;
      }
      offset += traj.length;
    }

    // Normalize advantages across the whole rollout (not per-trajectory — that's the
    // whole point of accumulating: more stable advantage statistics).
    this.normalizeAdvantages(pooledAdvantages);

    const n = pooledSteps.length;

    // Precompute conv features once for the whole rollout (frozen backbone).
    let featuresTensor: tf.Tensor | null = null;
    if (this.splitNet) {
      featuresTensor = tf.tidy(() => {
        const ns = this.splitNet!.numScales;
        const neighborhoods = tf.tensor(
          pooledSteps.map((s) => Array.from(s.neighborhood)),
          [n, 7, 7, 7, ns],
        );
        return this.extractFeatures(neighborhoods);
      });
    }

    let totalLoss = 0;
    let numUpdates = 0;

    for (let epoch = 0; epoch < this.config.numEpochs; epoch++) {
      const indices = this.shuffledIndices(n);
      for (let start = 0; start < n; start += this.config.minibatchSize) {
        const end = Math.min(start + this.config.minibatchSize, n);
        const batchIdx = indices.slice(start, end);

        const loss = this.optimizer.minimize(() => {
          return this.computeMinibatchLoss(
            pooledSteps,
            batchIdx,
            pooledAdvantages,
            pooledReturns,
            pooledOldLogProbs,
            featuresTensor,
          );
        }, true) as tf.Scalar | null;

        if (loss) {
          totalLoss += loss.dataSync()[0];
          numUpdates++;
          loss.dispose();
        }
      }
    }

    featuresTensor?.dispose();
    this.rolloutBuffer = [];
    return numUpdates > 0 ? totalLoss / numUpdates : 0;
  }

  private computeMinibatchLoss(
    pooledSteps: TrajectoryStep[],
    batchIdx: number[],
    advantages: Float32Array,
    returns: Float32Array,
    oldLogProbs: Float32Array,
    featuresTensor: tf.Tensor | null,
  ): tf.Scalar {
    const bs = batchIdx.length;

    // Forward pass on this minibatch
    let policyTensor: tf.Tensor;
    let valueTensor: tf.Tensor;

    if (this.flatNet) {
      const states = tf.tensor2d(
        batchIdx.map((i) => {
          const step = pooledSteps[i];
          const arr = new Float32Array(STATE_DIM);
          arr.set(step.neighborhood);
          arr[step.neighborhood.length] = step.direction[0];
          arr[step.neighborhood.length + 1] = step.direction[1];
          arr[step.neighborhood.length + 2] = step.direction[2];
          return Array.from(arr);
        }),
        [bs, STATE_DIM],
      );
      const out = this.flatNet.model.predict(states) as tf.Tensor[];
      policyTensor = out[0];
      valueTensor = out[1];
    } else if (this.convNet) {
      const size = this.convNet.neighborhoodSize;
      const ns = this.convNet.numScales;
      const flat = new Float32Array(bs * size * size * size * ns);
      for (let b = 0; b < bs; b++) {
        flat.set(pooledSteps[batchIdx[b]].neighborhood, b * size * size * size * ns);
      }
      const neighborhoods = tf.tensor(flat, [bs, size, size, size, ns]);
      const directions = tf.tensor2d(
        batchIdx.map((i) => pooledSteps[i].direction),
        [bs, 3],
      );
      const out = this.convNet.model.predict([neighborhoods, directions]) as tf.Tensor[];
      policyTensor = out[0];
      valueTensor = out[1];
    } else {
      const gatherIdx = tf.tensor1d(batchIdx, 'int32');
      const batchFeatures = featuresTensor!.gather(gatherIdx);
      gatherIdx.dispose();
      const directions = tf.tensor2d(
        batchIdx.map((i) => pooledSteps[i].direction),
        [bs, 3],
      );
      const out = this.splitNet!.headModel.predict([batchFeatures, directions]) as tf.Tensor[];
      policyTensor = out[0];
      valueTensor = out[1];
    }

    // Value loss (MSE vs. returns)
    const values = valueTensor.squeeze();
    const batchReturns = tf.tensor1d(batchIdx.map((i) => returns[i]));
    const valueLoss = batchReturns.sub(values).square().mean();

    // Policy loss — PPO clipped surrogate
    const batchAdv = tf.tensor1d(batchIdx.map((i) => advantages[i]));
    const batchOldLogProbs = tf.tensor1d(batchIdx.map((i) => oldLogProbs[i]));
    const actions = batchIdx.map((i) => pooledSteps[i].action);
    const actionMask = tf.oneHot(tf.tensor1d(actions, 'int32'), NUM_ACTIONS);
    const selectedProbs = policyTensor.mul(actionMask).sum(1);
    const newLogProbs = selectedProbs.add(1e-8).log();

    const ratio = newLogProbs.sub(batchOldLogProbs).exp();
    const clipLow = 1 - this.config.clipEpsilon;
    const clipHigh = 1 + this.config.clipEpsilon;
    const unclipped = ratio.mul(batchAdv);
    const clipped = ratio.clipByValue(clipLow, clipHigh).mul(batchAdv);
    const policyLoss = tf.minimum(unclipped, clipped).mean().neg();

    // Entropy bonus
    const entropy = policyTensor.add(1e-8).log().mul(policyTensor).sum(1).mean().neg();

    return policyLoss
      .add(valueLoss.mul(this.config.valueLossCoeff))
      .sub(entropy.mul(this.config.entropyCoeff)) as tf.Scalar;
  }

  private shuffledIndices(n: number): number[] {
    const arr = Array.from({ length: n }, (_, i) => i);
    for (let i = n - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [arr[i], arr[j]] = [arr[j], arr[i]];
    }
    return arr;
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
    this.convNet?.model.dispose();
    this.optimizer.dispose();
    this.rolloutBuffer = [];
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
