import * as tf from '@tensorflow/tfjs';
import { BrainEnv, setObservationConfig, NEIGHBORHOOD_SIZE, STRIDES } from '../env/BrainEnv';
import type { BrainEnvConfig, StartDistanceCurriculum } from '../env/BrainEnv';
import type { GoalVectorModel } from '../lib/goalVectorModel';
import { DQNAgent } from '../agent/DQNAgent';
import { A2CAgent } from '../agent/A2CAgent';
import type { A2CConfig } from '../agent/A2CAgent';
import { PPOAgent } from '../agent/PPOAgent';
import type { PPOConfig } from '../agent/PPOAgent';
import { OracleAgent } from '../agent/OracleAgent';
import { RandomAgent } from '../agent/RandomAgent';
import { Trajectory } from '../agent/Trajectory';
import type { Action } from '../env/types';
import type { Agent } from '../agent/types';
import type { Landmark } from '../env/types';
import type { AgentType } from '../components/ControlPanel';

export interface EpisodeResult {
  episode: number;
  totalReward: number;
  finalDistance: number;
  steps: number;
  success: boolean;
  epsilon: number;
}

export interface ExperimentConfig {
  landmark: Landmark;
  agentType: AgentType;
  numEpisodes: number;
  seed?: number;
  envConfig?: BrainEnvConfig;
  a2cConfig?: Partial<A2CConfig>;
  ppoConfig?: Partial<PPOConfig>;
  /** Odd integer >= 3. Sets the observation cube edge length. */
  neighborhoodSize?: number;
  /** Sampling strides (resolutions) for multi-scale observations. */
  strides?: number[];
  /** Optional linear curriculum over episode starting radius. */
  curriculum?: StartDistanceCurriculum;
  /**
   * If set, replace the env's oracle direction signal with this trained
   * goal-vector model's prediction. The runner resolves the landmark's
   * one-hot index from the model's metadata.
   */
  goalVectorModel?: GoalVectorModel | null;
}

export interface ExperimentResult {
  config: {
    landmark: string;
    agentType: AgentType;
    numEpisodes: number;
    seed?: number;
    maxStartDistance?: number;
    maxSteps?: number;
    zeroDirection?: boolean;
    directionScale?: number;
    neighborhoodSize?: number;
    strides?: number[];
    a2cConfig?: Partial<A2CConfig>;
    ppoConfig?: Partial<PPOConfig>;
    curriculum?: StartDistanceCurriculum;
    goalVector?: 'predicted' | 'oracle' | 'zero';
  };
  episodes: EpisodeResult[];
  startTime: string;
  endTime: string;
}

export type ProgressCallback = (info: {
  configIndex: number;
  totalConfigs: number;
  episode: number;
  totalEpisodes: number;
  landmark: string;
  agentType: AgentType;
}) => void;

export type ConfigCompleteCallback = (result: ExperimentResult, configIndex: number) => void;

/** Unique key for a config so we can identify completed ones for resume. */
export function configKey(
  agentType: string,
  landmark: string,
  neighborhoodSize?: number,
  strides?: number[],
  extras?: {
    trunk?: string;
    zeroDirection?: boolean;
    seed?: number;
    directionScale?: number;
    curriculum?: StartDistanceCurriculum;
    goalVector?: 'predicted' | 'oracle' | 'zero';
  },
): string {
  const ns = neighborhoodSize ?? 7;
  const s = strides ? `s${strides.join(',')}` : 's1';
  const trunk = extras?.trunk ?? 'flat';
  const zd = extras?.zeroDirection ? ':nodir' : '';
  const seed = extras?.seed !== undefined ? `:seed${extras.seed}` : '';
  const ds = extras?.directionScale !== undefined && extras.directionScale !== 1
    ? `:ds${extras.directionScale}`
    : '';
  const cur = extras?.curriculum
    ? `:cur${extras.curriculum.start}-${extras.curriculum.end}@${extras.curriculum.annealEpisodes}`
    : '';
  const gv = extras?.goalVector && extras.goalVector !== 'oracle'
    ? `:gv${extras.goalVector}`
    : '';
  return `${agentType}:${landmark}:n${ns}:${s}:${trunk}${zd}${ds}${cur}${gv}${seed}`;
}

const SUCCESS_RADIUS = 3;

export class ExperimentRunner {
  private volumeData: ArrayLike<number>;
  private dims: [number, number, number];
  private aborted: boolean = false;

  constructor(volumeData: ArrayLike<number>, dims: [number, number, number]) {
    this.volumeData = volumeData;
    this.dims = dims;
  }

  abort(): void {
    this.aborted = true;
  }

  private goalVectorLabel(config: ExperimentConfig): 'predicted' | 'oracle' | 'zero' {
    if (config.envConfig?.zeroDirection) return 'zero';
    if (config.goalVectorModel) return 'predicted';
    return 'oracle';
  }

  async runAll(
    configs: ExperimentConfig[],
    onProgress?: ProgressCallback,
    onConfigComplete?: ConfigCompleteCallback,
    skipKeys?: Set<string>,
  ): Promise<ExperimentResult[]> {
    this.aborted = false;
    const results: ExperimentResult[] = [];

    for (let ci = 0; ci < configs.length; ci++) {
      if (this.aborted) break;
      const config = configs[ci];
      const key = configKey(
        config.agentType,
        config.landmark.name,
        config.neighborhoodSize,
        config.strides ?? config.envConfig?.strides,
        {
          trunk: config.ppoConfig?.trunk,
          zeroDirection: config.envConfig?.zeroDirection,
          seed: config.seed,
          directionScale: config.envConfig?.directionScale,
          curriculum: config.curriculum,
          goalVector: this.goalVectorLabel(config),
        },
      );

      // Skip already-completed configs (resume support)
      if (skipKeys?.has(key)) {
        console.log(`[SKIP] ${key} — already completed`);
        continue;
      }

      const numTensorsBefore = tf.memory().numTensors;

      const result = await this.runSingle(config, (episode) => {
        onProgress?.({
          configIndex: ci,
          totalConfigs: configs.length,
          episode,
          totalEpisodes: config.numEpisodes,
          landmark: config.landmark.name,
          agentType: config.agentType,
        });
      });
      results.push(result);
      onConfigComplete?.(result, ci);

      // Log memory state after each config for leak detection
      const mem = tf.memory();
      const leaked = mem.numTensors - numTensorsBefore;
      console.log(
        `[MEM] after ${key}: ${mem.numTensors} tensors, ${((mem.numBytes ?? 0) / 1e6).toFixed(1)} MB` +
          (leaked > 0 ? ` (leaked ${leaked} tensors)` : ''),
      );

      // Yield to allow GC between configs
      await new Promise((r) => setTimeout(r, 50));
    }

    return results;
  }

  private async runSingle(
    config: ExperimentConfig,
    onEpisode?: (episode: number) => void,
  ): Promise<ExperimentResult> {
    // Apply per-experiment neighborhood size and strides BEFORE constructing agents,
    // because DQN/A2C/PPO read STATE_DIM at network-build time.
    const ns = config.neighborhoodSize ?? NEIGHBORHOOD_SIZE;
    const s = config.strides ?? config.envConfig?.strides ?? STRIDES;
    setObservationConfig(ns, s);

    // Ensure env config matches the global strides for sampling consistency.
    // If a curriculum is set, seed the env's initial maxStartDistance with the
    // curriculum's starting value so episode 0 begins at the small radius.
    const goalVectorTargetIndex = config.goalVectorModel
      ? config.goalVectorModel.landmarkIndex(config.landmark.name)
      : -1;
    if (config.goalVectorModel && goalVectorTargetIndex < 0) {
      console.warn(
        `[goal-vector] landmark '${config.landmark.name}' not in model metadata; ` +
          `falling back to oracle direction for this config.`,
      );
    }
    const envConfig: BrainEnvConfig = {
      ...config.envConfig,
      strides: s,
      ...(config.curriculum ? { maxStartDistance: config.curriculum.start } : {}),
      goalVectorModel: goalVectorTargetIndex >= 0 ? config.goalVectorModel : null,
      goalVectorTargetIndex,
    };

    const env = new BrainEnv(this.volumeData, this.dims, config.landmark.mniVoxel, envConfig);

    let agent: Agent;
    if (config.agentType === 'a2c') {
      agent = await A2CAgent.create(config.a2cConfig);
    } else if (config.agentType === 'ppo') {
      agent = await PPOAgent.create(config.ppoConfig);
    } else if (config.agentType === 'oracle') {
      agent = new OracleAgent();
    } else if (config.agentType === 'random') {
      agent = new RandomAgent();
    } else {
      agent = new DQNAgent();
    }

    const episodes: EpisodeResult[] = [];
    const startTime = new Date().toISOString();

    for (let ep = 0; ep < config.numEpisodes; ep++) {
      if (this.aborted) break;

      // Advance curriculum: linearly interpolate start radius over the first
      // `annealEpisodes` episodes, then hold at `end`.
      if (config.curriculum) {
        const { start, end, annealEpisodes } = config.curriculum;
        const frac = Math.min(1, annealEpisodes > 0 ? ep / annealEpisodes : 1);
        env.setMaxStartDistance(start + (end - start) * frac);
      }

      const result = await this.runEpisode(env, agent, config.agentType);
      episodes.push({ episode: ep, ...result });
      agent.decayEpsilon();

      console.log(
        `[${config.agentType.toUpperCase()} ${config.landmark.name}] ep ${ep} reward=${result.totalReward.toFixed(2)} dist=${result.finalDistance.toFixed(1)} steps=${result.steps}${result.success ? ' SUCCESS' : ''}`,
      );

      onEpisode?.(ep);

      // Yield to UI every 5 episodes so the browser doesn't freeze
      if (ep % 5 === 0) {
        await new Promise((r) => setTimeout(r, 0));
      }
    }

    const endTime = new Date().toISOString();
    agent.dispose();

    return {
      config: {
        landmark: config.landmark.name,
        agentType: config.agentType,
        numEpisodes: config.numEpisodes,
        seed: config.seed,
        maxStartDistance: envConfig.maxStartDistance,
        maxSteps: envConfig.maxSteps,
        zeroDirection: envConfig.zeroDirection,
        directionScale: envConfig.directionScale,
        neighborhoodSize: ns,
        strides: s,
        a2cConfig: config.a2cConfig,
        ppoConfig: config.ppoConfig,
        curriculum: config.curriculum,
        goalVector: this.goalVectorLabel(config),
      },
      episodes,
      startTime,
      endTime,
    };
  }

  private async runEpisode(
    env: BrainEnv,
    agent: Agent,
    agentType: AgentType,
  ): Promise<Omit<EpisodeResult, 'episode'>> {
    let state = env.reset();
    let totalReward = 0;
    let done = false;
    let stepCount = 0;

    const isDQN = agentType === 'dqn';
    const isPolicyGradient = agentType === 'a2c' || agentType === 'ppo';
    const trajectory = isPolicyGradient ? new Trajectory() : null;

    while (!done) {
      const actionResult = agent.selectAction(state);
      const action = actionResult.action as Action;
      const result = env.step(action);
      stepCount++;

      if (isDQN) {
        const dqnAgent = agent as DQNAgent;
        dqnAgent.store({
          state: env.stateToArray(state),
          action,
          reward: result.reward,
          nextState: env.stateToArray(result.state),
          done: result.done,
        });
        await dqnAgent.train();
      }

      if (isPolicyGradient && trajectory && actionResult.extra) {
        trajectory.push({
          neighborhood: state.neighborhood,
          direction: state.direction,
          action,
          reward: result.reward,
          value: actionResult.extra.value,
          logProb: actionResult.extra.logProb,
        });
      }

      totalReward += result.reward;
      state = result.state;
      done = result.done;
    }

    // A2C / PPO train at end of episode
    if (isPolicyGradient && trajectory && trajectory.length > 0) {
      await agent.train(trajectory);
    }

    const finalDistance = (() => {
      const pos = env.currentPosition;
      const tgt = env.targetPosition;
      const dx = pos.x - tgt.x;
      const dy = pos.y - tgt.y;
      const dz = pos.z - tgt.z;
      return Math.sqrt(dx * dx + dy * dy + dz * dz);
    })();

    return {
      totalReward,
      finalDistance,
      steps: stepCount,
      success: finalDistance <= SUCCESS_RADIUS,
      epsilon: agent.epsilon,
    };
  }
}

const STORAGE_KEY = 'niivuerl_experiment_results';

/** Save a single completed result to localStorage (append). */
export function saveResultToStorage(result: ExperimentResult): void {
  try {
    const existing = loadResultsFromStorage();
    existing.push(result);
    localStorage.setItem(STORAGE_KEY, JSON.stringify(existing));
  } catch (err) {
    console.warn('Failed to save result to localStorage:', err);
  }
}

/** Load all saved results from localStorage. */
export function loadResultsFromStorage(): ExperimentResult[] {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return [];
    return JSON.parse(raw) as ExperimentResult[];
  } catch {
    return [];
  }
}

/** Get set of configKeys that are already completed in storage. */
export function getCompletedKeys(): Set<string> {
  const results = loadResultsFromStorage();
  return new Set(
    results.map((r) =>
      configKey(
        r.config.agentType,
        r.config.landmark,
        r.config.neighborhoodSize,
        r.config.strides,
        {
          trunk: r.config.ppoConfig?.trunk,
          zeroDirection: r.config.zeroDirection,
          seed: r.config.seed,
          directionScale: r.config.directionScale,
          curriculum: r.config.curriculum,
          goalVector: r.config.goalVector,
        },
      ),
    ),
  );
}

/** Clear saved results from localStorage. */
export function clearStoredResults(): void {
  localStorage.removeItem(STORAGE_KEY);
}

export function downloadResults(results: ExperimentResult[]): void {
  const json = JSON.stringify(results, null, 2);
  const blob = new Blob([json], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `experiment_results_${new Date().toISOString().slice(0, 19).replace(/:/g, '-')}.json`;
  a.click();
  URL.revokeObjectURL(url);
}
