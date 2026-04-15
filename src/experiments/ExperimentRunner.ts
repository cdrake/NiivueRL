import * as tf from '@tensorflow/tfjs';
import { BrainEnv, setNeighborhoodSize, NEIGHBORHOOD_SIZE } from '../env/BrainEnv';
import type { BrainEnvConfig } from '../env/BrainEnv';
import { DQNAgent } from '../agent/DQNAgent';
import { A2CAgent } from '../agent/A2CAgent';
import type { A2CConfig } from '../agent/A2CAgent';
import { PPOAgent } from '../agent/PPOAgent';
import type { PPOConfig } from '../agent/PPOAgent';
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
    neighborhoodSize?: number;
    a2cConfig?: Partial<A2CConfig>;
    ppoConfig?: Partial<PPOConfig>;
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
  extras?: { trunk?: string; zeroDirection?: boolean },
): string {
  const ns = neighborhoodSize ?? 7;
  const trunk = extras?.trunk ?? 'flat';
  const zd = extras?.zeroDirection ? ':nodir' : '';
  return `${agentType}:${landmark}:n${ns}:${trunk}${zd}`;
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
      const key = configKey(config.agentType, config.landmark.name, config.neighborhoodSize, {
        trunk: config.ppoConfig?.trunk,
        zeroDirection: config.envConfig?.zeroDirection,
      });

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
    // Apply per-experiment neighborhood size BEFORE constructing agents,
    // because DQN/A2C/PPO read STATE_DIM at network-build time.
    if (config.neighborhoodSize && config.neighborhoodSize !== NEIGHBORHOOD_SIZE) {
      setNeighborhoodSize(config.neighborhoodSize);
    }
    const env = new BrainEnv(this.volumeData, this.dims, config.landmark.mniVoxel, config.envConfig);

    let agent: Agent;
    if (config.agentType === 'a2c') {
      agent = await A2CAgent.create(config.a2cConfig);
    } else if (config.agentType === 'ppo') {
      agent = await PPOAgent.create(config.ppoConfig);
    } else {
      agent = new DQNAgent();
    }

    const episodes: EpisodeResult[] = [];
    const startTime = new Date().toISOString();

    for (let ep = 0; ep < config.numEpisodes; ep++) {
      if (this.aborted) break;

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
        maxStartDistance: config.envConfig?.maxStartDistance,
        maxSteps: config.envConfig?.maxSteps,
        zeroDirection: config.envConfig?.zeroDirection,
        neighborhoodSize: config.neighborhoodSize ?? NEIGHBORHOOD_SIZE,
        a2cConfig: config.a2cConfig,
        ppoConfig: config.ppoConfig,
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
      configKey(r.config.agentType, r.config.landmark, r.config.neighborhoodSize, {
        trunk: r.config.ppoConfig?.trunk,
        zeroDirection: r.config.zeroDirection,
      }),
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
