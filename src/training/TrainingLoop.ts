import { BrainEnv } from '../env/BrainEnv';
import { DQNAgent } from '../agent/DQNAgent';
import { A2CAgent } from '../agent/A2CAgent';
import { PPOAgent } from '../agent/PPOAgent';
import { Trajectory } from '../agent/Trajectory';
import type { Action, Vec3 } from '../env/types';
import type { Agent } from '../agent/types';
import { Niivue } from '@niivue/niivue';
import type { ClipPlaneManager } from '../visualization/ClipPlaneManager';

export interface TrainingMetrics {
  episode: number;
  totalReward: number;
  epsilon: number;
  distance: number;
  steps: number;
  rewardHistory: number[];
}

export type MetricsCallback = (metrics: TrainingMetrics) => void;
export type PositionCallback = (pos: Vec3) => void;

function delay(ms: number): Promise<void> {
  return new Promise(resolve => setTimeout(resolve, ms));
}

export class TrainingLoop {
  private env: BrainEnv;
  private agent: Agent;
  private nv: Niivue;
  private running: boolean;
  private episode: number;
  private rewardHistory: number[];
  private onMetrics: MetricsCallback;
  private onPosition: PositionCallback;
  private stepDelay: number;
  private clipPlaneManager: ClipPlaneManager | null;

  constructor(
    env: BrainEnv,
    agent: Agent,
    nv: Niivue,
    onMetrics: MetricsCallback,
    onPosition: PositionCallback,
    clipPlaneManager?: ClipPlaneManager,
  ) {
    this.env = env;
    this.agent = agent;
    this.nv = nv;
    this.running = false;
    this.episode = 0;
    this.rewardHistory = [];
    this.onMetrics = onMetrics;
    this.onPosition = onPosition;
    this.stepDelay = 10;
    this.clipPlaneManager = clipPlaneManager ?? null;
  }

  setClipPlaneManager(manager: ClipPlaneManager | null): void {
    this.clipPlaneManager = manager;
  }

  setStepDelay(ms: number): void {
    this.stepDelay = ms;
  }

  async start(): Promise<void> {
    if (this.running) return;
    this.running = true;

    while (this.running) {
      await this.runEpisode();
      this.agent.decayEpsilon();
      this.episode++;
    }
  }

  stop(): void {
    this.running = false;
  }

  reset(): void {
    this.stop();
    this.episode = 0;
    this.rewardHistory = [];
  }

  private async runEpisode(): Promise<void> {
    let state = this.env.reset();
    let totalReward = 0;
    let done = false;

    // Clear drawing overlay for new episode
    this.clearPath();
    this.clipPlaneManager?.clearClipPlanes();

    const isDQN = this.agent instanceof DQNAgent;
    const isPolicyGradient =
      this.agent instanceof A2CAgent || this.agent instanceof PPOAgent;
    const trajectory = isPolicyGradient ? new Trajectory() : null;

    while (!done && this.running) {
      const result_action = this.agent.selectAction(state);
      const action = result_action.action as Action;
      const result = this.env.step(action);

      if (isDQN) {
        const dqnAgent = this.agent as DQNAgent;
        dqnAgent.store({
          state: this.env.stateToArray(state),
          action,
          reward: result.reward,
          nextState: this.env.stateToArray(result.state),
          done: result.done,
        });
        await dqnAgent.train();
      }

      if (isPolicyGradient && trajectory && result_action.extra) {
        trajectory.push({
          neighborhood: state.neighborhood,
          direction: state.direction,
          action,
          reward: result.reward,
          value: result_action.extra.value,
          logProb: result_action.extra.logProb,
        });
      }

      totalReward += result.reward;
      state = result.state;
      done = result.done;

      // Update visualization
      this.updateCrosshair(result.info.position);
      this.clipPlaneManager?.updateClipPlanes(result.info.position);
      this.paintPath(result.info.position);
      this.onPosition(result.info.position);

      await delay(this.stepDelay);
    }

    // A2C / PPO train at end of episode
    if (isPolicyGradient && trajectory && trajectory.length > 0) {
      await this.agent.train(trajectory);
    }

    this.rewardHistory.push(totalReward);
    this.onMetrics({
      episode: this.episode,
      totalReward,
      epsilon: this.agent.epsilon,
      distance: this.distanceCurrent(),
      steps: 0,
      rewardHistory: [...this.rewardHistory],
    });
  }

  private distanceCurrent(): number {
    const pos = this.env.currentPosition;
    const tgt = this.env.targetPosition;
    const dx = pos.x - tgt.x;
    const dy = pos.y - tgt.y;
    const dz = pos.z - tgt.z;
    return Math.sqrt(dx * dx + dy * dy + dz * dz);
  }

  private updateCrosshair(pos: Vec3): void {
    const vol = this.nv.volumes[0];
    if (!vol) return;
    const dims = vol.dims!;
    const frac = [
      pos.x / (dims[1] - 1),
      pos.y / (dims[2] - 1),
      pos.z / (dims[3] - 1),
    ];
    this.nv.scene.crosshairPos = frac as [number, number, number];
    this.nv.updateGLVolume();
  }

  private paintPath(pos: Vec3): void {
    if (!this.nv.drawBitmap) return;
    const vol = this.nv.volumes[0];
    if (!vol) return;
    const dims = vol.dims!;
    const idx = pos.x + pos.y * dims[1] + pos.z * dims[1] * dims[2];
    if (idx >= 0 && idx < this.nv.drawBitmap.length) {
      this.nv.drawBitmap[idx] = 1;
      this.nv.refreshDrawing(false);
    }
  }

  private clearPath(): void {
    if (!this.nv.drawBitmap) return;
    this.nv.drawBitmap.fill(0);
    this.nv.refreshDrawing(false);
  }
}
