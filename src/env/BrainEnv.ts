import { Action, NUM_ACTIONS } from './types';
import type { State, StepResult, Vec3 } from './types';

export let NEIGHBORHOOD_SIZE = 7;
export let NEIGHBORHOOD_HALF = 3;
export let STRIDES = [1];
export let NEIGHBORHOOD_TOTAL = NEIGHBORHOOD_SIZE ** 3; // 343 per scale
export let STATE_DIM = NEIGHBORHOOD_TOTAL * STRIDES.length + 3; // (343 * scales) + 3

/**
 * Reconfigure the observation neighborhood size and scales. Affects the
 * global STATE_DIM read at runtime by all agents. Must be called before
 * constructing the agent/network for a given experiment.
 */
export function setObservationConfig(size: number, strides: number[]): void {
  if (size < 3 || size % 2 === 0) {
    throw new Error(`Neighborhood size must be an odd integer >= 3, got ${size}`);
  }
  if (strides.length === 0) {
    throw new Error('Strides must be a non-empty array of integers >= 1');
  }
  NEIGHBORHOOD_SIZE = size;
  NEIGHBORHOOD_HALF = (size - 1) / 2;
  STRIDES = [...strides];
  NEIGHBORHOOD_TOTAL = size ** 3;
  STATE_DIM = NEIGHBORHOOD_TOTAL * STRIDES.length + 3;
}

/** @deprecated Use setObservationConfig instead. */
export function setNeighborhoodSize(size: number): void {
  setObservationConfig(size, STRIDES);
}
const DEFAULT_MAX_STEPS = 200;
const SUCCESS_RADIUS = 3;
const STEP_PENALTY = -0.1;
const SUCCESS_BONUS = 10;
const DEFAULT_MAX_START_DISTANCE = Infinity; // no limit by default

const ACTION_DELTAS: Record<Action, [number, number, number]> = {
  [Action.PosX]: [1, 0, 0],
  [Action.NegX]: [-1, 0, 0],
  [Action.PosY]: [0, 1, 0],
  [Action.NegY]: [0, -1, 0],
  [Action.PosZ]: [0, 0, 1],
  [Action.NegZ]: [0, 0, -1],
};

export interface BrainEnvConfig {
  maxSteps?: number;
  maxStartDistance?: number;
  /** Ablation: replace direction-to-target vector with zeros in the state. */
  zeroDirection?: boolean;
  /** Sampling strides (resolutions) for multi-scale observations. */
  strides?: number[];
  /**
   * Multiplier applied to the normalized direction-to-target vector before it
   * enters the network. Raising this above 1 helps the policy attend to the
   * 3-dim direction signal when it's concatenated with a large voxel patch.
   * Default 1 (unscaled).
   */
  directionScale?: number;
}

/**
 * Curriculum schedule for the starting radius. If set, the runner will call
 * `env.setMaxStartDistance(r(ep))` before each `reset()`, with
 *   r(ep) = start + (end - start) * min(1, ep / (annealEpisodes))
 * Clamped to [start, end].
 */
export interface StartDistanceCurriculum {
  start: number;
  end: number;
  annealEpisodes: number;
}

export class BrainEnv {
  private volumeData: Float32Array | Uint8Array | Int16Array;
  private dims: [number, number, number];
  private target: Vec3;
  private position: Vec3;
  private steps: number;
  private voxelMin: number;
  private voxelMax: number;
  private maxSteps: number;
  private maxStartDistance: number;
  private zeroDirection: boolean;
  private strides: number[];
  private directionScale: number;

  constructor(
    volumeData: ArrayLike<number>,
    dims: [number, number, number],
    target: Vec3,
    config?: BrainEnvConfig,
  ) {
    this.volumeData = volumeData as Float32Array;
    this.dims = dims;
    this.target = target;
    this.position = { x: 0, y: 0, z: 0 };
    this.steps = 0;
    this.maxSteps = config?.maxSteps ?? DEFAULT_MAX_STEPS;
    this.maxStartDistance = config?.maxStartDistance ?? DEFAULT_MAX_START_DISTANCE;
    this.zeroDirection = config?.zeroDirection ?? false;
    this.strides = config?.strides ?? [1];
    this.directionScale = config?.directionScale ?? 1;

    // Compute normalization range
    let min = Infinity, max = -Infinity;
    for (let i = 0; i < volumeData.length; i++) {
      const v = volumeData[i];
      if (v < min) min = v;
      if (v > max) max = v;
    }
    this.voxelMin = min;
    this.voxelMax = max === min ? min + 1 : max;
  }

  get numActions(): number {
    return NUM_ACTIONS;
  }

  get stateDim(): number {
    return STATE_DIM;
  }

  get currentPosition(): Vec3 {
    return { ...this.position };
  }

  get targetPosition(): Vec3 {
    return { ...this.target };
  }

  setTarget(target: Vec3): void {
    this.target = { ...target };
  }

  setMaxStartDistance(v: number): void {
    this.maxStartDistance = v;
  }

  reset(): State {
    if (this.maxStartDistance < Infinity) {
      // Start at a random position within maxStartDistance of the target
      // Use rejection sampling within a bounding box
      const r = this.maxStartDistance;
      for (;;) {
        const x = Math.floor(this.target.x + (Math.random() * 2 - 1) * r);
        const y = Math.floor(this.target.y + (Math.random() * 2 - 1) * r);
        const z = Math.floor(this.target.z + (Math.random() * 2 - 1) * r);
        if (x < 0 || x >= this.dims[0] || y < 0 || y >= this.dims[1] || z < 0 || z >= this.dims[2]) continue;
        const dx = x - this.target.x, dy = y - this.target.y, dz = z - this.target.z;
        const dist = Math.sqrt(dx * dx + dy * dy + dz * dz);
        if (dist <= r && dist > SUCCESS_RADIUS) {
          this.position = { x, y, z };
          break;
        }
      }
    } else {
      // Start from a random position anywhere in the volume
      this.position = {
        x: Math.floor(Math.random() * this.dims[0]),
        y: Math.floor(Math.random() * this.dims[1]),
        z: Math.floor(Math.random() * this.dims[2]),
      };
    }
    this.steps = 0;
    return this.getState();
  }

  step(action: Action): StepResult {
    const oldDist = this.distanceToTarget();
    const delta = ACTION_DELTAS[action];

    this.position.x = Math.min(Math.max(0, this.position.x + delta[0]), this.dims[0] - 1);
    this.position.y = Math.min(Math.max(0, this.position.y + delta[1]), this.dims[1] - 1);
    this.position.z = Math.min(Math.max(0, this.position.z + delta[2]), this.dims[2] - 1);
    this.steps++;

    const newDist = this.distanceToTarget();
    let reward = -(newDist - oldDist) + STEP_PENALTY;
    let done = false;

    if (newDist <= SUCCESS_RADIUS) {
      reward += SUCCESS_BONUS;
      done = true;
    } else if (this.steps >= this.maxSteps) {
      done = true;
    }

    return {
      state: this.getState(),
      reward,
      done,
      info: {
        position: { ...this.position },
        distance: newDist,
        steps: this.steps,
      },
    };
  }

  private distanceToTarget(): number {
    const dx = this.position.x - this.target.x;
    const dy = this.position.y - this.target.y;
    const dz = this.position.z - this.target.z;
    return Math.sqrt(dx * dx + dy * dy + dz * dz);
  }

  private getVoxel(x: number, y: number, z: number): number {
    if (x < 0 || x >= this.dims[0] || y < 0 || y >= this.dims[1] || z < 0 || z >= this.dims[2]) {
      return 0;
    }
    const idx = x + y * this.dims[0] + z * this.dims[0] * this.dims[1];
    return (this.volumeData[idx] - this.voxelMin) / (this.voxelMax - this.voxelMin);
  }

  private getState(): State {
    const neighborhood = new Float32Array(NEIGHBORHOOD_TOTAL * this.strides.length);
    let offset = 0;
    for (const stride of this.strides) {
      for (let dz = -NEIGHBORHOOD_HALF; dz <= NEIGHBORHOOD_HALF; dz++) {
        for (let dy = -NEIGHBORHOOD_HALF; dy <= NEIGHBORHOOD_HALF; dy++) {
          for (let dx = -NEIGHBORHOOD_HALF; dx <= NEIGHBORHOOD_HALF; dx++) {
            neighborhood[offset++] = this.getVoxel(
              this.position.x + dx * stride,
              this.position.y + dy * stride,
              this.position.z + dz * stride,
            );
          }
        }
      }
    }

    // Normalized direction to target, scaled so it isn't drowned out by the
    // much-larger voxel patch in the concatenated input (see directionScale).
    const dist = this.distanceToTarget();
    const k = this.directionScale;
    const direction: [number, number, number] = (this.zeroDirection || dist === 0)
      ? [0, 0, 0]
      : [
          (k * (this.target.x - this.position.x)) / dist,
          (k * (this.target.y - this.position.y)) / dist,
          (k * (this.target.z - this.position.z)) / dist,
        ];

    return { neighborhood, direction };
  }

  stateToArray(state: State): Float32Array {
    const arr = new Float32Array(STATE_DIM);
    arr.set(state.neighborhood);
    arr[state.neighborhood.length] = state.direction[0];
    arr[state.neighborhood.length + 1] = state.direction[1];
    arr[state.neighborhood.length + 2] = state.direction[2];
    return arr;
  }
}
