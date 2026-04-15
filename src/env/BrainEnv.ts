import { Action, NUM_ACTIONS } from './types';
import type { State, StepResult, Vec3 } from './types';

export let NEIGHBORHOOD_SIZE = 7;
export let NEIGHBORHOOD_HALF = 3;
export let NEIGHBORHOOD_TOTAL = NEIGHBORHOOD_SIZE ** 3; // 343
export let STATE_DIM = NEIGHBORHOOD_TOTAL + 3; // 346

/**
 * Reconfigure the observation neighborhood size (must be odd). Affects the
 * global STATE_DIM read at runtime by all agents. Must be called before
 * constructing the agent/network for a given experiment. ESM live bindings
 * propagate the update to all importers that dereference these names at
 * runtime (which every current site does, inside methods).
 */
export function setNeighborhoodSize(size: number): void {
  if (size < 3 || size % 2 === 0) {
    throw new Error(`Neighborhood size must be an odd integer >= 3, got ${size}`);
  }
  NEIGHBORHOOD_SIZE = size;
  NEIGHBORHOOD_HALF = (size - 1) / 2;
  NEIGHBORHOOD_TOTAL = size ** 3;
  STATE_DIM = NEIGHBORHOOD_TOTAL + 3;
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
    const neighborhood = new Float32Array(NEIGHBORHOOD_TOTAL);
    let i = 0;
    for (let dz = -NEIGHBORHOOD_HALF; dz <= NEIGHBORHOOD_HALF; dz++) {
      for (let dy = -NEIGHBORHOOD_HALF; dy <= NEIGHBORHOOD_HALF; dy++) {
        for (let dx = -NEIGHBORHOOD_HALF; dx <= NEIGHBORHOOD_HALF; dx++) {
          neighborhood[i++] = this.getVoxel(
            this.position.x + dx,
            this.position.y + dy,
            this.position.z + dz,
          );
        }
      }
    }

    // Normalized direction to target (zeroed if ablation enabled)
    const dist = this.distanceToTarget();
    const direction: [number, number, number] = (this.zeroDirection || dist === 0)
      ? [0, 0, 0]
      : [
          (this.target.x - this.position.x) / dist,
          (this.target.y - this.position.y) / dist,
          (this.target.z - this.position.z) / dist,
        ];

    return { neighborhood, direction };
  }

  stateToArray(state: State): Float32Array {
    const arr = new Float32Array(STATE_DIM);
    arr.set(state.neighborhood);
    arr[NEIGHBORHOOD_TOTAL] = state.direction[0];
    arr[NEIGHBORHOOD_TOTAL + 1] = state.direction[1];
    arr[NEIGHBORHOOD_TOTAL + 2] = state.direction[2];
    return arr;
  }
}
