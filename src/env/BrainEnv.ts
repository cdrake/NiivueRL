import { Action, NUM_ACTIONS } from './types';
import type { State, StepResult, Vec3 } from './types';

const NEIGHBORHOOD_SIZE = 7;
const NEIGHBORHOOD_HALF = 3;
const NEIGHBORHOOD_TOTAL = NEIGHBORHOOD_SIZE ** 3; // 343
export const STATE_DIM = NEIGHBORHOOD_TOTAL + 3; // 346
const MAX_STEPS = 200;
const SUCCESS_RADIUS = 3;
const STEP_PENALTY = -0.1;
const SUCCESS_BONUS = 10;

const ACTION_DELTAS: Record<Action, [number, number, number]> = {
  [Action.PosX]: [1, 0, 0],
  [Action.NegX]: [-1, 0, 0],
  [Action.PosY]: [0, 1, 0],
  [Action.NegY]: [0, -1, 0],
  [Action.PosZ]: [0, 0, 1],
  [Action.NegZ]: [0, 0, -1],
};

export class BrainEnv {
  private volumeData: Float32Array | Uint8Array | Int16Array;
  private dims: [number, number, number];
  private target: Vec3;
  private position: Vec3;
  private steps: number;
  private voxelMin: number;
  private voxelMax: number;

  constructor(
    volumeData: ArrayLike<number>,
    dims: [number, number, number],
    target: Vec3,
  ) {
    this.volumeData = volumeData as Float32Array;
    this.dims = dims;
    this.target = target;
    this.position = { x: 0, y: 0, z: 0 };
    this.steps = 0;

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
    // Start from a random position in the volume
    this.position = {
      x: Math.floor(Math.random() * this.dims[0]),
      y: Math.floor(Math.random() * this.dims[1]),
      z: Math.floor(Math.random() * this.dims[2]),
    };
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
    } else if (this.steps >= MAX_STEPS) {
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

    // Normalized direction to target
    const dist = this.distanceToTarget();
    const direction: [number, number, number] = dist > 0
      ? [
          (this.target.x - this.position.x) / dist,
          (this.target.y - this.position.y) / dist,
          (this.target.z - this.position.z) / dist,
        ]
      : [0, 0, 0];

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
