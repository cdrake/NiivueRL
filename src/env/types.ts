export const Action = {
  PosX: 0,
  NegX: 1,
  PosY: 2,
  NegY: 3,
  PosZ: 4,
  NegZ: 5,
} as const;

export type Action = (typeof Action)[keyof typeof Action];

export const NUM_ACTIONS = 6;

export interface Vec3 {
  x: number;
  y: number;
  z: number;
}

export interface State {
  neighborhood: Float32Array; // 7x7x7 = 343 voxel values
  direction: [number, number, number]; // normalized direction to target
}

export interface StepResult {
  state: State;
  reward: number;
  done: boolean;
  info: {
    position: Vec3;
    distance: number;
    steps: number;
  };
}

export interface Landmark {
  name: string;
  index: number;
  color: [number, number, number];
  mniVoxel: Vec3;
}
