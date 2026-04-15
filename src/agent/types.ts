import type { State } from '../env/types';

export interface AgentActionResult {
  action: number;
  extra?: Record<string, number>;
}

export interface Agent {
  epsilon: number;
  selectAction(state: State): AgentActionResult;
  train(data: unknown): Promise<number>;
  decayEpsilon(): void;
  dispose(): void;
}
