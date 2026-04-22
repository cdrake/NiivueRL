import type { State } from '../env/types';
import { Action } from '../env/types';
import type { Agent, AgentActionResult } from './types';

/**
 * Non-learning oracle baseline: picks the action whose axis has the largest
 * absolute direction-to-target component, with sign matching the direction.
 * If the zero-direction ablation is on (direction = [0,0,0]), falls back to
 * a random action so the agent still moves.
 */
export class OracleAgent implements Agent {
  epsilon = 0;

  selectAction(state: State): AgentActionResult {
    const [dx, dy, dz] = state.direction;
    const ax = Math.abs(dx), ay = Math.abs(dy), az = Math.abs(dz);
    if (ax === 0 && ay === 0 && az === 0) {
      return { action: Math.floor(Math.random() * 6) };
    }
    let action: number;
    if (ax >= ay && ax >= az) {
      action = dx >= 0 ? Action.PosX : Action.NegX;
    } else if (ay >= az) {
      action = dy >= 0 ? Action.PosY : Action.NegY;
    } else {
      action = dz >= 0 ? Action.PosZ : Action.NegZ;
    }
    return { action };
  }

  async train(): Promise<number> {
    return 0;
  }

  decayEpsilon(): void {
    // no-op
  }

  dispose(): void {
    // no-op
  }
}
