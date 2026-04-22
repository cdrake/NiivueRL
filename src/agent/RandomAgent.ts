import type { Agent, AgentActionResult } from './types';

/** Non-learning baseline: uniform random action. */
export class RandomAgent implements Agent {
  epsilon = 0;

  selectAction(): AgentActionResult {
    return { action: Math.floor(Math.random() * 6) };
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
