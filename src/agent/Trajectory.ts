export interface TrajectoryStep {
  neighborhood: Float32Array;
  direction: [number, number, number];
  action: number;
  reward: number;
  value: number;
  logProb: number;
}

export class Trajectory {
  steps: TrajectoryStep[] = [];

  push(step: TrajectoryStep): void {
    this.steps.push(step);
  }

  get length(): number {
    return this.steps.length;
  }

  clear(): void {
    this.steps = [];
  }

  computeReturns(gamma: number): Float32Array {
    const n = this.steps.length;
    const returns = new Float32Array(n);
    let G = 0;
    for (let t = n - 1; t >= 0; t--) {
      G = this.steps[t].reward + gamma * G;
      returns[t] = G;
    }
    return returns;
  }

  computeAdvantages(gamma: number, lambda: number): Float32Array {
    const n = this.steps.length;
    const advantages = new Float32Array(n);
    let gae = 0;
    for (let t = n - 1; t >= 0; t--) {
      const nextValue = t < n - 1 ? this.steps[t + 1].value : 0;
      const delta = this.steps[t].reward + gamma * nextValue - this.steps[t].value;
      gae = delta + gamma * lambda * gae;
      advantages[t] = gae;
    }
    return advantages;
  }
}
