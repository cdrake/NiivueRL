export interface Experience {
  state: Float32Array;
  action: number;
  reward: number;
  nextState: Float32Array;
  done: boolean;
}

export class ReplayBuffer {
  private buffer: Experience[];
  private capacity: number;
  private position: number;

  constructor(capacity: number = 10000) {
    this.buffer = [];
    this.capacity = capacity;
    this.position = 0;
  }

  push(experience: Experience): void {
    if (this.buffer.length < this.capacity) {
      this.buffer.push(experience);
    } else {
      this.buffer[this.position] = experience;
    }
    this.position = (this.position + 1) % this.capacity;
  }

  sample(batchSize: number): Experience[] {
    const batch: Experience[] = [];
    for (let i = 0; i < batchSize; i++) {
      const idx = Math.floor(Math.random() * this.buffer.length);
      batch.push(this.buffer[idx]);
    }
    return batch;
  }

  get size(): number {
    return this.buffer.length;
  }
}
