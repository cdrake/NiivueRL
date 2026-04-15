import type { Niivue } from '@niivue/niivue';
import type { Vec3 } from '../env/types';
import { NEIGHBORHOOD_HALF } from '../env/BrainEnv';

export class ClipPlaneManager {
  private nv: Niivue;
  private enabled: boolean;

  constructor(nv: Niivue) {
    this.nv = nv;
    this.enabled = true;
  }

  setEnabled(enabled: boolean): void {
    this.enabled = enabled;
    if (!enabled) {
      this.clearClipPlanes();
    }
  }

  updateClipPlanes(position: Vec3): void {
    if (!this.enabled) return;

    const vol = this.nv.volumes[0];
    if (!vol) return;
    const dims = vol.dims!;

    // Convert voxel position to fractional coordinates
    const fracX = position.x / (dims[1] - 1);
    const fracY = position.y / (dims[2] - 1);
    const fracZ = position.z / (dims[3] - 1);

    // Compute fractional extent of neighborhood
    const extX = NEIGHBORHOOD_HALF / (dims[1] - 1);
    const extY = NEIGHBORHOOD_HALF / (dims[2] - 1);
    const extZ = NEIGHBORHOOD_HALF / (dims[3] - 1);

    // Use Niivue clip plane to show a box around the neighborhood
    // setClipPlane takes [azimuth, elevation, depth] - we use a simple approach
    // by setting a single clip plane near the agent position
    // The depth parameter controls how much to clip (0-1 range)
    const depth = Math.min(extX, extY, extZ) * 2;
    this.nv.setClipPlane([0, 0, depth]);

    // Move crosshair to agent position to center the clip
    this.nv.scene.crosshairPos = [fracX, fracY, fracZ] as [number, number, number];
  }

  clearClipPlanes(): void {
    // Depth of 2 disables clipping in Niivue
    this.nv.setClipPlane([0, 0, 2]);
  }
}
