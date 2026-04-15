import type { Landmark } from '../env/types';

// Subcortical landmarks from brainchop-models colormap.json
// MNI voxel coordinates are approximate centroids in the t1_crop volume (256x256x256 → cropped)
// These will be validated at runtime against actual volume dimensions.
const colormapR = [0, 245, 205, 120, 196, 220, 230, 0, 122, 236, 12, 204, 42, 119, 220, 103, 255, 165];
const colormapG = [0, 245, 62, 18, 58, 248, 148, 118, 186, 13, 48, 182, 204, 159, 216, 255, 165, 42];
const colormapB = [0, 245, 78, 134, 250, 164, 34, 14, 220, 176, 255, 142, 164, 176, 20, 255, 0, 42];
const labels = [
  'Unknown', 'Cerebral-White-Matter', 'Cerebral-Cortex', 'Lateral-Ventricle',
  'Inferior-Lateral-Ventricle', 'Cerebellum-White-Matter', 'Cerebellum-Cortex',
  'Thalamus', 'Caudate', 'Putamen', 'Pallidum', '3rd-Ventricle',
  '4th-Ventricle', 'Brain-Stem', 'Hippocampus', 'Amygdala',
  'Accumbens-area', 'VentralDC',
];

// Approximate MNI voxel coordinates for subcortical structures in a standard 1mm MNI brain.
// These are rough centroids and will be clamped to valid dims at runtime.
const mniCoords: Record<string, [number, number, number]> = {
  'Thalamus':                   [99, 134, 82],
  'Caudate':                    [108, 155, 90],
  'Putamen':                    [115, 145, 82],
  'Pallidum':                   [110, 140, 80],
  'Hippocampus':                [110, 115, 68],
  'Amygdala':                   [115, 115, 62],
  'Brain-Stem':                 [90, 118, 58],
  'Cerebellum-White-Matter':    [100, 100, 45],
  'Cerebellum-Cortex':          [108, 90, 40],
  'Lateral-Ventricle':          [96, 145, 92],
  'Inferior-Lateral-Ventricle': [108, 120, 62],
  '3rd-Ventricle':              [90, 140, 82],
  '4th-Ventricle':              [90, 108, 50],
  'Accumbens-area':             [112, 155, 60],
  'VentralDC':                  [102, 130, 72],
};

export const LANDMARKS: Landmark[] = labels
  .map((name, index) => {
    const coords = mniCoords[name];
    if (!coords) return null;
    return {
      name,
      index,
      color: [colormapR[index], colormapG[index], colormapB[index]] as [number, number, number],
      mniVoxel: { x: coords[0], y: coords[1], z: coords[2] },
    };
  })
  .filter((l): l is Landmark => l !== null);

// Mapping from landmark names to bcmodel label indices (subcortical MeshNet output)
export const LANDMARK_LABEL_INDEX: Record<string, number> = {
  'Thalamus': 7,
  'Caudate': 8,
  'Putamen': 9,
  'Pallidum': 10,
  'Hippocampus': 14,
  'Amygdala': 15,
  'Brain-Stem': 13,
  'Cerebellum-White-Matter': 5,
  'Cerebellum-Cortex': 6,
  'Lateral-Ventricle': 3,
  'Inferior-Lateral-Ventricle': 4,
  '3rd-Ventricle': 11,
  '4th-Ventricle': 12,
  'Accumbens-area': 16,
  'VentralDC': 17,
};

// Bcmodel colormap arrays (18 entries each, indexed by label)
export const BCMODEL_COLORMAP = {
  R: colormapR,
  G: colormapG,
  B: colormapB,
  labels,
};

export function clampLandmarkToVolume(
  landmark: Landmark,
  dims: [number, number, number],
): Landmark {
  return {
    ...landmark,
    mniVoxel: {
      x: Math.min(Math.max(0, landmark.mniVoxel.x), dims[0] - 1),
      y: Math.min(Math.max(0, landmark.mniVoxel.y), dims[1] - 1),
      z: Math.min(Math.max(0, landmark.mniVoxel.z), dims[2] - 1),
    },
  };
}
