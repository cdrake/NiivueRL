import type { Landmark } from '../env/types';

// Subcortical landmarks from brainchop-models colormap.json. The centroid
// table below was generated from AOMIC ID1000 sub-0083's aseg.mgz — a held-out
// val subject the goal-vector network never saw during training. We deploy
// against this subject's brain.mgz so the predicted-direction signal is
// in-distribution. See scripts/goal_vector/prep_aomic_deployment.py (or
// /tmp/prep_aomic_deployment.py) for the generator and public/aomic_test_
// landmarks.json for the JSON dump.
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

// Voxel-space centroids for AOMIC ID1000 sub-0083 (held-out val subject).
const mniCoords: Record<string, [number, number, number]> = {
  'Thalamus':                   [130, 140, 125],
  'Caudate':                    [129, 128, 152],
  'Putamen':                    [131, 139, 145],
  'Pallidum':                   [130, 142, 141],
  'Hippocampus':                [129, 159, 127],
  'Amygdala':                   [129, 158, 145],
  'Brain-Stem':                 [131, 177, 126],
  'Cerebellum-White-Matter':    [132, 187, 105],
  'Cerebellum-Cortex':          [130, 193, 98],
  'Lateral-Ventricle':          [128, 135, 127],
  'Inferior-Lateral-Ventricle': [125, 158, 129],
  '3rd-Ventricle':              [130, 145, 134],
  '4th-Ventricle':              [132, 185, 114],
  'Accumbens-area':             [129, 142, 157],
  'VentralDC':                  [130, 154, 133],
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
