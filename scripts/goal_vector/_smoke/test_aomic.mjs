/**
 * End-to-end smoke test: load model + the served AOMIC volume, extract patches
 * via the same indexing as src/lib/goalVectorModel.ts (flat = x + y*W + z*W*H),
 * predict + post-normalize, compare to oracle direction. Should match the Python
 * probe's ~0.99 cosine.
 *
 * Run with the Vite dev server up on :5176.
 */
import * as tf from '@tensorflow/tfjs';
import * as nifti from 'nifti-reader-js';
import { gunzipSync } from 'fflate';

const BASE = 'http://localhost:5176';
const MODEL_URL = `${BASE}/goal_vector_model`;
const VOL_URL = `${BASE}/aomic_test.nii.gz`;
const TARGETS = [
  { name: 'Thalamus',          c: [130, 140, 125] },
  { name: 'Lateral-Ventricle', c: [128, 135, 127] },
];
const N_SAMPLES = 200;
const SAMPLE_RADIUS = 50;

function rng(seed) {
  let s = seed | 0;
  return () => {
    s = (1664525 * s + 1013904223) | 0;
    return ((s >>> 0) % 1_000_000) / 1_000_000;
  };
}

function readNifti(buffer) {
  const u8 = new Uint8Array(buffer);
  const decompressed = nifti.isCompressed(u8.buffer) ? gunzipSync(u8) : u8;
  const buf = decompressed.buffer.slice(decompressed.byteOffset, decompressed.byteOffset + decompressed.byteLength);
  if (!nifti.isNIFTI(buf)) throw new Error('not a NIfTI');
  const hdr = nifti.readHeader(buf);
  const imgRaw = nifti.readImage(hdr, buf);
  // Datatype 4 = INT16 (typical for brain.mgz/AOMIC); 16 = FLOAT32.
  const dims = [hdr.dims[1], hdr.dims[2], hdr.dims[3]];
  let img;
  if (hdr.datatypeCode === 4) img = new Int16Array(imgRaw);
  else if (hdr.datatypeCode === 2) img = new Uint8Array(imgRaw);
  else if (hdr.datatypeCode === 16) img = new Float32Array(imgRaw);
  else throw new Error(`unhandled dtype ${hdr.datatypeCode}`);
  return { dims, img };
}

const volBuf = await (await fetch(VOL_URL)).arrayBuffer();
const { dims, img } = readNifti(volBuf);
console.log('volume:', dims, 'len:', img.length);

let voxelMin = Infinity, voxelMax = -Infinity;
for (let i = 0; i < img.length; i++) { const v = img[i]; if (v < voxelMin) voxelMin = v; if (v > voxelMax) voxelMax = v; }
console.log('voxelMin:', voxelMin, 'voxelMax:', voxelMax);

const meta = await (await fetch(`${MODEL_URL}/metadata.json`)).json();
const model = await tf.loadLayersModel(`${MODEL_URL}/model.json`);
console.log('model loaded; output_post_normalize:', meta.output_post_normalize);

const [W, H, D] = dims;
const r = meta.patch_radius, ps = meta.patch_size, n_scales = meta.n_scales, n_landmarks = meta.n_landmarks;
const denom = (voxelMax === voxelMin) ? 1 : (voxelMax - voxelMin);

function predictAt(pos, targetIdx) {
  const patchData = new Float32Array(ps * ps * ps * n_scales);
  let idx = 0;
  for (let ix = -r; ix <= r; ix++) {
    for (let iy = -r; iy <= r; iy++) {
      for (let iz = -r; iz <= r; iz++) {
        for (let c = 0; c < n_scales; c++) {
          const s = meta.strides[c];
          const x = Math.min(Math.max(0, pos[0] + ix * s), W - 1);
          const y = Math.min(Math.max(0, pos[1] + iy * s), H - 1);
          const z = Math.min(Math.max(0, pos[2] + iz * s), D - 1);
          const flat = x + y * W + z * W * H;
          patchData[idx++] = (img[flat] - voxelMin) / denom;
        }
      }
    }
  }
  const posData = new Float32Array([
    (2 * pos[0]) / W - 1,
    (2 * pos[1]) / H - 1,
    (2 * pos[2]) / D - 1,
  ]);
  const tgt = new Float32Array(n_landmarks); tgt[targetIdx] = 1;
  const out = tf.tidy(() => {
    const p = tf.tensor(patchData, [1, ps, ps, ps, n_scales]);
    const po = tf.tensor(posData, [1, 3]);
    const t = tf.tensor(tgt, [1, n_landmarks]);
    const o = model.predict([p, po, t]);
    return o.dataSync();
  });
  let dx = out[0], dy = out[1], dz = out[2];
  if (meta.output_post_normalize) {
    const n = Math.hypot(dx, dy, dz) || 1;
    dx /= n; dy /= n; dz /= n;
  }
  return [dx, dy, dz];
}

const rand = rng(42);
for (const { name, c } of TARGETS) {
  const targetIdx = meta.landmark_names.indexOf(name);
  if (targetIdx < 0) { console.log('!! unknown landmark', name); continue; }
  let sumCos = 0, n = 0;
  const cosines = [];
  for (let i = 0; i < N_SAMPLES; i++) {
    const off = [
      (rand() * 2 - 1) * SAMPLE_RADIUS,
      (rand() * 2 - 1) * SAMPLE_RADIUS,
      (rand() * 2 - 1) * SAMPLE_RADIUS,
    ];
    const margin = Math.max(...meta.strides) * r;
    const pos = [
      Math.max(margin, Math.min(W - margin - 1, Math.round(c[0] + off[0]))),
      Math.max(margin, Math.min(H - margin - 1, Math.round(c[1] + off[1]))),
      Math.max(margin, Math.min(D - margin - 1, Math.round(c[2] + off[2]))),
    ];
    const [dx, dy, dz] = predictAt(pos, targetIdx);
    const tx = c[0] - pos[0], ty = c[1] - pos[1], tz = c[2] - pos[2];
    const tn = Math.hypot(tx, ty, tz) || 1;
    const cos = (dx * tx + dy * ty + dz * tz) / tn;
    sumCos += cos; n++;
    cosines.push(cos);
  }
  cosines.sort((a, b) => a - b);
  const median = cosines[Math.floor(cosines.length / 2)];
  console.log(`${name.padEnd(20)} mean_cos=${(sumCos / n).toFixed(4)}  median=${median.toFixed(4)}  n=${n}`);
}
