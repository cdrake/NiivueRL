import * as tf from '@tensorflow/tfjs';
console.log('tf version:', tf.version_core);
const baseUrl = 'http://localhost:5176/goal_vector_model_wide';
const metaResp = await fetch(`${baseUrl}/metadata.json`);
const meta = await metaResp.json();
console.log('patch_size:', meta.patch_size, 'n_scales:', meta.n_scales, 'hierarchical:', meta.hierarchical);
console.log('best_val_cos:', meta.best_val_cos);
const model = await tf.loadLayersModel(`${baseUrl}/model.json`);
console.log('model inputs:', model.inputs.map((i) => `${i.name}: ${i.shape}`));
console.log('model outputs:', model.outputs.map((o) => `${o.name}: ${o.shape}`));
const ps = meta.patch_size, ns = meta.n_scales, nl = meta.n_landmarks;
const perScale = meta.per_scale_inputs === true;
const pos = tf.tensor([[0, 0, 0]]);
const target = tf.oneHot([4], nl);
const inputs = perScale
  ? [...Array(ns)].map(() => tf.zeros([1, ps, ps, ps, 1])).concat([pos, target])
  : [tf.zeros([1, ps, ps, ps, ns]), pos, target];
const out = model.predict(inputs);
const data = await out.data();
console.log('raw output:', Array.from(data));
const n = Math.hypot(data[0], data[1], data[2]) || 1;
console.log('normalized:', [data[0] / n, data[1] / n, data[2] / n]);
