export interface BcmodelTensor {
  data: Float32Array;
  shape: number[];
}

export interface BcmodelFile {
  header: Record<string, unknown>;
  tensors: Record<string, BcmodelTensor>;
}

export async function loadBcmodel(source: string | ArrayBuffer): Promise<BcmodelFile> {
  let buffer: ArrayBuffer;
  if (typeof source === 'string') {
    const response = await fetch(source);
    if (!response.ok) {
      throw new Error(`Failed to fetch ${source}: ${response.statusText}`);
    }
    buffer = await response.arrayBuffer();
  } else if (source instanceof ArrayBuffer) {
    buffer = source;
  } else {
    throw new TypeError('source must be a URL string or ArrayBuffer');
  }
  return parseBcmodel(buffer);
}

function parseBcmodel(buffer: ArrayBuffer): BcmodelFile {
  const view = new DataView(buffer);
  const headerSize = view.getUint32(0, true);
  if (view.getUint32(4, true) !== 0) {
    throw new Error('Header size exceeds 4GB — unsupported');
  }

  const headerBytes = new Uint8Array(buffer, 8, headerSize);
  const headerString = new TextDecoder().decode(headerBytes);
  const header = JSON.parse(headerString);

  if (!header.bcmodel_version) {
    throw new Error('Not a valid .bcmodel file: missing bcmodel_version');
  }

  const dataOffset = 8 + headerSize;
  const tensors: Record<string, BcmodelTensor> = {};

  for (const [name, info] of Object.entries(header.tensors as Record<string, { data_offsets: [number, number]; shape: number[] }>)) {
    const [begin, end] = info.data_offsets;
    const count = (end - begin) / 4;
    const data = new Float32Array(buffer, dataOffset + begin, count);
    tensors[name] = { data, shape: info.shape };
  }

  return { header, tensors };
}

export function transposeConvKernel(
  data: Float32Array,
  shape: number[],
): { data: Float32Array; shape: number[] } {
  if (shape.length !== 5) {
    throw new Error(`Expected 5D tensor, got ${shape.length}D`);
  }
  const [O, I, D, H, W] = shape;
  const newShape = [D, H, W, I, O];
  const result = new Float32Array(data.length);

  for (let o = 0; o < O; o++)
    for (let i = 0; i < I; i++)
      for (let d = 0; d < D; d++)
        for (let h = 0; h < H; h++)
          for (let w = 0; w < W; w++) {
            const srcIdx = ((((o * I + i) * D + d) * H + h) * W + w);
            const dstIdx = ((((d * H + h) * W + w) * I + i) * O + o);
            result[dstIdx] = data[srcIdx];
          }

  return { data: result, shape: newShape };
}
