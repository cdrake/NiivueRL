import { forwardRef, useEffect, useImperativeHandle, useRef } from 'react';
import { Niivue, NVImage, SHOW_RENDER } from '@niivue/niivue';
import { BCMODEL_COLORMAP } from '../lib/landmarks';
import { createParcellationNifti } from '../lib/meshnetInference';
import type { NIFTI1 } from 'nifti-reader-js';

const NUM_LABELS = 18;

export interface NiivueViewerHandle {
  nv: Niivue | null;
  setParcellationOverlay(labels: Uint8Array): void;
  highlightRegion(labelIndex: number | null): void;
}

interface Props {
  showRender3D?: boolean;
  onReady?: () => void;
}

const NiivueViewer = forwardRef<NiivueViewerHandle, Props>(({ showRender3D, onReady }, ref) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const nvRef = useRef<Niivue | null>(null);
  const onReadyRef = useRef(onReady);
  onReadyRef.current = onReady;

  useImperativeHandle(ref, () => ({
    get nv() {
      return nvRef.current;
    },
    setParcellationOverlay(labels: Uint8Array) {
      const nv = nvRef.current;
      if (!nv || !nv.volumes.length) return;

      const hdr = nv.volumes[0].hdr as NIFTI1;
      const niftiBuf = createParcellationNifti(labels, hdr);

      // Build LUT: all labels hidden initially (alpha=0)
      const lut = new Uint8ClampedArray(NUM_LABELS * 4);
      for (let i = 1; i < NUM_LABELS; i++) {
        lut[i * 4] = BCMODEL_COLORMAP.R[i];
        lut[i * 4 + 1] = BCMODEL_COLORMAP.G[i];
        lut[i * 4 + 2] = BCMODEL_COLORMAP.B[i];
        lut[i * 4 + 3] = 0;
      }

      NVImage.loadFromUrl({
        url: niftiBuf,
        opacity: 0.4,
        name: 'parcellation',
      }).then((image) => {
        image.colormapLabel = {
          lut,
          labels: BCMODEL_COLORMAP.labels,
        };
        nv.addVolume(image);
        nv.updateGLVolume();
      });
    },
    highlightRegion(labelIndex: number | null) {
      const nv = nvRef.current;
      if (!nv || nv.volumes.length < 2) return;

      const overlay = nv.volumes[1];
      if (!overlay.colormapLabel) return;

      const lut = overlay.colormapLabel.lut;
      for (let i = 1; i < NUM_LABELS; i++) {
        lut[i * 4 + 3] = (labelIndex !== null && i === labelIndex) ? 255 : 0;
      }
      nv.updateGLVolume();
    },
  }));

  useEffect(() => {
    if (!canvasRef.current) return;
    // Guard against StrictMode double-mount: a single WebGL context can't be
    // re-attached, so reuse the existing Niivue instance if one was created.
    if (nvRef.current) {
      onReadyRef.current?.();
      return;
    }

    const nv = new Niivue({
      backColor: [0.15, 0.15, 0.2, 1],
      crosshairColor: [1, 0, 0, 0.7],
      show3Dcrosshair: true,
    });

    nv.attachToCanvas(canvasRef.current);
    nv.setSliceType(nv.sliceTypeMultiplanar);
    nvRef.current = nv;

    (window as unknown as { nv: Niivue }).nv = nv;
    nv.loadVolumes([{ url: `${import.meta.env.BASE_URL}aomic_test.nii.gz` }])
      .then(() => {
        console.log('[NiivueViewer] volumes loaded:', nv.volumes.length, 'dims:', nv.volumes[0]?.dims);
        nv.createEmptyDrawing();
        nv.updateGLVolume();
        onReadyRef.current?.();
      })
      .catch((err) => console.error('[NiivueViewer] loadVolumes failed:', err));
  }, []);

  // Update 3D render mode when prop changes
  useEffect(() => {
    const nv = nvRef.current;
    if (!nv) return;

    if (showRender3D) {
      nv.opts.multiplanarShowRender = SHOW_RENDER.ALWAYS;
    } else {
      nv.opts.multiplanarShowRender = SHOW_RENDER.AUTO;
    }
    nv.updateGLVolume();
  }, [showRender3D]);

  return (
    <canvas
      ref={canvasRef}
      style={{ width: '100%', height: '100%' }}
    />
  );
});

NiivueViewer.displayName = 'NiivueViewer';
export default NiivueViewer;
