import { forwardRef, useEffect, useImperativeHandle, useRef } from 'react';
import { Niivue } from '@niivue/niivue';

export interface NiivueViewerHandle {
  nv: Niivue | null;
}

const NiivueViewer = forwardRef<NiivueViewerHandle>((_props, ref) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const nvRef = useRef<Niivue | null>(null);

  useImperativeHandle(ref, () => ({
    get nv() {
      return nvRef.current;
    },
  }));

  useEffect(() => {
    if (!canvasRef.current) return;

    const nv = new Niivue({
      backColor: [0.15, 0.15, 0.2, 1],
      crosshairColor: [1, 0, 0, 0.7],
      show3Dcrosshair: true,
    });

    nv.attachToCanvas(canvasRef.current);
    nv.setSliceType(nv.sliceTypeMultiplanar);

    nv.loadVolumes([{ url: '/t1_crop.nii.gz' }]).then(() => {
      nv.createEmptyDrawing();
      nvRef.current = nv;
    });

    return () => {
      nv.closeDrawing();
    };
  }, []);

  return (
    <canvas
      ref={canvasRef}
      style={{ width: '100%', height: '100%' }}
    />
  );
});

NiivueViewer.displayName = 'NiivueViewer';
export default NiivueViewer;
