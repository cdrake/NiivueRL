import { useEffect, useRef, useState } from 'react';
import { Link } from 'react-router-dom';
import { Niivue } from '@niivue/niivue';

interface ExperimentMeta {
  slug: string;
  title: string;
  description: string;
}

interface AalManifest {
  R: number[];
  G: number[];
  B: number[];
  A?: number[];
  I?: number[];
  labels: string[];
}

export default function LandingPage() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const nvRef = useRef<Niivue | null>(null);
  const [experiments, setExperiments] = useState<ExperimentMeta[]>([]);
  const [aalLabels, setAalLabels] = useState<string[]>([]);
  const [hoverRegion, setHoverRegion] = useState<string>('');
  const [loadError, setLoadError] = useState<string | null>(null);

  useEffect(() => {
    const base = import.meta.env.BASE_URL;
    fetch(`${base}experiments/index.json`)
      .then((r) => r.json())
      .then((d) => setExperiments(d.experiments ?? []))
      .catch((e) => setLoadError(String(e)));
  }, []);

  useEffect(() => {
    if (!canvasRef.current || nvRef.current) return;
    const nv = new Niivue({
      backColor: [0.06, 0.06, 0.14, 1],
      crosshairColor: [1, 0.4, 0.4, 0.7],
      show3Dcrosshair: false,
    });
    nv.attachToCanvas(canvasRef.current);
    nv.setSliceType(nv.sliceTypeMultiplanar);
    nvRef.current = nv;
    const base = import.meta.env.BASE_URL;

    (async () => {
      try {
        await nv.loadVolumes([
          { url: `${base}mni152.nii.gz` },
          { url: `${base}aal.nii.gz` },
        ]);
        const cmap: AalManifest = await fetch(`${base}aal.json`).then((r) => r.json());
        // Niivue expects an object with R/G/B/A/I/labels; setColormapLabel
        // converts that into an internal LUT keyed on voxel value.
        const overlay = nv.volumes[1] as unknown as {
          setColormapLabel: (cmap: AalManifest) => void;
        };
        overlay.setColormapLabel(cmap);
        nv.setOpacity(1, 110 / 255);
        // Thin region outlines so the colored atlas reads as a parcellation
        // rather than a flat blanket of color.
        const nvAtlas = nv as unknown as { setAtlasOutline?: (w: number) => void };
        nvAtlas.setAtlasOutline?.(0.5);
        nv.updateGLVolume();
        setAalLabels(cmap.labels ?? []);
      } catch (err) {
        console.error('atlas load failed:', err);
        setLoadError(String(err));
      }
    })();

    const handleMove = (e: MouseEvent) => {
      if (!nv.volumes[1] || !canvasRef.current) return;
      try {
        const pos = nv.getNoPaddingNoBorderCanvasRelativeMousePosition(e, nv.gl.canvas);
        if (!pos) return;
        const dpr = nv.uiData.dpr ?? window.devicePixelRatio ?? 1;
        const frac = nv.canvasPos2frac([pos.x * dpr, pos.y * dpr]);
        if (!frac || frac[0] < 0) return;
        const mm = nv.frac2mm(frac);
        const overlay = nv.volumes[1];
        const vox = overlay.mm2vox([mm[0], mm[1], mm[2]]);
        const idx = overlay.getValue(vox[0], vox[1], vox[2]);
        if (Number.isFinite(idx) && idx > 0) {
          const overlayAny = overlay as unknown as { colormapLabel?: { labels?: string[] } };
          const name = overlayAny.colormapLabel?.labels?.[idx as number];
          if (name) setHoverRegion(name);
        }
      } catch {
        /* canvas not ready yet */
      }
    };
    canvasRef.current.addEventListener('mousemove', handleMove);
    const cv = canvasRef.current;
    return () => {
      cv?.removeEventListener('mousemove', handleMove);
    };
  }, []);

  return (
    <div style={{ height: '100vh', display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
      <header style={{ padding: '20px 28px', borderBottom: '1px solid #2a2a5a', background: '#16213e' }}>
        <h1 style={{ fontSize: 22, color: '#4fc3f7', marginBottom: 4 }}>NiivueRL</h1>
        <p style={{ fontSize: 13, color: '#aaa' }}>
          Reinforcement-learning agents that navigate a 3D MRI volume to find anatomical landmarks.{' '}
          <a href="https://github.com/cdrake/NiivueRL" style={{ color: '#88f' }}>source</a>{' · '}
          <Link to="/notes" style={{ color: '#88f' }}>notes</Link>{' · '}
          <Link to="/interactive" style={{ color: '#88f' }}>interactive demo</Link>{' · '}
          <Link to="/configure" style={{ color: '#88f' }}>custom experiment</Link>
        </p>
      </header>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', flex: 1, minHeight: 0 }}>
        <section style={{ padding: 24, overflowY: 'auto', borderRight: '1px solid #2a2a5a' }}>
          <h2 style={{ fontSize: 15, textTransform: 'uppercase', letterSpacing: 0.5, color: '#888', marginBottom: 12 }}>
            Experiments
          </h2>
          {loadError && <div style={{ color: '#f88' }}>Failed to load: {loadError}</div>}
          <ul style={{ listStyle: 'none', display: 'flex', flexDirection: 'column', gap: 10 }}>
            {experiments.map((e) => (
              <li
                key={e.slug}
                style={{
                  background: '#16213e',
                  border: '1px solid #2a2a5a',
                  borderRadius: 6,
                  padding: 12,
                }}
              >
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', gap: 12 }}>
                  <Link
                    to={`/run/${e.slug}`}
                    style={{ color: '#4fc3f7', fontWeight: 600, fontSize: 14, textDecoration: 'none' }}
                  >
                    {e.title}
                  </Link>
                  <code style={{ fontSize: 11, color: '#666' }}>{e.slug}</code>
                </div>
                <p style={{ fontSize: 12.5, color: '#bbb', marginTop: 6, lineHeight: 1.45 }}>{e.description}</p>
              </li>
            ))}
          </ul>
        </section>

        <section style={{ display: 'flex', flexDirection: 'column', minHeight: 0 }}>
          <div style={{ flex: 1, minHeight: 0, position: 'relative', background: '#0f0f23' }}>
            <canvas ref={canvasRef} style={{ width: '100%', height: '100%', display: 'block' }} />
            {hoverRegion && (
              <div
                style={{
                  position: 'absolute',
                  top: 8,
                  right: 8,
                  background: 'rgba(0,0,0,0.7)',
                  color: '#fff',
                  padding: '4px 10px',
                  borderRadius: 4,
                  fontSize: 12,
                  pointerEvents: 'none',
                }}
              >
                {hoverRegion}
              </div>
            )}
          </div>
          <div style={{ padding: '10px 16px 12px', borderTop: '1px solid #2a2a5a', background: '#16213e', flexShrink: 0 }}>
            <h3 style={{ fontSize: 12, textTransform: 'uppercase', letterSpacing: 0.5, color: '#888', marginBottom: 4 }}>
              AAL atlas
            </h3>
            <p style={{ fontSize: 12, color: '#aaa', lineHeight: 1.4 }}>
              MNI152 T1 with the Automated Anatomical Labeling parcellation overlaid (
              {aalLabels.length > 0 ? `${aalLabels.length - 1} regions` : 'loading…'}). Hover the volume to identify regions. Agents navigate to the 15 subcortical landmarks listed under each experiment.
            </p>
          </div>
        </section>
      </div>
    </div>
  );
}
