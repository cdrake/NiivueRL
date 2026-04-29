import { useEffect, useRef, useState } from 'react';
import { Link } from 'react-router-dom';
import { Niivue } from '@niivue/niivue';
import { LANDMARKS } from '../lib/landmarks';

interface ExperimentMeta {
  slug: string;
  title: string;
  description: string;
}

export default function LandingPage() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const nvRef = useRef<Niivue | null>(null);
  const [experiments, setExperiments] = useState<ExperimentMeta[]>([]);
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
    nv.loadVolumes([{ url: `${base}aomic_test.nii.gz` }])
      .then(() => {
        nv.updateGLVolume();
      })
      .catch((err) => console.error('atlas load failed:', err));
  }, []);

  return (
    <div style={{ minHeight: '100vh', display: 'flex', flexDirection: 'column' }}>
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
          {loadError && <div style={{ color: '#f88' }}>Failed to load manifest: {loadError}</div>}
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
          </div>
          <div style={{ padding: '12px 16px', borderTop: '1px solid #2a2a5a', background: '#16213e', maxHeight: 220, overflowY: 'auto' }}>
            <h3 style={{ fontSize: 12, textTransform: 'uppercase', letterSpacing: 0.5, color: '#888', marginBottom: 6 }}>
              Landmarks ({LANDMARKS.length})
            </h3>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
              {LANDMARKS.map((l) => (
                <span
                  key={l.name}
                  style={{
                    fontSize: 11,
                    padding: '2px 8px',
                    borderRadius: 10,
                    background: `rgb(${l.color[0]},${l.color[1]},${l.color[2]})`,
                    color: l.color[0] + l.color[1] + l.color[2] > 380 ? '#000' : '#fff',
                  }}
                >
                  {l.name}
                </span>
              ))}
            </div>
          </div>
        </section>
      </div>
    </div>
  );
}
