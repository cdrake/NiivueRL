import { useCallback, useRef, useState } from 'react';
import NiivueViewer from './components/NiivueViewer';
import type { NiivueViewerHandle } from './components/NiivueViewer';
import LandmarkSelector from './components/LandmarkSelector';
import ControlPanel from './components/ControlPanel';
import type { AgentType } from './components/ControlPanel';
import MetricsPanel from './components/MetricsPanel';
import ExperimentPanel from './experiments/ExperimentPanel';
import { BrainEnv } from './env/BrainEnv';
import { DQNAgent } from './agent/DQNAgent';
import { A2CAgent } from './agent/A2CAgent';
import type { A2CConfig } from './agent/A2CAgent';
import { PPOAgent } from './agent/PPOAgent';
import { TrainingLoop } from './training/TrainingLoop';
import { ClipPlaneManager } from './visualization/ClipPlaneManager';
import type { TrainingMetrics } from './training/TrainingLoop';
import type { Agent } from './agent/types';
import type { Landmark, Vec3 } from './env/types';
import { clampLandmarkToVolume, LANDMARK_LABEL_INDEX } from './lib/landmarks';
import { runParcellation } from './lib/meshnetInference';
import './App.css';

export default function App() {
  const viewerRef = useRef<NiivueViewerHandle>(null);
  const trainingRef = useRef<TrainingLoop | null>(null);
  const agentRef = useRef<Agent | null>(null);
  const envRef = useRef<BrainEnv | null>(null);
  const clipPlaneRef = useRef<ClipPlaneManager | null>(null);

  const [landmark, setLandmark] = useState<Landmark | null>(null);
  const [running, setRunning] = useState(false);
  const [metrics, setMetrics] = useState<TrainingMetrics | null>(null);
  const [speed, setSpeed] = useState(10);
  const [, setAgentPos] = useState<Vec3 | null>(null);
  const [agentType, setAgentType] = useState<AgentType>('dqn');
  const [a2cConfig, setA2cConfig] = useState<Partial<A2CConfig>>({});
  const [showNeighborhood, setShowNeighborhood] = useState(false);
  const [loading, setLoading] = useState(false);
  const [parcLoading, setParcLoading] = useState(false);
  const [parcReady, setParcReady] = useState(false);
  const [mode, setMode] = useState<'interactive' | 'experiment'>('interactive');
  const [volumeInfo, setVolumeInfo] = useState<{ data: ArrayLike<number>; dims: [number, number, number] } | null>(null);

  const handleViewerReady = useCallback(() => {
    const nv = viewerRef.current?.nv;
    if (!nv || !nv.volumes.length) return;

    const vol = nv.volumes[0];
    const dims = vol.dims!;

    // Store volume data for experiment runner
    setVolumeInfo({
      data: vol.img! as unknown as Float32Array,
      dims: [dims[1], dims[2], dims[3]],
    });
  }, []);

  const handleRunParcellation = useCallback(async () => {
    const nv = viewerRef.current?.nv;
    if (!nv || !nv.volumes.length) return;

    const vol = nv.volumes[0];
    const dims = vol.dims!;

    setParcLoading(true);
    try {
      const labels = await runParcellation(
        vol.img! as unknown as Float32Array,
        [dims[1], dims[2], dims[3]],
      );
      viewerRef.current?.setParcellationOverlay(labels);
      setParcReady(true);
    } catch (err) {
      console.error('Parcellation failed:', err);
      alert(`Parcellation failed: ${err}\n\nThis usually means WebGL ran out of texture memory. Try reloading the page, or skip parcellation (experiments don't need it).`);
    } finally {
      setParcLoading(false);
    }
  }, []);

  const handleLandmarkSelect = useCallback((lm: Landmark) => {
    const nv = viewerRef.current?.nv;
    if (!nv || !nv.volumes.length) return;

    const vol = nv.volumes[0];
    const dims = vol.dims!;
    const clamped = clampLandmarkToVolume(lm, [dims[1], dims[2], dims[3]]);
    setLandmark(clamped);

    // Move crosshair to target
    const frac = [
      clamped.mniVoxel.x / (dims[1] - 1),
      clamped.mniVoxel.y / (dims[2] - 1),
      clamped.mniVoxel.z / (dims[3] - 1),
    ];
    nv.scene.crosshairPos = frac as [number, number, number];
    nv.updateGLVolume();

    // Highlight parcellation region for this landmark
    if (parcReady) {
      viewerRef.current?.highlightRegion(LANDMARK_LABEL_INDEX[clamped.name] ?? null);
    }

    // Update environment target if it exists
    if (envRef.current) {
      envRef.current.setTarget(clamped.mniVoxel);
    }
  }, [parcReady]);

  const handleStart = useCallback(async () => {
    const nv = viewerRef.current?.nv;
    if (!nv || !nv.volumes.length || !landmark) return;

    const vol = nv.volumes[0];
    const dims = vol.dims!;
    const imgData = vol.img!;

    if (!envRef.current) {
      envRef.current = new BrainEnv(
        imgData as unknown as Float32Array,
        [dims[1], dims[2], dims[3]],
        landmark.mniVoxel,
      );
    } else {
      envRef.current.setTarget(landmark.mniVoxel);
    }

    if (!agentRef.current) {
      if (agentType === 'a2c') {
        setLoading(true);
        try {
          agentRef.current = await A2CAgent.create(a2cConfig);
        } finally {
          setLoading(false);
        }
      } else if (agentType === 'ppo') {
        setLoading(true);
        try {
          agentRef.current = await PPOAgent.create(a2cConfig);
        } finally {
          setLoading(false);
        }
      } else {
        agentRef.current = new DQNAgent();
      }
    }

    // Set up clip plane manager
    if (showNeighborhood && !clipPlaneRef.current) {
      clipPlaneRef.current = new ClipPlaneManager(nv);
    }
    if (clipPlaneRef.current) {
      clipPlaneRef.current.setEnabled(showNeighborhood);
    }

    if (!trainingRef.current) {
      trainingRef.current = new TrainingLoop(
        envRef.current,
        agentRef.current,
        nv,
        (m) => setMetrics({ ...m }),
        (pos) => setAgentPos({ ...pos }),
        clipPlaneRef.current ?? undefined,
      );
    }

    trainingRef.current.setStepDelay(speed);
    setRunning(true);
    trainingRef.current.start();
  }, [landmark, speed, agentType, showNeighborhood, a2cConfig]);

  const handleStop = useCallback(() => {
    trainingRef.current?.stop();
    setRunning(false);
  }, []);

  const handleReset = useCallback(() => {
    trainingRef.current?.reset();
    agentRef.current?.dispose();
    agentRef.current = null;
    envRef.current = null;
    trainingRef.current = null;
    clipPlaneRef.current = null;
    setMetrics(null);
    setAgentPos(null);
    setRunning(false);
  }, []);

  const handleSpeedChange = useCallback((ms: number) => {
    setSpeed(ms);
    trainingRef.current?.setStepDelay(ms);
  }, []);

  const handleAgentTypeChange = useCallback((type: AgentType) => {
    setAgentType(type);
    // Reset agent when type changes
    if (agentRef.current) {
      agentRef.current.dispose();
      agentRef.current = null;
      trainingRef.current = null;
    }
  }, []);

  const handleShowNeighborhoodChange = useCallback((show: boolean) => {
    setShowNeighborhood(show);
    if (clipPlaneRef.current) {
      clipPlaneRef.current.setEnabled(show);
    }
    if (show && !clipPlaneRef.current) {
      const nv = viewerRef.current?.nv;
      if (nv) {
        clipPlaneRef.current = new ClipPlaneManager(nv);
        trainingRef.current?.setClipPlaneManager(clipPlaneRef.current);
      }
    }
  }, []);

  return (
    <div className="app">
      <div className="viewer-panel">
        <NiivueViewer ref={viewerRef} showRender3D={showNeighborhood} onReady={handleViewerReady} />
        {parcLoading && (
          <div style={{ position: 'absolute', top: 8, left: 8, background: 'rgba(0,0,0,0.7)', color: '#fff', padding: '4px 10px', borderRadius: 4, fontSize: 13 }}>
            Running parcellation...
          </div>
        )}
      </div>
      <div className="control-panel">
        <h2>NiivueRL</h2>
        <div style={{ display: 'flex', gap: 4, marginBottom: 12 }}>
          <button
            onClick={() => setMode('interactive')}
            style={{ fontWeight: mode === 'interactive' ? 'bold' : 'normal', flex: 1 }}
          >
            Interactive
          </button>
          <button
            onClick={() => setMode('experiment')}
            style={{ fontWeight: mode === 'experiment' ? 'bold' : 'normal', flex: 1 }}
          >
            Experiments
          </button>
        </div>
        {mode === 'interactive' ? (
          <>
            {!parcReady && (
              <div style={{ marginBottom: 8 }}>
                <button onClick={handleRunParcellation} disabled={parcLoading || !volumeInfo}>
                  {parcLoading ? 'Running parcellation...' : 'Run parcellation (landmark highlight)'}
                </button>
              </div>
            )}
            <LandmarkSelector
              selected={landmark}
              onSelect={handleLandmarkSelect}
              disabled={running}
            />
            <ControlPanel
              running={running}
              onStart={handleStart}
              onStop={handleStop}
              onReset={handleReset}
              speed={speed}
              onSpeedChange={handleSpeedChange}
              disabled={!landmark}
              agentType={agentType}
              onAgentTypeChange={handleAgentTypeChange}
              showNeighborhood={showNeighborhood}
              onShowNeighborhoodChange={handleShowNeighborhoodChange}
              loading={loading}
              a2cConfig={a2cConfig}
              onA2cConfigChange={setA2cConfig}
            />
            <MetricsPanel metrics={metrics} />
          </>
        ) : (
          <ExperimentPanel
            volumeData={volumeInfo?.data ?? null}
            dims={volumeInfo?.dims ?? null}
          />
        )}
      </div>
    </div>
  );
}
