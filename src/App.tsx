import { useCallback, useRef, useState } from 'react';
import NiivueViewer from './components/NiivueViewer';
import type { NiivueViewerHandle } from './components/NiivueViewer';
import LandmarkSelector from './components/LandmarkSelector';
import ControlPanel from './components/ControlPanel';
import MetricsPanel from './components/MetricsPanel';
import { BrainEnv } from './env/BrainEnv';
import { DQNAgent } from './agent/DQNAgent';
import { TrainingLoop } from './training/TrainingLoop';
import type { TrainingMetrics } from './training/TrainingLoop';
import type { Landmark, Vec3 } from './env/types';
import { clampLandmarkToVolume } from './lib/landmarks';
import './App.css';

export default function App() {
  const viewerRef = useRef<NiivueViewerHandle>(null);
  const trainingRef = useRef<TrainingLoop | null>(null);
  const agentRef = useRef<DQNAgent | null>(null);
  const envRef = useRef<BrainEnv | null>(null);

  const [landmark, setLandmark] = useState<Landmark | null>(null);
  const [running, setRunning] = useState(false);
  const [metrics, setMetrics] = useState<TrainingMetrics | null>(null);
  const [speed, setSpeed] = useState(10);
  const [, setAgentPos] = useState<Vec3 | null>(null);

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

    // Update environment target if it exists
    if (envRef.current) {
      envRef.current.setTarget(clamped.mniVoxel);
    }
  }, []);

  const handleStart = useCallback(() => {
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
      agentRef.current = new DQNAgent();
    }

    if (!trainingRef.current) {
      trainingRef.current = new TrainingLoop(
        envRef.current,
        agentRef.current,
        nv,
        (m) => setMetrics({ ...m }),
        (pos) => setAgentPos({ ...pos }),
      );
    }

    trainingRef.current.setStepDelay(speed);
    setRunning(true);
    trainingRef.current.start();
  }, [landmark, speed]);

  const handleStop = useCallback(() => {
    trainingRef.current?.stop();
    setRunning(false);
  }, []);

  const handleReset = useCallback(() => {
    trainingRef.current?.reset();
    agentRef.current = new DQNAgent();
    envRef.current = null;
    trainingRef.current = null;
    setMetrics(null);
    setAgentPos(null);
  }, []);

  const handleSpeedChange = useCallback((ms: number) => {
    setSpeed(ms);
    trainingRef.current?.setStepDelay(ms);
  }, []);

  return (
    <div className="app">
      <div className="viewer-panel">
        <NiivueViewer ref={viewerRef} />
      </div>
      <div className="control-panel">
        <h2>NiivueRL</h2>
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
        />
        <MetricsPanel metrics={metrics} />
      </div>
    </div>
  );
}
