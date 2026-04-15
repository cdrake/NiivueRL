import { useCallback, useRef, useState } from 'react';
import {
  ExperimentRunner,
  downloadResults,
  saveResultToStorage,
  loadResultsFromStorage,
  getCompletedKeys,
  clearStoredResults,
  configKey,
} from './ExperimentRunner';
import { LANDMARKS } from '../lib/landmarks';
import { clampLandmarkToVolume } from '../lib/landmarks';
import type { ExperimentConfig, ExperimentResult } from './ExperimentRunner';
import type { AgentType } from '../components/ControlPanel';
import type { A2CConfig } from '../agent/A2CAgent';
import { DEFAULT_A2C_CONFIG } from '../agent/A2CAgent';
import type { PPOConfig } from '../agent/PPOAgent';
import { DEFAULT_PPO_CONFIG } from '../agent/PPOAgent';

interface ExperimentPanelProps {
  volumeData: ArrayLike<number> | null;
  dims: [number, number, number] | null;
}

// Diverse subset: large/deep, small/curved, large/easy, distinct, small/hard
const DEFAULT_LANDMARKS = [
  'Thalamus',
  'Hippocampus',
  'Lateral-Ventricle',
  'Brain-Stem',
  'Putamen',
];

const AGENT_TYPES: AgentType[] = ['dqn', 'a2c', 'ppo'];
const DEFAULT_EPISODES = 300;
const DEFAULT_MAX_START_DISTANCE = 50;
const NEIGHBORHOOD_SIZE_OPTIONS = [5, 7, 9, 11, 13, 15, 19, 25];
const DEFAULT_NEIGHBORHOOD_SIZE = 7;

export default function ExperimentPanel({ volumeData, dims }: ExperimentPanelProps) {
  const [running, setRunning] = useState(false);
  const [progress, setProgress] = useState('');
  const [results, setResults] = useState<ExperimentResult[] | null>(null);
  const [numEpisodes, setNumEpisodes] = useState(DEFAULT_EPISODES);
  const [selectedAgents, setSelectedAgents] = useState<AgentType[]>(['dqn']);
  const [selectedLandmarks, setSelectedLandmarks] = useState<string[]>(DEFAULT_LANDMARKS);
  const [maxStartDistance, setMaxStartDistance] = useState(DEFAULT_MAX_START_DISTANCE);
  const [neighborhoodSize, setNeighborhoodSize] = useState(DEFAULT_NEIGHBORHOOD_SIZE);
  const [zeroDirection, setZeroDirection] = useState(false);
  const [a2cLr, setA2cLr] = useState(DEFAULT_A2C_CONFIG.lr);
  const [a2cEntropy, setA2cEntropy] = useState(DEFAULT_A2C_CONFIG.entropyCoeff);
  const [a2cUseConv, setA2cUseConv] = useState(DEFAULT_A2C_CONFIG.useConvBackbone);
  const [ppoLr, setPpoLr] = useState(DEFAULT_PPO_CONFIG.lr);
  const [ppoEntropy, setPpoEntropy] = useState(DEFAULT_PPO_CONFIG.entropyCoeff);
  const [ppoClipEpsilon, setPpoClipEpsilon] = useState(DEFAULT_PPO_CONFIG.clipEpsilon);
  const [ppoNumEpochs, setPpoNumEpochs] = useState(DEFAULT_PPO_CONFIG.numEpochs);
  const [ppoMinibatchSize, setPpoMinibatchSize] = useState(DEFAULT_PPO_CONFIG.minibatchSize);
  const [ppoRolloutSize, setPpoRolloutSize] = useState(DEFAULT_PPO_CONFIG.rolloutSize);
  const [ppoUseConv, setPpoUseConv] = useState(DEFAULT_PPO_CONFIG.useConvBackbone);
  const [ppoTrunk, setPpoTrunk] = useState<'flat' | 'meshnet' | 'conv'>(DEFAULT_PPO_CONFIG.trunk ?? 'flat');
  const runnerRef = useRef<ExperimentRunner | null>(null);

  // Check how many configs are already saved
  const savedResults = loadResultsFromStorage();
  const completedKeys = getCompletedKeys();

  const handleRun = useCallback(async () => {
    if (!volumeData || !dims) return;

    setRunning(true);
    // Merge any previously saved results into state
    const prior = loadResultsFromStorage();
    const skipKeys = new Set(
      prior.map((r) =>
        configKey(r.config.agentType, r.config.landmark, r.config.neighborhoodSize, {
          trunk: r.config.ppoConfig?.trunk,
          zeroDirection: r.config.zeroDirection,
        }),
      ),
    );

    const runner = new ExperimentRunner(volumeData, dims);
    runnerRef.current = runner;

    const a2cConfig: Partial<A2CConfig> = {
      lr: a2cLr,
      entropyCoeff: a2cEntropy,
      useConvBackbone: a2cUseConv,
    };

    const ppoConfig: Partial<PPOConfig> = {
      lr: ppoLr,
      entropyCoeff: ppoEntropy,
      clipEpsilon: ppoClipEpsilon,
      numEpochs: ppoNumEpochs,
      minibatchSize: ppoMinibatchSize,
      rolloutSize: ppoRolloutSize,
      useConvBackbone: ppoUseConv,
      trunk: ppoTrunk,
    };

    const configs: ExperimentConfig[] = [];
    for (const agentType of selectedAgents) {
      for (const lmName of selectedLandmarks) {
        const lm = LANDMARKS.find((l) => l.name === lmName);
        if (!lm) continue;
        const clamped = clampLandmarkToVolume(lm, dims);
        configs.push({
          landmark: clamped,
          agentType,
          numEpisodes,
          envConfig: { maxStartDistance, zeroDirection },
          neighborhoodSize,
          ...(agentType === 'a2c' ? { a2cConfig } : {}),
          ...(agentType === 'ppo' ? { ppoConfig } : {}),
        });
      }
    }

    try {
      const newResults = await runner.runAll(
        configs,
        (info) => {
          const skipped = skipKeys.size;
          setProgress(
            `[${info.configIndex + 1}/${info.totalConfigs}] ${info.agentType.toUpperCase()} → ${info.landmark}: episode ${info.episode + 1}/${info.totalEpisodes}` +
              (skipped > 0 ? ` (${skipped} resumed from cache)` : ''),
          );
        },
        (result) => {
          // Save each completed config immediately to localStorage
          saveResultToStorage(result);
          console.log(`[SAVED] ${result.config.agentType}:${result.config.landmark} → localStorage`);
        },
        skipKeys,
      );

      // Combine prior + new results for display/download
      const allResults = [...prior, ...newResults];
      setResults(allResults);
      setProgress(`Done! ${allResults.length} configs total (${newResults.length} new, ${prior.length} resumed).`);
    } catch (err) {
      // Even on crash, saved results are safe in localStorage
      const saved = loadResultsFromStorage();
      setResults(saved.length > 0 ? saved : null);
      setProgress(`Error: ${err}. ${saved.length} configs saved — click "Run" to resume.`);
    } finally {
      setRunning(false);
      runnerRef.current = null;
    }
  }, [
    volumeData,
    dims,
    numEpisodes,
    selectedAgents,
    selectedLandmarks,
    maxStartDistance,
    neighborhoodSize,
    zeroDirection,
    a2cLr,
    a2cEntropy,
    a2cUseConv,
    ppoLr,
    ppoEntropy,
    ppoClipEpsilon,
    ppoNumEpochs,
    ppoMinibatchSize,
    ppoRolloutSize,
    ppoUseConv,
    ppoTrunk,
  ]);

  const handleAbort = useCallback(() => {
    runnerRef.current?.abort();
  }, []);

  const handleDownload = useCallback(() => {
    // Download from localStorage (includes all saved results)
    const all = loadResultsFromStorage();
    if (all.length > 0) {
      downloadResults(all);
    } else if (results) {
      downloadResults(results);
    }
  }, [results]);

  const handleClearCache = useCallback(() => {
    if (confirm('Clear all saved experiment results from cache?')) {
      clearStoredResults();
      setResults(null);
      setProgress('Cache cleared.');
    }
  }, []);

  const toggleAgent = (agent: AgentType) => {
    setSelectedAgents((prev) =>
      prev.includes(agent) ? prev.filter((a) => a !== agent) : [...prev, agent],
    );
  };

  const toggleLandmark = (name: string) => {
    setSelectedLandmarks((prev) =>
      prev.includes(name) ? prev.filter((n) => n !== name) : [...prev, name],
    );
  };

  // Count how many of the currently selected configs are already done
  const totalSelected = selectedAgents.length * selectedLandmarks.length;
  const alreadyDone = selectedAgents.reduce((count, agent) => {
    const extras = { trunk: agent === 'ppo' ? ppoTrunk : undefined, zeroDirection };
    return (
      count +
      selectedLandmarks.filter((lm) =>
        completedKeys.has(configKey(agent, lm, neighborhoodSize, extras)),
      ).length
    );
  }, 0);
  const remaining = totalSelected - alreadyDone;

  return (
    <div style={{ padding: '8px 0' }}>
      <h3 style={{ margin: '0 0 8px' }}>Experiments</h3>

      <div style={{ marginBottom: 8 }}>
        <label style={{ fontSize: 13 }}>
          Episodes per config:{' '}
          <input
            type="number"
            value={numEpisodes}
            onChange={(e) => setNumEpisodes(Number(e.target.value))}
            min={10}
            max={5000}
            step={50}
            style={{ width: 70 }}
            disabled={running}
          />
        </label>
      </div>

      <div style={{ marginBottom: 8 }}>
        <label style={{ fontSize: 13 }}>
          Neighborhood size (voxels, odd):{' '}
          <select
            value={neighborhoodSize}
            onChange={(e) => setNeighborhoodSize(Number(e.target.value))}
            disabled={running}
            style={{ fontSize: 13 }}
          >
            {NEIGHBORHOOD_SIZE_OPTIONS.map((n) => (
              <option key={n} value={n}>
                {n}³ ({n ** 3} voxels)
              </option>
            ))}
          </select>
        </label>
      </div>

      <div style={{ marginBottom: 8 }}>
        <label style={{ fontSize: 13 }}>
          <input
            type="checkbox"
            checked={zeroDirection}
            onChange={(e) => setZeroDirection(e.target.checked)}
            disabled={running}
          />
          Zero direction vector (ablation)
        </label>
      </div>

      <div style={{ marginBottom: 8 }}>
        <label style={{ fontSize: 13 }}>
          Max start distance (voxels):{' '}
          <input
            type="number"
            value={maxStartDistance}
            onChange={(e) => setMaxStartDistance(Number(e.target.value))}
            min={10}
            max={200}
            step={10}
            style={{ width: 70 }}
            disabled={running}
          />
        </label>
      </div>

      <div style={{ marginBottom: 8 }}>
        <div style={{ fontSize: 13, fontWeight: 'bold', marginBottom: 4 }}>Algorithms:</div>
        {AGENT_TYPES.map((agent) => (
          <label key={agent} style={{ fontSize: 13, marginRight: 12 }}>
            <input
              type="checkbox"
              checked={selectedAgents.includes(agent)}
              onChange={() => toggleAgent(agent)}
              disabled={running}
            />
            {agent.toUpperCase()}
          </label>
        ))}
      </div>

      {selectedAgents.includes('a2c') && (
        <div style={{ marginBottom: 8, padding: 8, background: '#f5f5f5', borderRadius: 4 }}>
          <div style={{ fontSize: 13, fontWeight: 'bold', marginBottom: 4 }}>A2C Settings:</div>
          <label style={{ fontSize: 12, display: 'block', marginBottom: 4 }}>
            Learning rate:{' '}
            <select
              value={a2cLr}
              onChange={(e) => setA2cLr(Number(e.target.value))}
              disabled={running}
              style={{ fontSize: 12 }}
            >
              <option value={0.0003}>3e-4</option>
              <option value={0.0001}>1e-4</option>
              <option value={0.00003}>3e-5</option>
            </select>
          </label>
          <label style={{ fontSize: 12, display: 'block', marginBottom: 4 }}>
            Entropy coeff:{' '}
            <select
              value={a2cEntropy}
              onChange={(e) => setA2cEntropy(Number(e.target.value))}
              disabled={running}
              style={{ fontSize: 12 }}
            >
              <option value={0.01}>0.01</option>
              <option value={0.05}>0.05</option>
              <option value={0.1}>0.1</option>
            </select>
          </label>
          <label style={{ fontSize: 12, display: 'block' }}>
            <input
              type="checkbox"
              checked={a2cUseConv}
              onChange={(e) => setA2cUseConv(e.target.checked)}
              disabled={running}
            />
            Use conv backbone (MeshNet)
          </label>
        </div>
      )}

      {selectedAgents.includes('ppo') && (
        <div style={{ marginBottom: 8, padding: 8, background: '#f5f5f5', borderRadius: 4 }}>
          <div style={{ fontSize: 13, fontWeight: 'bold', marginBottom: 4 }}>PPO Settings:</div>
          <label style={{ fontSize: 12, display: 'block', marginBottom: 4 }}>
            Learning rate:{' '}
            <select
              value={ppoLr}
              onChange={(e) => setPpoLr(Number(e.target.value))}
              disabled={running}
              style={{ fontSize: 12 }}
            >
              <option value={0.0003}>3e-4</option>
              <option value={0.0001}>1e-4</option>
              <option value={0.00003}>3e-5</option>
            </select>
          </label>
          <label style={{ fontSize: 12, display: 'block', marginBottom: 4 }}>
            Entropy coeff:{' '}
            <select
              value={ppoEntropy}
              onChange={(e) => setPpoEntropy(Number(e.target.value))}
              disabled={running}
              style={{ fontSize: 12 }}
            >
              <option value={0.01}>0.01</option>
              <option value={0.05}>0.05</option>
              <option value={0.1}>0.1</option>
            </select>
          </label>
          <label style={{ fontSize: 12, display: 'block', marginBottom: 4 }}>
            Clip epsilon:{' '}
            <select
              value={ppoClipEpsilon}
              onChange={(e) => setPpoClipEpsilon(Number(e.target.value))}
              disabled={running}
              style={{ fontSize: 12 }}
            >
              <option value={0.1}>0.1</option>
              <option value={0.2}>0.2</option>
              <option value={0.3}>0.3</option>
            </select>
          </label>
          <label style={{ fontSize: 12, display: 'block', marginBottom: 4 }}>
            Num epochs:{' '}
            <select
              value={ppoNumEpochs}
              onChange={(e) => setPpoNumEpochs(Number(e.target.value))}
              disabled={running}
              style={{ fontSize: 12 }}
            >
              <option value={1}>1</option>
              <option value={2}>2</option>
              <option value={4}>4</option>
              <option value={8}>8</option>
            </select>
          </label>
          <label style={{ fontSize: 12, display: 'block', marginBottom: 4 }}>
            Minibatch size:{' '}
            <select
              value={ppoMinibatchSize}
              onChange={(e) => setPpoMinibatchSize(Number(e.target.value))}
              disabled={running}
              style={{ fontSize: 12 }}
            >
              <option value={16}>16</option>
              <option value={32}>32</option>
              <option value={64}>64</option>
              <option value={128}>128</option>
              <option value={256}>256</option>
            </select>
          </label>
          <label style={{ fontSize: 12, display: 'block', marginBottom: 4 }}>
            Rollout size (episodes):{' '}
            <select
              value={ppoRolloutSize}
              onChange={(e) => setPpoRolloutSize(Number(e.target.value))}
              disabled={running}
              style={{ fontSize: 12 }}
            >
              <option value={1}>1</option>
              <option value={2}>2</option>
              <option value={4}>4</option>
              <option value={8}>8</option>
              <option value={16}>16</option>
            </select>
          </label>
          <label style={{ fontSize: 12, display: 'block', marginBottom: 4 }}>
            Trunk:{' '}
            <select
              value={ppoTrunk}
              onChange={(e) => setPpoTrunk(e.target.value as 'flat' | 'meshnet' | 'conv')}
              disabled={running}
              style={{ fontSize: 12 }}
            >
              <option value="flat">Flat MLP</option>
              <option value="meshnet">MeshNet (frozen conv)</option>
              <option value="conv">Trainable 3D conv (16 filters × 2)</option>
            </select>
          </label>
          <label style={{ fontSize: 12, display: 'block' }}>
            <input
              type="checkbox"
              checked={ppoUseConv}
              onChange={(e) => setPpoUseConv(e.target.checked)}
              disabled={running || ppoTrunk !== 'meshnet'}
            />
            (legacy) Use conv backbone (MeshNet)
          </label>
        </div>
      )}

      <div style={{ marginBottom: 8 }}>
        <div style={{ fontSize: 13, fontWeight: 'bold', marginBottom: 4 }}>Landmarks:</div>
        <div style={{ maxHeight: 150, overflowY: 'auto', fontSize: 12 }}>
          {LANDMARKS.map((lm) => (
            <label key={lm.name} style={{ display: 'block' }}>
              <input
                type="checkbox"
                checked={selectedLandmarks.includes(lm.name)}
                onChange={() => toggleLandmark(lm.name)}
                disabled={running}
              />
              {lm.name}
              {completedKeys.has(
                configKey(selectedAgents[0] ?? '', lm.name, neighborhoodSize, {
                  trunk: selectedAgents[0] === 'ppo' ? ppoTrunk : undefined,
                  zeroDirection,
                }),
              ) && ' (cached)'}
            </label>
          ))}
        </div>
      </div>

      <div style={{ marginBottom: 8, fontSize: 12, color: '#888' }}>
        Total: {totalSelected} configs x {numEpisodes} episodes
        {alreadyDone > 0 && (
          <span style={{ color: '#2a7' }}> — {alreadyDone} cached, {remaining} remaining</span>
        )}
      </div>

      {savedResults.length > 0 && !running && (
        <div style={{ marginBottom: 8, fontSize: 12, padding: '4px 8px', background: '#e8f5e9', borderRadius: 4 }}>
          {savedResults.length} config(s) saved in cache. Click Run to resume, or{' '}
          <button onClick={handleClearCache} style={{ fontSize: 11 }}>
            clear cache
          </button>
        </div>
      )}

      {!running ? (
        <button
          onClick={handleRun}
          disabled={!volumeData || selectedAgents.length === 0 || selectedLandmarks.length === 0}
          style={{ marginRight: 8 }}
        >
          {alreadyDone > 0 ? `Resume (${remaining} remaining)` : 'Run Experiments'}
        </button>
      ) : (
        <button onClick={handleAbort} style={{ marginRight: 8 }}>
          Abort
        </button>
      )}

      {(results || savedResults.length > 0) && !running && (
        <button onClick={handleDownload}>Download Results</button>
      )}

      {progress && (
        <div style={{ marginTop: 8, fontSize: 12, fontFamily: 'monospace' }}>{progress}</div>
      )}
    </div>
  );
}
