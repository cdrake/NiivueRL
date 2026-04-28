import { useCallback, useEffect, useRef, useState } from 'react';
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
import { GoalVectorModel } from '../lib/goalVectorModel';

type GoalVectorMode = 'oracle' | 'predicted' | 'zero';

interface ExperimentPanelProps {
  volumeData: ArrayLike<number> | null;
  dims: [number, number, number] | null;
}

/** Schema for /experiments/<name>.json files. All fields optional. */
interface ExperimentSpec {
  name?: string;
  description?: string;
  agents?: string[];
  landmarks?: string[];
  episodes?: number;
  replicates?: number;
  neighborhood?: number;
  strides?: number[];
  dirScale?: number | number[];
  zeroDir?: boolean;
  maxStartDist?: number;
  trunk?: 'flat' | 'meshnet' | 'conv';
  ppo?: { lr?: number; entropy?: number; clip?: number; epochs?: number; mb?: number; roll?: number };
  a2c?: { lr?: number; entropy?: number };
  /** Linear curriculum over starting radius: start -> end over anneal episodes. */
  curriculum?: { start: number; end: number; anneal: number };
  /**
   * Source of the direction-to-target signal in the env's state vector.
   *   - 'oracle' (default): exact (target - position) / |·|, only available
   *     because the env knows where the answer is — clinically unacceptable.
   *   - 'predicted': output of a pre-trained CNN that takes a local T1 patch +
   *     position + target one-hot and predicts a unit direction. This is the
   *     deploy-time signal.
   *   - 'zero': ablation, [0, 0, 0] direction.
   */
  goalVector?: GoalVectorMode;
  /** URL prefix for the tfjs goal-vector model directory (model.json + metadata.json). */
  goalVectorModelUrl?: string;
  autorun?: boolean;
  autodownload?: boolean;
  clearCache?: boolean;
}

const DEFAULT_GOAL_VECTOR_MODEL_URL = '/goal_vector_model';

// Diverse subset: large/deep, small/curved, large/easy, distinct, small/hard
const DEFAULT_LANDMARKS = [
  'Thalamus',
  'Hippocampus',
  'Lateral-Ventricle',
  'Brain-Stem',
  'Putamen',
];

const AGENT_TYPES: AgentType[] = ['dqn', 'a2c', 'ppo', 'oracle', 'random'];
const DEFAULT_EPISODES = 300;
const DEFAULT_MAX_START_DISTANCE = 50;
const NEIGHBORHOOD_SIZE_OPTIONS = [3, 5, 7, 9, 11, 13, 15, 19, 25];
const DEFAULT_NEIGHBORHOOD_SIZE = 7;
const DEFAULT_STRIDES = '1';
const DEFAULT_REPLICATES = 3;
const DEFAULT_DIRECTION_SCALE = 1;

// --- URL query-string helpers --------------------------------------------
// Supported params (all optional; unset → use default):
//   agents=ppo,oracle          landmarks=Thalamus,Hippocampus
//   episodes=600               replicates=3
//   neighborhood=7             strides=1,4
//   dirScale=10                zeroDir=1
//   maxStartDist=50            trunk=flat|meshnet|conv
//   ppoLr=0.0003  ppoEntropy=0.01  ppoClip=0.2  ppoEpochs=4  ppoMb=64  ppoRoll=4
//   a2cLr=0.0003  a2cEntropy=0.01
//   autorun=1                  autodownload=1           clearCache=1
const URL_PARAMS = typeof window !== 'undefined'
  ? new URLSearchParams(window.location.search)
  : new URLSearchParams();

const qNum = (k: string, fallback: number): number => {
  const v = URL_PARAMS.get(k);
  const n = v !== null ? Number(v) : NaN;
  return Number.isFinite(n) ? n : fallback;
};
const qStr = (k: string, fallback: string): string => URL_PARAMS.get(k) ?? fallback;
const qBool = (k: string, fallback: boolean): boolean => {
  const v = URL_PARAMS.get(k);
  if (v === null) return fallback;
  return v === '1' || v === 'true' || v === 'yes';
};
const qList = (k: string, fallback: string[]): string[] => {
  const v = URL_PARAMS.get(k);
  return v !== null ? v.split(',').map((s) => s.trim()).filter(Boolean) : fallback;
};

export default function ExperimentPanel({ volumeData, dims }: ExperimentPanelProps) {
  const [running, setRunning] = useState(false);
  const [progress, setProgress] = useState('');
  const [results, setResults] = useState<ExperimentResult[] | null>(null);
  const [numEpisodes, setNumEpisodes] = useState(() => qNum('episodes', DEFAULT_EPISODES));
  const [selectedAgents, setSelectedAgents] = useState<AgentType[]>(
    () => qList('agents', ['dqn']).filter((a): a is AgentType => (AGENT_TYPES as string[]).includes(a)),
  );
  const [selectedLandmarks, setSelectedLandmarks] = useState<string[]>(() => qList('landmarks', DEFAULT_LANDMARKS));
  const [maxStartDistance, setMaxStartDistance] = useState(() => qNum('maxStartDist', DEFAULT_MAX_START_DISTANCE));
  const [neighborhoodSize, setNeighborhoodSize] = useState(() => qNum('neighborhood', DEFAULT_NEIGHBORHOOD_SIZE));
  const [stridesInput, setStridesInput] = useState(() => qStr('strides', DEFAULT_STRIDES));
  const [replicates, setReplicates] = useState(() => Math.max(1, qNum('replicates', DEFAULT_REPLICATES)));
  const [directionScaleInput, setDirectionScaleInput] = useState(() =>
    qStr('dirScale', String(DEFAULT_DIRECTION_SCALE)),
  );
  const [zeroDirection, setZeroDirection] = useState(() => qBool('zeroDir', false));
  const [a2cLr, setA2cLr] = useState(() => qNum('a2cLr', DEFAULT_A2C_CONFIG.lr));
  const [a2cEntropy, setA2cEntropy] = useState(() => qNum('a2cEntropy', DEFAULT_A2C_CONFIG.entropyCoeff));
  const [a2cUseConv, setA2cUseConv] = useState(DEFAULT_A2C_CONFIG.useConvBackbone);
  const [ppoLr, setPpoLr] = useState(() => qNum('ppoLr', DEFAULT_PPO_CONFIG.lr));
  const [ppoEntropy, setPpoEntropy] = useState(() => qNum('ppoEntropy', DEFAULT_PPO_CONFIG.entropyCoeff));
  const [ppoClipEpsilon, setPpoClipEpsilon] = useState(() => qNum('ppoClip', DEFAULT_PPO_CONFIG.clipEpsilon));
  const [ppoNumEpochs, setPpoNumEpochs] = useState(() => qNum('ppoEpochs', DEFAULT_PPO_CONFIG.numEpochs));
  const [ppoMinibatchSize, setPpoMinibatchSize] = useState(() => qNum('ppoMb', DEFAULT_PPO_CONFIG.minibatchSize));
  const [ppoRolloutSize, setPpoRolloutSize] = useState(() => qNum('ppoRoll', DEFAULT_PPO_CONFIG.rolloutSize));
  const [ppoUseConv, setPpoUseConv] = useState(DEFAULT_PPO_CONFIG.useConvBackbone);
  const [ppoTrunk, setPpoTrunk] = useState<'flat' | 'meshnet' | 'conv'>(
    () => qStr('trunk', DEFAULT_PPO_CONFIG.trunk ?? 'flat') as 'flat' | 'meshnet' | 'conv',
  );
  // Curriculum over starting radius (start, end, annealEpisodes). Unset = off.
  const [curriculum, setCurriculum] = useState<{ start: number; end: number; anneal: number } | null>(null);
  // Goal-vector source: oracle (default), predicted (CNN), or zero (ablation).
  const [goalVectorMode, setGoalVectorMode] = useState<GoalVectorMode>(() =>
    (qStr('goalVector', 'oracle') as GoalVectorMode),
  );
  const [goalVectorModelUrl, setGoalVectorModelUrl] = useState<string>(() =>
    qStr('goalVectorModelUrl', DEFAULT_GOAL_VECTOR_MODEL_URL),
  );
  const goalVectorModelRef = useRef<GoalVectorModel | null>(null);
  const runnerRef = useRef<ExperimentRunner | null>(null);
  const autorunRef = useRef<boolean>(qBool('autorun', false));
  const autodownloadRef = useRef<boolean>(qBool('autodownload', false));
  const experimentName = qStr('experiment', '');
  const [experimentReady, setExperimentReady] = useState<boolean>(experimentName === '');
  const [experimentSpec, setExperimentSpec] = useState<{ name?: string; description?: string } | null>(null);

  // One-shot cache clear via ?clearCache=1 (runs before first real render uses storage)
  const clearedRef = useRef(false);
  if (!clearedRef.current && qBool('clearCache', false)) {
    clearedRef.current = true;
    clearStoredResults();
  }

  // ?experiment=<name> — fetch /experiments/<name>.json and apply its settings
  useEffect(() => {
    if (!experimentName) return;
    let cancelled = false;
    fetch(`/experiments/${experimentName}.json`)
      .then((r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status} for /experiments/${experimentName}.json`);
        return r.json();
      })
      .then((spec: ExperimentSpec) => {
        if (cancelled) return;
        if (spec.clearCache) clearStoredResults();
        if (Array.isArray(spec.agents)) {
          setSelectedAgents(spec.agents.filter((a): a is AgentType => (AGENT_TYPES as string[]).includes(a)));
        }
        if (Array.isArray(spec.landmarks)) setSelectedLandmarks(spec.landmarks);
        if (typeof spec.episodes === 'number') setNumEpisodes(spec.episodes);
        if (typeof spec.replicates === 'number') setReplicates(Math.max(1, spec.replicates));
        if (typeof spec.neighborhood === 'number') setNeighborhoodSize(spec.neighborhood);
        if (Array.isArray(spec.strides)) setStridesInput(spec.strides.join(','));
        if (typeof spec.dirScale === 'number') setDirectionScaleInput(String(spec.dirScale));
        else if (Array.isArray(spec.dirScale)) setDirectionScaleInput(spec.dirScale.join(','));
        if (typeof spec.zeroDir === 'boolean') setZeroDirection(spec.zeroDir);
        if (spec.goalVector === 'oracle' || spec.goalVector === 'predicted' || spec.goalVector === 'zero') {
          setGoalVectorMode(spec.goalVector);
        }
        if (typeof spec.goalVectorModelUrl === 'string') setGoalVectorModelUrl(spec.goalVectorModelUrl);
        if (typeof spec.maxStartDist === 'number') setMaxStartDistance(spec.maxStartDist);
        if (spec.trunk === 'flat' || spec.trunk === 'meshnet' || spec.trunk === 'conv') setPpoTrunk(spec.trunk);
        if (spec.ppo) {
          if (typeof spec.ppo.lr === 'number') setPpoLr(spec.ppo.lr);
          if (typeof spec.ppo.entropy === 'number') setPpoEntropy(spec.ppo.entropy);
          if (typeof spec.ppo.clip === 'number') setPpoClipEpsilon(spec.ppo.clip);
          if (typeof spec.ppo.epochs === 'number') setPpoNumEpochs(spec.ppo.epochs);
          if (typeof spec.ppo.mb === 'number') setPpoMinibatchSize(spec.ppo.mb);
          if (typeof spec.ppo.roll === 'number') setPpoRolloutSize(spec.ppo.roll);
        }
        if (spec.a2c) {
          if (typeof spec.a2c.lr === 'number') setA2cLr(spec.a2c.lr);
          if (typeof spec.a2c.entropy === 'number') setA2cEntropy(spec.a2c.entropy);
        }
        if (spec.curriculum &&
            typeof spec.curriculum.start === 'number' &&
            typeof spec.curriculum.end === 'number' &&
            typeof spec.curriculum.anneal === 'number') {
          setCurriculum({
            start: spec.curriculum.start,
            end: spec.curriculum.end,
            anneal: spec.curriculum.anneal,
          });
        }
        if (spec.autorun) autorunRef.current = true;
        if (spec.autodownload) autodownloadRef.current = true;
        setExperimentSpec({ name: spec.name ?? experimentName, description: spec.description });
        setExperimentReady(true);
        console.log(`[experiment] loaded ${experimentName}`, spec);
      })
      .catch((err) => {
        if (cancelled) return;
        console.error('[experiment] failed to load', err);
        setExperimentReady(true); // unblock UI even on failure
      });
    return () => {
      cancelled = true;
    };
  }, [experimentName]);

  // Parse strides from comma-separated string
  const strides = stridesInput
    .split(',')
    .map((s) => parseInt(s.trim()))
    .filter((n) => !isNaN(n) && n >= 1);

  // Parse direction scales from comma-separated string (supports sweep)
  const directionScales = (() => {
    const parsed = directionScaleInput
      .split(',')
      .map((s) => parseFloat(s.trim()))
      .filter((n) => !isNaN(n) && n >= 0);
    return parsed.length > 0 ? parsed : [DEFAULT_DIRECTION_SCALE];
  })();

  // Check how many configs are already saved
  const savedResults = loadResultsFromStorage();
  const completedKeys = getCompletedKeys();

  const handleRun = useCallback(async () => {
    if (!volumeData || !dims || strides.length === 0) return;

    setRunning(true);

    // Lazy-load the goal-vector model (cached across runs).
    let goalVectorModel: GoalVectorModel | null = null;
    if (goalVectorMode === 'predicted') {
      try {
        if (!goalVectorModelRef.current) {
          setProgress(`Loading goal-vector model from ${goalVectorModelUrl}...`);
          goalVectorModelRef.current = await GoalVectorModel.load(goalVectorModelUrl);
          console.log('[goal-vector] model loaded', goalVectorModelRef.current.metadata);
        }
        goalVectorModel = goalVectorModelRef.current;
      } catch (err) {
        setProgress(`Failed to load goal-vector model: ${err}. Aborting.`);
        setRunning(false);
        return;
      }
      // Sanity-check: at the chosen landmarks, sample a few offsets and report
      // mean cosine of the predicted unit vector vs the oracle direction. This
      // makes it obvious in the console when the deployment volume is OOD or
      // the patch indexing has drifted from training.
      if (goalVectorModel && volumeData && dims) {
        let voxelMin = Infinity, voxelMax = -Infinity;
        for (let i = 0; i < volumeData.length; i++) {
          const v = volumeData[i];
          if (v < voxelMin) voxelMin = v;
          if (v > voxelMax) voxelMax = v;
        }
        const rand = (seed => () => { seed = (1664525 * seed + 1013904223) | 0; return ((seed >>> 0) % 1_000_000) / 1_000_000; })(42);
        for (const lmName of selectedLandmarks) {
          const lm = LANDMARKS.find((l) => l.name === lmName);
          if (!lm) continue;
          const idx = goalVectorModel.landmarkIndex(lmName);
          if (idx < 0) continue;
          const c = lm.mniVoxel;
          let sumCos = 0; const n = 50;
          for (let i = 0; i < n; i++) {
            const ox = (rand() * 2 - 1) * 30, oy = (rand() * 2 - 1) * 30, oz = (rand() * 2 - 1) * 30;
            const pos = {
              x: Math.max(3, Math.min(dims[0] - 4, Math.round(c.x + ox))),
              y: Math.max(3, Math.min(dims[1] - 4, Math.round(c.y + oy))),
              z: Math.max(3, Math.min(dims[2] - 4, Math.round(c.z + oz))),
            };
            const [dx, dy, dz] = goalVectorModel.predict(volumeData, dims, voxelMin, voxelMax, pos, idx);
            const tx = c.x - pos.x, ty = c.y - pos.y, tz = c.z - pos.z;
            const tn = Math.hypot(tx, ty, tz) || 1;
            sumCos += (dx * tx + dy * ty + dz * tz) / tn;
          }
          console.log(`[goal-vector sanity] ${lmName}: mean_cos=${(sumCos / n).toFixed(4)} over ${n} samples`);
        }
      }
    }
    const effectiveZeroDir = goalVectorMode === 'zero' || zeroDirection;

    // Merge any previously saved results into state
    const prior = loadResultsFromStorage();
    const skipKeys = new Set(
      prior.map((r) =>
        configKey(r.config.agentType, r.config.landmark, r.config.neighborhoodSize, r.config.strides, {
          trunk: r.config.ppoConfig?.trunk,
          zeroDirection: r.config.zeroDirection,
          seed: r.config.seed,
          directionScale: r.config.directionScale,
          curriculum: r.config.curriculum,
          goalVector: r.config.goalVector,
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
    for (const ds of directionScales) {
      for (const agentType of selectedAgents) {
        for (const lmName of selectedLandmarks) {
          const lm = LANDMARKS.find((l) => l.name === lmName);
          if (!lm) continue;
          const clamped = clampLandmarkToVolume(lm, dims);
          for (let seed = 0; seed < replicates; seed++) {
            configs.push({
              landmark: clamped,
              agentType,
              numEpisodes,
              seed,
              envConfig: { maxStartDistance, zeroDirection: effectiveZeroDir, directionScale: ds },
              neighborhoodSize,
              strides,
              ...(curriculum
                ? { curriculum: { start: curriculum.start, end: curriculum.end, annealEpisodes: curriculum.anneal } }
                : {}),
              ...(agentType === 'a2c' ? { a2cConfig } : {}),
              ...(agentType === 'ppo' ? { ppoConfig } : {}),
              ...(goalVectorModel ? { goalVectorModel } : {}),
            });
          }
        }
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
    strides,
    replicates,
    directionScaleInput,
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
    curriculum,
    goalVectorMode,
    goalVectorModelUrl,
  ]);

  const handleAbort = useCallback(() => {
    runnerRef.current?.abort();
  }, []);

  // ?autorun=1 (or experiment spec autorun:true) — kick off handleRun once
  // the volume is ready AND the experiment spec (if any) has been applied.
  useEffect(() => {
    if (autorunRef.current && volumeData && dims && experimentReady && !running) {
      autorunRef.current = false;
      void handleRun();
    }
  }, [volumeData, dims, experimentReady, running, handleRun]);

  const handleDownload = useCallback(() => {
    // Download from localStorage (includes all saved results)
    const all = loadResultsFromStorage();
    if (all.length > 0) {
      downloadResults(all);
    } else if (results) {
      downloadResults(results);
    }
  }, [results]);

  // ?autodownload=1 — trigger download once the run finishes
  useEffect(() => {
    if (autodownloadRef.current && !autorunRef.current && !running && results) {
      autodownloadRef.current = false;
      handleDownload();
    }
  }, [results, running, handleDownload]);

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
  const totalSelected =
    directionScales.length * selectedAgents.length * selectedLandmarks.length * replicates;
  const curriculumForKey = curriculum
    ? { start: curriculum.start, end: curriculum.end, annealEpisodes: curriculum.anneal }
    : undefined;
  const effectiveZeroDirForKey = goalVectorMode === 'zero' || zeroDirection;
  const alreadyDone = selectedAgents.reduce((count, agent) => {
    const trunk = agent === 'ppo' ? ppoTrunk : undefined;
    let agentCount = 0;
    for (const ds of directionScales) {
      for (const lm of selectedLandmarks) {
        for (let seed = 0; seed < replicates; seed++) {
          if (completedKeys.has(configKey(agent, lm, neighborhoodSize, strides, {
            trunk, zeroDirection: effectiveZeroDirForKey, seed, directionScale: ds,
            curriculum: curriculumForKey, goalVector: goalVectorMode,
          }))) {
            agentCount++;
          }
        }
      }
    }
    return count + agentCount;
  }, 0);
  const remaining = totalSelected - alreadyDone;

  return (
    <div style={{ padding: '8px 0' }}>
      <h3 style={{ margin: '0 0 8px' }}>Experiments</h3>

      {experimentSpec && (
        <div style={{ marginBottom: 8, padding: '6px 8px', background: '#e3f2fd', borderRadius: 4, fontSize: 12 }}>
          <div><strong>Loaded:</strong> {experimentSpec.name}</div>
          {experimentSpec.description && <div style={{ color: '#555' }}>{experimentSpec.description}</div>}
        </div>
      )}

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

      <div style={{ marginBottom: 8, display: 'flex', gap: 16 }}>
        <label style={{ fontSize: 13 }}>
          Neighborhood:{' '}
          <select
            value={neighborhoodSize}
            onChange={(e) => setNeighborhoodSize(Number(e.target.value))}
            disabled={running}
            style={{ fontSize: 13 }}
          >
            {NEIGHBORHOOD_SIZE_OPTIONS.map((n) => (
              <option key={n} value={n}>
                {n}³
              </option>
            ))}
          </select>
        </label>

        <label style={{ fontSize: 13 }}>
          Strides (e.g. 1,2,4):{' '}
          <input
            type="text"
            value={stridesInput}
            onChange={(e) => setStridesInput(e.target.value)}
            placeholder="1"
            style={{ width: 60 }}
            disabled={running}
          />
        </label>

        <label style={{ fontSize: 13 }}>
          Replicates:{' '}
          <input
            type="number"
            value={replicates}
            onChange={(e) => setReplicates(Math.max(1, Number(e.target.value)))}
            min={1}
            max={20}
            step={1}
            style={{ width: 50 }}
            disabled={running}
          />
        </label>
      </div>

      <div style={{ marginBottom: 8, display: 'flex', gap: 16, alignItems: 'center' }}>
        <label style={{ fontSize: 13 }}>
          Direction scale (sweep: 10,30,100):{' '}
          <input
            type="text"
            value={directionScaleInput}
            onChange={(e) => setDirectionScaleInput(e.target.value)}
            placeholder="1"
            style={{ width: 100 }}
            disabled={running || zeroDirection}
          />
        </label>
        <label style={{ fontSize: 13 }}>
          <input
            type="checkbox"
            checked={zeroDirection}
            onChange={(e) => setZeroDirection(e.target.checked)}
            disabled={running || goalVectorMode !== 'oracle'}
          />
          Zero direction (ablation)
        </label>
        <label style={{ fontSize: 13 }}>
          Goal vector:{' '}
          <select
            value={goalVectorMode}
            onChange={(e) => setGoalVectorMode(e.target.value as GoalVectorMode)}
            disabled={running}
            style={{ fontSize: 13 }}
          >
            <option value="oracle">oracle (cheating)</option>
            <option value="predicted">predicted (CNN)</option>
            <option value="zero">zero (ablation)</option>
          </select>
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
            disabled={running || !!curriculum}
          />
        </label>
        {curriculum && (
          <span style={{ marginLeft: 8, fontSize: 12, color: '#2a7' }}>
            curriculum active: {curriculum.start} → {curriculum.end} over {curriculum.anneal} eps
          </span>
        )}
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
          {LANDMARKS.map((lm) => {
            const agent = selectedAgents[0] ?? '';
            const trunk = agent === 'ppo' ? ppoTrunk : undefined;
            let cached = 0;
            for (const ds of directionScales) {
              for (let seed = 0; seed < replicates; seed++) {
                if (completedKeys.has(configKey(agent, lm.name, neighborhoodSize, strides, {
                  trunk, zeroDirection: effectiveZeroDirForKey, seed, directionScale: ds,
                  curriculum: curriculumForKey, goalVector: goalVectorMode,
                }))) {
                  cached++;
                }
              }
            }
            const totalPerLm = directionScales.length * replicates;
            return (
              <label key={lm.name} style={{ display: 'block' }}>
                <input
                  type="checkbox"
                  checked={selectedLandmarks.includes(lm.name)}
                  onChange={() => toggleLandmark(lm.name)}
                  disabled={running}
                />
                {lm.name}
                {cached > 0 && ` (${cached}/${totalPerLm} cached)`}
              </label>
            );
          })}
        </div>
      </div>

      <div style={{ marginBottom: 8, fontSize: 12, color: '#888' }}>
        Total: {directionScales.length}×{selectedAgents.length}×{selectedLandmarks.length}×{replicates} = {totalSelected} configs × {numEpisodes} episodes
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
