import type { A2CConfig } from '../agent/A2CAgent';
import { DEFAULT_A2C_CONFIG } from '../agent/A2CAgent';

export type AgentType = 'dqn' | 'a2c' | 'ppo';

interface Props {
  running: boolean;
  onStart: () => void;
  onStop: () => void;
  onReset: () => void;
  speed: number;
  onSpeedChange: (ms: number) => void;
  disabled?: boolean;
  agentType: AgentType;
  onAgentTypeChange: (type: AgentType) => void;
  showNeighborhood: boolean;
  onShowNeighborhoodChange: (show: boolean) => void;
  loading?: boolean;
  a2cConfig: Partial<A2CConfig>;
  onA2cConfigChange: (config: Partial<A2CConfig>) => void;
}

export default function ControlPanel({
  running,
  onStart,
  onStop,
  onReset,
  speed,
  onSpeedChange,
  disabled,
  agentType,
  onAgentTypeChange,
  showNeighborhood,
  onShowNeighborhoodChange,
  loading,
  a2cConfig,
  onA2cConfigChange,
}: Props) {
  const mergedA2c = { ...DEFAULT_A2C_CONFIG, ...a2cConfig };

  return (
    <div className="panel-section">
      <h3>Controls</h3>
      <label className="agent-label">
        Agent:
        <select
          value={agentType}
          onChange={(e) => onAgentTypeChange(e.target.value as AgentType)}
          disabled={running || loading}
        >
          <option value="dqn">DQN</option>
          <option value="a2c">A2C</option>
          <option value="ppo">PPO</option>
        </select>
      </label>
      {(agentType === 'a2c' || agentType === 'ppo') && (
        <div style={{ fontSize: 12, marginBottom: 8 }}>
          <label style={{ display: 'block', marginBottom: 2 }}>
            LR:{' '}
            <select
              value={mergedA2c.lr}
              onChange={(e) => onA2cConfigChange({ ...a2cConfig, lr: Number(e.target.value) })}
              disabled={running || loading}
              style={{ fontSize: 12 }}
            >
              <option value={0.0003}>3e-4</option>
              <option value={0.0001}>1e-4</option>
              <option value={0.00003}>3e-5</option>
            </select>
          </label>
          <label style={{ display: 'block', marginBottom: 2 }}>
            Entropy:{' '}
            <select
              value={mergedA2c.entropyCoeff}
              onChange={(e) => onA2cConfigChange({ ...a2cConfig, entropyCoeff: Number(e.target.value) })}
              disabled={running || loading}
              style={{ fontSize: 12 }}
            >
              <option value={0.01}>0.01</option>
              <option value={0.05}>0.05</option>
              <option value={0.1}>0.1</option>
            </select>
          </label>
          <label style={{ display: 'block' }}>
            <input
              type="checkbox"
              checked={mergedA2c.useConvBackbone}
              onChange={(e) => onA2cConfigChange({ ...a2cConfig, useConvBackbone: e.target.checked })}
              disabled={running || loading}
            />
            Conv backbone
          </label>
        </div>
      )}
      <div className="button-row">
        <button onClick={onStart} disabled={running || disabled || loading}>
          {loading ? 'Loading...' : 'Start'}
        </button>
        <button onClick={onStop} disabled={!running}>
          Stop
        </button>
        <button onClick={onReset} disabled={running || loading}>
          Reset
        </button>
      </div>
      <label className="speed-label">
        Speed: {speed}ms/step
        <input
          type="range"
          min={0}
          max={100}
          value={speed}
          onChange={(e) => onSpeedChange(Number(e.target.value))}
        />
      </label>
      <label className="neighborhood-label">
        <input
          type="checkbox"
          checked={showNeighborhood}
          onChange={(e) => onShowNeighborhoodChange(e.target.checked)}
        />
        Show Neighborhood
      </label>
    </div>
  );
}
