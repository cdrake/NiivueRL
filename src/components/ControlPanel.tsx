interface Props {
  running: boolean;
  onStart: () => void;
  onStop: () => void;
  onReset: () => void;
  speed: number;
  onSpeedChange: (ms: number) => void;
  disabled?: boolean;
}

export default function ControlPanel({
  running,
  onStart,
  onStop,
  onReset,
  speed,
  onSpeedChange,
  disabled,
}: Props) {
  return (
    <div className="panel-section">
      <h3>Controls</h3>
      <div className="button-row">
        <button onClick={onStart} disabled={running || disabled}>
          Start
        </button>
        <button onClick={onStop} disabled={!running}>
          Stop
        </button>
        <button onClick={onReset} disabled={running}>
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
    </div>
  );
}
