import { LANDMARKS } from '../lib/landmarks';
import type { Landmark } from '../env/types';

interface Props {
  selected: Landmark | null;
  onSelect: (landmark: Landmark) => void;
  disabled?: boolean;
}

export default function LandmarkSelector({ selected, onSelect, disabled }: Props) {
  return (
    <div className="panel-section">
      <h3>Target Landmark</h3>
      <select
        value={selected?.name ?? ''}
        onChange={(e) => {
          const lm = LANDMARKS.find(l => l.name === e.target.value);
          if (lm) onSelect(lm);
        }}
        disabled={disabled}
      >
        <option value="" disabled>Select landmark...</option>
        {LANDMARKS.map(lm => (
          <option key={lm.name} value={lm.name}>
            {lm.name}
          </option>
        ))}
      </select>
    </div>
  );
}
