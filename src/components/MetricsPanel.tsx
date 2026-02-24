import { useEffect, useRef } from 'react';
import type { TrainingMetrics } from '../training/TrainingLoop';

interface Props {
  metrics: TrainingMetrics | null;
}

export default function MetricsPanel({ metrics }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    if (!canvasRef.current || !metrics?.rewardHistory.length) return;
    drawChart(canvasRef.current, metrics.rewardHistory);
  }, [metrics?.rewardHistory.length]);

  return (
    <div className="panel-section">
      <h3>Metrics</h3>
      <div className="metrics-grid">
        <span>Episode:</span>
        <span>{metrics?.episode ?? 0}</span>
        <span>Epsilon:</span>
        <span>{metrics?.epsilon?.toFixed(3) ?? '1.000'}</span>
        <span>Reward:</span>
        <span>{metrics?.totalReward?.toFixed(1) ?? '0.0'}</span>
        <span>Distance:</span>
        <span>{metrics?.distance?.toFixed(1) ?? '-'} vox</span>
      </div>
      <canvas
        ref={canvasRef}
        width={220}
        height={120}
        className="reward-chart"
      />
    </div>
  );
}

function drawChart(canvas: HTMLCanvasElement, data: number[]) {
  const ctx = canvas.getContext('2d');
  if (!ctx || data.length < 2) return;

  const w = canvas.width;
  const h = canvas.height;
  const padding = 5;

  ctx.clearRect(0, 0, w, h);

  const min = Math.min(...data);
  const max = Math.max(...data);
  const range = max === min ? 1 : max - min;

  ctx.strokeStyle = '#444';
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(padding, h - padding);
  ctx.lineTo(w - padding, h - padding);
  ctx.stroke();

  ctx.strokeStyle = '#4fc3f7';
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  for (let i = 0; i < data.length; i++) {
    const x = padding + (i / (data.length - 1)) * (w - 2 * padding);
    const y = h - padding - ((data[i] - min) / range) * (h - 2 * padding);
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  ctx.stroke();
}
