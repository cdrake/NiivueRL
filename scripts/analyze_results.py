#!/usr/bin/env python3
"""Analyze experiment results and generate figures for the progress report."""

import json
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def load_results(path: str) -> list[dict]:
    """Load one results file, or merge many if a directory is given."""
    if os.path.isdir(path):
        merged = []
        for fn in sorted(os.listdir(path)):
            if fn.endswith('.json'):
                with open(os.path.join(path, fn)) as f:
                    merged.extend(json.load(f))
        return merged
    with open(path) as f:
        return json.load(f)

def neighborhood_size(run: dict) -> int:
    return int(run['config'].get('neighborhoodSize', 7))

def strides_label(run: dict) -> str:
    s = run['config'].get('strides', [1])
    return ','.join(map(str, s))

def direction_scale(run: dict) -> float:
    return float(run['config'].get('directionScale', 1))

def curriculum_info(run: dict) -> tuple:
    """Return (start, end, anneal) tuple, or (0,0,0) sentinel if no curriculum.
    Use a sentinel (not None) so group keys remain totally ordered."""
    cur = run['config'].get('curriculum')
    if not cur:
        return (0, 0, 0)
    return (cur.get('start', 0), cur.get('end', 0), cur.get('annealEpisodes', 0))

def has_curriculum(run: dict) -> bool:
    return curriculum_info(run) != (0, 0, 0)

def run_label(run: dict) -> str:
    """Algorithm label including neighborhood size, strides, dirScale, trunk, and curriculum."""
    agent = run['config']['agentType']
    n = neighborhood_size(run)
    s = strides_label(run)
    trunk = (run['config'].get('ppoConfig') or {}).get('trunk')
    ds = direction_scale(run)
    parts = [agent.upper()]
    if n != 7:
        parts.append(f"n{n}")
    if s != '1':
        parts.append(f"s{s}")
    if trunk and trunk != 'flat':
        parts.append(trunk)
    if ds != 1:
        parts.append(f"ds{ds:g}")
    if run['config'].get('zeroDirection'):
        parts.append('nodir')
    if has_curriculum(run):
        cur = curriculum_info(run)
        parts.append(f"cur{cur[0]}-{cur[1]}@{cur[2]}")
    return '-'.join(parts)

def group_key(run: dict) -> tuple:
    """Key that groups replicates (same config, different seed) together."""
    c = run['config']
    trunk = (c.get('ppoConfig') or {}).get('trunk') or 'flat'
    return (
        c['agentType'],
        c['landmark'],
        neighborhood_size(run),
        strides_label(run),
        trunk,
        bool(c.get('zeroDirection', False)),
        direction_scale(run),
        curriculum_info(run),
    )

def group_runs(results: list[dict]) -> dict[tuple, list[dict]]:
    """Group runs by (agent, landmark, n, strides, trunk, zeroDir) — i.e. replicates together."""
    groups: dict[tuple, list[dict]] = {}
    for r in results:
        groups.setdefault(group_key(r), []).append(r)
    return groups

def _stack(metric: str, runs: list[dict]) -> np.ndarray:
    """Stack per-episode values across replicates, truncating to shortest run."""
    arrays = [np.array([ep[metric] for ep in r['episodes']]) for r in runs]
    min_len = min(len(a) for a in arrays)
    return np.stack([a[:min_len] for a in arrays], axis=0)  # (n_seeds, n_episodes)

def _smooth_row(arr: np.ndarray, window: int) -> np.ndarray:
    if arr.shape[-1] < window:
        return arr
    kernel = np.ones(window) / window
    return np.apply_along_axis(lambda x: np.convolve(x, kernel, mode='valid'), -1, arr)

def _mean_ci(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (mean, half-width of 95% CI) across the seed axis (axis=0)."""
    n = arr.shape[0]
    mean = arr.mean(axis=0)
    if n <= 1:
        return mean, np.zeros_like(mean)
    sem = arr.std(axis=0, ddof=1) / np.sqrt(n)
    return mean, 1.96 * sem

def smooth(values: list[float], window: int = 20) -> np.ndarray:
    """Simple moving average smoothing."""
    arr = np.array(values)
    if len(arr) < window:
        return arr
    kernel = np.ones(window) / window
    return np.convolve(arr, kernel, mode='valid')

def _grid(n):
    import math
    if n <= 5:
        return 1, n
    cols = 5
    rows = math.ceil(n / cols)
    return rows, cols

def color_for(run):
    base_colors = {'dqn': '#2196F3', 'a2c': '#FF5722', 'ppo': '#4CAF50',
                   'oracle': '#000000', 'random': '#BDBDBD'}
    c = base_colors.get(run['config']['agentType'], '#999')
    # Shade based on strides: multi-scale is darker
    s = run['config'].get('strides', [1])
    shade = 1.0
    if len(s) > 1:
        shade = 0.7  # Multi-scale variant

    # Further adjust by neighborhood size if not 7
    n = neighborhood_size(run)
    if n > 7: shade *= 0.8
    elif n < 7: shade *= 1.1

    # Further adjust by dirScale: higher dirScale -> darker
    ds = direction_scale(run)
    if ds >= 100: shade *= 0.55
    elif ds >= 30: shade *= 0.75
    elif ds >= 10: shade *= 0.9
    shade = max(0.2, min(1.0, shade))

    # Darken by shade factor (mix toward black)
    r, g, b = int(c[1:3], 16), int(c[3:5], 16), int(c[5:7], 16)
    return f'#{int(r*shade):02x}{int(g*shade):02x}{int(b*shade):02x}'

def _plot_metric(
    results: list[dict],
    output_dir: str,
    metric: str,
    *,
    filename: str,
    ylabel: str,
    title: str,
    window: int = 20,
    transform=None,
    ylim=None,
):
    """Shared machinery: one subplot per landmark, one line per (agent×config) group with CI band across seeds."""
    landmarks = sorted(set(r['config']['landmark'] for r in results))
    nrows, ncols = _grid(len(landmarks))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 2.6 * nrows), sharey=True)
    axes = np.atleast_1d(axes).flatten()

    grouped = group_runs(results)

    for ax, lm in zip(axes, landmarks):
        for key, runs in sorted(grouped.items()):
            if key[1] != lm:
                continue
            arr = _stack(metric, runs)
            if transform is not None:
                arr = transform(arr)
            smoothed = _smooth_row(arr, window)
            mean, ci = _mean_ci(smoothed)
            episodes = np.arange(mean.shape[-1]) + window // 2
            color = color_for(runs[0])
            label = run_label(runs[0]) + (f' (n={len(runs)})' if len(runs) > 1 else '')
            ax.plot(episodes, mean, label=label, color=color, linewidth=1.5)
            if len(runs) > 1:
                ax.fill_between(episodes, mean - ci, mean + ci, color=color, alpha=0.2, linewidth=0)

        ax.set_title(lm, fontsize=9)
        ax.set_xlabel('Episode', fontsize=8)
        ax.set_ylabel(ylabel, fontsize=8)
        if ylim is not None:
            ax.set_ylim(*ylim)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=7)

    for ax in axes[len(landmarks):]:
        ax.axis('off')
    fig.suptitle(title, fontsize=12, fontweight='bold')
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, filename), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {filename}")

def plot_reward_curves(results: list[dict], output_dir: str):
    _plot_metric(
        results, output_dir,
        metric='totalReward',
        filename='reward_curves.png',
        ylabel='Reward (smoothed)',
        title='Reward Curves by Landmark (mean ± 95% CI across seeds)',
        window=20,
    )

def plot_success_rate(results: list[dict], output_dir: str):
    _plot_metric(
        results, output_dir,
        metric='success',
        filename='success_rate.png',
        ylabel='Success Rate (%)',
        title='Success Rate by Landmark (30-episode window, mean ± 95% CI)',
        window=30,
        transform=lambda arr: arr.astype(float) * 100.0,
        ylim=(-1, 101),
    )

def plot_final_distance(results: list[dict], output_dir: str):
    _plot_metric(
        results, output_dir,
        metric='finalDistance',
        filename='final_distance.png',
        ylabel='Final Distance (voxels)',
        title='Final Distance to Target by Landmark (mean ± 95% CI)',
        window=30,
    )

def generate_summary_table(results: list[dict], output_dir: str):
    """Summary table aggregated across replicates (mean ± std over seeds, last 50 eps each)."""
    tail = 50
    rows = []

    for key, runs in sorted(group_runs(results).items()):
        # Per-seed mean over the last `tail` episodes, then mean±std across seeds.
        reward_means = [np.mean([ep['totalReward'] for ep in r['episodes'][-tail:]]) for r in runs]
        dist_means = [np.mean([ep['finalDistance'] for ep in r['episodes'][-tail:]]) for r in runs]
        succ_means = [np.mean([1.0 if ep['success'] else 0.0 for ep in r['episodes'][-tail:]]) for r in runs]
        step_means = [np.mean([ep['steps'] for ep in r['episodes'][-tail:]]) for r in runs]

        def fmt(xs, pct=False, decimals=1):
            m, s = float(np.mean(xs)), float(np.std(xs, ddof=1)) if len(xs) > 1 else 0.0
            if pct:
                return f"{m*100:.{decimals}f}% ± {s*100:.{decimals}f}" if len(xs) > 1 else f"{m*100:.{decimals}f}%"
            return f"{m:.{decimals}f} ± {s:.{decimals}f}" if len(xs) > 1 else f"{m:.{decimals}f}"

        rows.append({
            'Landmark': key[1],
            'Agent': run_label(runs[0]),
            'Seeds': str(len(runs)),
            'Reward': fmt(reward_means),
            'Success': fmt(succ_means, pct=True),
            'Distance': fmt(dist_means),
            'Steps': fmt(step_means, decimals=0),
        })

    # Print as formatted table
    if not rows:
        print("No results to summarize.")
        return

    headers = list(rows[0].keys())
    col_widths = {h: max(len(h), max(len(str(r[h])) for r in rows)) for h in headers}

    header_line = ' | '.join(h.ljust(col_widths[h]) for h in headers)
    sep_line = '-+-'.join('-' * col_widths[h] for h in headers)

    lines = [header_line, sep_line]
    for row in sorted(rows, key=lambda r: (r['Landmark'], r['Agent'])):
        lines.append(' | '.join(str(row[h]).ljust(col_widths[h]) for h in headers))

    table_text = '\n'.join(lines)
    print('\nSummary (last 50 episodes):\n')
    print(table_text)

    with open(os.path.join(output_dir, 'summary_table.txt'), 'w') as f:
        f.write(table_text)
    print(f"\nSaved summary_table.txt")

def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_results.py <results.json> [output_dir]")
        sys.exit(1)

    results_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else 'docs/figures'

    os.makedirs(output_dir, exist_ok=True)
    results = load_results(results_path)

    print(f"Loaded {len(results)} experiment configs")
    for r in results:
        c = r['config']
        eps = r['episodes']
        successes = sum(1 for e in eps if e['success'])
        print(f"  {run_label(r)}: {len(eps)} episodes, {successes} successes ({successes/len(eps)*100:.1f}%)")

    plot_reward_curves(results, output_dir)
    plot_success_rate(results, output_dir)
    plot_final_distance(results, output_dir)
    generate_summary_table(results, output_dir)

if __name__ == '__main__':
    main()
