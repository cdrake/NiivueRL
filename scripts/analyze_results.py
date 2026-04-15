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

def run_label(run: dict) -> str:
    """Algorithm label including neighborhood size and trunk type."""
    agent = run['config']['agentType']
    n = neighborhood_size(run)
    trunk = (run['config'].get('ppoConfig') or {}).get('trunk')
    parts = [agent.upper()]
    if n != 7:
        parts.append(f"n{n}")
    if trunk and trunk != 'flat':
        parts.append(trunk)
    if run['config'].get('zeroDirection'):
        parts.append('nodir')
    return '-'.join(parts)

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

def plot_reward_curves(results: list[dict], output_dir: str):
    """Plot reward curves for each algorithm, grouped by landmark."""
    landmarks = sorted(set(r['config']['landmark'] for r in results))
    agents = sorted(set(r['config']['agentType'] for r in results))

    nrows, ncols = _grid(len(landmarks))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 2.6 * nrows), sharey=True)
    axes = np.atleast_1d(axes).flatten()

    # Base colors per algorithm; neighborhood-size variants are shaded from there
    base_colors = {'dqn': '#2196F3', 'a2c': '#FF5722', 'ppo': '#4CAF50'}
    size_shades = {5: 0.6, 7: 1.0, 9: 0.85, 11: 0.7, 13: 0.55, 15: 0.4, 19: 0.3, 25: 0.2}
    def color_for(run):
        c = base_colors.get(run['config']['agentType'], '#999')
        shade = size_shades.get(neighborhood_size(run), 1.0)
        # Darken by shade factor (mix toward black)
        r, g, b = int(c[1:3], 16), int(c[3:5], 16), int(c[5:7], 16)
        return f'#{int(r*shade):02x}{int(g*shade):02x}{int(b*shade):02x}'

    for ax, lm in zip(axes, landmarks):
        for r in results:
            if r['config']['landmark'] != lm:
                continue
            rewards = [ep['totalReward'] for ep in r['episodes']]
            smoothed = smooth(rewards)
            episodes = np.arange(len(smoothed)) + 10  # offset by half window
            ax.plot(episodes, smoothed, label=run_label(r),
                    color=color_for(r), linewidth=1.5)

        ax.set_title(lm, fontsize=9)
        ax.set_xlabel('Episode', fontsize=8)
        ax.set_ylabel('Reward (smoothed)', fontsize=8)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=7)

    for ax in axes[len(landmarks):]:
        ax.axis('off')
    fig.suptitle('Reward Curves by Landmark', fontsize=12, fontweight='bold')
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'reward_curves.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved reward_curves.png")

def plot_success_rate(results: list[dict], output_dir: str):
    """Plot success rate over training (rolling window)."""
    landmarks = sorted(set(r['config']['landmark'] for r in results))
    agents = sorted(set(r['config']['agentType'] for r in results))

    nrows, ncols = _grid(len(landmarks))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 2.6 * nrows), sharey=True)
    axes = np.atleast_1d(axes).flatten()

    # Base colors per algorithm; neighborhood-size variants are shaded from there
    base_colors = {'dqn': '#2196F3', 'a2c': '#FF5722', 'ppo': '#4CAF50'}
    size_shades = {5: 0.6, 7: 1.0, 9: 0.85, 11: 0.7, 13: 0.55, 15: 0.4, 19: 0.3, 25: 0.2}
    def color_for(run):
        c = base_colors.get(run['config']['agentType'], '#999')
        shade = size_shades.get(neighborhood_size(run), 1.0)
        # Darken by shade factor (mix toward black)
        r, g, b = int(c[1:3], 16), int(c[3:5], 16), int(c[5:7], 16)
        return f'#{int(r*shade):02x}{int(g*shade):02x}{int(b*shade):02x}'

    for ax, lm in zip(axes, landmarks):
        for r in results:
            if r['config']['landmark'] != lm:
                continue
            successes = [1.0 if ep['success'] else 0.0 for ep in r['episodes']]
            smoothed = smooth(successes, window=30)
            episodes = np.arange(len(smoothed)) + 15
            ax.plot(episodes, smoothed * 100, label=run_label(r),
                    color=color_for(r), linewidth=1.5)

        ax.set_title(lm, fontsize=9)
        ax.set_xlabel('Episode', fontsize=8)
        ax.set_ylabel('Success Rate (%)', fontsize=8)
        ax.set_ylim(-1, 15)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=7)

    for ax in axes[len(landmarks):]:
        ax.axis('off')
    fig.suptitle('Success Rate by Landmark (30-episode window)', fontsize=12, fontweight='bold')
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'success_rate.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved success_rate.png")

def plot_final_distance(results: list[dict], output_dir: str):
    """Plot final distance over training."""
    landmarks = sorted(set(r['config']['landmark'] for r in results))

    nrows, ncols = _grid(len(landmarks))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 2.6 * nrows), sharey=True)
    axes = np.atleast_1d(axes).flatten()

    # Base colors per algorithm; neighborhood-size variants are shaded from there
    base_colors = {'dqn': '#2196F3', 'a2c': '#FF5722', 'ppo': '#4CAF50'}
    size_shades = {5: 0.6, 7: 1.0, 9: 0.85, 11: 0.7, 13: 0.55, 15: 0.4, 19: 0.3, 25: 0.2}
    def color_for(run):
        c = base_colors.get(run['config']['agentType'], '#999')
        shade = size_shades.get(neighborhood_size(run), 1.0)
        # Darken by shade factor (mix toward black)
        r, g, b = int(c[1:3], 16), int(c[3:5], 16), int(c[5:7], 16)
        return f'#{int(r*shade):02x}{int(g*shade):02x}{int(b*shade):02x}'

    for ax, lm in zip(axes, landmarks):
        for r in results:
            if r['config']['landmark'] != lm:
                continue
            distances = [ep['finalDistance'] for ep in r['episodes']]
            smoothed = smooth(distances, window=30)
            episodes = np.arange(len(smoothed)) + 15
            ax.plot(episodes, smoothed, label=run_label(r),
                    color=color_for(r), linewidth=1.5)

        ax.set_title(lm, fontsize=9)
        ax.set_xlabel('Episode', fontsize=8)
        ax.set_ylabel('Final Distance (voxels)', fontsize=8)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=7)

    for ax in axes[len(landmarks):]:
        ax.axis('off')
    fig.suptitle('Final Distance to Target by Landmark', fontsize=12, fontweight='bold')
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'final_distance.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved final_distance.png")

def generate_summary_table(results: list[dict], output_dir: str):
    """Generate a summary table of final performance metrics."""
    # Use last 50 episodes for summary stats
    tail = 50
    rows = []

    for r in results:
        eps = r['episodes'][-tail:]
        agent = run_label(r)
        landmark = r['config']['landmark']

        rewards = [ep['totalReward'] for ep in eps]
        distances = [ep['finalDistance'] for ep in eps]
        successes = [ep['success'] for ep in eps]
        steps_list = [ep['steps'] for ep in eps]

        rows.append({
            'Landmark': landmark,
            'Agent': agent,
            'Mean Reward': f"{np.mean(rewards):.1f}",
            'Success Rate': f"{np.mean(successes) * 100:.1f}%",
            'Mean Distance': f"{np.mean(distances):.1f}",
            'Mean Steps': f"{np.mean(steps_list):.0f}",
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
        print(f"  {c['agentType'].upper()} / {c['landmark']}: {len(eps)} episodes, {successes} successes ({successes/len(eps)*100:.1f}%)")

    plot_reward_curves(results, output_dir)
    plot_success_rate(results, output_dir)
    plot_final_distance(results, output_dir)
    generate_summary_table(results, output_dir)

if __name__ == '__main__':
    main()
