#!/usr/bin/env python3
"""Generate the predicted-vs-oracle comparison figure for the report.

Reads the two stored result JSONs in docs/ (predicted and oracle goal-vector
runs on AOMIC sub-0083) and produces a single PNG with one panel per landmark
showing rolling success-rate curves with mean +/- 95% CI across seeds.
"""
import argparse
import json
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
DEFAULT_PRED = REPO / "docs" / "predicted_vs_oracle_predicted.json"
DEFAULT_PRED_WIDE = REPO / "docs" / "predicted_vs_oracle_predicted_wide.json"
DEFAULT_ORAC = REPO / "docs" / "predicted_vs_oracle_oracle.json"
DEFAULT_OUT = REPO / "docs" / "figures" / "predicted-vs-oracle"


def rolling(arr: np.ndarray, w: int) -> np.ndarray:
    if w <= 1 or len(arr) < w:
        return arr.astype(float)
    cs = np.cumsum(np.insert(arr.astype(float), 0, 0.0))
    return (cs[w:] - cs[:-w]) / w


def load_runs(path: Path):
    """Return dict[landmark] -> list of per-seed success arrays (one per ep)."""
    runs = json.loads(Path(path).read_text())
    by_lm = defaultdict(list)
    for r in runs:
        lm = r["config"]["landmark"]
        succ = np.array([1 if e["success"] else 0 for e in r["episodes"]])
        by_lm[lm].append(succ)
    return by_lm


def stack_curves(seed_arrs, window: int):
    n_eps = min(len(a) for a in seed_arrs)
    rolled = np.stack([rolling(a[:n_eps], window) for a in seed_arrs])
    mean = rolled.mean(axis=0)
    se = rolled.std(axis=0, ddof=1) / np.sqrt(rolled.shape[0]) if rolled.shape[0] > 1 else np.zeros_like(mean)
    ci95 = 1.96 * se
    x = np.arange(window - 1, n_eps)
    return x, mean, ci95


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--predicted", type=Path, default=DEFAULT_PRED)
    ap.add_argument("--predicted-wide", type=Path, default=DEFAULT_PRED_WIDE,
                    help="optional third arm: wide-FOV predicted goal-vector results")
    ap.add_argument("--oracle", type=Path, default=DEFAULT_ORAC)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--window", type=int, default=30)
    args = ap.parse_args()

    pred = load_runs(args.predicted)
    orac = load_runs(args.oracle)
    have_wide = args.predicted_wide.exists()
    pred_wide = load_runs(args.predicted_wide) if have_wide else {}

    arms = [("predicted (7^3)", pred, "tab:green"),
            ("oracle", orac, "tab:gray")]
    if have_wide:
        arms.insert(1, ("predicted (15^3, multi-scale)", pred_wide, "tab:blue"))

    common = set(pred.keys()) & set(orac.keys())
    if have_wide:
        common &= set(pred_wide.keys())
    landmarks = sorted(common)
    if not landmarks:
        raise SystemExit("no landmarks shared across all input runs")

    args.out.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, len(landmarks), figsize=(5.5 * len(landmarks), 4), sharey=True)
    if len(landmarks) == 1:
        axes = [axes]

    for ax, lm in zip(axes, landmarks):
        for label, src, color in arms:
            x, m, ci = stack_curves(src[lm], args.window)
            ax.plot(x, m * 100, color=color, label=f"{label} (n={len(src[lm])} seeds)", lw=2)
            ax.fill_between(x, (m - ci) * 100, (m + ci) * 100, color=color, alpha=0.15)
        ax.set_title(lm)
        ax.set_xlabel("Episode")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper left", frameon=False)
    axes[0].set_ylabel(f"Success rate (%, {args.window}-ep rolling)")

    title = "Predicted goal vector vs. oracle (AOMIC sub-0083, PPO, k=10, curriculum 20→50)"
    if have_wide:
        title = "Predicted (7^3 vs. wide-FOV 15^3) goal vector vs. oracle (AOMIC sub-0083, PPO, k=10)"
    fig.suptitle(title)
    fig.tight_layout()
    out_png = args.out / "success_rate.png"
    fig.savefig(out_png, dpi=140)
    print(f"wrote {out_png}")

    # Also write a small summary table.
    summary = []
    for lm in landmarks:
        for label, src, _color in arms:
            seeds = src[lm]
            last100 = np.array([a[-100:].mean() for a in seeds])
            summary.append((lm, label, last100.mean(), last100.std(ddof=1), [f"{s:.2f}" for s in last100]))
    out_txt = args.out / "summary_table.txt"
    with open(out_txt, "w") as f:
        f.write(f"{'landmark':<20} {'mode':<32} mean_last100   std    per_seed\n")
        for lm, lab, m, s, ps in summary:
            f.write(f"{lm:<20} {lab:<32} {m*100:>11.1f}%  {s*100:>5.1f}  {','.join(ps)}\n")
    print(f"wrote {out_txt}")


if __name__ == "__main__":
    main()
