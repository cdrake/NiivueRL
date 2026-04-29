#!/usr/bin/env python3
"""Plot CDFs and histograms of per-step angular errors for one or more
goal-vector models, plus a summary table. Inputs are JSON files written by
scripts/goal_vector/eval_per_step_error.py.

Usage:
    python3 scripts/plot_per_step_error.py \
        --eval-json docs/per_step_error_baseline.json \
        --eval-json docs/per_step_error_wide.json \
        --out-dir docs/figures/per_step_error
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# Colour-cycle that stays distinguishable for up to ~6 variants.
COLOURS = [
    "#1f77b4",  # baseline blue
    "#d62728",  # red
    "#2ca02c",  # green
    "#ff7f0e",  # orange
    "#9467bd",  # purple
    "#8c564b",  # brown
]


def _label_for(record: dict) -> str:
    """Human-readable legend label: derive a short name from model_dir + tag
    the mean cosine so readers can compare to the val-cos numbers in the
    report."""
    name = Path(record["model_dir"]).name
    # Trim the timestamp suffix (e.g. _20260428_180017) to keep legends short.
    parts = name.split("_")
    if len(parts) >= 2 and parts[-1].isdigit() and parts[-2].isdigit():
        name = "_".join(parts[:-2])
    return f"{name}  (mean cos = {record['mean_cosine']:.4f})"


def plot_cdf(records: list[dict], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(1200 / 140, 600 / 140), dpi=140)
    for i, rec in enumerate(records):
        errs = np.sort(np.asarray(rec["errors_deg"], dtype=float))
        cdf = np.linspace(1.0 / len(errs), 1.0, len(errs))
        ax.plot(errs, cdf, label=_label_for(rec),
                color=COLOURS[i % len(COLOURS)], linewidth=2)
    ax.set_xlabel("angular error (deg)")
    ax.set_ylabel("CDF")
    ax.set_title("Per-step angular error CDF (held-out subject)")
    ax.set_xlim(0, 180)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    # Reference verticals at 30 and 60 degrees -- the tail-fraction thresholds
    # quoted in the summary table.
    for x in (30.0, 60.0):
        ax.axvline(x, color="grey", linestyle=":", alpha=0.5)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    print(f"[plot] wrote {path}")


def plot_histogram(records: list[dict], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(1200 / 140, 600 / 140), dpi=140)
    bins = np.arange(0.0, 180.0 + 5.0, 5.0)
    for i, rec in enumerate(records):
        errs = np.asarray(rec["errors_deg"], dtype=float)
        ax.hist(errs, bins=bins, alpha=0.5, label=_label_for(rec),
                color=COLOURS[i % len(COLOURS)],
                edgecolor=COLOURS[i % len(COLOURS)],
                linewidth=1.0,
                density=True)
    ax.set_xlabel("angular error (deg)")
    ax.set_ylabel("density (per deg)")
    ax.set_title("Per-step angular error distribution (5 deg bins)")
    ax.set_xlim(0, 180)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    print(f"[plot] wrote {path}")


def write_summary(records: list[dict], path: Path) -> None:
    rows = []
    header = ("model", "mean_cos", "mean_err", "P50", "P75", "P90", "P95",
              ">30deg", ">60deg")
    rows.append(header)
    for rec in records:
        errs = np.asarray(rec["errors_deg"], dtype=float)
        name = Path(rec["model_dir"]).name
        # Trim timestamp suffix.
        parts = name.split("_")
        if len(parts) >= 2 and parts[-1].isdigit() and parts[-2].isdigit():
            name = "_".join(parts[:-2])
        rows.append((
            name,
            f"{rec['mean_cosine']:.4f}",
            f"{rec['mean_err_deg']:.2f}",
            f"{np.median(errs):.2f}",
            f"{np.percentile(errs, 75):.2f}",
            f"{np.percentile(errs, 90):.2f}",
            f"{np.percentile(errs, 95):.2f}",
            f"{100 * (errs > 30.0).mean():.1f}%",
            f"{100 * (errs > 60.0).mean():.1f}%",
        ))
    # Compute column widths and write a fixed-width table.
    widths = [max(len(r[i]) for r in rows) for i in range(len(header))]
    lines = []
    for ri, row in enumerate(rows):
        line = "  ".join(cell.ljust(widths[i]) for i, cell in enumerate(row))
        lines.append(line)
        if ri == 0:
            lines.append("  ".join("-" * widths[i] for i in range(len(header))))
    path.write_text("\n".join(lines) + "\n")
    print(f"[plot] wrote {path}")
    print()
    print("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--eval-json", type=Path, action="append", required=True,
                    help="path to an eval_per_step_error.py JSON; pass multiple times")
    ap.add_argument("--out-dir", type=Path,
                    default=Path("docs/figures/per_step_error"))
    args = ap.parse_args()

    records = []
    for jp in args.eval_json:
        if not jp.exists():
            raise SystemExit(f"missing {jp}")
        records.append(json.loads(jp.read_text()))
    if not records:
        raise SystemExit("no eval JSONs supplied")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    plot_cdf(records, args.out_dir / "cdf.png")
    plot_histogram(records, args.out_dir / "histogram.png")
    write_summary(records, args.out_dir / "summary_table.txt")


if __name__ == "__main__":
    main()
