#!/usr/bin/env python3
"""Parse src/lib/landmarks.ts and write data/mni152_landmarks.json.

The TS source maps landmark name -> [x, y, z] integer voxel indices in the
volume that the browser env loads (public/t1_crop.nii.gz). We dump those
coordinates as JSON so the Python training scripts (mni_sampler.py,
train_goal_net.py) can sample patches around the same centroids.

The TS file uses the spelling "Inferior-Lateral-Ventricle" while the trainer's
LANDMARKS dict uses the FreeSurfer-style "Inf-Lat-Vent". This script remaps to
the trainer spelling so the JSON keys match LANDMARKS exactly. The output keeps
the order of LANDMARKS in train_goal_net.py.

Usage (from repo root):
    python scripts/goal_vector/dump_mni_landmarks.py
        [--ts-source src/lib/landmarks.ts]
        [--out data/mni152_landmarks.json]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

# Same 15 landmark order as scripts/goal_vector/train_goal_net.py LANDMARKS.
TRAINER_NAMES = [
    "Lateral-Ventricle",
    "Inf-Lat-Vent",
    "Cerebellum-White-Matter",
    "Cerebellum-Cortex",
    "Thalamus",
    "Caudate",
    "Putamen",
    "Pallidum",
    "3rd-Ventricle",
    "4th-Ventricle",
    "Brain-Stem",
    "Hippocampus",
    "Amygdala",
    "Accumbens-area",
    "VentralDC",
]

# Map TS spelling -> trainer spelling. Only the ventricle name differs today.
TS_TO_TRAINER = {
    "Inferior-Lateral-Ventricle": "Inf-Lat-Vent",
}


def parse_landmarks_ts(ts_path: Path) -> dict[str, list[int]]:
    """Extract the mniCoords table from src/lib/landmarks.ts.

    Looks for lines like:
        'Thalamus':                   [130, 140, 125],
    inside the const mniCoords definition. We don't try to be a full TS parser;
    a regex over the whole file is enough because every centroid line is on a
    single line with the canonical [x, y, z] form.
    """
    text = ts_path.read_text()
    # Restrict to the mniCoords block to avoid grabbing unrelated arrays.
    m = re.search(
        r"mniCoords\s*:\s*Record<[^>]+>\s*=\s*\{([^}]+)\};", text, re.DOTALL
    )
    if not m:
        # Fallback: take the whole file but require the [x, y, z] integer form.
        block = text
    else:
        block = m.group(1)
    out: dict[str, list[int]] = {}
    pattern = re.compile(
        r"['\"]([A-Za-z0-9 \-]+)['\"]\s*:\s*\[\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*\]"
    )
    for name, x, y, z in pattern.findall(block):
        out[name] = [int(x), int(y), int(z)]
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--ts-source",
        type=Path,
        default=REPO / "src" / "lib" / "landmarks.ts",
        help="path to src/lib/landmarks.ts",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=REPO / "data" / "mni152_landmarks.json",
        help="output JSON path",
    )
    args = ap.parse_args()

    if not args.ts_source.exists():
        print(f"ERROR: {args.ts_source} not found", file=sys.stderr)
        return 1

    raw = parse_landmarks_ts(args.ts_source)
    if not raw:
        print(
            f"ERROR: no landmark coordinates parsed from {args.ts_source}",
            file=sys.stderr,
        )
        return 2

    # Remap TS spellings to trainer spellings.
    remapped: dict[str, list[int]] = {}
    for ts_name, coords in raw.items():
        trainer_name = TS_TO_TRAINER.get(ts_name, ts_name)
        remapped[trainer_name] = coords

    # Verify all 15 trainer names are present.
    missing = [n for n in TRAINER_NAMES if n not in remapped]
    if missing:
        print(
            f"ERROR: missing landmark(s) after remap: {missing}", file=sys.stderr
        )
        print(f"  parsed names: {sorted(remapped)}", file=sys.stderr)
        return 3

    # Reorder to match TRAINER_NAMES.
    ordered = {n: remapped[n] for n in TRAINER_NAMES}

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(ordered, indent=2) + "\n")
    print(f"[ok] wrote {len(ordered)} landmarks to {args.out}")
    for name, c in ordered.items():
        print(f"  {name:30s} {c}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
