#!/usr/bin/env python3
"""Download N subjects' brain.mgz + aseg.mgz from AOMIC ID1000 (ds003097).

Files land in data/aomic_id1000/sub-XXXX/ . The S3 bucket is public; no
credentials needed. Existing files are skipped.
"""
import argparse
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

BUCKET = "s3://openneuro.org/ds003097/derivatives/freesurfer"
REPO = Path(__file__).resolve().parents[2]
ROOT = REPO / "data" / "aomic_id1000"
FILES = ("brain.mgz", "aseg.mgz")


def fetch_subject(sub_id: str) -> str:
    out = ROOT / sub_id
    out.mkdir(parents=True, exist_ok=True)
    for f in FILES:
        dst = out / f
        if dst.exists() and dst.stat().st_size > 0:
            continue
        src = f"{BUCKET}/{sub_id}/mri/{f}"
        subprocess.run(
            ["aws", "s3", "cp", "--no-sign-request", src, str(dst)],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
    return sub_id


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=20, help="number of subjects")
    ap.add_argument("--start", type=int, default=1, help="first subject index (1-based)")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    subs = [f"sub-{i:04d}" for i in range(args.start, args.start + args.n)]
    print(f"downloading {len(subs)} subjects from AOMIC ID1000 -> {ROOT}")

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(fetch_subject, s): s for s in subs}
        done = 0
        for f in as_completed(futures):
            sub = futures[f]
            try:
                f.result()
                done += 1
                print(f"  [{done}/{len(subs)}] {sub}")
            except subprocess.CalledProcessError as e:
                stderr = e.stderr.decode(errors="replace") if e.stderr else ""
                print(f"  FAILED {sub}: {stderr.strip()[:200]}")


if __name__ == "__main__":
    main()
