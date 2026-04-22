#!/usr/bin/env bash
# Build docs/project_presentation.pdf from docs/project_presentation.md
# via pandoc + beamer (xelatex). Run from repo root or from inside docs/.

set -euo pipefail

cd "$(dirname "$0")"

pandoc project_presentation.md \
  -o project_presentation.pdf \
  -t beamer \
  --pdf-engine=xelatex \
  --slide-level=1 \
  --resource-path=.:./figures \
  -V classoption:aspectratio=169

echo "Built $(pwd)/project_presentation.pdf"
