#!/usr/bin/env bash
# Build docs/project_report.pdf from docs/project_report.md via pandoc.
# Run from the repo root or from inside docs/.

set -euo pipefail

cd "$(dirname "$0")"

pandoc project_report.md \
  -o project_report.pdf \
  --pdf-engine=xelatex \
  -V geometry:"margin=0.7in,top=0.55in,bottom=0.55in" \
  -V fontsize=10pt \
  -V linkcolor=blue \
  -V urlcolor=blue \
  -V header-includes='\usepackage{float}\floatplacement{figure}{H}' \
  --resource-path=.:./figures

echo "Built $(pwd)/project_report.pdf"
