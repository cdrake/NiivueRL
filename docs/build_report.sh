#!/usr/bin/env bash
# Build docs/project_report.pdf from docs/project_report.md via pandoc.
# Run from the repo root or from inside docs/.

set -euo pipefail

cd "$(dirname "$0")"

pandoc project_report.md \
  -o project_report.pdf \
  --pdf-engine=xelatex \
  -V geometry:margin=1in \
  -V fontsize=11pt \
  -V linkcolor=blue \
  -V urlcolor=blue \
  --resource-path=.:./figures \
  --toc-depth=2

echo "Built $(pwd)/project_report.pdf"
