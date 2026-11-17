#!/usr/bin/env bash
# Compile paper/main.tex out of tree (build artefacts never touch the repo).
# Uses $TECTONIC if set, else `tectonic` on PATH, else latexmk/pdflatex.
set -euo pipefail
cd "$(dirname "$0")/.."
build="${BUILD_DIR:-$(mktemp -d)}"
mkdir -p "$build"; rm -rf "$build/paper"; cp -r paper "$build/paper"
cd "$build/paper"
if [[ -n "${TECTONIC:-}" ]] || command -v tectonic >/dev/null; then
  "${TECTONIC:-tectonic}" -X compile main.tex --keep-logs
else
  latexmk -pdf -interaction=nonstopmode main.tex
fi
echo "PDF: $build/paper/main.pdf"
