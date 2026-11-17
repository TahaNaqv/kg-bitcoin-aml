#!/usr/bin/env bash
# Package the LaTeX paper for submission (upload-ready for Overleaf / IEEE).
set -euo pipefail
cd "$(dirname "$0")/.."
out="kg-bitcoin-aml-paper.zip"
rm -f "$out"
(cd paper && zip -qr "../$out" main.tex main.pdf IEEEtran.cls sections tables figures -x '*.aux' '*.log')
echo "wrote $out"; unzip -l "$out"
