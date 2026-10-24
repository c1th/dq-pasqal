#!/usr/bin/env bash
# Build report.pdf from report.md (needs pandoc + a TeX distribution with xelatex and extsizes).
set -e
cd "$(dirname "$0")"
pandoc report.md -o report.pdf --pdf-engine=xelatex -V documentclass=extarticle -V fontsize=9pt \
  -V geometry:margin=1.7cm -V mainfont="DejaVu Serif" -V monofont="DejaVu Sans Mono" -V colorlinks=true
echo "wrote report/report.pdf ($(pdfinfo report.pdf 2>/dev/null | grep Pages || echo 'page count unknown'))"
