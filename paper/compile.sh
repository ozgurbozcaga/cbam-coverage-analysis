#!/bin/bash
# Compile the CBAM spillovers paper
# Run from project root: bash paper/compile.sh

cd "$(dirname "$0")"
pdflatex cbam_spillovers.tex
bibtex cbam_spillovers
pdflatex cbam_spillovers.tex
pdflatex cbam_spillovers.tex
echo "Done. Output: paper/cbam_spillovers.pdf"
