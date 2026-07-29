#!/bin/bash

# Exit on error
set -e

echo "=== Building Generative AI for Bioimage Analysis Slide Deck ==="

SLIDE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SLIDE_DIR"

if command -v latexmk &> /dev/null; then
    echo "[+] Compiling main.tex using latexmk..."
    latexmk -pdf -interaction=nonstopmode main.tex
elif command -v pdflatex &> /dev/null; then
    echo "[+] Compiling main.tex using pdflatex (Pass 1)..."
    pdflatex -interaction=nonstopmode main.tex
    echo "[+] Compiling main.tex using pdflatex (Pass 2)..."
    pdflatex -interaction=nonstopmode main.tex
else
    echo "[-] Error: Neither latexmk nor pdflatex was found in system PATH."
    echo "    Please install TeX Live, MacTeX, or MikTeX to compile the slides."
    exit 1
fi

echo "[+] Cleaning up auxiliary build files..."
rm -f *.aux *.log *.nav *.out *.snm *.toc *.vrb *.fls *.fdb_latexmk

echo "=== Build Complete: ${SLIDE_DIR}/main.pdf ==="
