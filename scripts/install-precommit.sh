#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if ! command -v pre-commit >/dev/null 2>&1; then
  echo "pre-commit not found. Install it with pip: python -m pip install --user pre-commit"
  exit 1
fi
pre-commit install
echo "pre-commit installed. Run 'pre-commit run --all-files' to format existing files."
