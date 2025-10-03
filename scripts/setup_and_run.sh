#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

echo "[1/6] Cleaning old virtualenvs..."
rm -rf .venv .venv38 .venv310 || true

echo "[2/6] Ensuring Miniforge (conda-forge) is installed..."
if ! command -v conda >/dev/null 2>&1; then
  if command -v brew >/dev/null 2>&1; then
    brew install --cask miniforge || true
  else
    echo "Homebrew not found. Please install Miniforge manually: https://github.com/conda-forge/miniforge" >&2
    exit 1
  fi
fi

# Find conda binary from Miniforge
if command -v conda >/dev/null 2>&1; then
  CONDA_BIN="$(command -v conda)"
else
  CONDA_BIN="$(ls -d /opt/homebrew/Caskroom/miniforge/*/Miniforge3/bin/conda 2>/dev/null | head -n1)"
fi

if [ -z "${CONDA_BIN:-}" ] || [ ! -x "$CONDA_BIN" ]; then
  echo "Could not locate Miniforge conda binary. Aborting." >&2
  exit 1
fi

echo "[3/6] Creating single env 'tfx-39' with Python 3.9..."
"$CONDA_BIN" env remove -y -n tfx-39 >/dev/null 2>&1 || true
"$CONDA_BIN" create -y -n tfx-39 -c conda-forge python=3.9

echo "[4/6] Installing TFX stack from conda-forge (prebuilt on Apple Silicon)..."
"$CONDA_BIN" install -y -n tfx-39 -c conda-forge \
  tfx=1.15 tensorflow=2.13 tensorflow-transform=1.15 \
  tensorflow-data-validation=1.15 tensorflow-model-analysis=0.45 \
  ml-metadata=1.15 protobuf=3.20 apache-beam=2.48

echo "[5/6] Installing pip extras..."
"$CONDA_BIN" run -n tfx-39 pip install -q \
  keras-tuner==1.4.6 pandas==1.5.3 numpy==1.24.4 \
  scikit-learn==1.2.2 pyarrow==12.0.1 tensorflow-serving-api==2.13.1 \
  tqdm==4.66.1 jinja2==3.1.2

echo "[6/6] Running pipeline (reduced trials/epochs for speed)..."
export TUNING_MAX_TRIALS=2
export BASELINE_EPOCHS=2
"$CONDA_BIN" run -n tfx-39 python orchestration/local_runner.py
