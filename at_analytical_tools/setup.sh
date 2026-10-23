#!/usr/bin/env bash
set -euo pipefail
VIEWER_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
CONDA_EXE="${AT_VIEWER_CONDA:-/root/miniconda3/condabin/conda}"
ENV_PYTHON="${AT_VIEWER_PYTHON:-/root/miniconda3/envs/at_viewer/bin/python}"
if [[ ! -x "$ENV_PYTHON" ]]; then
    "$CONDA_EXE" create --name at_viewer python=3.9 pip -y
fi
PIP_CACHE_DIR="$VIEWER_DIR/.pip-cache" "$ENV_PYTHON" -m pip install -r "$VIEWER_DIR/requirements.lock.txt"
export npm_config_cache="$VIEWER_DIR/.npm-cache"
cd "$VIEWER_DIR/frontend"
npm ci
npm run build
printf '\n安装完成。运行：bash %s/run_viewer.sh <项目目录>\n' "$VIEWER_DIR"
