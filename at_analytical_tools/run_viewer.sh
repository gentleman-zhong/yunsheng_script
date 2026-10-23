#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
VIEWER_PYTHON="${AT_VIEWER_PYTHON:-/root/miniconda3/envs/at_viewer/bin/python}"
if [[ $# -lt 1 ]]; then
    echo "用法：bash run_viewer.sh <项目目录> [--host 0.0.0.0] [--port 8080] [--reindex]" >&2
    exit 2
fi
if [[ ! -x "$VIEWER_PYTHON" ]]; then
    echo "缺少 at_viewer Conda 环境，请先按 README.md 安装。" >&2
    exit 1
fi
export PYTHONDONTWRITEBYTECODE=1
ALGORITHM_ROOT="${AT_VIEWER_ODM_ROOT:-/code}"
export PYTHONPATH="$ALGORITHM_ROOT/SuperBuild/install/bin/opensfm${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$ALGORITHM_ROOT/SuperBuild/install/lib:$(dirname -- "$VIEWER_PYTHON")/../lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
exec "$VIEWER_PYTHON" -B "$ROOT_DIR/backend/server.py" "$@"
