#!/usr/bin/env bash
# Kinematic-twin demo for any URDF robot in configs/robots/*.yaml.
#   bash webapp/run.sh [robot]      e.g. so101 | ur7e | piper     (PORT=8765 by default)
# Legacy robosuite/Panda version: python webapp/server_robosuite.py
set -e
cd "$(dirname "$0")/.."
source /root/miniconda3/etc/profile.d/conda.sh
conda activate shadow
MUJOCO_GL=egl exec python webapp/server.py --robot "${1:-so101}" --port "${PORT:-8765}"
