#!/usr/bin/env bash

set -euo pipefail

# make sure that the user has permissions to access the serial port
# sudo chmod 666 $robot-port

cd "$(dirname "$0")/.."
source .venv/bin/activate
PYTHONPATH=src python -m xlerobot_personality.head_calibrate \
  --config configs/real_head.example.yaml
