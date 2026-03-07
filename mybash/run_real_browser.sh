#!/usr/bin/env bash

set -euo pipefail

cd "$(dirname "$0")/.."
source .venv/bin/activate
PYTHONPATH=src python -m xlerobot_personality.main \
  --config configs/real_head.example.yaml \
  --web-preview \
  --no-visualize
