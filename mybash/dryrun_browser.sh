#!/usr/bin/env bash

set -euo pipefail

cd "$(dirname "$0")/.."
source .venv/bin/activate
PYTHONPATH=src python -m xlerobot_personality.main \
  --config configs/local_demo.yaml \
  --dry-run \
  --web-preview \
  --no-visualize
