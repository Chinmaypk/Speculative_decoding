#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"

accelerate launch -m eagle3_draft.train --config configs/gemma4_example.yaml
