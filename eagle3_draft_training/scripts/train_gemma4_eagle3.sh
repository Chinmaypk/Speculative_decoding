#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"

CONFIG_PATH="${1:-configs/gemma4_example.yaml}"
accelerate launch -m eagle3_draft.train --config "$CONFIG_PATH"
