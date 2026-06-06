#!/usr/bin/env bash
set -euo pipefail

# Multi-GPU launcher. Run from Linux/WSL with CUDA visible.
# Example:
#   NUM_PROCESSES=4 TARGET_MODEL=/models/my-gemma4-e2b-it TRAIN_FILE=data/train.jsonl ./finetune_gemma4_draft_multigpu.sh

TARGET_MODEL="${TARGET_MODEL:-/path/to/your-finetuned-gemma-4-E2B-it}"
DRAFT_MODEL="${DRAFT_MODEL:-google/gemma-4-E2B-it-assistant}"
TRAIN_FILE="${TRAIN_FILE:-data/train.jsonl}"
EVAL_FILE="${EVAL_FILE:-}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/gemma4-e2b-assistant-distilled-multigpu}"
SAVE_FINAL_DIR="${SAVE_FINAL_DIR:-$OUTPUT_DIR/final_drafter}"

NUM_PROCESSES="${NUM_PROCESSES:-2}"
MIXED_PRECISION="${MIXED_PRECISION:-bf16}"
MAX_LENGTH="${MAX_LENGTH:-2048}"
PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-1}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-8}"
LEARNING_RATE="${LEARNING_RATE:-2e-5}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-1}"
WARMUP_RATIO="${WARMUP_RATIO:-0.03}"
TEMPERATURE="${TEMPERATURE:-1.0}"
KL_WEIGHT="${KL_WEIGHT:-1.0}"
CE_WEIGHT="${CE_WEIGHT:-0.1}"
SAVE_STEPS="${SAVE_STEPS:-500}"
EVAL_STEPS="${EVAL_STEPS:-500}"
LOGGING_STEPS="${LOGGING_STEPS:-10}"
DEBUG_LOSS_STEPS="${DEBUG_LOSS_STEPS:-10}"
TENSORBOARD_LOG_DIR="${TENSORBOARD_LOG_DIR:-$OUTPUT_DIR/tensorboard}"

PRECISION_FLAG="${PRECISION_FLAG:---bf16}"
LORA_FLAG="${LORA_FLAG:---use-lora}"
GRADIENT_CHECKPOINTING_FLAG="${GRADIENT_CHECKPOINTING_FLAG:---gradient-checkpointing}"

if [[ "$TARGET_MODEL" == "/path/to/your-finetuned-gemma-4-E2B-it" ]]; then
  echo "Set TARGET_MODEL to your fine-tuned Gemma 4 target model path or HF id." >&2
  exit 1
fi

if [[ ! -f "$TRAIN_FILE" ]]; then
  echo "TRAIN_FILE does not exist: $TRAIN_FILE" >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR"

args=(
  --target-model "$TARGET_MODEL"
  --draft-model "$DRAFT_MODEL"
  --train-file "$TRAIN_FILE"
  --output-dir "$OUTPUT_DIR"
  --save-final-dir "$SAVE_FINAL_DIR"
  --training-mode multi_gpu
  --max-length "$MAX_LENGTH"
  --per-device-train-batch-size "$PER_DEVICE_BATCH_SIZE"
  --gradient-accumulation-steps "$GRADIENT_ACCUMULATION_STEPS"
  --learning-rate "$LEARNING_RATE"
  --num-train-epochs "$NUM_TRAIN_EPOCHS"
  --warmup-ratio "$WARMUP_RATIO"
  --temperature "$TEMPERATURE"
  --kl-weight "$KL_WEIGHT"
  --ce-weight "$CE_WEIGHT"
  --logging-steps "$LOGGING_STEPS"
  --debug-loss-steps "$DEBUG_LOSS_STEPS"
  --save-steps "$SAVE_STEPS"
  --eval-steps "$EVAL_STEPS"
  --tensorboard-log-dir "$TENSORBOARD_LOG_DIR"
)

if [[ -n "$EVAL_FILE" ]]; then
  if [[ ! -f "$EVAL_FILE" ]]; then
    echo "EVAL_FILE does not exist: $EVAL_FILE" >&2
    exit 1
  fi
  args+=(--eval-file "$EVAL_FILE")
fi

if [[ -n "$PRECISION_FLAG" ]]; then
  args+=("$PRECISION_FLAG")
fi

if [[ -n "$LORA_FLAG" ]]; then
  args+=("$LORA_FLAG")
fi

if [[ -n "$GRADIENT_CHECKPOINTING_FLAG" ]]; then
  args+=("$GRADIENT_CHECKPOINTING_FLAG")
fi

accelerate launch \
  --num_processes "$NUM_PROCESSES" \
  --mixed_precision "$MIXED_PRECISION" \
  scripts/train_gemma4_draft.py "${args[@]}"
