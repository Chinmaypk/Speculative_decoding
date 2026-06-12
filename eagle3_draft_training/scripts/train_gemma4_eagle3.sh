#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"

# ============================================================
# Edit this section only
# ============================================================

ASSISTANT_MODEL_PATH="/home/c.kulkarni/hf_models/google/gemma-4-E2B-it-assistant"
TARGET_MODEL_PATH="/home/c.kulkarni/hf_models/google/gemma-4-E2B-it"

TRAIN_JSONL="/path/to/train.jsonl"
TEST_JSONL=""  # optional: /path/to/test.jsonl

OUTPUT_DIR="outputs/gemma4_eagle3_draft"
TOKENIZED_DATA_DIR="data/tokenized/gemma4"
CONFIG_PATH="configs/gemma4_runtime.yaml"
TENSORBOARD_LOG_DIR="${OUTPUT_DIR}/tensorboard"
BEST_CHECKPOINT_DIR="${OUTPUT_DIR}/best-checkpoint"

TRUST_REMOTE_CODE="true"
AUTO_CONFIGURE_FROM_TARGET="true"
RUN_PREPARE_DATA="true"
RUN_TRAIN="true"
RUN_EVAL="true"
RUN_TEST_GENERATION="false"
SAVE_TEST_PREDICTIONS="true"

MAX_LENGTH="2048"
EVAL_RATIO="0.05"
SEED="42"

# Manual drafter settings. These are overwritten when AUTO_CONFIGURE_FROM_TARGET=true.
TARGET_HIDDEN_LAYER_INDICES="[8, 16, 26]"
DRAFT_HIDDEN_SIZE="1024"
DRAFT_NUM_LAYERS="4"
DRAFT_NUM_HEADS="8"
DRAFT_INTERMEDIATE_SIZE="4096"
DROPOUT="0.0"

LEARNING_RATE="0.0002"
WEIGHT_DECAY="0.01"
WARMUP_STEPS="100"
GRADIENT_ACCUMULATION_STEPS="4"
PER_DEVICE_TRAIN_BATCH_SIZE="1"
PER_DEVICE_EVAL_BATCH_SIZE="1"
NUM_TRAIN_EPOCHS="1"
MAX_STEPS="-1"
LOGGING_STEPS="10"
EVAL_STEPS="200"
SAVE_STEPS="500"
BF16="true"
FP16="false"
SCHEDULED_SAMPLING_PROB="0.0"
GRADIENT_CHECKPOINTING="true"

RESUME_DRAFT_CHECKPOINT_PATH=""  # optional: outputs/gemma4_eagle3_draft/checkpoint-500 or draft_model.pt
EVAL_CHECKPOINT_PATH=""          # optional. Empty means best-checkpoint.
EVAL_DATA_DIR=""                 # optional. Empty means test split if TEST_JSONL exists, else eval split.
GEN_RESPONSE_TEXT="Create a chart showing monthly revenue"
GEN_INPUT_JSONL=""              # optional batch generation input
GEN_OUTPUT_JSONL=""             # optional. Empty means best-checkpoint/test_predictions.jsonl
GEN_MAX_NEW_TOKENS="256"

# ============================================================
# Do not edit below unless changing pipeline behavior
# ============================================================

bool_flag() {
  local value="$1"
  if [[ "$value" == "true" || "$value" == "1" || "$value" == "yes" ]]; then
    return 0
  fi
  return 1
}

TRUST_ARG=()
if bool_flag "$TRUST_REMOTE_CODE"; then
  TRUST_ARG=(--trust_remote_code)
fi

mkdir -p "$(dirname "$CONFIG_PATH")" "$OUTPUT_DIR" "$TOKENIZED_DATA_DIR"

cat > "$CONFIG_PATH" <<YAML
assistant_model_path: ${ASSISTANT_MODEL_PATH}
target_model_path: ${TARGET_MODEL_PATH}
train_data_dir: ${TOKENIZED_DATA_DIR}/train
eval_data_dir: ${TOKENIZED_DATA_DIR}/eval
test_data_dir: ${TOKENIZED_DATA_DIR}/test
output_dir: ${OUTPUT_DIR}
tensorboard_log_dir: ${TENSORBOARD_LOG_DIR}
resume_draft_checkpoint_path: ${RESUME_DRAFT_CHECKPOINT_PATH:-null}
target_hidden_layer_indices: ${TARGET_HIDDEN_LAYER_INDICES}
max_length: ${MAX_LENGTH}
draft_hidden_size: ${DRAFT_HIDDEN_SIZE}
draft_num_layers: ${DRAFT_NUM_LAYERS}
draft_num_heads: ${DRAFT_NUM_HEADS}
draft_intermediate_size: ${DRAFT_INTERMEDIATE_SIZE}
dropout: ${DROPOUT}
learning_rate: ${LEARNING_RATE}
weight_decay: ${WEIGHT_DECAY}
warmup_steps: ${WARMUP_STEPS}
gradient_accumulation_steps: ${GRADIENT_ACCUMULATION_STEPS}
per_device_train_batch_size: ${PER_DEVICE_TRAIN_BATCH_SIZE}
per_device_eval_batch_size: ${PER_DEVICE_EVAL_BATCH_SIZE}
num_train_epochs: ${NUM_TRAIN_EPOCHS}
max_steps: ${MAX_STEPS}
logging_steps: ${LOGGING_STEPS}
eval_steps: ${EVAL_STEPS}
save_steps: ${SAVE_STEPS}
seed: ${SEED}
bf16: ${BF16}
fp16: ${FP16}
scheduled_sampling_prob: ${SCHEDULED_SAMPLING_PROB}
gradient_checkpointing: ${GRADIENT_CHECKPOINTING}
trust_remote_code: ${TRUST_REMOTE_CODE}
YAML

if bool_flag "$AUTO_CONFIGURE_FROM_TARGET"; then
  python scripts/configure_from_assistant.py \
    --assistant_model_path "$ASSISTANT_MODEL_PATH" \
    --target_model_path "$TARGET_MODEL_PATH" \
    --config "$CONFIG_PATH" \
    "${TRUST_ARG[@]}" \
    --overwrite
fi

if bool_flag "$RUN_PREPARE_DATA"; then
  if [[ "$TRAIN_JSONL" == "/path/to/train.jsonl" || ! -f "$TRAIN_JSONL" ]]; then
    echo "TRAIN_JSONL is not set to a real file: $TRAIN_JSONL" >&2
    exit 1
  fi
  PREP_ARGS=(
    --target_model_path "$TARGET_MODEL_PATH"
    --train_jsonl "$TRAIN_JSONL"
    --output_dir "$TOKENIZED_DATA_DIR"
    --max_length "$MAX_LENGTH"
    --eval_ratio "$EVAL_RATIO"
    --seed "$SEED"
  )
  if [[ -n "$TEST_JSONL" ]]; then
    PREP_ARGS+=(--test_jsonl "$TEST_JSONL")
  fi
  if bool_flag "$TRUST_REMOTE_CODE"; then
    PREP_ARGS+=(--trust_remote_code)
  fi
  python scripts/prepare_jsonl.py "${PREP_ARGS[@]}"
fi

if bool_flag "$RUN_TRAIN"; then
  accelerate launch -m eagle3_draft.train --config "$CONFIG_PATH"
fi

resolve_checkpoint() {
  if [[ -n "$EVAL_CHECKPOINT_PATH" ]]; then
    echo "$EVAL_CHECKPOINT_PATH"
    return
  fi
  if [[ -f "${BEST_CHECKPOINT_DIR}/draft_model.pt" ]]; then
    echo "$BEST_CHECKPOINT_DIR"
    return
  fi
  latest=$(find "$OUTPUT_DIR" -maxdepth 1 -type d -name 'checkpoint-*' | sort -V | tail -n 1 || true)
  if [[ -z "$latest" ]]; then
    echo "No checkpoint found under $OUTPUT_DIR" >&2
    exit 1
  fi
  echo "$latest"
}

resolve_eval_data_dir() {
  if [[ -n "$EVAL_DATA_DIR" ]]; then
    echo "$EVAL_DATA_DIR"
    return
  fi
  if [[ -n "$TEST_JSONL" && -f "${TOKENIZED_DATA_DIR}/test/data.pt" ]]; then
    echo "${TOKENIZED_DATA_DIR}/test"
    return
  fi
  echo "${TOKENIZED_DATA_DIR}/eval"
}

if bool_flag "$RUN_EVAL"; then
  CKPT=$(resolve_checkpoint)
  DATA_DIR=$(resolve_eval_data_dir)
  python -m eagle3_draft.eval \
    --config "$CONFIG_PATH" \
    --checkpoint_path "$CKPT" \
    --data_dir "$DATA_DIR"
fi

if bool_flag "$SAVE_TEST_PREDICTIONS"; then
  if [[ -n "$TEST_JSONL" ]]; then
    CKPT=$(resolve_checkpoint)
    mkdir -p "$CKPT"
    python -m eagle3_draft.test_generation \
      --config "$CONFIG_PATH" \
      --checkpoint_path "$CKPT" \
      --input_jsonl "$TEST_JSONL" \
      --output_jsonl "${CKPT}/test_predictions.jsonl" \
      --max_new_tokens "$GEN_MAX_NEW_TOKENS"
  else
    echo "SAVE_TEST_PREDICTIONS=true but TEST_JSONL is empty; skipping test prediction export."
  fi
fi

if bool_flag "$RUN_TEST_GENERATION"; then
  CKPT=$(resolve_checkpoint)
  if [[ -n "$GEN_INPUT_JSONL" ]]; then
    OUT_JSONL="$GEN_OUTPUT_JSONL"
    if [[ -z "$OUT_JSONL" ]]; then
      OUT_JSONL="${CKPT}/predictions.jsonl"
    fi
    python -m eagle3_draft.test_generation \
      --config "$CONFIG_PATH" \
      --checkpoint_path "$CKPT" \
      --input_jsonl "$GEN_INPUT_JSONL" \
      --output_jsonl "$OUT_JSONL" \
      --max_new_tokens "$GEN_MAX_NEW_TOKENS"
  else
    python -m eagle3_draft.test_generation \
      --config "$CONFIG_PATH" \
      --checkpoint_path "$CKPT" \
      --response_text "$GEN_RESPONSE_TEXT" \
      --max_new_tokens "$GEN_MAX_NEW_TOKENS"
  fi
fi

echo "Pipeline finished. Config used: $CONFIG_PATH"
echo "Best checkpoint: $BEST_CHECKPOINT_DIR"
echo "TensorBoard: tensorboard --logdir $TENSORBOARD_LOG_DIR"
