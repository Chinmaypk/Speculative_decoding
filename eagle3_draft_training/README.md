# EAGLE-3 draft model training

This folder contains a compact, Gemma-compatible training recipe for an EAGLE-3-style draft model.

## Naming convention

- `assistant_model_path`: path or Hugging Face ID of your existing assistant model. This is kept separate for clarity and future assisted-generation integration.
- `target_model_path`: path or Hugging Face ID of the frozen target model used for tokenization, hidden states, embeddings, vocabulary size, and drafter supervision.
- `target_model`: the in-memory frozen model object created from `target_model_path`.
- `resume_draft_checkpoint_path`: path to an existing **drafter** checkpoint if you want to continue fine-tuning the draft model.

Older configs using `model_name_or_path` still load as a fallback for `target_model_path`, but new configs should use `target_model_path` explicitly.

## Current local paths

```yaml
assistant_model_path: /home/c.kulkarni/hf_models/google/gemma-4-E2B-it-assistant
target_model_path: /home/c.kulkarni/hf_models/google/gemma-4-E2B-it
```

## What is implemented

EAGLE-3 changes the earlier EAGLE idea in two important ways:

1. **Direct token prediction**: the drafter predicts next-token logits directly instead of predicting only top-layer features.
2. **Multi-layer feature fusion**: the drafter uses low-, middle-, and high-level hidden states from the frozen target model.

This implementation follows those ideas:

- freezes the target model;
- reads hidden states from configurable target layers;
- fuses selected layers through learned projections;
- conditions on the previous token embedding;
- trains a lightweight Transformer drafter to predict the target `genui_json` tokens;
- supports supervised JSONL data with `response_text` as input and `genui_json` as output;
- splits only the training JSONL into train/eval and supports a separate held-out test JSONL;
- supports TensorBoard logging, standalone evaluation, generation testing, target-model auto-configuration, and resuming/fine-tuning an existing drafter checkpoint.

## Folder layout

```text
eagle3_draft_training/
  README.md
  requirements.txt
  configs/
    gemma4_example.yaml
  scripts/
    configure_from_assistant.py
    prepare_jsonl.py
    train_gemma4_eagle3.sh
  src/eagle3_draft/
    __init__.py
    config.py
    data.py
    eval.py
    model.py
    test_generation.py
    train.py
```

## Install

```bash
cd eagle3_draft_training
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Login to Hugging Face if either model checkpoint is gated:

```bash
huggingface-cli login
```

## Auto-configure from your target model

```bash
python scripts/configure_from_assistant.py \
  --assistant_model_path /home/c.kulkarni/hf_models/google/gemma-4-E2B-it-assistant \
  --target_model_path /home/c.kulkarni/hf_models/google/gemma-4-E2B-it \
  --config configs/gemma4_example.yaml \
  --trust_remote_code \
  --overwrite
```

This inspects the target model config and overwrites the YAML values for:

```yaml
assistant_model_path
target_model_path
target_hidden_layer_indices
draft_hidden_size
draft_num_heads
draft_num_layers
draft_intermediate_size
```

The utility chooses low/mid/high hidden-state indices based on the target model `num_hidden_layers`, so you do not need to manually guess valid Gemma/Gemma-like layer indices.

## Expected JSONL format

Each JSONL line must be one JSON object with these keys:

```json
{"response_text": "assistant natural language response here", "genui_json": {"type": "..."}}
```

`response_text` is used as the input/prompt. `genui_json` is used as the supervised output. During training, prompt tokens are masked with `-100`, so loss is computed only on output JSON tokens.

## Prepare data

Pass one JSONL for training. The script splits this into `train` and `eval` using `--eval_ratio`. Pass a second JSONL only when you want a separate held-out test set.

```bash
python scripts/prepare_jsonl.py \
  --target_model_path /home/c.kulkarni/hf_models/google/gemma-4-E2B-it \
  --train_jsonl data/raw/train.jsonl \
  --test_jsonl data/raw/test.jsonl \
  --output_dir data/tokenized/gemma4 \
  --max_length 2048 \
  --eval_ratio 0.05 \
  --trust_remote_code
```

The preprocessing tokenizer should match the target model vocabulary, so use `target_model_path` here.

This creates:

```text
data/tokenized/gemma4/train/data.pt   # from train_jsonl
data/tokenized/gemma4/eval/data.pt    # split from train_jsonl
data/tokenized/gemma4/test/data.pt    # from test_jsonl, if provided
```

If `--test_jsonl` is omitted, only train/eval are created.

## Train with TensorBoard logging

```bash
bash scripts/train_gemma4_eagle3.sh configs/gemma4_example.yaml
```

or directly:

```bash
accelerate launch -m eagle3_draft.train --config configs/gemma4_example.yaml
```

TensorBoard logs are written to:

```text
outputs/gemma4_eagle3_draft/tensorboard
```

Open them with:

```bash
tensorboard --logdir outputs/gemma4_eagle3_draft/tensorboard
```

Logged metrics include:

- `train/loss`
- `train/accuracy`
- `train/perplexity`
- `train/lr`
- `eval/loss`
- `eval/accuracy`
- `eval/perplexity`

## Standalone evaluation

```bash
python -m eagle3_draft.eval \
  --config configs/gemma4_example.yaml \
  --checkpoint_path outputs/gemma4_eagle3_draft/checkpoint-500 \
  --data_dir data/tokenized/gemma4/test
```

## Test generation

Single input:

```bash
python -m eagle3_draft.test_generation \
  --config configs/gemma4_example.yaml \
  --checkpoint_path outputs/gemma4_eagle3_draft/checkpoint-500 \
  --response_text "Create a chart showing monthly revenue"
```

Batch JSONL:

```bash
python -m eagle3_draft.test_generation \
  --config configs/gemma4_example.yaml \
  --checkpoint_path outputs/gemma4_eagle3_draft/checkpoint-500 \
  --input_jsonl data/raw/test.jsonl \
  --output_jsonl outputs/predictions.jsonl
```

## Fine-tuning from an existing drafter checkpoint

If you already have a trained drafter checkpoint and want to continue fine-tuning it, set:

```yaml
resume_draft_checkpoint_path: outputs/gemma4_eagle3_draft/checkpoint-500
```

That path can be either a checkpoint folder or the direct file path to `draft_model.pt`.

## Important notes

- This trains the **draft / drafter model**, not the full assistant or target model.
- The target model is frozen and is only used to produce hidden states and embeddings.
- For production-grade serving, you still need integration with a verification engine such as SGLang, vLLM, or a custom speculative decoding loop.
- Official EAGLE-3 serving frameworks may expect checkpoint metadata/tree configs that are different from this lightweight trainer. Treat this folder as a training starting point.
