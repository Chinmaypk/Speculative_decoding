# EAGLE-3 draft model training

This folder contains a compact, Gemma-compatible training recipe for an EAGLE-3-style draft model.

It is intentionally independent from the upstream `SafeAILab/EAGLE` codebase so it can be adapted to new target models such as Gemma without copying target-model-specific internals.

## What is implemented

EAGLE-3 changes the earlier EAGLE idea in two important ways:

1. **Direct token prediction**: the drafter predicts next-token logits directly instead of predicting only top-layer features.
2. **Multi-layer feature fusion**: the drafter uses low-, middle-, and high-level hidden states from the frozen target model.

This implementation follows those ideas:

- freezes the target/assistant model;
- reads hidden states from configurable target layers;
- fuses selected layers through learned projections;
- conditions on the previous token embedding;
- trains a lightweight Transformer drafter to predict the target `genui_json` tokens;
- supports supervised JSONL data with `response_text` as input and `genui_json` as output;
- supports train/eval/test splits, TensorBoard logging, standalone evaluation, generation testing, and resuming/fine-tuning an existing drafter checkpoint.

## Folder layout

```text
eagle3_draft_training/
  README.md
  requirements.txt
  configs/
    gemma4_example.yaml
  scripts/
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

Login to Hugging Face if the target model checkpoint is gated:

```bash
huggingface-cli login
```

## Expected training data

Each JSONL line must be one JSON object with these keys:

```json
{"response_text": "assistant natural language response here", "genui_json": {"type": "..."}}
```

`response_text` is used as the input/prompt. `genui_json` is used as the supervised output. During training, prompt tokens are masked with `-100`, so loss is computed only on output JSON tokens.

## Prepare data

```bash
python scripts/prepare_jsonl.py \
  --model_name_or_path google/gemma-4-e2b-it \
  --input_jsonl data/raw/train.jsonl \
  --output_dir data/tokenized/gemma4 \
  --max_length 2048 \
  --eval_ratio 0.05 \
  --test_ratio 0.05 \
  --trust_remote_code
```

This creates:

```text
data/tokenized/gemma4/train/data.pt
data/tokenized/gemma4/eval/data.pt
data/tokenized/gemma4/test/data.pt
```

Change `google/gemma-4-e2b-it` to your actual assistant/target model path or Hugging Face checkpoint.

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

## Fine-tuning from an existing assistant/drafter checkpoint

Yes, if you already have an assistant/target model path, set:

```yaml
model_name_or_path: /path/to/your/assistant_model
```

If you already have a trained drafter checkpoint and want to continue fine-tuning it, set:

```yaml
resume_draft_checkpoint_path: outputs/gemma4_eagle3_draft/checkpoint-500
```

That path can be either a checkpoint folder or the direct file path to `draft_model.pt`.

## Important notes

- This trains the **draft / drafter model**, not the full target model.
- The target/assistant model is frozen and is only used to produce hidden states and embeddings.
- For production-grade serving, you still need integration with a verification engine such as SGLang, vLLM, or a custom speculative decoding loop.
- Official EAGLE-3 serving frameworks may expect checkpoint metadata/tree configs that are different from this lightweight trainer. Treat this folder as a training starting point.
