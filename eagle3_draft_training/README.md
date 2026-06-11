# EAGLE-3 draft model training

This folder contains a compact, Gemma-compatible training recipe for an EAGLE-3-style draft model.

It is intentionally independent from the upstream `SafeAILab/EAGLE` codebase so it can be adapted to new target models such as Gemma without copying target-model-specific internals.

## What is implemented

EAGLE-3 changes the earlier EAGLE idea in two important ways:

1. **Direct token prediction**: the drafter predicts next-token logits directly instead of predicting only top-layer features.
2. **Multi-layer feature fusion**: the drafter uses low-, middle-, and high-level hidden states from the frozen target model.

This implementation follows those ideas:

- freezes the target model;
- reads hidden states from configurable target layers;
- fuses selected layers through learned projections;
- conditions on the previous token embedding;
- trains a lightweight Transformer drafter to predict the next target token;
- includes an optional scheduled-sampling approximation to reduce exposure bias.

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
    model.py
    train.py
```

## Install

```bash
cd eagle3_draft_training
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Login to Hugging Face if the target Gemma checkpoint is gated:

```bash
huggingface-cli login
```

## Prepare data

Input can be plain JSONL where each line contains either:

```json
{"text": "Your complete training text here"}
```

or chat-style records:

```json
{"messages": [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello"}]}
```

Convert it into tokenized `.pt` shards:

```bash
python scripts/prepare_jsonl.py \
  --model_name_or_path google/gemma-4-e2b-it \
  --input_jsonl data/raw/train.jsonl \
  --output_dir data/tokenized/gemma4 \
  --max_length 2048
```

Change `google/gemma-4-e2b-it` to your actual Gemma 4 checkpoint name or local fine-tuned model path if different.

## Train

```bash
bash scripts/train_gemma4_eagle3.sh
```

or directly:

```bash
accelerate launch -m eagle3_draft.train --config configs/gemma4_example.yaml
```

## Important notes

- This trains the **draft / drafter model**, not the full Gemma target model.
- The target model is frozen and is only used to produce hidden states and labels.
- For production-grade serving, you still need integration with a verification engine such as SGLang, vLLM, or a custom speculative decoding loop.
- For large Gemma checkpoints, use BF16 and DeepSpeed/FSDP/TPU equivalents as needed.
- Official EAGLE-3 serving frameworks may expect checkpoint metadata/tree configs that are different from this lightweight trainer. Treat this folder as a training starting point.

## Minimal workflow

1. Put raw instruction/chat JSONL under `data/raw/train.jsonl`.
2. Edit `configs/gemma4_example.yaml`.
3. Run `prepare_jsonl.py`.
4. Run `train_gemma4_eagle3.sh`.
5. Use the saved drafter checkpoint from `outputs/gemma4_eagle3_draft`.
