# Gemma 4 Draft Model Fine-Tuning

Gemma 4 speculative decoding uses matching `*-assistant` MTP drafters, not a
normal smaller chat model. The target model still verifies every proposed token,
so output quality is unchanged at inference time, but speed depends on how often
the assistant proposes tokens that the target accepts.

If you fine-tuned `google/gemma-4-E2B-it`, fine-tune the matching assistant,
for example `google/gemma-4-E2B-it-assistant`, against your fine-tuned target.
The script in `scripts/train_gemma4_draft.py` freezes the target and trains the
assistant with next-token KL distillation from target logits, plus a small
cross-entropy term on the ground-truth tokens.

## Install

```bash
pip install -U torch transformers accelerate datasets peft
```

You also need Hugging Face access to the gated Gemma 4 checkpoints if you are
loading them from the Hub.

## Data

Use the same distribution you used to fine-tune the target. JSONL rows may use
one of these shapes:

```json
{"text": "<fully formatted training text>"}
{"prompt": "Question: ", "completion": "Answer"}
{"messages": [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello"}]}
```

For chat data, the target model processor applies the Gemma 4 chat template.

## Single-GPU Training

```bash
TARGET_MODEL=/path/to/your-finetuned-gemma-4-E2B-it \
TRAIN_FILE=data/train.jsonl \
EVAL_FILE=data/valid.jsonl \
OUTPUT_DIR=outputs/gemma4-e2b-assistant-distilled \
./finetune_gemma4_draft.sh
```

The single-GPU wrapper calls:

```bash
python scripts/train_gemma4_draft.py ... --training-mode single_gpu
```

## Multi-GPU Training

Use `accelerate` for proper multi-process training:

```bash
NUM_PROCESSES=4 \
TARGET_MODEL=/path/to/your-finetuned-gemma-4-E2B-it \
TRAIN_FILE=data/train.jsonl \
EVAL_FILE=data/valid.jsonl \
OUTPUT_DIR=outputs/gemma4-e2b-assistant-distilled-multigpu \
./finetune_gemma4_draft_multigpu.sh
```

For a full assistant fine-tune, omit `--use-lora`, but expect higher VRAM use.
In multi-GPU mode each process loads both the frozen target and trainable
drafter, so LoRA is the practical default.

## Logs and Saved Drafter

The trainer prints debug lines with the `[gemma4-draft]` prefix, and Hugging
Face Trainer prints step loss in the terminal. Extra distillation metrics are
logged as:

- `train/kl_loss`
- `train/ce_loss`
- `train/active_tokens`

TensorBoard logs are written to:

```text
<OUTPUT_DIR>/tensorboard
```

Launch TensorBoard with:

```bash
tensorboard --logdir outputs
```

The drafter is saved in two places:

```text
<OUTPUT_DIR>
<OUTPUT_DIR>/final_drafter
```

## Smoke Test

Before a real run, validate data formatting and tokenizer loading:

```bash
python scripts/train_gemma4_draft.py \
  --target-model /path/to/your-finetuned-gemma-4-E2B-it \
  --draft-model google/gemma-4-E2B-it-assistant \
  --train-file data/train.jsonl \
  --output-dir outputs/dry-run \
  --max-train-examples 2 \
  --dry-run
```

After training, pass the resulting assistant to generation:

```python
outputs = target_model.generate(**inputs, assistant_model=assistant_model, max_new_tokens=256)
```

Measure acceptance rate and tokens/sec on your serving stack; the best checkpoint
is usually the one with the highest accepted draft tokens per second, not
necessarily the lowest language-model loss.

You can also run the included local smoke test. With greedy decoding,
`same_greedy_output=True` is the first sanity check that the assistant is wired
into speculative generation correctly:

```bash
python scripts/check_gemma4_draft.py \
  --target-model /path/to/your-finetuned-gemma-4-E2B-it \
  --assistant-model outputs/gemma4-e2b-assistant-distilled \
  --bf16
```

## vLLM Speculative Inference

For vLLM, Gemma 4 assistant checkpoints should be used as MTP speculators:

```bash
pip install -r requirements-vllm.txt

python scripts/vllm_speculative_infer.py \
  --target-model /path/to/your-finetuned-gemma-4-E2B-it \
  --drafter-model outputs/gemma4-e2b-assistant-distilled \
  --num-speculative-tokens 3 \
  --tensor-parallel-size 1 \
  --dtype bfloat16 \
  --prompt "Write a short Python function that reverses a string."
```

If you trained the assistant with LoRA, merge the LoRA adapter into the assistant
base model before passing it to vLLM as `--drafter-model`.
