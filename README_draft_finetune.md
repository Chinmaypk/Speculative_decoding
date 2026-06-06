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

## Run

```bash
python scripts/train_gemma4_draft.py \
  --target-model /path/to/your-finetuned-gemma-4-E2B-it \
  --draft-model google/gemma-4-E2B-it-assistant \
  --train-file data/train.jsonl \
  --eval-file data/valid.jsonl \
  --output-dir outputs/gemma4-e2b-assistant-distilled \
  --bf16 \
  --use-lora \
  --per-device-train-batch-size 1 \
  --gradient-accumulation-steps 16 \
  --learning-rate 2e-5 \
  --num-train-epochs 1
```

For a full assistant fine-tune, omit `--use-lora`, but expect higher VRAM use.

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
