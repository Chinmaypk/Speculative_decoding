#!/usr/bin/env python
"""Fine-tune a Gemma 4 MTP assistant/draft model against a target model.

This script is intended for the case where you already fine-tuned the target
Gemma 4 model and now want the matching `*-assistant` drafter to follow that
fine-tuned target more closely for speculative decoding.
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class JsonlExample:
    text: str


def read_jsonl(path: str | os.PathLike[str]) -> Iterable[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: each JSONL row must be an object")
            yield value


def example_to_text(example: dict[str, Any], processor: Any | None = None) -> str:
    """Convert common SFT JSONL shapes into a single training string."""
    if "text" in example:
        text = example["text"]
        if not isinstance(text, str) or not text.strip():
            raise ValueError("`text` must be a non-empty string")
        return text

    if "messages" in example:
        messages = example["messages"]
        if not isinstance(messages, list) or not messages:
            raise ValueError("`messages` must be a non-empty list")
        if processor is None or not hasattr(processor, "apply_chat_template"):
            raise ValueError("chat-template examples require an AutoProcessor/AutoTokenizer")
        return processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )

    if "prompt" in example and "completion" in example:
        prompt = example["prompt"]
        completion = example["completion"]
        if not isinstance(prompt, str) or not isinstance(completion, str):
            raise ValueError("`prompt` and `completion` must be strings")
        return prompt + completion

    raise ValueError("expected one of: `text`, `messages`, or `prompt` + `completion`")


def load_texts(
    data_path: str | os.PathLike[str],
    processor: Any | None,
    max_examples: int | None = None,
) -> list[JsonlExample]:
    texts: list[JsonlExample] = []
    for row in read_jsonl(data_path):
        texts.append(JsonlExample(example_to_text(row, processor)))
        if max_examples is not None and len(texts) >= max_examples:
            break
    if not texts:
        raise ValueError(f"no training examples found in {data_path}")
    return texts


class DraftDistillDataset:
    def __init__(
        self,
        examples: list[JsonlExample],
        tokenizer: Any,
        max_length: int,
    ) -> None:
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        encoded = self.tokenizer(
            self.examples[idx].text,
            truncation=True,
            max_length=self.max_length,
            return_tensors=None,
        )
        return {"input_ids": encoded["input_ids"]}


def collate_batch(features: list[dict[str, Any]], tokenizer: Any) -> dict[str, Any]:
    import torch

    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    max_len = max(len(item["input_ids"]) for item in features)
    input_ids = []
    attention_mask = []
    for item in features:
        ids = list(item["input_ids"])
        pad = max_len - len(ids)
        input_ids.append(ids + [pad_id] * pad)
        attention_mask.append([1] * len(ids) + [0] * pad)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
    }


def distillation_loss(
    draft_logits: Any,
    target_logits: Any,
    labels: Any,
    attention_mask: Any,
    temperature: float,
    kl_weight: float,
    ce_weight: float,
) -> Any:
    import torch.nn.functional as F

    # Predict token t+1 from position t, matching standard CausalLM training.
    draft = draft_logits[:, :-1, :].float()
    target = target_logits[:, :-1, :].float()
    shifted_labels = labels[:, 1:]
    shifted_mask = attention_mask[:, 1:].bool()

    if shifted_mask.sum().item() == 0:
        return draft.sum() * 0.0

    draft_active = draft[shifted_mask]
    target_active = target[shifted_mask]
    label_active = shifted_labels[shifted_mask]

    temp = float(temperature)
    kl = F.kl_div(
        F.log_softmax(draft_active / temp, dim=-1),
        F.softmax(target_active / temp, dim=-1),
        reduction="batchmean",
    ) * (temp * temp)
    ce = F.cross_entropy(draft_active, label_active)
    return kl_weight * kl + ce_weight * ce


def distillation_loss_parts(
    draft_logits: Any,
    target_logits: Any,
    labels: Any,
    attention_mask: Any,
    temperature: float,
    kl_weight: float,
    ce_weight: float,
) -> dict[str, Any]:
    import torch.nn.functional as F

    draft = draft_logits[:, :-1, :].float()
    target = target_logits[:, :-1, :].float()
    shifted_labels = labels[:, 1:]
    shifted_mask = attention_mask[:, 1:].bool()
    active_tokens = shifted_mask.sum()

    if active_tokens.item() == 0:
        zero = draft.sum() * 0.0
        return {"loss": zero, "kl_loss": zero, "ce_loss": zero, "active_tokens": active_tokens}

    draft_active = draft[shifted_mask]
    target_active = target[shifted_mask]
    label_active = shifted_labels[shifted_mask]

    temp = float(temperature)
    kl = F.kl_div(
        F.log_softmax(draft_active / temp, dim=-1),
        F.softmax(target_active / temp, dim=-1),
        reduction="batchmean",
    ) * (temp * temp)
    ce = F.cross_entropy(draft_active, label_active)
    return {
        "loss": kl_weight * kl + ce_weight * ce,
        "kl_loss": kl.detach(),
        "ce_loss": ce.detach(),
        "active_tokens": active_tokens.detach(),
    }


def make_draft_distillation_trainer_class() -> Any:
    from transformers import Trainer

    class DraftDistillationTrainer(Trainer):
        def __init__(
            self,
            *args: Any,
            target_model: Any,
            temperature: float = 1.0,
            kl_weight: float = 1.0,
            ce_weight: float = 0.1,
            debug_loss_steps: int = 10,
            **kwargs: Any,
        ) -> None:
            super().__init__(*args, **kwargs)
            self.target_model = target_model
            self.temperature = temperature
            self.kl_weight = kl_weight
            self.ce_weight = ce_weight
            self.debug_loss_steps = max(1, debug_loss_steps)

        def compute_loss(
            self,
            model: Any,
            inputs: dict[str, Any],
            return_outputs: bool = False,
            **_: Any,
        ) -> Any:
            import torch

            labels = inputs["input_ids"]
            attention_mask = inputs["attention_mask"]
            model_device = next(model.parameters()).device
            if next(self.target_model.parameters()).device != model_device:
                self.target_model.to(model_device)

            with torch.no_grad():
                target_outputs = self.target_model(**inputs)
            draft_outputs = model(**inputs)
            parts = distillation_loss_parts(
                draft_outputs.logits,
                target_outputs.logits,
                labels,
                attention_mask,
                self.temperature,
                self.kl_weight,
                self.ce_weight,
            )
            if self.state.global_step % self.debug_loss_steps == 0:
                self.log(
                    {
                        "train/kl_loss": parts["kl_loss"].item(),
                        "train/ce_loss": parts["ce_loss"].item(),
                        "train/active_tokens": float(parts["active_tokens"].item()),
                    }
                )
            return (parts["loss"], draft_outputs) if return_outputs else parts["loss"]

    return DraftDistillationTrainer


def is_main_process() -> bool:
    return int(os.environ.get("RANK", "0")) == 0


def debug_print(message: str) -> None:
    if is_main_process():
        print(f"[gemma4-draft] {message}", flush=True)


def count_trainable_parameters(model: Any) -> tuple[int, int]:
    trainable = 0
    total = 0
    for parameter in model.parameters():
        count = parameter.numel()
        total += count
        if parameter.requires_grad:
            trainable += count
    return trainable, total


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune a Gemma 4 assistant draft model with target-logit distillation."
    )
    parser.add_argument("--target-model", required=True, help="Fine-tuned target model path or HF id")
    parser.add_argument("--draft-model", required=True, help="Matching Gemma 4 *-assistant model path or HF id")
    parser.add_argument("--train-file", required=True, help="JSONL with text, messages, or prompt/completion rows")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--save-final-dir",
        help="Optional final drafter save directory. Defaults to --output-dir/final_drafter.",
    )
    parser.add_argument("--eval-file")
    parser.add_argument(
        "--training-mode",
        choices=["single_gpu", "multi_gpu"],
        default="single_gpu",
        help="Use single_gpu for normal python launch, multi_gpu for accelerate/torchrun launch.",
    )
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--max-train-examples", type=int)
    parser.add_argument("--max-eval-examples", type=int)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--kl-weight", type=float, default=1.0)
    parser.add_argument("--ce-weight", type=float, default=0.1)
    parser.add_argument("--bf16", action="store_true", help="Use bf16 training/loading")
    parser.add_argument("--fp16", action="store_true", help="Use fp16 training/loading")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--use-lora", action="store_true", help="Train a PEFT LoRA adapter instead of full weights")
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--debug-loss-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=500)
    parser.add_argument("--eval-steps", type=int, default=500)
    parser.add_argument("--tensorboard-log-dir", help="Defaults to <output-dir>/tensorboard")
    parser.add_argument("--disable-tensorboard", action="store_true")
    parser.add_argument("--debug-samples", type=int, default=1, help="Print this many rendered samples")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true", help="Load and tokenize data, then exit before model loading")
    return parser.parse_args()


def resolve_dtype(args: argparse.Namespace) -> Any:
    import torch

    if args.bf16 and args.fp16:
        raise ValueError("choose only one of --bf16 or --fp16")
    if args.bf16:
        return torch.bfloat16
    if args.fp16:
        return torch.float16
    return "auto"


def main() -> None:
    args = parse_args()

    from transformers import AutoModelForCausalLM, AutoProcessor, TrainingArguments, set_seed

    set_seed(args.seed)
    debug_print(f"training_mode={args.training_mode}")
    debug_print(f"target_model={args.target_model}")
    debug_print(f"draft_model={args.draft_model}")
    debug_print(f"train_file={args.train_file}")
    if args.eval_file:
        debug_print(f"eval_file={args.eval_file}")

    if args.training_mode == "multi_gpu" and int(os.environ.get("WORLD_SIZE", "1")) == 1:
        debug_print("multi_gpu mode selected, but WORLD_SIZE=1. Launch with accelerate or torchrun.")

    processor = AutoProcessor.from_pretrained(args.target_model)
    tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    debug_print(f"tokenizer vocab_size={len(tokenizer)} pad_token_id={tokenizer.pad_token_id}")

    train_examples = load_texts(args.train_file, processor, args.max_train_examples)
    eval_examples = (
        load_texts(args.eval_file, processor, args.max_eval_examples) if args.eval_file else None
    )

    train_dataset = DraftDistillDataset(train_examples, tokenizer, args.max_length)
    eval_dataset = DraftDistillDataset(eval_examples, tokenizer, args.max_length) if eval_examples else None

    sample_tokens = train_dataset[0]["input_ids"]
    if len(sample_tokens) < 2:
        raise ValueError("first example tokenizes to fewer than two tokens")
    debug_print(f"loaded_train_examples={len(train_dataset)}")
    if eval_dataset is not None:
        debug_print(f"loaded_eval_examples={len(eval_dataset)}")
    debug_print(f"first_tokenized_length={len(sample_tokens)}")
    for idx in range(min(args.debug_samples, len(train_examples))):
        preview = train_examples[idx].text.replace("\n", "\\n")[:500]
        debug_print(f"sample[{idx}]={preview}")

    if args.dry_run:
        debug_print("dry run passed")
        return

    dtype = resolve_dtype(args)
    debug_print(f"loading models dtype={dtype}")
    model_load_kwargs = {"torch_dtype": dtype}
    if args.training_mode == "single_gpu":
        debug_print("single_gpu mode: Trainer will place the drafter and frozen target on one device")
    else:
        debug_print("multi_gpu mode: each process loads target+drafter; do not use device_map='auto'")

    target_model = AutoModelForCausalLM.from_pretrained(
        args.target_model,
        **model_load_kwargs,
    )
    draft_model = AutoModelForCausalLM.from_pretrained(
        args.draft_model,
        **model_load_kwargs,
    )

    target_model.eval()
    for parameter in target_model.parameters():
        parameter.requires_grad_(False)
    debug_print("frozen target model parameters")

    if args.gradient_checkpointing:
        draft_model.gradient_checkpointing_enable()
        draft_model.config.use_cache = False
        debug_print("enabled drafter gradient checkpointing")

    if args.use_lora:
        from peft import LoraConfig, get_peft_model

        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules="all-linear",
        )
        draft_model = get_peft_model(draft_model, lora_config)
        if is_main_process():
            draft_model.print_trainable_parameters()

    trainable_params, total_params = count_trainable_parameters(draft_model)
    debug_print(
        "drafter parameters: "
        f"trainable={trainable_params:,} total={total_params:,} "
        f"ratio={100 * trainable_params / max(total_params, 1):.4f}%"
    )

    evaluation_strategy = "steps" if eval_dataset is not None else "no"
    tensorboard_log_dir = args.tensorboard_log_dir or str(Path(args.output_dir) / "tensorboard")
    report_to = [] if args.disable_tensorboard else ["tensorboard"]
    training_kwargs = {
        "output_dir": args.output_dir,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "num_train_epochs": args.num_train_epochs,
        "warmup_ratio": args.warmup_ratio,
        "weight_decay": args.weight_decay,
        "logging_steps": args.logging_steps,
        "logging_first_step": True,
        "logging_strategy": "steps",
        "save_steps": args.save_steps,
        "eval_steps": args.eval_steps,
        "save_total_limit": 3,
        "bf16": args.bf16,
        "fp16": args.fp16,
        "remove_unused_columns": False,
        "report_to": report_to,
        "logging_dir": tensorboard_log_dir,
    }
    strategy_arg = (
        "eval_strategy"
        if "eval_strategy" in inspect.signature(TrainingArguments).parameters
        else "evaluation_strategy"
    )
    training_kwargs[strategy_arg] = evaluation_strategy
    training_args = TrainingArguments(**training_kwargs)

    debug_print(f"output_dir={args.output_dir}")
    debug_print(f"final_save_dir={args.save_final_dir or str(Path(args.output_dir) / 'final_drafter')}")
    if not args.disable_tensorboard:
        debug_print(f"tensorboard_log_dir={tensorboard_log_dir}")
    debug_print("starting training")

    TrainerClass = make_draft_distillation_trainer_class()
    trainer = TrainerClass(
        model=draft_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=lambda features: collate_batch(features, tokenizer),
        target_model=target_model,
        temperature=args.temperature,
        kl_weight=args.kl_weight,
        ce_weight=args.ce_weight,
        debug_loss_steps=args.debug_loss_steps,
    )
    train_result = trainer.train()

    debug_print(f"training finished: {train_result.metrics}")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    final_dir = args.save_final_dir or str(Path(args.output_dir) / "final_drafter")
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    debug_print(f"saved drafter checkpoint to {args.output_dir}")
    debug_print(f"saved final drafter checkpoint to {final_dir}")

    config_path = Path(args.output_dir) / "draft_distillation_config.json"
    config_path.write_text(
        json.dumps(
            {
                "target_model": args.target_model,
                "draft_model": args.draft_model,
                "training_mode": args.training_mode,
                "temperature": args.temperature,
                "kl_weight": args.kl_weight,
                "ce_weight": args.ce_weight,
                "max_length": args.max_length,
                "tensorboard_log_dir": None if args.disable_tensorboard else tensorboard_log_dir,
                "final_save_dir": final_dir,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    debug_print(f"wrote config to {config_path}")


if __name__ == "__main__":
    main()
