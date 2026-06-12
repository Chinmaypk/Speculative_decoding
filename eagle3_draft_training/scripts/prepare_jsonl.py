#!/usr/bin/env python
"""Prepare JSONL records for EAGLE-3 drafter training.

Expected input format, one JSON object per line:
  {"response_text": "...", "genui_json": {...}}

Usage:
  - Pass --train_jsonl for training data. It is split into train/eval.
  - Optionally pass --test_jsonl for a separate held-out test set.

The model receives `response_text` as the prompt/input and learns to generate
`genui_json` as the supervised target. Prompt tokens are masked with -100.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm
from transformers import AutoTokenizer


DEFAULT_SYSTEM_PROMPT = (
    "Convert the assistant response into the correct GenUI JSON. "
    "Return only valid JSON."
)


def normalize_target(value: Any) -> str:
    if isinstance(value, str):
        try:
            return json.dumps(json.loads(value), ensure_ascii=False, separators=(",", ":"))
        except json.JSONDecodeError:
            return value.strip()
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def build_prompt(response_text: str, system_prompt: str) -> str:
    return (
        f"{system_prompt}\n\n"
        f"### response_text\n{response_text.strip()}\n\n"
        f"### genui_json\n"
    )


def encode_example(record: dict[str, Any], tokenizer: AutoTokenizer, max_length: int, system_prompt: str) -> dict[str, Any] | None:
    if "response_text" not in record or "genui_json" not in record:
        raise ValueError("Each JSONL line must contain 'response_text' and 'genui_json'.")

    response_text = str(record["response_text"])
    target_text = normalize_target(record["genui_json"])
    if not response_text.strip() or not target_text.strip():
        return None

    prompt = build_prompt(response_text, system_prompt)
    prompt_ids = tokenizer(prompt, add_special_tokens=True)["input_ids"]
    target_ids = tokenizer(target_text + tokenizer.eos_token, add_special_tokens=False)["input_ids"]

    input_ids = (prompt_ids + target_ids)[:max_length]
    prompt_length = min(len(prompt_ids), len(input_ids))
    labels = [-100] * prompt_length + input_ids[prompt_length:]

    if len(input_ids) < 8 or all(label == -100 for label in labels):
        return None

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "prompt_length": prompt_length,
        "response_text": response_text,
        "target_text": target_text,
    }


def load_jsonl_examples(
    jsonl_path: str | Path,
    tokenizer: AutoTokenizer,
    max_length: int,
    system_prompt: str,
    split_name: str,
) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line_number, line in enumerate(tqdm(f, desc=f"Tokenizing {split_name}"), start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                example = encode_example(record, tokenizer, max_length, system_prompt)
            except Exception as exc:
                raise ValueError(f"Failed to parse {jsonl_path} line {line_number}: {exc}") from exc
            if example is not None:
                examples.append(example)
    return examples


def save_split(examples: list[dict[str, Any]], output_dir: Path, name: str) -> None:
    split_dir = output_dir / name
    split_dir.mkdir(parents=True, exist_ok=True)
    torch.save(examples, split_dir / "data.pt")
    with open(split_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump({"num_examples": len(examples)}, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_model_path", required=True, help="Target model path used for tokenizer/vocabulary.")
    parser.add_argument("--train_jsonl", required=True, help="Training JSONL. This is split into train/eval only.")
    parser.add_argument("--test_jsonl", default=None, help="Optional separate held-out test JSONL. Not mixed into train/eval.")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--eval_ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--system_prompt", default=DEFAULT_SYSTEM_PROMPT)
    parser.add_argument("--trust_remote_code", action="store_true")
    args = parser.parse_args()

    if not 0.0 <= args.eval_ratio < 1.0:
        raise ValueError("--eval_ratio must be >= 0.0 and < 1.0")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(
        args.target_model_path,
        trust_remote_code=args.trust_remote_code,
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_source_examples = load_jsonl_examples(
        args.train_jsonl,
        tokenizer,
        args.max_length,
        args.system_prompt,
        split_name="train_jsonl",
    )
    if not train_source_examples:
        raise ValueError("No valid examples were produced from --train_jsonl.")

    random.Random(args.seed).shuffle(train_source_examples)
    n_total = len(train_source_examples)
    n_eval = int(n_total * args.eval_ratio)
    eval_examples = train_source_examples[:n_eval]
    train_examples = train_source_examples[n_eval:]

    if not train_examples:
        train_examples = train_source_examples
        eval_examples = []

    test_examples: list[dict[str, Any]] = []
    if args.test_jsonl:
        test_examples = load_jsonl_examples(
            args.test_jsonl,
            tokenizer,
            args.max_length,
            args.system_prompt,
            split_name="test_jsonl",
        )

    save_split(train_examples, output_dir, "train")
    save_split(eval_examples, output_dir, "eval")
    if args.test_jsonl:
        save_split(test_examples, output_dir, "test")
    tokenizer.save_pretrained(output_dir / "tokenizer")

    print(
        f"Saved train={len(train_examples)}, eval={len(eval_examples)}, test={len(test_examples)} "
        f"under {output_dir}"
    )


if __name__ == "__main__":
    main()
