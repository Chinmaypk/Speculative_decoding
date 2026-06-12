#!/usr/bin/env python
"""Prepare JSONL records for EAGLE-3 drafter training.

Expected input format, one JSON object per line:
  {"response_text": "...", "genui_json": {...}}

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


def save_split(examples: list[dict[str, Any]], output_dir: Path, name: str) -> None:
    split_dir = output_dir / name
    split_dir.mkdir(parents=True, exist_ok=True)
    torch.save(examples, split_dir / "data.pt")
    with open(split_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump({"num_examples": len(examples)}, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--input_jsonl", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--eval_ratio", type=float, default=0.05)
    parser.add_argument("--test_ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--system_prompt", default=DEFAULT_SYSTEM_PROMPT)
    parser.add_argument("--trust_remote_code", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=args.trust_remote_code,
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    examples: list[dict[str, Any]] = []
    with open(args.input_jsonl, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Tokenizing response_text -> genui_json"):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            example = encode_example(record, tokenizer, args.max_length, args.system_prompt)
            if example is not None:
                examples.append(example)

    if not examples:
        raise ValueError("No valid examples were produced from the JSONL file.")

    random.Random(args.seed).shuffle(examples)
    n_total = len(examples)
    n_test = int(n_total * args.test_ratio)
    n_eval = int(n_total * args.eval_ratio)
    test_examples = examples[:n_test]
    eval_examples = examples[n_test : n_test + n_eval]
    train_examples = examples[n_test + n_eval :]

    if not train_examples:
        train_examples = examples
        eval_examples = []
        test_examples = []

    save_split(train_examples, output_dir, "train")
    save_split(eval_examples, output_dir, "eval")
    save_split(test_examples, output_dir, "test")
    tokenizer.save_pretrained(output_dir / "tokenizer")

    print(
        f"Saved train={len(train_examples)}, eval={len(eval_examples)}, test={len(test_examples)} "
        f"under {output_dir}"
    )


if __name__ == "__main__":
    main()
