#!/usr/bin/env python
"""Prepare JSONL text/chat records for EAGLE-3 drafter training.

Expected input line formats:
  {"text": "..."}
  {"messages": [{"role": "user", "content": "..."}, ...]}

The script writes `data.pt` containing a list of token id tensors.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm
from transformers import AutoTokenizer


def render_record(record: dict[str, Any], tokenizer: AutoTokenizer) -> str:
    if "text" in record and record["text"]:
        return str(record["text"])

    if "messages" in record and record["messages"]:
        messages = record["messages"]
        if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
        parts: list[str] = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            parts.append(f"{role}: {content}")
        return "\n".join(parts)

    raise ValueError("Each JSONL line must contain either a non-empty 'text' or 'messages' field.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--input_jsonl", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_length", type=int, default=2048)
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

    examples: list[torch.Tensor] = []
    with open(args.input_jsonl, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Tokenizing"):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            text = render_record(record, tokenizer)
            encoded = tokenizer(
                text,
                truncation=True,
                max_length=args.max_length,
                add_special_tokens=True,
            )["input_ids"]
            if len(encoded) >= 8:
                examples.append(torch.tensor(encoded, dtype=torch.long))

    torch.save(examples, output_dir / "data.pt")
    tokenizer.save_pretrained(output_dir / "tokenizer")
    print(f"Saved {len(examples)} examples to {output_dir / 'data.pt'}")


if __name__ == "__main__":
    main()
