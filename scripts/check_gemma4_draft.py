#!/usr/bin/env python
"""Smoke-test a Gemma 4 assistant drafter with Transformers generation."""

from __future__ import annotations

import argparse
import time


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check target vs target+assistant generation.")
    parser.add_argument("--target-model", required=True)
    parser.add_argument("--assistant-model", required=True)
    parser.add_argument("--prompt", default="Write a short Python function that reverses a string.")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    return parser.parse_args()


def resolve_dtype(args: argparse.Namespace):
    import torch

    if args.bf16 and args.fp16:
        raise ValueError("choose only one of --bf16 or --fp16")
    if args.bf16:
        return torch.bfloat16
    if args.fp16:
        return torch.float16
    return "auto"


def timed_generate(model, inputs, max_new_tokens: int, assistant_model=None):
    start = time.perf_counter()
    outputs = model.generate(
        **inputs,
        assistant_model=assistant_model,
        do_sample=False,
        max_new_tokens=max_new_tokens,
    )
    return outputs, time.perf_counter() - start


def main() -> None:
    from transformers import AutoModelForCausalLM, AutoProcessor

    args = parse_args()
    dtype = resolve_dtype(args)
    processor = AutoProcessor.from_pretrained(args.target_model)
    target = AutoModelForCausalLM.from_pretrained(
        args.target_model,
        torch_dtype=dtype,
        device_map="auto",
    ).eval()
    assistant = AutoModelForCausalLM.from_pretrained(
        args.assistant_model,
        torch_dtype=dtype,
        device_map="auto",
    ).eval()

    messages = [{"role": "user", "content": args.prompt}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=text, return_tensors="pt").to(target.device)
    input_len = inputs["input_ids"].shape[-1]

    base_outputs, base_seconds = timed_generate(target, inputs, args.max_new_tokens)
    draft_outputs, draft_seconds = timed_generate(
        target,
        inputs,
        args.max_new_tokens,
        assistant_model=assistant,
    )

    base_text = processor.decode(base_outputs[0][input_len:], skip_special_tokens=True)
    draft_text = processor.decode(draft_outputs[0][input_len:], skip_special_tokens=True)
    print(f"base_seconds={base_seconds:.3f}")
    print(f"assistant_seconds={draft_seconds:.3f}")
    print(f"same_greedy_output={base_text == draft_text}")
    print(draft_text)


if __name__ == "__main__":
    main()
