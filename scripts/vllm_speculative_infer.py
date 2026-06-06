#!/usr/bin/env python
"""Run Gemma 4 speculative decoding in vLLM with a trained assistant drafter.

Gemma 4 `*-assistant` checkpoints are MTP drafters in vLLM, so this script uses
`speculative_config={"method": "mtp", ...}` instead of generic draft-model
speculation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Offline vLLM inference with Gemma 4 MTP speculative decoding."
    )
    parser.add_argument("--target-model", required=True, help="Target Gemma 4 model path or HF id")
    parser.add_argument(
        "--drafter-model",
        required=True,
        help="Fine-tuned Gemma 4 assistant drafter path or HF id. Merge LoRA adapters first.",
    )
    parser.add_argument(
        "--prompt",
        default="Write a short Python function that checks whether a string is a palindrome.",
    )
    parser.add_argument(
        "--messages-json",
        help="Optional JSON file containing a chat messages list. Overrides --prompt.",
    )
    parser.add_argument("--system-prompt", default="You are a helpful assistant.")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--num-speculative-tokens", type=int, default=3)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--draft-tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--max-model-len", type=int)
    parser.add_argument("--dtype", default="bfloat16", choices=["auto", "half", "float16", "bfloat16", "float"])
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--raw-prompt",
        action="store_true",
        help="Send --prompt directly to vLLM without applying the Gemma chat template.",
    )
    parser.add_argument(
        "--print-request",
        action="store_true",
        help="Print the rendered prompt and speculative_config before generation.",
    )
    return parser.parse_args()


def load_messages(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.messages_json:
        path = Path(args.messages_json)
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, list):
            raise ValueError("--messages-json must contain a JSON list of chat messages")
        return value
    messages: list[dict[str, Any]] = []
    if args.system_prompt:
        messages.append({"role": "system", "content": args.system_prompt})
    messages.append({"role": "user", "content": args.prompt})
    return messages


def render_prompt(args: argparse.Namespace) -> str:
    if args.raw_prompt:
        return args.prompt

    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(
        args.target_model,
        trust_remote_code=args.trust_remote_code,
    )
    return processor.apply_chat_template(
        load_messages(args),
        tokenize=False,
        add_generation_prompt=True,
    )


def build_llm_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    speculative_config: dict[str, Any] = {
        "method": "mtp",
        "model": args.drafter_model,
        "num_speculative_tokens": args.num_speculative_tokens,
    }
    if args.draft_tensor_parallel_size > 1:
        speculative_config["draft_tensor_parallel_size"] = args.draft_tensor_parallel_size

    llm_kwargs: dict[str, Any] = {
        "model": args.target_model,
        "dtype": args.dtype,
        "tensor_parallel_size": args.tensor_parallel_size,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "speculative_config": speculative_config,
        "trust_remote_code": args.trust_remote_code,
        "seed": args.seed,
    }
    if args.max_model_len is not None:
        llm_kwargs["max_model_len"] = args.max_model_len
    return llm_kwargs


def main() -> None:
    from vllm import LLM, SamplingParams

    args = parse_args()
    prompt = render_prompt(args)
    llm_kwargs = build_llm_kwargs(args)

    if args.print_request:
        print("Rendered prompt:")
        print(prompt)
        print("\nspeculative_config:")
        print(json.dumps(llm_kwargs["speculative_config"], indent=2))
        print()

    llm = LLM(**llm_kwargs)
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_new_tokens,
    )
    outputs = llm.generate([prompt], sampling_params)
    completion = outputs[0].outputs[0].text
    print(completion)


if __name__ == "__main__":
    main()
