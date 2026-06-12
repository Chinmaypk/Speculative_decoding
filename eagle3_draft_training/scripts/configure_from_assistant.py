#!/usr/bin/env python
"""Inspect an assistant model and overwrite drafter config safely.

Example:
  python scripts/configure_from_assistant.py \
    --assistant_model_path /path/to/assistant_model \
    --config configs/gemma4_example.yaml \
    --overwrite
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import yaml
from transformers import AutoConfig


def pick_layer_indices(num_layers: int) -> list[int]:
    """Pick low/mid/high hidden-state indices for EAGLE-3-style fusion.

    Hugging Face hidden_states usually includes embeddings at index 0 and layer
    outputs from 1..num_hidden_layers. We avoid 0 and choose three internal
    semantic levels.
    """
    if num_layers < 3:
        raise ValueError(f"Need at least 3 hidden layers, got {num_layers}")
    candidates = [
        max(1, round(num_layers * 0.25)),
        max(1, round(num_layers * 0.50)),
        max(1, round(num_layers * 0.85)),
    ]
    indices: list[int] = []
    for idx in candidates:
        idx = min(max(1, idx), num_layers)
        if idx not in indices:
            indices.append(idx)
    while len(indices) < 3:
        candidate = min(num_layers, indices[-1] + 1)
        if candidate not in indices:
            indices.append(candidate)
        else:
            break
    return indices[:3]


def choose_draft_width(hidden_size: int) -> int:
    """Choose a compact drafter width from assistant-model hidden size."""
    if hidden_size <= 1024:
        return hidden_size
    raw = min(2048, max(1024, hidden_size // 2))
    return int(math.ceil(raw / 128) * 128)


def choose_num_heads(draft_hidden_size: int) -> int:
    for heads in [16, 12, 8, 4, 2, 1]:
        if draft_hidden_size % heads == 0:
            return heads
    return 1


def inspect_model(assistant_model_path: str, trust_remote_code: bool) -> dict[str, Any]:
    cfg = AutoConfig.from_pretrained(assistant_model_path, trust_remote_code=trust_remote_code)
    hidden_size = int(getattr(cfg, "hidden_size"))
    vocab_size = int(getattr(cfg, "vocab_size"))
    num_layers = int(getattr(cfg, "num_hidden_layers"))
    draft_hidden_size = choose_draft_width(hidden_size)
    draft_num_heads = choose_num_heads(draft_hidden_size)
    return {
        "model_type": getattr(cfg, "model_type", None),
        "architectures": getattr(cfg, "architectures", None),
        "assistant_hidden_size": hidden_size,
        "assistant_vocab_size": vocab_size,
        "assistant_num_hidden_layers": num_layers,
        "assistant_num_attention_heads": getattr(cfg, "num_attention_heads", None),
        "assistant_num_key_value_heads": getattr(cfg, "num_key_value_heads", None),
        "assistant_intermediate_size": getattr(cfg, "intermediate_size", None),
        "recommended": {
            "assistant_model_path": assistant_model_path,
            "target_hidden_layer_indices": pick_layer_indices(num_layers),
            "draft_hidden_size": draft_hidden_size,
            "draft_num_heads": draft_num_heads,
            "draft_num_layers": 4,
            "draft_intermediate_size": draft_hidden_size * 4,
        },
    }


def update_yaml_config(config_path: Path, model_info: dict[str, Any]) -> None:
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    config.pop("model_name_or_path", None)
    config.update(model_info["recommended"])
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--assistant_model_path", required=True)
    parser.add_argument("--config", default="configs/gemma4_example.yaml")
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite the YAML config with recommended values.")
    args = parser.parse_args()

    info = inspect_model(args.assistant_model_path, args.trust_remote_code)
    print(yaml.safe_dump(info, sort_keys=False))

    if args.overwrite:
        update_yaml_config(Path(args.config), info)
        print(f"Updated {args.config} with recommended drafter settings.")


if __name__ == "__main__":
    main()
