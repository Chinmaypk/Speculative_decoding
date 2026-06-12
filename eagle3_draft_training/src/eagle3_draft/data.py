from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset


class SupervisedJsonlDataset(Dataset):
    """Loads tokenized response_text -> genui_json examples.

    Each item is a dict with:
      input_ids: prompt + target tokens
      labels: -100 for prompt tokens and target token ids for output tokens
      prompt_length: number of prompt tokens
    """

    def __init__(self, data_dir: str | Path):
        data_path = Path(data_dir) / "data.pt"
        if not data_path.exists():
            raise FileNotFoundError(f"Could not find tokenized data at {data_path}")
        self.examples: list[dict[str, Any]] = torch.load(data_path, map_location="cpu")
        if not self.examples:
            raise ValueError(f"No training examples found in {data_path}")

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.examples[idx]


# Backward-compatible alias for earlier code/users.
TokenizedSequenceDataset = SupervisedJsonlDataset


class SupervisedDataCollator:
    """Pads supervised examples and preserves target-only labels."""

    def __init__(self, pad_token_id: int, max_length: int):
        self.pad_token_id = pad_token_id
        self.max_length = max_length

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        trimmed: list[dict[str, torch.Tensor | int]] = []
        for item in batch:
            input_ids = item["input_ids"][: self.max_length]
            labels = item["labels"][: self.max_length]
            prompt_length = min(int(item.get("prompt_length", 0)), input_ids.numel())
            trimmed.append({"input_ids": input_ids, "labels": labels, "prompt_length": prompt_length})

        max_len = max(item["input_ids"].numel() for item in trimmed)  # type: ignore[union-attr]
        input_ids = torch.full((len(trimmed), max_len), self.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((len(trimmed), max_len), dtype=torch.long)
        labels = torch.full((len(trimmed), max_len), -100, dtype=torch.long)
        prompt_lengths = torch.zeros((len(trimmed),), dtype=torch.long)

        for i, item in enumerate(trimmed):
            ids = item["input_ids"]
            labs = item["labels"]
            assert isinstance(ids, torch.Tensor)
            assert isinstance(labs, torch.Tensor)
            input_ids[i, : ids.numel()] = ids
            attention_mask[i, : ids.numel()] = 1
            labels[i, : labs.numel()] = labs
            prompt_lengths[i] = int(item["prompt_length"])

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "prompt_lengths": prompt_lengths,
        }


# Backward-compatible alias for earlier code/users.
CausalLMCollator = SupervisedDataCollator
