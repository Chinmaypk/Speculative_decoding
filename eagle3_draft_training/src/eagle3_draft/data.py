from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import Dataset


class TokenizedSequenceDataset(Dataset):
    """Loads tokenized examples produced by scripts/prepare_jsonl.py."""

    def __init__(self, data_dir: str | Path):
        data_path = Path(data_dir) / "data.pt"
        if not data_path.exists():
            raise FileNotFoundError(f"Could not find tokenized data at {data_path}")
        self.examples: list[torch.Tensor] = torch.load(data_path, map_location="cpu")
        if not self.examples:
            raise ValueError(f"No training examples found in {data_path}")

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.examples[idx]


class CausalLMCollator:
    """Pads variable-length token sequences and builds next-token labels."""

    def __init__(self, pad_token_id: int, max_length: int):
        self.pad_token_id = pad_token_id
        self.max_length = max_length

    def __call__(self, batch: list[torch.Tensor]) -> dict[str, torch.Tensor]:
        trimmed = [x[: self.max_length] for x in batch]
        max_len = max(x.numel() for x in trimmed)
        input_ids = torch.full(
            (len(trimmed), max_len),
            fill_value=self.pad_token_id,
            dtype=torch.long,
        )
        attention_mask = torch.zeros((len(trimmed), max_len), dtype=torch.long)

        for i, item in enumerate(trimmed):
            input_ids[i, : item.numel()] = item
            attention_mask[i, : item.numel()] = 1

        # The trainer uses labels[:, 1:] against logits[:, :-1].
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
