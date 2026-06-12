from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import yaml


@dataclass
class Eagle3TrainingConfig:
    train_data_dir: str
    output_dir: str
    target_hidden_layer_indices: list[int]

    # Preferred name: this is the frozen assistant/target model path used for
    # tokenization, hidden states, and embeddings.
    assistant_model_path: str | None = None

    # Deprecated fallback kept only for older config files.
    model_name_or_path: str | None = None

    eval_data_dir: str | None = None
    test_data_dir: str | None = None
    resume_draft_checkpoint_path: str | None = None
    tensorboard_log_dir: str | None = None

    max_length: int = 2048
    draft_hidden_size: int = 1024
    draft_num_layers: int = 4
    draft_num_heads: int = 8
    draft_intermediate_size: int = 4096
    dropout: float = 0.0

    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_steps: int = 100
    gradient_accumulation_steps: int = 4
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    num_train_epochs: int = 1
    max_steps: int = -1
    logging_steps: int = 10
    eval_steps: int = 200
    save_steps: int = 500
    seed: int = 42

    bf16: bool = True
    fp16: bool = False
    scheduled_sampling_prob: float = 0.0
    gradient_checkpointing: bool = True
    trust_remote_code: bool = True

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Eagle3TrainingConfig":
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        if raw is None:
            raw = {}
        allowed = {field.name for field in fields(cls)}
        unknown = set(raw) - allowed
        if unknown:
            raise ValueError(f"Unknown config keys: {sorted(unknown)}")
        cfg = cls(**raw)
        if cfg.assistant_model_path is None and cfg.model_name_or_path is not None:
            cfg.assistant_model_path = cfg.model_name_or_path
        if cfg.assistant_model_path is None:
            raise ValueError("Set 'assistant_model_path' in the YAML config.")
        return cfg

    @property
    def resolved_assistant_model_path(self) -> str:
        if self.assistant_model_path is None:
            raise ValueError("assistant_model_path is not configured.")
        return self.assistant_model_path

    def to_dict(self) -> dict[str, Any]:
        data = {field.name: getattr(self, field.name) for field in fields(self)}
        data["resolved_assistant_model_path"] = self.resolved_assistant_model_path
        return data
