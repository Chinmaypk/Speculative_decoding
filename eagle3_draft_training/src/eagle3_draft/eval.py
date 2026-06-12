from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import Eagle3TrainingConfig
from .data import SupervisedDataCollator, SupervisedJsonlDataset
from .model import Eagle3DraftModel
from .train import forward_drafter


def load_draft_model(target_model: torch.nn.Module, cfg: Eagle3TrainingConfig, checkpoint_path: str) -> Eagle3DraftModel:
    draft_model = Eagle3DraftModel(
        target_config=target_model.config,
        target_hidden_layer_indices=cfg.target_hidden_layer_indices,
        draft_hidden_size=cfg.draft_hidden_size,
        draft_num_layers=cfg.draft_num_layers,
        draft_num_heads=cfg.draft_num_heads,
        draft_intermediate_size=cfg.draft_intermediate_size,
        dropout=cfg.dropout,
    )
    path = Path(checkpoint_path)
    if path.is_dir():
        path = path / "draft_model.pt"
    draft_model.load_state_dict(torch.load(path, map_location="cpu"), strict=True)
    return draft_model


@torch.no_grad()
def run_eval(cfg: Eagle3TrainingConfig, checkpoint_path: str, data_dir: str) -> dict[str, float]:
    accelerator = Accelerator(mixed_precision="bf16" if cfg.bf16 else "fp16" if cfg.fp16 else "no")
    target_model_path = cfg.resolved_target_model_path
    tokenizer = AutoTokenizer.from_pretrained(target_model_path, trust_remote_code=cfg.trust_remote_code, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if cfg.bf16 else torch.float16 if cfg.fp16 else torch.float32
    target_model = AutoModelForCausalLM.from_pretrained(
        target_model_path,
        torch_dtype=dtype,
        trust_remote_code=cfg.trust_remote_code,
        output_hidden_states=True,
    )
    target_model.eval()
    target_model.requires_grad_(False)

    draft_model = load_draft_model(target_model, cfg, checkpoint_path)
    draft_model.eval()

    dataset = SupervisedJsonlDataset(data_dir)
    collator = SupervisedDataCollator(tokenizer.pad_token_id, cfg.max_length)
    dataloader = DataLoader(dataset, batch_size=cfg.per_device_eval_batch_size, shuffle=False, collate_fn=collator)
    target_model, draft_model, dataloader = accelerator.prepare(target_model, draft_model, dataloader)

    losses: list[torch.Tensor] = []
    correct = torch.tensor(0.0, device=accelerator.device)
    total = torch.tensor(0.0, device=accelerator.device)

    for batch in tqdm(dataloader, desc="Evaluating", disable=not accelerator.is_main_process):
        logits = forward_drafter(
            target_model,
            draft_model,
            batch["input_ids"],
            batch["attention_mask"],
            cfg.target_hidden_layer_indices,
        )
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = batch["labels"][:, 1:].contiguous()
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
            reduction="mean",
        )
        losses.append(accelerator.gather_for_metrics(loss.detach()).mean())
        valid = shift_labels != -100
        preds = shift_logits.argmax(dim=-1)
        correct += accelerator.gather_for_metrics(((preds == shift_labels) & valid).sum()).sum()
        total += accelerator.gather_for_metrics(valid.sum()).sum()

    mean_loss = torch.stack(losses).mean().item() if losses else 0.0
    accuracy = (correct / total).item() if total.item() > 0 else 0.0
    return {
        "loss": mean_loss,
        "accuracy": accuracy,
        "perplexity": math.exp(min(mean_loss, 20.0)),
        "target_tokens": float(total.item()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint_path", required=True)
    parser.add_argument("--data_dir", default=None)
    args = parser.parse_args()

    cfg = Eagle3TrainingConfig.from_yaml(args.config)
    data_dir = args.data_dir or cfg.eval_data_dir or cfg.test_data_dir
    if not data_dir:
        raise ValueError("Pass --data_dir or set eval_data_dir/test_data_dir in the config.")
    metrics = run_eval(cfg, args.checkpoint_path, data_dir)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
