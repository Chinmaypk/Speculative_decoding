from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import set_seed
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

from .config import Eagle3TrainingConfig
from .data import SupervisedDataCollator, SupervisedJsonlDataset
from .model import Eagle3DraftModel


def get_input_embeddings(model: torch.nn.Module, input_ids: torch.Tensor) -> torch.Tensor:
    return model.get_input_embeddings()(input_ids)


def gather_selected_hidden_states(all_hidden_states: tuple[torch.Tensor, ...], layer_indices: list[int]) -> list[torch.Tensor]:
    selected: list[torch.Tensor] = []
    for idx in layer_indices:
        try:
            selected.append(all_hidden_states[idx])
        except IndexError as exc:
            raise IndexError(
                f"Hidden-state layer index {idx} is invalid. Model returned {len(all_hidden_states)} hidden states."
            ) from exc
    return selected


def build_draft_model(target_model: torch.nn.Module, cfg: Eagle3TrainingConfig) -> Eagle3DraftModel:
    draft_model = Eagle3DraftModel(
        target_config=target_model.config,
        target_hidden_layer_indices=cfg.target_hidden_layer_indices,
        draft_hidden_size=cfg.draft_hidden_size,
        draft_num_layers=cfg.draft_num_layers,
        draft_num_heads=cfg.draft_num_heads,
        draft_intermediate_size=cfg.draft_intermediate_size,
        dropout=cfg.dropout,
    )
    if cfg.resume_draft_checkpoint_path:
        checkpoint_path = Path(cfg.resume_draft_checkpoint_path)
        if checkpoint_path.is_dir():
            checkpoint_path = checkpoint_path / "draft_model.pt"
        state = torch.load(checkpoint_path, map_location="cpu")
        draft_model.load_state_dict(state, strict=True)
        print(f"Loaded existing drafter checkpoint from {checkpoint_path}")
    return draft_model


def compute_loss_and_metrics(logits: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
    )
    with torch.no_grad():
        valid = shift_labels != -100
        total = valid.sum().item()
        if total == 0:
            accuracy = 0.0
        else:
            preds = shift_logits.argmax(dim=-1)
            accuracy = ((preds == shift_labels) & valid).sum().item() / total
        ppl = math.exp(min(float(loss.item()), 20.0))
    return loss, {"loss": float(loss.item()), "accuracy": accuracy, "perplexity": ppl}


def forward_drafter(
    target_model: torch.nn.Module,
    draft_model: Eagle3DraftModel,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    layer_indices: list[int],
) -> torch.Tensor:
    with torch.no_grad():
        target_outputs = target_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )
        selected_hidden_states = gather_selected_hidden_states(target_outputs.hidden_states, layer_indices)
        previous_token_embeddings = get_input_embeddings(target_model, input_ids)
    return draft_model(
        selected_hidden_states=selected_hidden_states,
        previous_token_embeddings=previous_token_embeddings,
        attention_mask=attention_mask,
    )


@torch.no_grad()
def evaluate(
    accelerator: Accelerator,
    target_model: torch.nn.Module,
    draft_model: Eagle3DraftModel,
    dataloader: DataLoader | None,
    cfg: Eagle3TrainingConfig,
) -> dict[str, float]:
    if dataloader is None:
        return {}
    draft_model.eval()
    losses: list[torch.Tensor] = []
    correct = torch.tensor(0.0, device=accelerator.device)
    total = torch.tensor(0.0, device=accelerator.device)

    for batch in dataloader:
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
    draft_model.train()
    return {
        "eval_loss": mean_loss,
        "eval_accuracy": accuracy,
        "eval_perplexity": math.exp(min(mean_loss, 20.0)),
    }


def maybe_apply_scheduled_sampling(draft_logits: torch.Tensor, input_ids: torch.Tensor, probability: float) -> torch.Tensor:
    if probability <= 0.0:
        return input_ids
    with torch.no_grad():
        predicted = draft_logits.argmax(dim=-1)
        sampled = input_ids.clone()
        mask = torch.rand_like(sampled.float()) < probability
        mask[:, 0] = False
        sampled[mask] = predicted[mask]
        return sampled


def save_checkpoint(accelerator: Accelerator, draft_model: Eagle3DraftModel, output_dir: str | Path, step: int, cfg: Eagle3TrainingConfig) -> None:
    checkpoint_dir = Path(output_dir) / f"checkpoint-{step}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    unwrapped = accelerator.unwrap_model(draft_model)
    accelerator.save(unwrapped.state_dict(), checkpoint_dir / "draft_model.pt")
    unwrapped.save_config(str(checkpoint_dir))
    if accelerator.is_main_process:
        with open(checkpoint_dir / "training_config.json", "w", encoding="utf-8") as f:
            json.dump(cfg.to_dict(), f, indent=2)


def make_loader(data_dir: str | None, tokenizer: AutoTokenizer, cfg: Eagle3TrainingConfig, train: bool) -> DataLoader | None:
    if not data_dir:
        return None
    data_path = Path(data_dir) / "data.pt"
    if not data_path.exists():
        return None
    dataset = SupervisedJsonlDataset(data_dir)
    collator = SupervisedDataCollator(tokenizer.pad_token_id, cfg.max_length)
    return DataLoader(
        dataset,
        batch_size=cfg.per_device_train_batch_size if train else cfg.per_device_eval_batch_size,
        shuffle=train,
        collate_fn=collator,
    )


def train(cfg: Eagle3TrainingConfig) -> None:
    set_seed(cfg.seed)
    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        mixed_precision="bf16" if cfg.bf16 else "fp16" if cfg.fp16 else "no",
    )
    writer = None
    if accelerator.is_main_process:
        log_dir = cfg.tensorboard_log_dir or str(Path(cfg.output_dir) / "tensorboard")
        writer = SummaryWriter(log_dir=log_dir)

    assistant_model_path = cfg.resolved_assistant_model_path
    tokenizer = AutoTokenizer.from_pretrained(assistant_model_path, trust_remote_code=cfg.trust_remote_code, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if cfg.bf16 else torch.float16 if cfg.fp16 else torch.float32
    target_model = AutoModelForCausalLM.from_pretrained(
        assistant_model_path,
        torch_dtype=dtype,
        trust_remote_code=cfg.trust_remote_code,
        output_hidden_states=True,
    )
    target_model.eval()
    target_model.requires_grad_(False)
    if cfg.gradient_checkpointing and hasattr(target_model, "gradient_checkpointing_enable"):
        target_model.gradient_checkpointing_enable()

    draft_model = build_draft_model(target_model, cfg)

    train_loader = make_loader(cfg.train_data_dir, tokenizer, cfg, train=True)
    eval_loader = make_loader(cfg.eval_data_dir, tokenizer, cfg, train=False)
    if train_loader is None:
        raise FileNotFoundError(f"No train data found in {cfg.train_data_dir}")

    optimizer = AdamW(draft_model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    steps_per_epoch = math.ceil(len(train_loader) / cfg.gradient_accumulation_steps)
    total_steps = cfg.max_steps if cfg.max_steps > 0 else steps_per_epoch * cfg.num_train_epochs
    scheduler = get_cosine_schedule_with_warmup(optimizer, cfg.warmup_steps, total_steps)

    target_model, draft_model, optimizer, train_loader, scheduler = accelerator.prepare(
        target_model, draft_model, optimizer, train_loader, scheduler
    )
    if eval_loader is not None:
        eval_loader = accelerator.prepare(eval_loader)

    global_step = 0
    completed_steps = 0
    progress = tqdm(total=total_steps, disable=not accelerator.is_main_process, desc="Training drafter")

    for _epoch in range(cfg.num_train_epochs):
        draft_model.train()
        for batch in train_loader:
            if cfg.max_steps > 0 and completed_steps >= cfg.max_steps:
                break

            with accelerator.accumulate(draft_model):
                input_ids = batch["input_ids"]
                attention_mask = batch["attention_mask"]
                labels = batch["labels"]

                logits = forward_drafter(target_model, draft_model, input_ids, attention_mask, cfg.target_hidden_layer_indices)

                if cfg.scheduled_sampling_prob > 0.0:
                    sampled_input_ids = maybe_apply_scheduled_sampling(logits.detach(), input_ids, cfg.scheduled_sampling_prob)
                    if not torch.equal(sampled_input_ids, input_ids):
                        logits = forward_drafter(
                            target_model, draft_model, sampled_input_ids, attention_mask, cfg.target_hidden_layer_indices
                        )

                loss, train_metrics = compute_loss_and_metrics(logits, labels)
                accelerator.backward(loss)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            if accelerator.sync_gradients:
                completed_steps += 1
                global_step += 1
                progress.update(1)

                if accelerator.is_main_process and global_step % cfg.logging_steps == 0:
                    lr = scheduler.get_last_lr()[0]
                    progress.set_postfix({"loss": f"{train_metrics['loss']:.4f}", "acc": f"{train_metrics['accuracy']:.4f}", "lr": lr})
                    if writer is not None:
                        writer.add_scalar("train/loss", train_metrics["loss"], global_step)
                        writer.add_scalar("train/accuracy", train_metrics["accuracy"], global_step)
                        writer.add_scalar("train/perplexity", train_metrics["perplexity"], global_step)
                        writer.add_scalar("train/lr", lr, global_step)

                if eval_loader is not None and cfg.eval_steps > 0 and global_step % cfg.eval_steps == 0:
                    metrics = evaluate(accelerator, target_model, draft_model, eval_loader, cfg)
                    if accelerator.is_main_process and writer is not None:
                        for key, value in metrics.items():
                            writer.add_scalar(key.replace("eval_", "eval/"), value, global_step)

                if cfg.save_steps > 0 and global_step % cfg.save_steps == 0:
                    save_checkpoint(accelerator, draft_model, cfg.output_dir, global_step, cfg)

        if cfg.max_steps > 0 and completed_steps >= cfg.max_steps:
            break

    progress.close()
    final_metrics = evaluate(accelerator, target_model, draft_model, eval_loader, cfg)
    save_checkpoint(accelerator, draft_model, cfg.output_dir, global_step, cfg)
    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        if writer is not None:
            for key, value in final_metrics.items():
                writer.add_scalar(key.replace("eval_", "eval/"), value, global_step)
            writer.close()
        print(f"Training complete. Final checkpoint saved under {cfg.output_dir}")
        if final_metrics:
            print(json.dumps(final_metrics, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = Eagle3TrainingConfig.from_yaml(args.config)
    os.makedirs(cfg.output_dir, exist_ok=True)
    train(cfg)


if __name__ == "__main__":
    main()
