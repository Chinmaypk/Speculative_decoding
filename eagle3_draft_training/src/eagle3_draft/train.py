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
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

from .config import Eagle3TrainingConfig
from .data import CausalLMCollator, TokenizedSequenceDataset
from .model import Eagle3DraftModel


def get_input_embeddings(model: torch.nn.Module, input_ids: torch.Tensor) -> torch.Tensor:
    return model.get_input_embeddings()(input_ids)


def gather_selected_hidden_states(
    all_hidden_states: tuple[torch.Tensor, ...],
    layer_indices: list[int],
) -> list[torch.Tensor]:
    selected: list[torch.Tensor] = []
    for idx in layer_indices:
        try:
            selected.append(all_hidden_states[idx])
        except IndexError as exc:
            raise IndexError(
                f"Hidden-state layer index {idx} is invalid. "
                f"Model returned {len(all_hidden_states)} hidden states."
            ) from exc
    return selected


def maybe_apply_scheduled_sampling(
    draft_logits: torch.Tensor,
    input_ids: torch.Tensor,
    probability: float,
) -> torch.Tensor:
    """Replace some previous tokens with drafter predictions.

    This is a lightweight approximation of training-time-test behavior. Keep it at
    0.0 initially, then increase slowly only after stable baseline training.
    """

    if probability <= 0.0 or not torch.is_grad_enabled():
        return input_ids

    with torch.no_grad():
        predicted = draft_logits.argmax(dim=-1)
        sampled = input_ids.clone()
        mask = torch.rand_like(sampled.float()) < probability
        # Keep first token and padding/ignored positions unchanged.
        mask[:, 0] = False
        sampled[mask] = predicted[mask]
        return sampled


def save_checkpoint(
    accelerator: Accelerator,
    draft_model: Eagle3DraftModel,
    output_dir: str | Path,
    step: int,
    cfg: Eagle3TrainingConfig,
) -> None:
    checkpoint_dir = Path(output_dir) / f"checkpoint-{step}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    unwrapped = accelerator.unwrap_model(draft_model)
    accelerator.save(unwrapped.state_dict(), checkpoint_dir / "draft_model.pt")
    unwrapped.save_config(str(checkpoint_dir))
    if accelerator.is_main_process:
        with open(checkpoint_dir / "training_config.json", "w", encoding="utf-8") as f:
            json.dump(cfg.to_dict(), f, indent=2)


def train(cfg: Eagle3TrainingConfig) -> None:
    set_seed(cfg.seed)
    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        mixed_precision="bf16" if cfg.bf16 else "fp16" if cfg.fp16 else "no",
    )

    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model_name_or_path,
        trust_remote_code=cfg.trust_remote_code,
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    target_model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name_or_path,
        torch_dtype=torch.bfloat16 if cfg.bf16 else torch.float16 if cfg.fp16 else torch.float32,
        trust_remote_code=cfg.trust_remote_code,
        output_hidden_states=True,
    )
    target_model.eval()
    target_model.requires_grad_(False)
    if cfg.gradient_checkpointing and hasattr(target_model, "gradient_checkpointing_enable"):
        # Target is frozen, so this mainly helps memory only if the target implementation supports it.
        target_model.gradient_checkpointing_enable()

    draft_model = Eagle3DraftModel(
        target_config=target_model.config,
        target_hidden_layer_indices=cfg.target_hidden_layer_indices,
        draft_hidden_size=cfg.draft_hidden_size,
        draft_num_layers=cfg.draft_num_layers,
        draft_num_heads=cfg.draft_num_heads,
        draft_intermediate_size=cfg.draft_intermediate_size,
        dropout=cfg.dropout,
    )

    dataset = TokenizedSequenceDataset(cfg.train_data_dir)
    collator = CausalLMCollator(tokenizer.pad_token_id, cfg.max_length)
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.per_device_train_batch_size,
        shuffle=True,
        collate_fn=collator,
    )

    optimizer = AdamW(draft_model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)

    steps_per_epoch = math.ceil(len(dataloader) / cfg.gradient_accumulation_steps)
    total_steps = cfg.max_steps if cfg.max_steps > 0 else steps_per_epoch * cfg.num_train_epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=cfg.warmup_steps,
        num_training_steps=total_steps,
    )

    target_model, draft_model, optimizer, dataloader, scheduler = accelerator.prepare(
        target_model,
        draft_model,
        optimizer,
        dataloader,
        scheduler,
    )

    global_step = 0
    completed_steps = 0
    progress = tqdm(total=total_steps, disable=not accelerator.is_main_process, desc="Training drafter")

    for epoch in range(cfg.num_train_epochs):
        for batch in dataloader:
            if cfg.max_steps > 0 and completed_steps >= cfg.max_steps:
                break

            with accelerator.accumulate(draft_model):
                input_ids = batch["input_ids"]
                attention_mask = batch["attention_mask"]
                labels = batch["labels"]

                with torch.no_grad():
                    target_outputs = target_model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        output_hidden_states=True,
                        use_cache=False,
                    )
                    selected_hidden_states = gather_selected_hidden_states(
                        target_outputs.hidden_states,
                        cfg.target_hidden_layer_indices,
                    )
                    previous_token_embeddings = get_input_embeddings(target_model, input_ids)

                logits = draft_model(
                    selected_hidden_states=selected_hidden_states,
                    previous_token_embeddings=previous_token_embeddings,
                    attention_mask=attention_mask,
                )

                if cfg.scheduled_sampling_prob > 0.0:
                    sampled_input_ids = maybe_apply_scheduled_sampling(
                        logits.detach(),
                        input_ids,
                        cfg.scheduled_sampling_prob,
                    )
                    if not torch.equal(sampled_input_ids, input_ids):
                        with torch.no_grad():
                            sampled_outputs = target_model(
                                input_ids=sampled_input_ids,
                                attention_mask=attention_mask,
                                output_hidden_states=True,
                                use_cache=False,
                            )
                            selected_hidden_states = gather_selected_hidden_states(
                                sampled_outputs.hidden_states,
                                cfg.target_hidden_layer_indices,
                            )
                            previous_token_embeddings = get_input_embeddings(target_model, sampled_input_ids)
                        logits = draft_model(
                            selected_hidden_states=selected_hidden_states,
                            previous_token_embeddings=previous_token_embeddings,
                            attention_mask=attention_mask,
                        )

                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = labels[:, 1:].contiguous()
                loss = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    ignore_index=-100,
                )

                accelerator.backward(loss)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            if accelerator.sync_gradients:
                completed_steps += 1
                global_step += 1
                progress.update(1)

                if accelerator.is_main_process and global_step % cfg.logging_steps == 0:
                    progress.set_postfix({"loss": f"{loss.item():.4f}", "lr": scheduler.get_last_lr()[0]})

                if cfg.save_steps > 0 and global_step % cfg.save_steps == 0:
                    save_checkpoint(accelerator, draft_model, cfg.output_dir, global_step, cfg)

        if cfg.max_steps > 0 and completed_steps >= cfg.max_steps:
            break

    progress.close()
    save_checkpoint(accelerator, draft_model, cfg.output_dir, global_step, cfg)
    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        print(f"Training complete. Final checkpoint saved under {cfg.output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = Eagle3TrainingConfig.from_yaml(args.config)
    os.makedirs(cfg.output_dir, exist_ok=True)
    train(cfg)


if __name__ == "__main__":
    main()
