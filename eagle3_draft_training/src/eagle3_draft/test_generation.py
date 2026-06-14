from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import Eagle3TrainingConfig
from .model import Eagle3DraftModel
from .train import forward_drafter


DEFAULT_SYSTEM_PROMPT = "Convert the assistant response into the correct GenUI JSON. Return only valid JSON."


def build_prompt(response_text: str, system_prompt: str = DEFAULT_SYSTEM_PROMPT) -> str:
    return f"{system_prompt}\n\n### response_text\n{response_text.strip()}\n\n### genui_json\n"


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
    draft_model.eval()
    return draft_model


@torch.no_grad()
def greedy_generate(
    target_model: torch.nn.Module,
    draft_model: Eagle3DraftModel,
    tokenizer: AutoTokenizer,
    cfg: Eagle3TrainingConfig,
    response_text: str,
    max_new_tokens: int,
    system_prompt: str,
) -> str:
    device = next(draft_model.parameters()).device
    input_ids = tokenizer(build_prompt(response_text, system_prompt), return_tensors="pt", add_special_tokens=True)["input_ids"].to(device)

    for _ in range(max_new_tokens):
        attention_mask = torch.ones_like(input_ids)
        logits = forward_drafter(target_model, draft_model, input_ids, attention_mask, cfg.target_hidden_layer_indices)
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        input_ids = torch.cat([input_ids, next_token], dim=-1)
        if next_token.item() == tokenizer.eos_token_id:
            break

    decoded = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    marker = "### genui_json\n"
    return decoded.split(marker, 1)[-1].strip() if marker in decoded else decoded.strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint_path", required=True)
    parser.add_argument("--response_text", default=None)
    parser.add_argument("--input_jsonl", default=None)
    parser.add_argument("--output_jsonl", default=None)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--system_prompt", default=DEFAULT_SYSTEM_PROMPT)
    args = parser.parse_args()

    cfg = Eagle3TrainingConfig.from_yaml(args.config)
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    target_model.eval().to(device)
    target_model.requires_grad_(False)
    draft_model = load_draft_model(target_model, cfg, args.checkpoint_path)
    draft_dtype = dtype if device.type == "cuda" else torch.float32
    draft_model.to(device=device, dtype=draft_dtype)

    if args.response_text:
        print(greedy_generate(target_model, draft_model, tokenizer, cfg, args.response_text, args.max_new_tokens, args.system_prompt))
        return

    if not args.input_jsonl or not args.output_jsonl:
        raise ValueError("Pass either --response_text or both --input_jsonl and --output_jsonl.")

    with open(args.input_jsonl, "r", encoding="utf-8") as fin, open(args.output_jsonl, "w", encoding="utf-8") as fout:
        for line in fin:
            record = json.loads(line)
            response_text = str(record["response_text"])
            pred = greedy_generate(target_model, draft_model, tokenizer, cfg, response_text, args.max_new_tokens, args.system_prompt)
            record["predicted_genui_json"] = pred
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
