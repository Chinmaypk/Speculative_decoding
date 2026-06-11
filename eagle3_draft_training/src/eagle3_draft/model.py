from __future__ import annotations

import torch
import torch.nn as nn
from transformers import PretrainedConfig


class Eagle3DraftModel(nn.Module):
    """Lightweight EAGLE-3-style drafter.

    Inputs:
      - selected hidden states from a frozen target model;
      - previous token embeddings from the target model embedding table.

    Output:
      - next-token logits over the target vocabulary.
    """

    def __init__(
        self,
        target_config: PretrainedConfig,
        target_hidden_layer_indices: list[int],
        draft_hidden_size: int,
        draft_num_layers: int,
        draft_num_heads: int,
        draft_intermediate_size: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.target_hidden_layer_indices = target_hidden_layer_indices
        self.target_hidden_size = int(getattr(target_config, "hidden_size"))
        self.vocab_size = int(getattr(target_config, "vocab_size"))
        self.draft_hidden_size = draft_hidden_size

        self.hidden_projections = nn.ModuleList(
            [nn.Linear(self.target_hidden_size, draft_hidden_size, bias=False) for _ in target_hidden_layer_indices]
        )
        self.token_projection = nn.Linear(self.target_hidden_size, draft_hidden_size, bias=False)
        self.fusion_norm = nn.LayerNorm(draft_hidden_size)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=draft_hidden_size,
            nhead=draft_num_heads,
            dim_feedforward=draft_intermediate_size,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(encoder_layer, num_layers=draft_num_layers)
        self.final_norm = nn.LayerNorm(draft_hidden_size)
        self.lm_head = nn.Linear(draft_hidden_size, self.vocab_size, bias=False)

    def forward(
        self,
        selected_hidden_states: list[torch.Tensor],
        previous_token_embeddings: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if len(selected_hidden_states) != len(self.hidden_projections):
            raise ValueError(
                f"Expected {len(self.hidden_projections)} hidden states, got {len(selected_hidden_states)}"
            )

        fused = self.token_projection(previous_token_embeddings)
        for hidden, projection in zip(selected_hidden_states, self.hidden_projections):
            fused = fused + projection(hidden)
        fused = fused / (len(selected_hidden_states) + 1)
        fused = self.fusion_norm(fused)

        seq_len = fused.size(1)
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=fused.device, dtype=torch.bool),
            diagonal=1,
        )
        key_padding_mask = None
        if attention_mask is not None:
            key_padding_mask = attention_mask == 0

        hidden = self.blocks(
            fused,
            mask=causal_mask,
            src_key_padding_mask=key_padding_mask,
            is_causal=True,
        )
        hidden = self.final_norm(hidden)
        return self.lm_head(hidden)

    def save_config(self, output_dir: str) -> None:
        import json
        from pathlib import Path

        Path(output_dir).mkdir(parents=True, exist_ok=True)
        with open(Path(output_dir) / "draft_config.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "target_hidden_layer_indices": self.target_hidden_layer_indices,
                    "target_hidden_size": self.target_hidden_size,
                    "draft_hidden_size": self.draft_hidden_size,
                    "vocab_size": self.vocab_size,
                },
                f,
                indent=2,
            )
