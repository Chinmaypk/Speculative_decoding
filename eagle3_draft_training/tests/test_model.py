import unittest

import torch
from transformers import PretrainedConfig

from eagle3_draft.model import Eagle3DraftModel


class Eagle3DraftModelTest(unittest.TestCase):
    def test_forward_accepts_bfloat16_target_features_with_float32_weights(self) -> None:
        target_config = PretrainedConfig()
        target_config.hidden_size = 8
        target_config.vocab_size = 16
        model = Eagle3DraftModel(
            target_config=target_config,
            target_hidden_layer_indices=[1, 2],
            draft_hidden_size=8,
            draft_num_layers=1,
            draft_num_heads=2,
            draft_intermediate_size=16,
        )
        selected_hidden_states = [
            torch.randn(1, 4, 8, dtype=torch.bfloat16),
            torch.randn(1, 4, 8, dtype=torch.bfloat16),
        ]
        previous_token_embeddings = torch.randn(1, 4, 8, dtype=torch.bfloat16)
        attention_mask = torch.ones(1, 4, dtype=torch.long)

        logits = model(selected_hidden_states, previous_token_embeddings, attention_mask)

        self.assertEqual(logits.dtype, torch.float32)
        self.assertEqual(logits.shape, (1, 4, 16))


if __name__ == "__main__":
    unittest.main()
