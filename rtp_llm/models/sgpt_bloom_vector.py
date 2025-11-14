from typing import Any, Dict, List

import numpy as np
import torch

from rtp_llm.model_factory_register import register_model
from rtp_llm.utils.base_model_datatypes import GenerateOutput
from rtp_llm.models.sgpt_bloom import SGPTBloom


class SGPTBloomVector(SGPTBloom):
    @torch.no_grad()
    def generate_weighted_hidden_states_stream(self, input_token_ids: torch.IntTensor):
        eos_token_id = self.config.special_tokens.eos_token_id
        batch_size = input_token_ids.shape[0]
        input_mask = torch.where(input_token_ids != eos_token_id, 1, 0)

        gen_output = list(self.generate_hidden_states_stream(input_token_ids))[0]
        hidden_states = gen_output.hidden_states

        weights = (
            torch.arange(start=1, end=hidden_states.shape[1] + 1)
            .unsqueeze(0)
            .unsqueeze(-1)
            .expand(hidden_states.size())
            .float()
            .to(hidden_states.device)
        )

        # input_mask_expanded.shape = [bs, seq_len, hid_dim]
        # input_mask.shape = [batch, len]
        # input_mask_expanded.shape = [batch, len, feat]
        input_mask_expanded = (
            input_mask.unsqueeze(-1).expand(hidden_states.size()).float()
        ).to(hidden_states.device)

        # Perform weighted mean pooling across seq_len: bs, seq_len, hidden_dim -> bs, hidden_dim
        sum_embeddings = torch.sum(hidden_states * input_mask_expanded * weights, dim=1)
        sum_mask = torch.sum(input_mask_expanded * weights, dim=1)

        embeddings = sum_embeddings / sum_mask

        embeddings = embeddings.cpu()

        norm = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

        yield GenerateOutput(
            norm,
            input_token_ids.unsqueeze(1),  # add beam dim
            torch.ones(batch_size),
            [{"decimals": 6}] * batch_size,
        )

    @torch.no_grad()
    def generate_stream(self, input_token_ids, input_lengths, generate_config):
        return self.generate_weighted_hidden_states_stream(
            input_token_ids=input_token_ids
        )


register_model("sgpt_bloom_vector", SGPTBloomVector)
