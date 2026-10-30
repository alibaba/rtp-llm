"""Slow QSA selection used only as a parity reference."""

import math
from typing import Optional

import torch

from rtp_llm.models_py.modules.qwen4_exp.qsa.indexer import (
    Qwen4ExpQSAIndexer,
    apply_partial_rope,
    exact_head_rms_norm,
)


class ReferenceQSAIndexer(Qwen4ExpQSAIndexer):
    def pooled_block_keys(
        self,
        raw_keys: torch.Tensor,
        block_token_indices: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        """``[num_blocks, ratio] token ids -> [num_blocks, D]`` block keys.

        ``raw_keys`` is one sequence's ``[T, D]``. Mean in fp32, cast back, then
        norm, then rotate at each block's first token position.
        """
        groups = raw_keys.index_select(0, block_token_indices.flatten())
        groups = groups.view(*block_token_indices.shape, self.head_dim)
        pooled = groups.float().mean(dim=1).to(raw_keys.dtype)
        pooled = exact_head_rms_norm(pooled, self.k_norm_gamma, self.norm_eps)
        starts = block_token_indices[:, 0]
        return apply_partial_rope(
            pooled, cos.index_select(0, starts), sin.index_select(0, starts)
        )

    def block_scores(
        self, q_row: torch.Tensor, block_keys: torch.Tensor
    ) -> torch.Tensor:
        """``q [H,D]`` against ``[num_blocks,D]`` -> ``[num_blocks]`` fp32.

        ReLU per head *then* an unweighted sum across heads. The DSv4 scoring
        kernel computes the same thing with a per-head weight vector, so this is
        its equal-weight special case with the weight folded into the scale.
        """
        scores = torch.matmul(q_row.float(), block_keys.float().transpose(-1, -2))
        return torch.relu(scores.transpose(-1, -2)).sum(dim=-1) / math.sqrt(
            self.head_dim
        )

    def select_for_row(
        self,
        q_row: torch.Tensor,
        raw_keys: torch.Tensor,
        visible_indices: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        """Token ids this query may attend to: scored blocks, then the tail."""
        num_blocks = visible_indices.shape[-1] // self.compress_ratio
        if num_blocks > 0:
            block_token_indices = visible_indices[
                : num_blocks * self.compress_ratio
            ].view(num_blocks, self.compress_ratio)
            block_keys = self.pooled_block_keys(raw_keys, block_token_indices, cos, sin)
            scores = self.block_scores(q_row, block_keys)
            chosen = scores.topk(min(self.block_topk, num_blocks), dim=0).indices
            # Kept in score order, matching upstream -- not sorted by position.
            selected = block_token_indices.index_select(0, chosen).flatten()
        else:
            selected = visible_indices.new_empty((0,))
        tail = visible_indices[num_blocks * self.compress_ratio :]
        return torch.cat([selected, tail]).to(torch.int32)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        visible: torch.Tensor,
        past_raw_keys: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """``[B,S,hidden] -> [B,S,max_selected] int32``, ``-1`` padded.

        ``cos``/``sin`` are ``[B, T, rotary_dim]`` over the full key range;
        ``visible`` is ``[B, S, T]`` bool. ``past_raw_keys`` ``[B, T-S, D]``
        prepends cached keys, so ``T`` is the full context length.
        """
        batch, seq_len, _ = hidden_states.shape
        q, raw_keys = self.project(hidden_states)
        if past_raw_keys is not None:
            raw_keys = torch.cat([past_raw_keys, raw_keys], dim=1)
        q = apply_partial_rope(
            q, cos[:, -seq_len:].unsqueeze(2), sin[:, -seq_len:].unsqueeze(2)
        )

        out = torch.full(
            (batch, seq_len, self.max_selected),
            -1,
            dtype=torch.int32,
            device=hidden_states.device,
        )
        for b in range(batch):
            for i in range(seq_len):
                visible_indices = torch.nonzero(visible[b, i], as_tuple=False).flatten()
                selected = self.select_for_row(
                    q[b, i], raw_keys[b], visible_indices, cos[b], sin[b]
                )
                out[b, i, : selected.numel()] = selected
        return out

    def selection_mask(
        self, selected_token_indices: torch.Tensor, kv_length: int
    ) -> torch.Tensor:
        """``[B,S,max_selected] -> [B,1,S,kv_length]`` bool, ``-1`` absorbed."""
        mask = torch.zeros(
            (*selected_token_indices.shape[:-1], kv_length + 1),
            dtype=torch.bool,
            device=selected_token_indices.device,
        )
        scatter_indices = torch.where(
            selected_token_indices >= 0, selected_token_indices, kv_length
        )
        mask = mask.scatter(-1, scatter_indices.long(), True)
        return mask[..., :kv_length].unsqueeze(1)
