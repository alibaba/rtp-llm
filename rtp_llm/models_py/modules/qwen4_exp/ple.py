import math
from typing import List, Optional

import torch
import torch.nn.functional as F
from torch import nn

from rtp_llm.models_py.modules.qwen4_exp.gated_residual import grouped_rms_norm


class Qwen4ExpNGramEmbedding(nn.Module):
    """Hashed n-gram lookup feeding the PLE layer.

    Each token is hashed once per n-gram order (2..``ngram_size``) and per head,
    giving ``(ngram_size - 1) * heads_per_ngram`` ids whose embeddings are
    concatenated. ``vocab_sizes`` / ``offsets`` / ``multipliers`` come straight
    from the checkpoint rather than being recomputed, so the prime search and the
    splitmix64 seeding upstream uses never need reproducing.

    The table is row-split across ``shards`` (128 equal parts in the released
    checkpoint). Lookup walks the shards rather than materialising the full
    table, which for the released model would be 102GB.
    """

    def __init__(
        self,
        shards: List[torch.Tensor],
        vocab_sizes: torch.Tensor,
        offsets: torch.Tensor,
        multipliers: torch.Tensor,
        *,
        ngram_size: int,
        eos_token_id: int,
        shard_indices: Optional[List[int]] = None,
        total_shards: Optional[int] = None,
        distributed_reduce: bool = False,
    ):
        super().__init__()
        if not shards:
            raise ValueError("ngram embedding needs at least one shard")
        metadata = {
            "vocab_sizes": vocab_sizes,
            "offsets": offsets,
            "multipliers": multipliers,
        }
        for name, tensor in metadata.items():
            if tensor.dtype != torch.int64:
                raise TypeError(
                    f"qwen4_exp PLE {name} must stay int64, got {tensor.dtype}"
                )
            if tensor.ndim != 1:
                raise ValueError(
                    f"qwen4_exp PLE {name} must be one-dimensional, got "
                    f"shape {tuple(tensor.shape)}"
                )
        rows = shards[0].shape[0]
        if any(shard.shape[0] != rows for shard in shards):
            raise ValueError("ngram embedding shards must have equal row counts")
        if ngram_size <= 1:
            raise ValueError(f"ngram_size must be greater than one, got {ngram_size}")
        heads = vocab_sizes.numel()
        if offsets.numel() != heads:
            raise ValueError(
                f"offsets ({offsets.numel()}) and vocab_sizes ({heads}) disagree"
            )
        if multipliers.numel() < ngram_size:
            raise ValueError(
                f"multipliers ({multipliers.numel()}) must cover ngram_size "
                f"{ngram_size}"
            )
        if heads % (ngram_size - 1) != 0:
            raise ValueError(
                f"{heads} ngram heads is not divisible by {ngram_size - 1} orders"
            )
        if shard_indices is None:
            shard_indices = list(range(len(shards)))
        if len(shard_indices) != len(shards):
            raise ValueError("shard_indices must have one entry per local shard")
        if len(set(shard_indices)) != len(shard_indices):
            raise ValueError("shard_indices must be unique")
        if total_shards is None:
            total_shards = len(shards)
        if total_shards <= 0 or any(
            index < 0 or index >= total_shards for index in shard_indices
        ):
            raise ValueError(
                f"shard_indices {shard_indices} are invalid for {total_shards} shards"
            )
        if len(shards) != total_shards and not distributed_reduce:
            raise ValueError(
                "a partial n-gram table requires distributed_reduce so remote "
                "shard lookups are combined across TP ranks"
            )
        total = int(offsets[-1].item() + vocab_sizes[-1].item())
        if rows * total_shards < total:
            raise ValueError(
                f"shards hold {rows * total_shards} rows but the hashed vocab needs "
                f"{total}"
            )

        self.shards = shards
        self.shard_indices = tuple(shard_indices)
        self.total_shards = total_shards
        self.distributed_reduce = distributed_reduce
        self.shard_rows = rows
        self.ngram_size = ngram_size
        self.context_len = ngram_size - 1
        self.ngram_heads = heads
        self.heads_per_ngram = heads // (ngram_size - 1)
        self.eos_token_id = eos_token_id
        self.vocab_sizes = vocab_sizes
        self.offsets = offsets
        self.multipliers = multipliers

    def _shift_right_ignore_eos(
        self, token_ids: torch.Tensor, shift: int
    ) -> torch.Tensor:
        """Shift right by ``shift``, restarting at every EOS boundary."""
        if shift == 0:
            return token_ids
        batch_size, seq_len = token_ids.shape
        positions = torch.arange(seq_len, device=token_ids.device, dtype=torch.long)
        eos_positions = torch.where(token_ids == self.eos_token_id, positions, -1)
        previous_eos_inclusive = torch.cummax(eos_positions, dim=1).values
        previous_eos = torch.cat(
            [
                eos_positions.new_full((batch_size, 1), -1),
                previous_eos_inclusive[:, :-1],
            ],
            dim=1,
        )
        position_in_segment = positions.unsqueeze(0) - (previous_eos + 1)
        source_positions = positions - shift
        gather_positions = (
            source_positions.clamp_min(0).unsqueeze(0).expand(batch_size, -1)
        )
        shifted = token_ids.gather(dim=1, index=gather_positions)
        valid = (position_in_segment >= shift) & (source_positions.unsqueeze(0) >= 0)
        return torch.where(valid, shifted, token_ids.new_full((), self.eos_token_id))

    def hashed_ids(self, token_history: torch.Tensor, seq_len: int) -> torch.Tensor:
        """Map a ``[B, context_len + seq_len]`` history to ``[B, seq_len, heads]``."""
        shifted = [
            self._shift_right_ignore_eos(token_history, shift)
            for shift in range(self.ngram_size)
        ]
        blocks = []
        for ngram in range(2, self.ngram_size + 1):
            start = (ngram - 2) * self.heads_per_ngram
            end = start + self.heads_per_ngram
            mixed = shifted[0] * self.multipliers[0]
            for position in range(1, ngram):
                mixed = torch.bitwise_xor(
                    mixed, shifted[position] * self.multipliers[position]
                )
            head_vocab = self.vocab_sizes[start:end].view(1, 1, -1)
            head_offsets = self.offsets[start:end].view(1, 1, -1)
            hashed = torch.remainder(mixed.unsqueeze(-1), head_vocab)
            blocks.append(hashed + head_offsets)
        return torch.cat(blocks, dim=-1)[:, -seq_len:]

    def _gather_local(self, ngram_ids: torch.Tensor) -> torch.Tensor:
        shard_idx = torch.div(ngram_ids, self.shard_rows, rounding_mode="floor")
        row_idx = ngram_ids - shard_idx * self.shard_rows
        out = self.shards[0].new_zeros(*ngram_ids.shape, self.shards[0].shape[-1])
        for global_index, shard in zip(self.shard_indices, self.shards):
            selected = shard_idx == global_index
            out[selected] = shard[row_idx[selected]]
        return out

    def gather(self, ngram_ids: torch.Tensor) -> torch.Tensor:
        out = self._gather_local(ngram_ids)
        if self.distributed_reduce:
            # Keep the correctness-only single-rank module importable without
            # loading the distributed runtime; production TP reaches this path.
            from rtp_llm.models_py.distributed.collective_torch import Group, all_reduce

            out = all_reduce(out, group=Group.TP)
        return out.flatten(-2)

    def forward(self, token_history: torch.Tensor, seq_len: int) -> torch.Tensor:
        return self.gather(self.hashed_ids(token_history.long(), seq_len))


class Qwen4ExpPLELayer(nn.Module):
    """Injects hashed n-gram features into every branch of the residual stream.

    Returns ``[B, S, hc_mult * hidden]``, which the decoder layer adds to the
    stream. The n-gram embedding is gated by the agreement between each branch's
    normalized activations and a per-branch key, then a dilated depthwise
    convolution mixes in local lexical context.
    """

    def __init__(
        self,
        ngram_embedding: Qwen4ExpNGramEmbedding,
        key_proj: torch.Tensor,
        value_proj: torch.Tensor,
        conv_weight: torch.Tensor,
        norm_key: torch.Tensor,
        norm_query: torch.Tensor,
        norm_conv: torch.Tensor,
        *,
        hc_mult: int,
        hidden_size: int,
        conv_kernel_size: int,
        norm_eps: float,
    ):
        super().__init__()
        hc_hidden = hc_mult * hidden_size
        if key_proj.shape[0] != hc_hidden:
            raise ValueError(f"key_proj out width {key_proj.shape[0]} != {hc_hidden}")
        if value_proj.shape[0] != hidden_size:
            raise ValueError(
                f"value_proj out width {value_proj.shape[0]} != {hidden_size}"
            )
        expected_conv = (hc_hidden, 1, conv_kernel_size)
        if tuple(conv_weight.shape) != expected_conv:
            raise ValueError(
                f"conv weight {tuple(conv_weight.shape)} != {expected_conv}"
            )

        self.ple_embedding = ngram_embedding
        self.hc_mult = hc_mult
        self.hidden_size = hidden_size
        self.hc_hidden_size = hc_hidden
        self.norm_eps = norm_eps
        # Dilation is the n-gram order, so the receptive field steps over whole
        # n-grams rather than adjacent tokens.
        self.conv_dilation = ngram_embedding.ngram_size
        self.short_conv_state_len = (conv_kernel_size - 1) * self.conv_dilation
        self.key_proj = key_proj
        self.value_proj = value_proj
        self.conv_weight = conv_weight
        self.norm_key = norm_key
        self.norm_query = norm_query
        self.norm_conv = norm_conv

    def _norm(self, x: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
        return grouped_rms_norm(x, gamma, self.hidden_size, self.norm_eps)

    def _short_conv(self, hidden_states: torch.Tensor) -> torch.Tensor:
        seq_len = hidden_states.shape[1]
        hidden_states = hidden_states.transpose(1, 2)
        # Always pad then slice: the dilation means the kernel reaches further
        # back than a plain causal conv of the same width.
        hidden_states = F.pad(hidden_states, (self.short_conv_state_len, 0))
        hidden_states = hidden_states[..., -(self.short_conv_state_len + seq_len) :]
        hidden_states = F.silu(
            F.conv1d(
                hidden_states,
                self.conv_weight,
                groups=self.hc_hidden_size,
                dilation=self.conv_dilation,
            )
        )
        return hidden_states.transpose(1, 2)

    def _gated_values(
        self,
        hyper_states: torch.Tensor,
        token_history: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> tuple:
        """Shared front half: n-gram embed, gate against the stream, normalize.

        Returns ``(gated_value, gated_value_normed)``, both ``[B, S, hc_hidden]``.
        The short conv is applied by the caller so prefill and decode can feed it
        different left context.
        """
        seq_len = hyper_states.shape[1]
        embeddings = self.ple_embedding(token_history, seq_len)

        branches = (self.hc_mult, self.hidden_size)
        key_normed = self._norm(
            F.linear(embeddings, self.key_proj), self.norm_key
        ).unflatten(-1, branches)
        query_normed = self._norm(hyper_states, self.norm_query).unflatten(-1, branches)
        value = F.linear(embeddings, self.value_proj)

        gate = (key_normed * query_normed).sum(dim=-1, keepdim=True) / math.sqrt(
            self.hidden_size
        )
        # Signed square root: compresses large agreements without losing sign.
        gate = gate.abs().clamp_min(1e-6).sqrt() * gate.sign()
        gated_value = torch.sigmoid(gate) * value.unsqueeze(-2)

        gated_value_normed = self._norm(gated_value.flatten(-2), self.norm_conv)
        gated_value = gated_value.flatten(-2)
        if padding_mask is not None:
            keep = padding_mask.unsqueeze(-1).to(gated_value.dtype)
            gated_value = gated_value * keep
            gated_value_normed = gated_value_normed * keep
        return gated_value, gated_value_normed

    def forward(
        self,
        hyper_states: torch.Tensor,
        token_history: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        output, _ = self.prefill(hyper_states, token_history, padding_mask)
        return output

    def prefill(
        self,
        hyper_states: torch.Tensor,
        token_history: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run PLE prefill and return output plus the decode conv state."""
        gated_value, gated_value_normed = self._gated_values(
            hyper_states, token_history, padding_mask
        )
        output = gated_value + self._short_conv(gated_value_normed)
        return output, self.prefill_conv_state(gated_value_normed)

    def prefill_with_state(
        self,
        hyper_states: torch.Tensor,
        token_history: torch.Tensor,
        conv_buffer: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Prefix-reuse prefill: continue the conv from a cached state buffer.

        ``token_history`` carries the cached n-gram context followed by the new
        tokens, so this is the chunked form of :meth:`decode_step` and returns
        the state the next chunk/decode must resume from.
        """
        output, candidate_inputs = self.decode_chunk(
            hyper_states, token_history, conv_buffer
        )
        state = torch.cat([conv_buffer, candidate_inputs], dim=1)[
            :, -self.short_conv_state_len :
        ]
        return output, state

    def prefill_conv_state(self, gated_value_normed: torch.Tensor) -> torch.Tensor:
        """The conv-input history decode must resume from: last state_len rows.

        Left-padded with zeros when the prompt is shorter than the state, exactly
        matching the zero left-context ``_short_conv`` assumes during prefill.
        """
        seq_len = gated_value_normed.shape[1]
        state_len = self.short_conv_state_len
        if seq_len >= state_len:
            return gated_value_normed[:, seq_len - state_len :]
        pad = gated_value_normed.new_zeros(
            gated_value_normed.shape[0], state_len - seq_len, self.hc_hidden_size
        )
        return torch.cat([pad, gated_value_normed], dim=1)

    def decode_step(
        self,
        hyper_state: torch.Tensor,
        token_history_step: torch.Tensor,
        conv_buffer: torch.Tensor,
    ) -> tuple:
        """One decode token: PLE output plus the rolled conv buffer.

        ``hyper_state`` is ``[B, 1, hc_hidden]``; ``token_history_step`` is
        ``[B, context_len + 1]`` (cached context ids followed by the new token);
        ``conv_buffer`` is ``[B, state_len, hc_hidden]`` of prior conv inputs.
        Unlike prefill's ``_short_conv``, the real cached history is used instead
        of a zero left pad.
        """
        gated_value, gated_value_normed = self._gated_values(
            hyper_state, token_history_step
        )
        conv_in = torch.cat([conv_buffer, gated_value_normed], dim=1).transpose(1, 2)
        conv_out = F.silu(
            F.conv1d(
                conv_in,
                self.conv_weight,
                groups=self.hc_hidden_size,
                dilation=self.conv_dilation,
            )
        ).transpose(1, 2)
        output = gated_value + conv_out
        new_buffer = torch.cat([conv_buffer, gated_value_normed], dim=1)[:, 1:]
        return output, new_buffer

    def decode_chunk(
        self,
        hyper_states: torch.Tensor,
        token_history: torch.Tensor,
        conv_buffer: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Evaluate a speculative decode chunk without committing its state.

        ``hyper_states`` contains the candidate rows ``[B, G, hc_hidden]`` and
        ``token_history`` contains the cached n-gram context followed by those
        same ``G`` token ids.  The returned second tensor is the normalized
        convolution input for each candidate row, not a materialised
        ``[B, G, state_len, hc_hidden]`` state tensor.  A caller can recover the
        state after accepting ``A`` rows by taking the last ``state_len`` rows
        of ``cat(conv_buffer, candidate_inputs)[:, :state_len + A]``.

        Concatenating the committed buffer with all candidate inputs and
        applying the dilated convolution once is numerically identical to
        calling :meth:`decode_step` in order, while keeping persistent cache
        mutation outside the target-verify forward.
        """
        if hyper_states.dim() != 3:
            raise ValueError(
                "qwen4_exp PLE decode chunk expects [B, G, hc_hidden] states"
            )
        batch_size, query_len, width = hyper_states.shape
        if query_len <= 0 or width != self.hc_hidden_size:
            raise ValueError(
                "qwen4_exp PLE decode chunk has invalid state geometry "
                f"{tuple(hyper_states.shape)}"
            )
        if token_history.dim() != 2 or tuple(token_history.shape) != (
            batch_size,
            self.ple_embedding.context_len + query_len,
        ):
            raise ValueError(
                "qwen4_exp PLE decode chunk token history must be "
                f"[{batch_size}, {self.ple_embedding.context_len + query_len}]"
            )
        if conv_buffer.dim() != 3 or tuple(conv_buffer.shape) != (
            batch_size,
            self.short_conv_state_len,
            self.hc_hidden_size,
        ):
            raise ValueError(
                "qwen4_exp PLE decode chunk conv buffer must be "
                f"[{batch_size}, {self.short_conv_state_len}, "
                f"{self.hc_hidden_size}]"
            )
        if (
            conv_buffer.device != hyper_states.device
            or token_history.device != hyper_states.device
        ):
            raise ValueError("qwen4_exp PLE decode chunk inputs must share one device")

        gated_value, candidate_inputs = self._gated_values(hyper_states, token_history)
        conv_in = torch.cat([conv_buffer, candidate_inputs], dim=1).transpose(1, 2)
        conv_out = F.silu(
            F.conv1d(
                conv_in,
                self.conv_weight,
                groups=self.hc_hidden_size,
                dilation=self.conv_dilation,
            )
        ).transpose(1, 2)
        if int(conv_out.shape[1]) != query_len:
            raise RuntimeError(
                "qwen4_exp PLE decode chunk convolution returned "
                f"{conv_out.shape[1]} rows, expected {query_len}"
            )
        return gated_value + conv_out, candidate_inputs
