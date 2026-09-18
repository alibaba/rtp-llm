"""QSA indexer: picks which KV tokens the sparse full-attention layer may read.

Reference implementation -- correctness only, no kernels. It exists to be the
truth baseline that the eventual fused path is validated against, the same role
``ple.py`` plays for the n-gram layer.

Selection is block-wise. Visible tokens are cut into ``compress_ratio``-sized
blocks; each complete block is represented by the *mean* of its raw keys (there
is no learned compressor in this checkpoint -- pooling is parameter-free), and
the top ``token_budget // compress_ratio`` blocks by score are expanded back into
token indices. The trailing incomplete block is always appended unscored, which
is why the output is ``token_budget + compress_ratio - 1`` wide rather than
``token_budget``.

Two orderings here are easy to get wrong and are load-bearing:

* ``k_layernorm`` is applied *after* pooling, not to the raw keys.
* the pooled key is rotated at the position of its block's **first** token,
  while q is rotated at the current position.

RoPE is partial: ``cos``/``sin`` are the host layer's, whose width is
``head_dim * partial_rotary_factor`` (64 for this model), so only the leading 64
of each 128-wide indexer head is rotated and the rest passes through.
"""

import math
from typing import Any, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from rtp_llm.models_py.modules.qwen4_exp.norm import exact_head_rms_norm


def _normalized_rope_style_name(style: Any) -> str:
    """Read pybind, string and integer RopeStyle representations lazily."""
    name = getattr(style, "name", None)
    if isinstance(name, str):
        value = name
    elif isinstance(style, str):
        value = style.rsplit(".", 1)[-1]
    elif isinstance(style, int):
        value = {1: "Base", 7: "Mrope"}.get(style, str(style))
    else:
        value = str(style).rsplit(".", 1)[-1]
    return {"base": "Base", "mrope": "Mrope"}.get(value.lower(), value)


def is_qsa_rope_style(rope_config: Any, expected_style: str) -> bool:
    """Match a QSA RoPE style across pybind enum, string and integer forms."""
    return _normalized_rope_style_name(
        rope_config.style
    ) == _normalized_rope_style_name(expected_style)


def _validate_indexer_rope_layout(rope_config: Any) -> None:
    if not bool(getattr(rope_config, "indexer_is_neox_style", True)):
        raise ValueError("qwen4 QSA requires indexer_is_neox_style=true")


def _rope_scale(rope_config: Any, rope_name: str) -> float:
    scale = float(getattr(rope_config, "scale", 1.0))
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError(f"qwen4 QSA {rope_name} scale must be finite and positive")
    return scale


def _position_matrix(
    position_ids: torch.Tensor,
    *,
    token_count: int,
    index_factor: int,
    device: torch.device,
) -> torch.Tensor:
    if token_count < 0:
        raise ValueError(f"token_count must be non-negative, got {token_count}")
    flat = position_ids.to(device=device, dtype=torch.long).contiguous().view(-1)
    expected = token_count * index_factor
    if int(flat.numel()) != expected:
        raise ValueError(
            f"qwen4 QSA position ids have {flat.numel()} values, expected "
            f"{expected} for {token_count} tokens"
        )
    return flat.view(token_count, index_factor)


def _validate_logical_cache_positions(
    positions: torch.Tensor,
    logical_positions: torch.Tensor,
    *,
    rope_name: str,
) -> None:
    expected = (
        logical_positions.to(device=positions.device, dtype=torch.long)
        .contiguous()
        .view(-1)
    )
    if int(expected.numel()) != int(positions.shape[0]):
        raise ValueError(
            f"qwen4 QSA {rope_name} logical cache positions have "
            f"{expected.numel()} values, expected {positions.shape[0]}"
        )
    if not bool(torch.equal(positions, expected.unsqueeze(1).expand_as(positions))):
        actual_head = positions[:6].reshape(-1).tolist()
        actual_tail = positions[-3:].reshape(-1).tolist()
        raise ValueError(
            f"qwen4 QSA {rope_name} positions must equal the logical cache positions"
            f" (rows={positions.shape[0]}, cols={positions.shape[1]},"
            f" actual[:6]={actual_head}, actual[-3:]={actual_tail},"
            f" expected_rows={expected.numel()})"
        )


def build_interleaved_mrope(
    position_ids: torch.Tensor,
    rope_config: Any,
    *,
    token_count: int,
    dtype: torch.dtype,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build qwen4's per-token partial-MRoPE ``(cos, sin)`` tensors.

    The engine transports MRoPE positions as a token-major flat vector
    ``[t0_T, t0_H, t0_W, t1_T, ...]``.  This helper deliberately implements
    only the released qwen4 contract: three axes, interleaved T/H/W frequency
    slots, and LLaMA-style half rotation.  Rejecting every other layout keeps
    the experimental QSA path from silently disagreeing with the dense FMHA
    RoPE kernel.

    Returns duplicated cos/sin values with shape ``[token_count, rope_dim]``;
    that is the shape consumed by :func:`apply_partial_rope`.
    """
    _validate_indexer_rope_layout(rope_config)
    index_factor = int(rope_config.index_factor)
    rope_dim = int(rope_config.dim)
    sections = (
        int(rope_config.mrope_dim1),
        int(rope_config.mrope_dim2),
        int(rope_config.mrope_dim3),
    )
    if index_factor != 3:
        raise ValueError(f"qwen4 QSA requires MRoPE index_factor=3, got {index_factor}")
    if not bool(rope_config.mrope_interleaved):
        raise ValueError("qwen4 QSA requires interleaved MRoPE")
    if rope_dim <= 0 or rope_dim % 2:
        raise ValueError(
            f"qwen4 QSA rope_dim must be positive and even, got {rope_dim}"
        )
    rotary_pairs = rope_dim // 2
    if sum(sections) != rotary_pairs:
        raise ValueError(
            f"qwen4 QSA MRoPE sections {sections} do not sum to {rotary_pairs}"
        )
    positions = _position_matrix(
        position_ids,
        token_count=token_count,
        index_factor=index_factor,
        device=device,
    )

    # Match the engine/upstream contract: temporal frequencies are the base;
    # H and W replace slots 1/2 of the T/H/W interleaving respectively.
    axes = torch.zeros(rotary_pairs, dtype=torch.long, device=device)
    axes[1 : 3 * sections[1] : 3] = 1
    axes[2 : 3 * sections[2] : 3] = 2
    if (
        int((axes == 1).sum().item()) != sections[1]
        or int((axes == 2).sum().item()) != sections[2]
    ):
        raise ValueError(
            "qwen4 QSA MRoPE H/W sections do not fit the interleaved rotary slots"
        )

    inv_freq = float(rope_config.base) ** (
        -2.0 * torch.arange(rotary_pairs, dtype=torch.float32, device=device) / rope_dim
    )
    scale = _rope_scale(rope_config, "MRoPE")
    angle = positions[:, axes].float().div(scale) * inv_freq.unsqueeze(0)
    cos_half = torch.cos(angle)
    sin_half = torch.sin(angle)
    return (
        torch.cat([cos_half, cos_half], dim=-1).to(dtype=dtype),
        torch.cat([sin_half, sin_half], dim=-1).to(dtype=dtype),
    )


def build_base_rope(
    position_ids: torch.Tensor,
    rope_config: Any,
    *,
    token_count: int,
    dtype: torch.dtype,
    device: torch.device,
    logical_positions: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build one-axis Base RoPE for the Qwen4 MTP draft indexer.

    The current MTP batch ABI may transport either one scalar position per draft
    token or the target model's three-axis position tuple.  The latter is safe
    for the Base-RoPE draft only for text: all three axes must agree, after which
    the first component is the scalar Base position.
    """
    _validate_indexer_rope_layout(rope_config)
    index_factor = int(rope_config.index_factor)
    rope_dim = int(rope_config.dim)
    if index_factor not in (1, 3):
        raise ValueError(
            "qwen4 QSA Base RoPE requires position index_factor 1 or 3, "
            f"got {index_factor}"
        )
    if rope_dim <= 0 or rope_dim % 2:
        raise ValueError(
            f"qwen4 QSA rope_dim must be positive and even, got {rope_dim}"
        )
    positions = _position_matrix(
        position_ids,
        token_count=token_count,
        index_factor=index_factor,
        device=device,
    )
    if index_factor == 3 and not bool(
        torch.equal(positions, positions[:, :1].expand_as(positions))
    ):
        raise ValueError(
            "qwen4 QSA Base RoPE factor=3 only supports draft text-only "
            "positions whose three axes are equal"
        )
    if logical_positions is not None:
        _validate_logical_cache_positions(
            positions, logical_positions, rope_name="Base RoPE"
        )

    scale = _rope_scale(rope_config, "Base RoPE")
    rotary_pairs = rope_dim // 2
    inv_freq = float(rope_config.base) ** (
        -2.0 * torch.arange(rotary_pairs, dtype=torch.float32, device=device) / rope_dim
    )
    angle = positions[:, 0].float().div(scale).unsqueeze(1) * inv_freq.unsqueeze(0)
    cos_half = torch.cos(angle)
    sin_half = torch.sin(angle)
    return (
        torch.cat([cos_half, cos_half], dim=-1).to(dtype=dtype),
        torch.cat([sin_half, sin_half], dim=-1).to(dtype=dtype),
    )


def build_qsa_rope(
    position_ids: torch.Tensor,
    rope_config: Any,
    *,
    token_count: int,
    dtype: torch.dtype,
    device: torch.device,
    logical_positions: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Dispatch the QSA indexer RoPE builder from ``rope_config.style``.

    Passing ``logical_positions`` additionally proves that every transported
    position component addresses the same absolute cache row.  Prefix-free
    target MRoPE prefill intentionally omits that check so its existing
    multimodal position behavior is unchanged.
    """
    _validate_indexer_rope_layout(rope_config)
    if is_qsa_rope_style(rope_config, "Base"):
        return build_base_rope(
            position_ids,
            rope_config,
            token_count=token_count,
            dtype=dtype,
            device=device,
            logical_positions=logical_positions,
        )
    if is_qsa_rope_style(rope_config, "Mrope"):
        if logical_positions is not None:
            positions = _position_matrix(
                position_ids,
                token_count=token_count,
                index_factor=int(rope_config.index_factor),
                device=device,
            )
            _validate_logical_cache_positions(
                positions, logical_positions, rope_name="text-only MRoPE"
            )
        return build_interleaved_mrope(
            position_ids,
            rope_config,
            token_count=token_count,
            dtype=dtype,
            device=device,
        )
    raise ValueError(f"qwen4 QSA does not support RoPE style {rope_config.style!r}")


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def apply_partial_rope(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    """Rotate the leading ``cos.shape[-1]`` features, pass the tail through.

    ``cos``/``sin`` must already be broadcastable to ``x``.
    """
    rotary_dim = cos.shape[-1]
    rotated = x[..., :rotary_dim] * cos + rotate_half(x[..., :rotary_dim]) * sin
    return torch.cat([rotated, x[..., rotary_dim:]], dim=-1)


class Qwen4ExpQSAIndexer(nn.Module):
    """Block-sparse token selection for one full-attention layer.

    ``qk_proj`` is the checkpoint's single fused ``index_qk_proj``
    ``[(n_heads + kv_heads) * head_dim, hidden]`` -- there is no q_lora here, so
    q comes straight off the hidden states. The norm gammas are raw checkpoint
    tensors; the ``+1`` is applied inside :func:`exact_head_rms_norm`.
    """

    def __init__(
        self,
        qk_proj: torch.Tensor,
        q_norm_gamma: torch.Tensor,
        k_norm_gamma: torch.Tensor,
        *,
        n_heads: int,
        kv_heads: int,
        head_dim: int,
        token_budget: int,
        compress_ratio: int,
        norm_eps: float,
    ):
        super().__init__()
        expected = (n_heads + kv_heads) * head_dim
        if qk_proj.shape[0] != expected:
            raise ValueError(
                f"index_qk_proj out width {qk_proj.shape[0]} != {expected}"
            )
        if kv_heads != 1:
            raise ValueError(f"indexer is MQA; kv_heads must be 1, got {kv_heads}")
        if token_budget % compress_ratio:
            raise ValueError(
                f"token_budget {token_budget} not divisible by "
                f"compress_ratio {compress_ratio}"
            )
        self.qk_proj = qk_proj
        self.q_norm_gamma = q_norm_gamma
        self.k_norm_gamma = k_norm_gamma
        self.n_heads = n_heads
        self.kv_heads = kv_heads
        self.head_dim = head_dim
        self.token_budget = token_budget
        self.compress_ratio = compress_ratio
        self.block_topk = token_budget // compress_ratio
        self.norm_eps = norm_eps
        # Widest possible result: every selected block contributes ratio tokens,
        # plus the unscored tail, which can hold at most ratio - 1 of them.
        self.max_selected = token_budget + compress_ratio - 1

    def project(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """``hidden -> (q [B,S,H,D], raw_keys [B,S,D])``, q normed but unrotated."""
        qk = F.linear(hidden_states, self.qk_proj)
        q, token_k = torch.split(
            qk,
            [self.n_heads * self.head_dim, self.kv_heads * self.head_dim],
            dim=-1,
        )
        shape = (*hidden_states.shape[:-1], -1, self.head_dim)
        q = exact_head_rms_norm(q.reshape(shape), self.q_norm_gamma, self.norm_eps)
        return q, token_k.reshape(shape).squeeze(-2)

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

    def pooled_block_keys_all(
        self, raw_keys: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> torch.Tensor:
        """Every complete block at once: ``[B,T,D] -> [B, T//ratio, D]``.

        This is the point of the vectorized path -- :meth:`pooled_block_keys`
        recomputes the same values once per query, which is O(S) redundant.
        """
        num_blocks = raw_keys.shape[1] // self.compress_ratio
        groups = raw_keys[:, : num_blocks * self.compress_ratio]
        groups = groups.unflatten(1, (num_blocks, self.compress_ratio))
        pooled = groups.float().mean(dim=2).to(raw_keys.dtype)
        pooled = exact_head_rms_norm(pooled, self.k_norm_gamma, self.norm_eps)
        starts = torch.arange(num_blocks, device=raw_keys.device) * self.compress_ratio
        return apply_partial_rope(
            pooled, cos.index_select(1, starts), sin.index_select(1, starts)
        )

    def forward_causal(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        past_raw_keys: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Loop-free causal selection: ``[B,S,hidden] -> [B,S,max_selected]``.

        Selects the same tokens as :meth:`forward` under a causal mask, but block
        keys are pooled once per sequence and every query is scored in one batched
        matmul. All shapes are static: no ``nonzero``, no data-dependent slicing.
        Causal visibility only -- prefill with ``past_raw_keys=None``, decode with
        ``S=1``; arbitrary masks still need :meth:`forward`.

        The output *layout* differs from :meth:`forward` on purpose: block tokens
        occupy the leading ``block_topk * ratio`` slots and the tail sits at a
        **fixed** offset, so a consumer can address both without per-row offsets,
        whereas :meth:`forward` packs everything to the left. The selected *set* is
        the same, so :meth:`selection_mask` agrees between the two.
        """
        q, raw_keys = self.project(hidden_states)
        return self.forward_causal_from_projected(q, raw_keys, cos, sin, past_raw_keys)

    def forward_causal_from_projected(
        self,
        q: torch.Tensor,
        raw_keys: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        past_raw_keys: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Causal selection from one already-computed indexer projection.

        ``q`` is normalized but unrotated ``[B,S,H,D]`` and ``raw_keys`` is
        ``[B,S,D]``.  Splitting projection from selection lets the model-level
        runtime write those same raw keys to the side cache without paying for
        a second large ``index_qk_proj`` GEMM.
        """
        if q.dim() != 4 or raw_keys.dim() != 3:
            raise ValueError("q/raw_keys must be [B,S,H,D] and [B,S,D]")
        batch, seq_len, heads, head_dim = q.shape
        if tuple(raw_keys.shape[:2]) != (batch, seq_len):
            raise ValueError("q and raw_keys batch/sequence shapes must match")
        if heads != self.n_heads or head_dim != self.head_dim:
            raise ValueError(
                f"q must have [{self.n_heads}, {self.head_dim}] head geometry"
            )
        ratio = self.compress_ratio
        device = q.device

        if past_raw_keys is not None:
            raw_keys = torch.cat([past_raw_keys, raw_keys], dim=1)
        past_len = raw_keys.shape[1] - seq_len
        q = apply_partial_rope(
            q, cos[:, past_len:].unsqueeze(2), sin[:, past_len:].unsqueeze(2)
        )

        out = torch.full(
            (batch, seq_len, self.max_selected), -1, dtype=torch.int32, device=device
        )
        positions = past_len + torch.arange(seq_len, device=device)
        # Complete blocks visible to each query; also where its tail starts.
        complete = (positions + 1) // ratio

        block_keys = self.pooled_block_keys_all(raw_keys, cos, sin)
        num_blocks = block_keys.shape[1]
        top_k = min(self.block_topk, num_blocks)
        if top_k > 0:
            # Same reduction order as block_scores: [B,S,H,nb] -> relu -> sum(H).
            scores = torch.matmul(
                q.float(), block_keys.float().transpose(-1, -2).unsqueeze(1)
            )
            scores = torch.relu(scores.transpose(-1, -2)).sum(dim=-1) / math.sqrt(
                self.head_dim
            )
            visible_block = torch.arange(num_blocks, device=device).unsqueeze(
                0
            ) < complete.unsqueeze(1)
            # Real scores are >= 0 (relu then sum), so -inf is an unambiguous
            # "not visible" sentinel that survives topk.
            scores = scores.masked_fill(~visible_block.unsqueeze(0), float("-inf"))

            top = scores.topk(top_k, dim=-1)
            chosen = torch.where(top.values > float("-inf"), top.indices, -1)
            tokens = chosen.unsqueeze(-1) * ratio + torch.arange(ratio, device=device)
            tokens = torch.where(chosen.unsqueeze(-1) >= 0, tokens, -1)
            out[..., : top_k * ratio] = tokens.flatten(-2).to(torch.int32)

        # Tail: the trailing incomplete block, always attended, never scored.
        tail_tokens = (complete * ratio).unsqueeze(-1) + torch.arange(
            ratio - 1, device=device
        )
        tail_tokens = torch.where(
            tail_tokens <= positions.unsqueeze(-1), tail_tokens, -1
        )
        tail_base = self.block_topk * ratio
        out[..., tail_base : tail_base + ratio - 1] = (
            tail_tokens.to(torch.int32).unsqueeze(0).expand(batch, -1, -1)
        )
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
