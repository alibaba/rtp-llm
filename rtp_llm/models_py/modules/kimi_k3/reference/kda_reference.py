"""Pure-Torch correctness backend for Kimi K3 Delta Attention (KDA).

This file deliberately provides two independent execution forms:

* ``kimi_kda_chunk`` composes token affine maps with a parallel prefix scan and
  is the prefill reference path.
* ``kimi_kda_recurrent`` applies the rank-one delta update token by token and is
  the decode reference path.

Keeping both forms is useful beyond bring-up: their agreement is a durable
oracle for the future Triton/CUDA chunk and recurrent kernels.  The dense
``K x K`` affine maps used by the chunk implementation are intentionally a
small-model reference, not a production-performance implementation.

TODO(Kimi-K3): replace the prefill and decode calls with optimized kernels once
their cache/state ABI is connected.  Retain these functions for CPU tests and
kernel differential validation.
"""

from __future__ import annotations

from typing import Literal, Optional, Tuple

import torch
import torch.nn.functional as F


# Keep this alias local so the file remains directly importable as a standalone
# pure-Torch oracle without initializing the compiled RTP module package.
KDAExecutionMode = Literal["prefill", "decode"]


def _validate_inputs(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    raw_gate: torch.Tensor,
    raw_beta: torch.Tensor,
    a_log: torch.Tensor,
    dt_bias: Optional[torch.Tensor],
    initial_state: Optional[torch.Tensor],
    cu_seqlens: Optional[torch.Tensor],
) -> Tuple[int, int, int, int, int]:
    if q.ndim != 4:
        raise ValueError(f"q must have shape [B,T,H,K], got {tuple(q.shape)}")
    if k.shape != q.shape:
        raise ValueError(f"k must match q, got q={tuple(q.shape)} k={tuple(k.shape)}")
    if v.ndim != 4 or v.shape[:3] != q.shape[:3]:
        raise ValueError(
            "v must have shape [B,T,H,V] with the same B/T/H as q; "
            f"got q={tuple(q.shape)} v={tuple(v.shape)}"
        )
    if raw_gate.shape != q.shape:
        raise ValueError(
            f"raw_gate must match q, got {tuple(raw_gate.shape)} and {tuple(q.shape)}"
        )

    batch, length, heads, key_dim = q.shape
    value_dim = v.shape[-1]
    if raw_beta.shape != (batch, length, heads):
        raise ValueError(
            "raw_beta must have shape [B,T,H], got "
            f"{tuple(raw_beta.shape)} instead of {(batch, length, heads)}"
        )
    if a_log.shape != (heads,):
        raise ValueError(
            f"a_log must have shape [H], got {tuple(a_log.shape)} instead of {(heads,)}"
        )
    if dt_bias is not None and dt_bias.numel() != heads * key_dim:
        raise ValueError(
            "dt_bias must contain H*K values, got "
            f"{dt_bias.numel()} instead of {heads * key_dim}"
        )

    sequence_count = batch
    if cu_seqlens is not None:
        if batch != 1:
            raise ValueError("packed KDA inputs require batch size 1")
        if cu_seqlens.ndim != 1 or cu_seqlens.numel() < 2:
            raise ValueError("cu_seqlens must be a one-dimensional [N+1] tensor")
        if cu_seqlens.dtype not in (torch.int32, torch.int64):
            raise ValueError("cu_seqlens must use an integer dtype")
        offsets = cu_seqlens.detach().cpu().tolist()
        if offsets[0] != 0 or offsets[-1] != length:
            raise ValueError(
                f"cu_seqlens must start at 0 and end at T={length}, got {offsets}"
            )
        if any(left > right for left, right in zip(offsets, offsets[1:])):
            raise ValueError("cu_seqlens must be non-decreasing")
        sequence_count = len(offsets) - 1

    if initial_state is not None:
        expected = (sequence_count, heads, key_dim, value_dim)
        if initial_state.shape != expected:
            raise ValueError(
                f"initial_state must have shape {expected}, got {tuple(initial_state.shape)}"
            )
    return batch, length, heads, key_dim, value_dim


def prepare_kimi_kda_inputs(
    q: torch.Tensor,
    k: torch.Tensor,
    raw_gate: torch.Tensor,
    raw_beta: torch.Tensor,
    a_log: torch.Tensor,
    dt_bias: Optional[torch.Tensor] = None,
    *,
    lower_bound: Optional[float] = None,
    scale: Optional[float] = None,
    norm_epsilon: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply K3's fused-kernel input transforms in explicit Torch operations.

    Returns normalized/scaled query, normalized key, per-key decay ``alpha``,
    and sigmoid beta.  Gate arithmetic is kept in fp32 just like FLA KDA.
    """

    if q.shape != k.shape or q.shape != raw_gate.shape or q.ndim != 4:
        raise ValueError("q, k and raw_gate must share shape [B,T,H,K]")
    if raw_beta.shape != q.shape[:-1]:
        raise ValueError("raw_beta must have shape [B,T,H]")

    heads, key_dim = q.shape[-2:]
    if a_log.shape != (heads,):
        raise ValueError(f"a_log must have shape {(heads,)}, got {tuple(a_log.shape)}")
    if dt_bias is not None and dt_bias.numel() != heads * key_dim:
        raise ValueError(f"dt_bias must contain {heads * key_dim} values")
    if lower_bound is not None and lower_bound >= 0:
        raise ValueError("KDA lower_bound is a negative log-decay bound")

    q_float = q.float()
    k_float = k.float()
    q_float = q_float * torch.rsqrt(
        q_float.square().sum(dim=-1, keepdim=True) + norm_epsilon
    )
    k_float = k_float * torch.rsqrt(
        k_float.square().sum(dim=-1, keepdim=True) + norm_epsilon
    )
    q_float = q_float * (key_dim**-0.5 if scale is None else scale)

    gate_input = raw_gate.float()
    if dt_bias is not None:
        gate_input = gate_input + dt_bias.float().reshape(heads, key_dim)
    rate = a_log.float().exp().reshape(1, 1, heads, 1)
    if lower_bound is None:
        log_decay = -rate * F.softplus(gate_input)
    else:
        log_decay = float(lower_bound) * torch.sigmoid(rate * gate_input)

    alpha = log_decay.exp()
    beta = raw_beta.float().sigmoid()
    return q_float, k_float, alpha, beta


def _sequence_ranges(
    batch: int, length: int, cu_seqlens: Optional[torch.Tensor]
) -> list[Tuple[int, int, int]]:
    if cu_seqlens is None:
        return [(batch_idx, 0, length) for batch_idx in range(batch)]
    offsets = cu_seqlens.detach().cpu().tolist()
    return [(0, int(start), int(end)) for start, end in zip(offsets, offsets[1:])]


def _initial_states(
    initial_state: Optional[torch.Tensor],
    sequence_count: int,
    heads: int,
    key_dim: int,
    value_dim: int,
    device: torch.device,
) -> torch.Tensor:
    if initial_state is None:
        return torch.zeros(
            sequence_count,
            heads,
            key_dim,
            value_dim,
            dtype=torch.float32,
            device=device,
        )
    return initial_state.float()


def _restore_output_layout(
    outputs: list[torch.Tensor],
    *,
    batch: int,
    length: int,
    heads: int,
    value_dim: int,
    packed: bool,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    if packed:
        if outputs:
            result = torch.cat(outputs, dim=0).unsqueeze(0)
        else:
            result = torch.empty(
                1, 0, heads, value_dim, dtype=torch.float32, device=device
            )
    else:
        result = torch.stack(outputs, dim=0) if outputs else torch.empty(
            batch, length, heads, value_dim, dtype=torch.float32, device=device
        )
    return result.to(dtype=dtype)


def kimi_kda_recurrent(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    raw_gate: torch.Tensor,
    raw_beta: torch.Tensor,
    a_log: torch.Tensor,
    dt_bias: Optional[torch.Tensor] = None,
    initial_state: Optional[torch.Tensor] = None,
    *,
    lower_bound: Optional[float] = None,
    scale: Optional[float] = None,
    cu_seqlens: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Token-recurrent KDA reference used by the decode path."""

    batch, length, heads, key_dim, value_dim = _validate_inputs(
        q, k, v, raw_gate, raw_beta, a_log, dt_bias, initial_state, cu_seqlens
    )
    q_work, k_work, alpha, beta = prepare_kimi_kda_inputs(
        q,
        k,
        raw_gate,
        raw_beta,
        a_log,
        dt_bias,
        lower_bound=lower_bound,
        scale=scale,
    )
    v_work = v.float()
    ranges = _sequence_ranges(batch, length, cu_seqlens)
    states = _initial_states(
        initial_state,
        len(ranges),
        heads,
        key_dim,
        value_dim,
        q.device,
    )

    sequence_outputs: list[torch.Tensor] = []
    final_states: list[torch.Tensor] = []
    for sequence_idx, (batch_idx, start, end) in enumerate(ranges):
        state = states[sequence_idx]
        token_outputs: list[torch.Tensor] = []
        for token_idx in range(start, end):
            token_alpha = alpha[batch_idx, token_idx]
            token_key = k_work[batch_idx, token_idx]
            token_value = v_work[batch_idx, token_idx]
            token_beta = beta[batch_idx, token_idx]

            decayed_state = state * token_alpha.unsqueeze(-1)
            value_residual = token_value - torch.einsum(
                "hkv,hk->hv", decayed_state, token_key
            )
            state = decayed_state + (
                token_beta[:, None, None]
                * token_key.unsqueeze(-1)
                * value_residual.unsqueeze(-2)
            )
            token_outputs.append(
                torch.einsum("hk,hkv->hv", q_work[batch_idx, token_idx], state)
            )

        if token_outputs:
            sequence_outputs.append(torch.stack(token_outputs, dim=0))
        else:
            sequence_outputs.append(
                torch.empty(
                    0,
                    heads,
                    value_dim,
                    dtype=torch.float32,
                    device=q.device,
                )
            )
        final_states.append(state)

    output = _restore_output_layout(
        sequence_outputs,
        batch=batch,
        length=length,
        heads=heads,
        value_dim=value_dim,
        packed=cu_seqlens is not None,
        dtype=q.dtype,
        device=q.device,
    )
    return output, torch.stack(final_states, dim=0)


def _token_affine_maps(
    key: torch.Tensor,
    value: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build ``S' = A @ S + B`` for all tokens in one chunk."""

    length, heads, key_dim = key.shape
    eye = torch.eye(key_dim, dtype=key.dtype, device=key.device).reshape(
        1, 1, key_dim, key_dim
    )
    erase = eye - (
        beta[:, :, None, None]
        * key.unsqueeze(-1)
        * key.unsqueeze(-2)
    )
    # The recurrent equation decays the state before applying the delta update,
    # hence A = (I - beta*k*k^T) @ Diag(alpha).  Multiplying columns by
    # alpha avoids materializing a diagonal matrix.
    transform = erase * alpha.unsqueeze(-2)
    injection = (
        beta[:, :, None, None]
        * key.unsqueeze(-1)
        * value.unsqueeze(-2)
    )
    if transform.shape != (length, heads, key_dim, key_dim):
        raise AssertionError("unexpected KDA affine transform shape")
    return transform, injection


def _inclusive_affine_scan(
    transform: torch.Tensor, injection: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Hillis-Steele scan for ordered affine-map composition.

    For token transforms ``F_i(S)=A_i S+B_i``, result ``i`` is the inclusive
    composition ``F_i o ... o F_0``.  Work is batched at every tree level, so
    this is structurally distinct from the recurrent rank-one oracle.
    """

    length = transform.shape[0]
    prefix_a, prefix_b = transform, injection
    offset = 1
    while offset < length:
        right_a = prefix_a[offset:]
        right_b = prefix_b[offset:]
        left_a = prefix_a[:-offset]
        left_b = prefix_b[:-offset]
        composed_a = torch.matmul(right_a, left_a)
        composed_b = torch.matmul(right_a, left_b) + right_b
        prefix_a = torch.cat((prefix_a[:offset], composed_a), dim=0)
        prefix_b = torch.cat((prefix_b[:offset], composed_b), dim=0)
        offset *= 2
    return prefix_a, prefix_b


def kimi_kda_chunk(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    raw_gate: torch.Tensor,
    raw_beta: torch.Tensor,
    a_log: torch.Tensor,
    dt_bias: Optional[torch.Tensor] = None,
    initial_state: Optional[torch.Tensor] = None,
    *,
    lower_bound: Optional[float] = None,
    scale: Optional[float] = None,
    cu_seqlens: Optional[torch.Tensor] = None,
    chunk_size: int = 64,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Chunk/parallel-prefix KDA reference used by the prefill path."""

    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    batch, length, heads, key_dim, value_dim = _validate_inputs(
        q, k, v, raw_gate, raw_beta, a_log, dt_bias, initial_state, cu_seqlens
    )
    q_work, k_work, alpha, beta = prepare_kimi_kda_inputs(
        q,
        k,
        raw_gate,
        raw_beta,
        a_log,
        dt_bias,
        lower_bound=lower_bound,
        scale=scale,
    )
    v_work = v.float()
    ranges = _sequence_ranges(batch, length, cu_seqlens)
    states = _initial_states(
        initial_state,
        len(ranges),
        heads,
        key_dim,
        value_dim,
        q.device,
    )

    sequence_outputs: list[torch.Tensor] = []
    final_states: list[torch.Tensor] = []
    for sequence_idx, (batch_idx, start, end) in enumerate(ranges):
        incoming_state = states[sequence_idx]
        chunk_outputs: list[torch.Tensor] = []
        for chunk_start in range(start, end, chunk_size):
            chunk_end = min(chunk_start + chunk_size, end)
            key_chunk = k_work[batch_idx, chunk_start:chunk_end]
            value_chunk = v_work[batch_idx, chunk_start:chunk_end]
            transform, injection = _token_affine_maps(
                key_chunk,
                value_chunk,
                alpha[batch_idx, chunk_start:chunk_end],
                beta[batch_idx, chunk_start:chunk_end],
            )
            prefix_a, prefix_b = _inclusive_affine_scan(transform, injection)
            token_states = torch.matmul(
                prefix_a, incoming_state.unsqueeze(0)
            ) + prefix_b
            chunk_outputs.append(
                torch.einsum(
                    "thk,thkv->thv",
                    q_work[batch_idx, chunk_start:chunk_end],
                    token_states,
                )
            )
            incoming_state = token_states[-1]

        if chunk_outputs:
            sequence_outputs.append(torch.cat(chunk_outputs, dim=0))
        else:
            sequence_outputs.append(
                torch.empty(
                    0,
                    heads,
                    value_dim,
                    dtype=torch.float32,
                    device=q.device,
                )
            )
        final_states.append(incoming_state)

    output = _restore_output_layout(
        sequence_outputs,
        batch=batch,
        length=length,
        heads=heads,
        value_dim=value_dim,
        packed=cu_seqlens is not None,
        dtype=q.dtype,
        device=q.device,
    )
    return output, torch.stack(final_states, dim=0)


def kimi_kda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    raw_gate: torch.Tensor,
    raw_beta: torch.Tensor,
    a_log: torch.Tensor,
    dt_bias: Optional[torch.Tensor] = None,
    initial_state: Optional[torch.Tensor] = None,
    *,
    mode: KDAExecutionMode,
    lower_bound: Optional[float] = None,
    scale: Optional[float] = None,
    cu_seqlens: Optional[torch.Tensor] = None,
    chunk_size: int = 64,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Dispatch K3 prefill to chunk form and decode to recurrent form."""

    common_args = (q, k, v, raw_gate, raw_beta, a_log, dt_bias, initial_state)
    common_kwargs = {
        "lower_bound": lower_bound,
        "scale": scale,
        "cu_seqlens": cu_seqlens,
    }
    if mode == "prefill":
        return kimi_kda_chunk(
            *common_args, chunk_size=chunk_size, **common_kwargs
        )
    if mode == "decode":
        return kimi_kda_recurrent(*common_args, **common_kwargs)
    raise ValueError(f"unsupported KDA execution mode {mode!r}")
