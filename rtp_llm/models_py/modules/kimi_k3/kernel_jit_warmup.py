"""Directed K3 JIT preparation after the model's cache resources are bound."""

from __future__ import annotations

import logging
import time
from types import SimpleNamespace

import torch

from rtp_llm.models_py.triton_kernels.linear_replay import (
    finalize_linear_replay,
    linear_serial_replay,
)
from rtp_llm.utils.warmup import model_warm_up_enabled


def _inactive_replay_inputs(group_id: int, device: torch.device) -> SimpleNamespace:
    """Use the same inactive lease convention as CUDA Graph capture."""
    i32 = lambda value: torch.tensor([value], device=device, dtype=torch.int32)
    i64 = lambda: torch.zeros(1, device=device, dtype=torch.int64)
    blocks = torch.full((group_id + 1, 1), -1, device=device, dtype=torch.int32)
    return SimpleNamespace(
        slot_ids=i32(-1),
        slot_generations=i64(),
        state_read_block_ids=blocks,
        active_block_ids=blocks,
        prev_accept_lengths=i32(0),
        history_valid_lengths=i32(0),
        history_epochs=i64(),
        verify_epochs=i64(),
        init_kinds=i32(0),
    )


@torch.inference_mode()
def _warmup_replay(model, init_resource) -> None:
    """Compile every reachable verify width against the actual replay pool.

    Inactive rows write only scratch output. The production state, logs and
    shared headers are never changed by this compile preparation.
    """
    if not init_resource.is_decode_role or model.kv_cache is None:
        return
    # The MTP draft updates its own recurrence; target verification owns replay.
    if type(model).__name__ != "KimiK3Model":
        return
    max_steps = max(int(model.config.gen_num_per_cycle) + 1, 1)
    prepared = set()
    for layer in model.layers:
        attention = layer.attention
        if attention.__class__.__name__ != "KimiK3KDA":
            continue
        for cache in model.kv_cache.get_layer_cache_groups(layer.index):
            replay_cache = cache.linear_replay
            if replay_cache is None:
                continue
            decoder = attention.decode
            kv = cache.kv_cache_base.reshape(cache.kv_cache_base.shape[0], -1)
            state = decoder._get_ssm_states(kv)
            conv = decoder._get_conv_states(kv)
            h, k = decoder.local_num_k_heads, decoder.head_k_dim
            hv, v = decoder.local_num_v_heads, decoder.head_v_dim
            device = state.device
            pool_key = (
                tuple(state.shape), tuple(state.stride()), str(state.dtype),
                tuple(conv.shape), tuple(conv.stride()), str(conv.dtype),
                tuple(replay_cache.k.shape), tuple(replay_cache.u.shape),
                tuple(replay_cache.g.shape), decoder.conv_weights.stride(),
                decoder.gate_lower_bound, str(device),
                attention.fa_width,
                str(attention.input.weight.dtype), tuple(attention.input.weight.shape),
                str(attention.f_b.weight.dtype), tuple(attention.f_b.weight.shape),
            )
            if pool_key in prepared:
                continue
            prepared.add(pool_key)
            if max_steps > replay_cache.k.shape[1]:
                raise ValueError("K3 verify width exceeds the bound replay log capacity")
            inputs = _inactive_replay_inputs(cache.group_id, device)
            start = time.perf_counter()
            for steps in range(1, max_steps + 1):
                tokens = steps
                # Run the same local projections to preserve beta/gate strides;
                # target verify then makes qkv contiguous before the replay split.
                hidden = torch.zeros(
                    (tokens, model.config.hidden_size), device=device,
                    dtype=model.embed_tokens.weight.dtype,
                )
                fused = attention.input(hidden)
                fused = fused[..., : 4 * attention.width + attention.fa_width + attention.heads]
                qkv, _, fa, beta = fused.split(
                    (3 * attention.width, attention.width, attention.fa_width, attention.heads),
                    dim=-1,
                )
                gate = attention._project_forget(fa)
                q, key, value = qkv.contiguous().split((h * k, h * k, hv * v), dim=-1)
                linear_serial_replay(
                    q, key, value, gate, beta, decoder.conv_weights,
                    decoder.alog, decoder.dt_bias, state, conv, replay_cache,
                    inputs, group_id=cache.group_id, vector_gate=True,
                    state_v_first=False, lower_bound=decoder.gate_lower_bound,
                )
                finalize_linear_replay(replay_cache, inputs, steps)
            torch.cuda.synchronize(device)
            logging.info(
                "K3 LINEAR replay JIT prepared: layer=%d group=%d steps=1..%d "
                "slots=%d blocks=%d capacity=%d state_stride=%d cost=%.3fs",
                layer.index, cache.group_id, max_steps, replay_cache.k.shape[0],
                state.shape[0], replay_cache.k.shape[1], state.stride(0),
                time.perf_counter() - start,
            )


def warmup_kimi_k3_kernel_jit(model, init_resource) -> None:
    if not model_warm_up_enabled() or not torch.cuda.is_available():
        return
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("K3 kernel JIT warmup cannot run during CUDA Graph capture")
    _warmup_replay(model, init_resource)
