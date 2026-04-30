"""
Unit tests for Qwen3NextGatedDeltaNet CP zigzag prefill correctness.

Verifies that the current production CP zigzag prefill path matches the
single-GPU non-CP reference (per-token output and SSM cache contents).
Uses real NCCL communication via multiprocessing.Process.

Note: this exercises the *current* production path through
``_forward_cp_prefill`` which uses ``chunk_gated_delta_rule_fwd_cp_zigzag``
and the new ``CpChunkAlignInfo.build`` signature
(``orig_lengths_cpu`` + ``padded_full_cu``).

Run with:
  bazel test //rtp_llm/models_py/modules/factory/attention/cuda_cp_impl/test:test_cp_linear_attn_zigzag
"""

import logging
import math
import multiprocessing as mp
import os
import unittest
from typing import Dict, List, Optional
from unittest.mock import patch

import torch

from rtp_llm.models_py.modules.factory.attention.cuda_cp_impl.test.cp_test_utils import (
    build_cp_attn_inputs,
    compute_rank_positions,
)
from rtp_llm.models_py.triton_kernels.causal_conv1d import (
    prepare_causal_conv1d_metadata,
)
from rtp_llm.test.utils.port_util import PortManager

logging.basicConfig(level=logging.INFO, format="%(message)s")


# ---------------------------------------------------------------------------
# Helpers (copied from test_cp_linear_attn_scan; kept local to avoid coupling)
# ---------------------------------------------------------------------------


class _AttnInputsWrapper:
    """Thin wrapper to override readonly pybind11 attributes for testing."""

    def __init__(self, wrapped, overrides: dict):
        object.__setattr__(self, "_wrapped", wrapped)
        object.__setattr__(self, "_overrides", overrides)

    def __getattr__(self, name):
        overrides = object.__getattribute__(self, "_overrides")
        if name in overrides:
            return overrides[name]
        return getattr(object.__getattribute__(self, "_wrapped"), name)

    def __setattr__(self, name, value):
        try:
            setattr(object.__getattribute__(self, "_wrapped"), name, value)
        except AttributeError:
            object.__getattribute__(self, "_overrides")[name] = value


def _add_device_tensors(inputs, device: torch.device):
    return _AttnInputsWrapper(
        inputs,
        {
            "prefix_lengths_d": inputs.prefix_lengths.to(device),
            "input_lengths_d": inputs.input_lengths.to(device),
        },
    )


def _noop_write_cache_store():
    import rtp_llm.ops.compute_ops as _co

    return patch.object(_co, "write_cache_store", lambda *a, **kw: None)


class MockLayerKVCache:
    def __init__(self, kv_cache_base: torch.Tensor, seq_size_per_block: int):
        self.kv_cache_base = kv_cache_base
        self.seq_size_per_block = seq_size_per_block


def allocate_kv_cache(
    sequence_lengths: List[int],
    tokens_per_block: int,
    num_v_heads: int,
    head_v_dim: int,
    head_k_dim: int,
    conv_kernel_dim: int,
    qkv_size: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    total_blocks = sum(math.ceil(sl / tokens_per_block) for sl in sequence_lengths)
    item_size = {torch.bfloat16: 2, torch.float16: 2, torch.float32: 4}[dtype]
    ssm_bytes = num_v_heads * head_v_dim * head_k_dim * item_size
    conv_bytes = (conv_kernel_dim - 1) * qkv_size * item_size
    block_elems = (ssm_bytes + conv_bytes) // item_size
    return torch.zeros(total_blocks, block_elems, dtype=dtype, device=device)


def _build_layer_weights(
    hidden_size: int,
    num_k_heads: int,
    num_v_heads: int,
    head_k_dim: int,
    head_v_dim: int,
    conv_kernel_dim: int,
    device: torch.device,
    dtype: torch.dtype = torch.bfloat16,
) -> Dict[str, Optional[torch.Tensor]]:
    from rtp_llm.utils.model_weight import W

    qkv_dim = head_k_dim * num_k_heads * 2 + head_v_dim * num_v_heads
    z_dim = head_v_dim * num_v_heads
    qkvz_scale = hidden_size**-0.5
    ba_scale = hidden_size**-0.5
    out_scale = (num_v_heads * head_v_dim) ** -0.5
    return {
        W.linear_attn_conv1d_w: torch.randn(
            qkv_dim, 1, conv_kernel_dim, device=device, dtype=dtype
        )
        * (conv_kernel_dim**-0.5),
        W.linear_attn_dt_b: torch.randn(num_v_heads, device=device, dtype=dtype) * 0.1,
        W.linear_attn_alog: torch.randn(num_v_heads, device=device, dtype=dtype) * 0.1,
        W.linear_attn_norm_w: torch.ones(head_v_dim, device=device, dtype=dtype),
        W.linear_attn_qkvz_w: torch.randn(
            hidden_size, qkv_dim + z_dim, device=device, dtype=dtype
        )
        * qkvz_scale,
        W.linear_attn_qkvz_s: None,
        W.linear_attn_ba_w: torch.randn(
            hidden_size, num_v_heads * 2, device=device, dtype=dtype
        )
        * ba_scale,
        W.linear_attn_out_w: torch.randn(
            num_v_heads * head_v_dim, hidden_size, device=device, dtype=dtype
        )
        * out_scale,
        W.linear_attn_out_s: None,
    }


def _build_nocp_attn_inputs(
    sequence_lengths: List[int],
    tokens_per_block: int,
    device: torch.device,
):
    from rtp_llm.ops.compute_ops import PyAttentionInputs, get_typemeta

    batch_size = len(sequence_lengths)
    inp = PyAttentionInputs()
    inp.is_prefill = True
    inp.input_lengths = torch.tensor(sequence_lengths, dtype=torch.int32, device="cpu")
    inp.prefix_lengths = torch.zeros(batch_size, dtype=torch.int32, device="cpu")
    inp.cache_store_inputs = None

    cu = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
    for i, sl in enumerate(sequence_lengths):
        cu[i + 1] = cu[i] + sl
    inp.cu_seqlens = cu

    max_blocks = max(math.ceil(sl / tokens_per_block) for sl in sequence_lengths)
    block_ids = torch.zeros(batch_size, max_blocks, dtype=torch.int32)
    offset = 0
    for i, sl in enumerate(sequence_lengths):
        nb = math.ceil(sl / tokens_per_block)
        block_ids[i, :nb] = torch.arange(offset, offset + nb, dtype=torch.int32)
        offset += nb
    inp.kv_cache_block_id_host = block_ids
    inp.kv_cache_kernel_block_id_host = block_ids
    inp.kv_cache_block_id_device = block_ids.to(device)
    inp.kv_cache_kernel_block_id_device = block_ids.to(device)
    inp.context_parallel_info = None
    inp.dtype = get_typemeta(torch.zeros(1, dtype=torch.bfloat16))
    return inp


def _zigzag_positions_for_rank(full_len: int, cp_size: int, rank: int) -> List[int]:
    """Zigzag layout for one sequence: rank holds [front-half-r] + [back-half-r]."""
    chunk_len = full_len // cp_size
    half = chunk_len // 2
    first = list(range(rank * half, (rank + 1) * half))
    second = list(range(full_len - (rank + 1) * half, full_len - rank * half))
    return first + second


def _build_padding_mask_padded(
    orig_seq_lengths: List[int], padded_seq_lengths: List[int]
) -> torch.Tensor:
    """Per-token mask in concatenated padded sequential layout: 1=real, 0=pad."""
    parts = []
    for orig_sl, padded_sl in zip(orig_seq_lengths, padded_seq_lengths):
        m = torch.zeros(padded_sl, dtype=torch.int32)
        m[:orig_sl] = 1
        parts.append(m)
    return torch.cat(parts)


def _build_restore_indices_padded(
    padded_seq_lengths: List[int], cp_size: int
) -> torch.Tensor:
    """Restore indices: ag layout (per-rank zigzag concat) → padded sequential."""
    ag_positions = []
    for rank in range(cp_size):
        for b, padded_sl in enumerate(padded_seq_lengths):
            positions = _zigzag_positions_for_rank(padded_sl, cp_size, rank)
            seq_offset = sum(padded_seq_lengths[:b])
            ag_positions.extend([p + seq_offset for p in positions])
    total = len(ag_positions)
    restore = [0] * total
    for ag_idx, orig_pos in enumerate(ag_positions):
        restore[orig_pos] = ag_idx
    return torch.tensor(restore, dtype=torch.int32)


def _compute_real_local_info(
    orig_seq_lengths: List[int], padded_seq_lengths: List[int], cp_size: int, rank: int
):
    """For one rank, return (real_local_idx, ref_orig_pos):
    real_local_idx[i] = local-buffer index that holds a real (non-pad) token
    ref_orig_pos[i]  = corresponding position in the original concatenated
                      (un-padded) sequence used by the non-CP reference.
    """
    real_local = []
    ref_pos = []
    local_offset = 0
    orig_offset = 0
    for orig_sl, padded_sl in zip(orig_seq_lengths, padded_seq_lengths):
        chunk_len = padded_sl // cp_size  # local tokens per rank for this seq
        padded_positions = _zigzag_positions_for_rank(padded_sl, cp_size, rank)
        for li, ppos in enumerate(padded_positions):
            if ppos < orig_sl:
                real_local.append(local_offset + li)
                ref_pos.append(orig_offset + ppos)
        local_offset += chunk_len
        orig_offset += orig_sl
    return (
        torch.tensor(real_local, dtype=torch.long),
        torch.tensor(ref_pos, dtype=torch.long),
    )


def _build_cp_attn_inputs_padded(
    orig_seq_lengths: List[int],
    padded_seq_lengths: List[int],
    cp_size: int,
    tokens_per_block: int,
    device: torch.device,
):
    """Production-faithful: cp_chunk_lengths reflect padded layout, padding
    metadata reflects the real per-seq pad count. Mirrors what C++ handleInputs
    produces for unaligned inputs.

    cp_test_utils.build_cp_attn_inputs hard-codes prefill_cp_padding_lengths=0,
    so it cannot exercise the need_pad / local_padding_mask path; this helper
    is the test-side replacement for that case.
    """
    from rtp_llm.ops.compute_ops import (
        PyAttentionInputs,
        PyContextParallelParams,
        get_typemeta,
    )

    batch_size = len(orig_seq_lengths)
    cp_chunk_lengths = [psl // cp_size for psl in padded_seq_lengths]

    inp = PyAttentionInputs()
    inp.is_prefill = True
    inp.input_lengths = torch.tensor(
        cp_chunk_lengths, dtype=torch.int32, device="cpu"
    ).pin_memory()
    inp.sequence_lengths = torch.tensor(
        orig_seq_lengths, dtype=torch.int32, device="cpu"
    ).pin_memory()
    inp.prefix_lengths = torch.zeros(batch_size, dtype=torch.int32, device="cpu")
    inp.cache_store_inputs = None

    cu = [0]
    for cl in cp_chunk_lengths:
        cu.append(cu[-1] + cl)
    inp.cu_seqlens = torch.tensor(cu, dtype=torch.int32, device=device)

    max_blocks = max(math.ceil(sl / tokens_per_block) for sl in orig_seq_lengths)
    block_ids = torch.zeros(batch_size, max_blocks, dtype=torch.int32)
    offset = 0
    for i, sl in enumerate(orig_seq_lengths):
        nb = math.ceil(sl / tokens_per_block)
        block_ids[i, :nb] = torch.arange(offset, offset + nb, dtype=torch.int32)
        offset += nb
    inp.kv_cache_block_id_host = block_ids
    inp.kv_cache_kernel_block_id_host = block_ids
    inp.kv_cache_block_id_device = block_ids.to(device)
    inp.kv_cache_kernel_block_id_device = block_ids.to(device)
    inp.dtype = get_typemeta(torch.zeros(1, dtype=torch.bfloat16))

    padding_lengths = [
        psl - sl for sl, psl in zip(orig_seq_lengths, padded_seq_lengths)
    ]

    cp_info = PyContextParallelParams()
    cp_info.prefill_cp_chunk_lengths = torch.tensor(cp_chunk_lengths, dtype=torch.int32)
    cp_info.prefill_cp_padding_lengths = torch.tensor(
        padding_lengths, dtype=torch.int32
    )
    cp_info.prefill_qkv_padding_mask = _build_padding_mask_padded(
        orig_seq_lengths, padded_seq_lengths
    ).to(device)
    cp_info.prefill_qkv_restore_indice = _build_restore_indices_padded(
        padded_seq_lengths, cp_size
    ).to(device)
    cp_info.prefill_actual_input_lengths_cpu = torch.tensor(
        orig_seq_lengths, dtype=torch.int32
    )
    cp_info.prefill_shuffle_indices = torch.tensor([], dtype=torch.int32)
    inp.context_parallel_info = cp_info
    return inp


def _build_cp_metadata(
    sequence_lengths: List[int],
    cp_chunk_lengths: List[int],
    cp_size: int,
    cp_rank: int,
    cp_attn_inputs,
    device: torch.device,
):
    """Mirror of `Qwen3NextModel._build_cp_linear_attn_metadata` for tests.

    Uses the *current* CpChunkAlignInfo.build signature: orig_lengths_cpu +
    padded_full_cu, and CPU-side need_pad detection."""
    from rtp_llm.models_py.model_desc.qwen3_next import (
        CpChunkAlignInfo,
        Qwen3NextMetadata,
    )

    batch_size = len(sequence_lengths)
    cp_info = cp_attn_inputs.context_parallel_info

    full_new_lengths = cp_info.prefill_actual_input_lengths_cpu  # CPU
    padded_chunk_lengths = (
        cp_info.prefill_cp_chunk_lengths
    )  # CPU/GPU; .to(device) below

    padded_full_cu = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
    padded_full_cu[1:] = (padded_chunk_lengths.to(device).long() * cp_size).cumsum(0)

    restore_indices = cp_info.prefill_qkv_restore_indice
    padding_mask = cp_info.prefill_qkv_padding_mask
    unpad_restore = restore_indices[padding_mask == 1]

    # local_padding_mask (only when C++ padded any seq): compute via inv-restore so
    # we know which local-rank slots correspond to real (vs pad) tokens.
    total_ag = padding_mask.shape[0]
    need_pad = int(full_new_lengths.sum().item()) != total_ag

    local_padding_mask = None
    if need_pad:
        local_chunk_total = total_ag // cp_size
        local_start = cp_rank * local_chunk_total
        inv_restore = torch.full((total_ag,), -1, dtype=torch.long, device=device)
        inv_restore[unpad_restore.long()] = torch.arange(
            unpad_restore.shape[0], device=device
        )
        local_padding_mask = (
            inv_restore[local_start : local_start + local_chunk_total] >= 0
        )

    chunk_align = CpChunkAlignInfo.build(
        cp_size=cp_size,
        cp_rank=cp_rank,
        device=device,
        orig_lengths_cpu=full_new_lengths.tolist(),
        padded_full_cu=padded_full_cu,
        local_padding_mask=local_padding_mask,
    )

    return Qwen3NextMetadata(
        cp_restore_indices=restore_indices,
        cp_chunk_align_info=chunk_align,
    )


# ---------------------------------------------------------------------------
# Worker (per-rank)
# ---------------------------------------------------------------------------


def _worker(
    rank: int,
    world_size: int,
    nccl_port: int,
    sequence_lengths: List[int],
    num_layers: int,
    num_k_heads: int,
    num_v_heads: int,
    head_k_dim: int,
    head_v_dim: int,
    conv_kernel_dim: int,
    tokens_per_block: int,
):
    import torch.distributed as dist

    from rtp_llm.models_py.distributed import collective_torch as ct
    from rtp_llm.models_py.distributed.collective_torch import Group
    from rtp_llm.models_py.model_desc.qwen3_next import Qwen3NextGatedDeltaNet
    from rtp_llm.models_py.triton_kernels.fla.block import store_ssm_state_to_block_map
    from rtp_llm.ops import DataType, LinearAttentionConfig, ParallelismConfig

    try:
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = str(nccl_port)
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(rank)

        par = ParallelismConfig()
        par.world_rank = rank
        par.world_size = world_size
        par.local_rank = rank
        par.tp_size = world_size
        par.dp_size = 1

        ct._parallelism_config = par
        ct._initialized = True
        ct._group_map[Group.DP_AND_TP] = dist.group.WORLD

        device = torch.device(f"cuda:{rank}")
        cp_size = world_size
        hidden_size = num_v_heads * head_v_dim
        total_tokens = sum(sequence_lengths)
        cp_chunk_lengths = [sl // cp_size for sl in sequence_lengths]
        dtype = torch.bfloat16
        qkv_dim = head_k_dim * num_k_heads * 2 + head_v_dim * num_v_heads

        linear_cfg = LinearAttentionConfig()
        linear_cfg.linear_num_key_heads = num_k_heads
        linear_cfg.linear_num_value_heads = num_v_heads
        linear_cfg.linear_key_head_dim = head_k_dim
        linear_cfg.linear_value_head_dim = head_v_dim
        linear_cfg.linear_conv_kernel_dim = conv_kernel_dim
        linear_cfg.ssm_state_dtype = DataType.TYPE_BF16
        linear_cfg.conv_state_dtype = DataType.TYPE_BF16

        torch.manual_seed(123)
        layers_cp = []
        layers_nocp = []
        for _ in range(num_layers):
            weights = _build_layer_weights(
                hidden_size,
                num_k_heads,
                num_v_heads,
                head_k_dim,
                head_v_dim,
                conv_kernel_dim,
                device,
            )
            m_cp = Qwen3NextGatedDeltaNet(
                linear_cfg, ParallelismConfig(), weights, layernorm_eps=1e-6
            ).to(device)
            m_nocp = Qwen3NextGatedDeltaNet(
                linear_cfg, ParallelismConfig(), weights, layernorm_eps=1e-6
            ).to(device)
            layers_cp.append(m_cp)
            layers_nocp.append(m_nocp)

        for m in layers_cp:
            m.parallelism_config.tp_size = cp_size
            m.parallelism_config.tp_rank = rank

        torch.manual_seed(42)
        full_hidden = torch.randn(total_tokens, hidden_size, device=device, dtype=dtype)

        # ---------------------------------------------------------------
        # 1. Non-CP reference (single-GPU full sequence)
        # ---------------------------------------------------------------
        nocp_inputs = _build_nocp_attn_inputs(
            sequence_lengths, tokens_per_block, device
        )
        nocp_inputs = _add_device_tensors(nocp_inputs, device)

        from rtp_llm.models_py.model_desc.qwen3_next import Qwen3NextMetadata

        nocp_conv_meta = prepare_causal_conv1d_metadata(
            query_start_loc=nocp_inputs.cu_seqlens, device=device
        )
        nocp_meta = Qwen3NextMetadata(prefill_conv1d_meta=nocp_conv_meta)

        with torch.no_grad(), _noop_write_cache_store():
            ref_h = full_hidden
            for layer in layers_nocp:
                kv = MockLayerKVCache(
                    allocate_kv_cache(
                        sequence_lengths,
                        tokens_per_block,
                        num_v_heads,
                        head_v_dim,
                        head_k_dim,
                        conv_kernel_dim,
                        qkv_dim,
                        dtype,
                        device,
                    ),
                    tokens_per_block,
                )
                ref_h = layer(ref_h, None, kv, nocp_inputs, nocp_meta)
        ref_output = ref_h

        # ---------------------------------------------------------------
        # 2. CP zigzag path (real NCCL)
        # ---------------------------------------------------------------
        all_rank_pos = compute_rank_positions(sequence_lengths, cp_size)
        rank_positions = all_rank_pos[rank]
        rank_idx = torch.tensor(rank_positions, device=device)
        local_hidden = full_hidden[rank_idx].contiguous()

        cp_attn_inputs = build_cp_attn_inputs(
            sequence_lengths,
            cp_chunk_lengths,
            cp_size,
            tokens_per_block,
            device=device,
        )
        # cp_test_utils doesn't set kv_cache_kernel_block_id_device
        cp_attn_inputs.kv_cache_kernel_block_id_device = (
            cp_attn_inputs.kv_cache_block_id_device
        )
        cp_attn_inputs = _add_device_tensors(cp_attn_inputs, device)

        cp_meta = _build_cp_metadata(
            sequence_lengths,
            cp_chunk_lengths,
            cp_size,
            rank,
            cp_attn_inputs,
            device,
        )

        with torch.no_grad(), _noop_write_cache_store():
            cp_h = local_hidden
            for layer in layers_cp:
                kv = MockLayerKVCache(
                    allocate_kv_cache(
                        sequence_lengths,
                        tokens_per_block,
                        num_v_heads,
                        head_v_dim,
                        head_k_dim,
                        conv_kernel_dim,
                        qkv_dim,
                        dtype,
                        device,
                    ),
                    tokens_per_block,
                )
                cp_h = layer(cp_h, None, kv, cp_attn_inputs, cp_meta)
        cp_output = cp_h

        # ---------------------------------------------------------------
        # 3. Compare per-token output
        # ---------------------------------------------------------------
        ref_local = ref_output[rank_idx]
        output_ok = torch.allclose(
            cp_output.float(), ref_local.float(), rtol=1e-2, atol=1e-2
        )
        diff = (cp_output.float() - ref_local.float()).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()

        # ---------------------------------------------------------------
        # 4. Verify SSM cache writes (full_h, final_state) match non-CP
        # ---------------------------------------------------------------
        from unittest.mock import patch as mock_patch

        captured_nocp = []
        captured_cp = []

        original_store = store_ssm_state_to_block_map

        def _capture_nocp(h, final_states, *args, **kwargs):
            captured_nocp.append((h.clone(), final_states.clone()))
            return original_store(h, final_states, *args, **kwargs)

        def _capture_cp(h, final_states, *args, **kwargs):
            captured_cp.append((h.clone(), final_states.clone()))
            return original_store(h, final_states, *args, **kwargs)

        STORE_PATH = (
            "rtp_llm.models_py.model_desc.qwen3_next.store_ssm_state_to_block_map"
        )

        with torch.no_grad(), _noop_write_cache_store():
            ref_h2 = full_hidden
            for layer in layers_nocp:
                kv = MockLayerKVCache(
                    allocate_kv_cache(
                        sequence_lengths,
                        tokens_per_block,
                        num_v_heads,
                        head_v_dim,
                        head_k_dim,
                        conv_kernel_dim,
                        qkv_dim,
                        dtype,
                        device,
                    ),
                    tokens_per_block,
                )
                with mock_patch(STORE_PATH, side_effect=_capture_nocp):
                    ref_h2 = layer(ref_h2, None, kv, nocp_inputs, nocp_meta)

        with torch.no_grad(), _noop_write_cache_store():
            cp_h2 = local_hidden
            for layer in layers_cp:
                kv = MockLayerKVCache(
                    allocate_kv_cache(
                        sequence_lengths,
                        tokens_per_block,
                        num_v_heads,
                        head_v_dim,
                        head_k_dim,
                        conv_kernel_dim,
                        qkv_dim,
                        dtype,
                        device,
                    ),
                    tokens_per_block,
                )
                with mock_patch(STORE_PATH, side_effect=_capture_cp):
                    cp_h2 = layer(cp_h2, None, kv, cp_attn_inputs, cp_meta)

        ssm_ok = True
        ssm_max_diff = 0.0
        for layer_idx in range(num_layers):
            nocp_h, nocp_fs = captured_nocp[layer_idx]
            cp_full_h, cp_global_fs = captured_cp[layer_idx]

            h_diff_val = (cp_full_h.float() - nocp_h.float()).abs().max().item()
            fs_diff_val = (cp_global_fs.float() - nocp_fs.float()).abs().max().item()
            ssm_max_diff = max(ssm_max_diff, h_diff_val, fs_diff_val)

            logging.info(
                f"  rank {rank} layer {layer_idx}: "
                f"full_h cp={cp_full_h.shape} nocp={nocp_h.shape} "
                f"h_diff={h_diff_val:.6f} fs_diff={fs_diff_val:.6f}"
            )

            if not (
                torch.allclose(cp_full_h.float(), nocp_h.float(), rtol=1e-2, atol=1e-2)
                and torch.allclose(
                    cp_global_fs.float(), nocp_fs.float(), rtol=1e-2, atol=1e-2
                )
            ):
                ssm_ok = False

        passed = output_ok and ssm_ok

        dist.barrier()
        logging.info(
            f"  rank {rank}: max_diff={max_diff:.6f} mean_diff={mean_diff:.6f} "
            f"ssm_max={ssm_max_diff:.6f} {'PASS' if passed else 'FAIL'}"
        )
        dist.barrier()
        torch.cuda.synchronize()
        dist.destroy_process_group()

        assert passed, (
            f"rank {rank} failed: max_diff={max_diff:.6f} "
            f"mean_diff={mean_diff:.6f} ssm_max={ssm_max_diff:.6f}"
        )

    except Exception as e:
        print(f"Rank {rank} error: {e}")
        import traceback

        traceback.print_exc()
        raise


# ---------------------------------------------------------------------------
# Worker (unaligned: production-faithful padding)
# ---------------------------------------------------------------------------


def _worker_unaligned(
    rank: int,
    world_size: int,
    nccl_port: int,
    sequence_lengths: List[int],
    num_layers: int,
    num_k_heads: int,
    num_v_heads: int,
    head_k_dim: int,
    head_v_dim: int,
    conv_kernel_dim: int,
    tokens_per_block: int,
):
    """Production-faithful unaligned worker: pads sequences/hidden like C++,
    runs CP zigzag, compares only at real (non-pad) token positions."""
    import torch.distributed as dist

    from rtp_llm.models_py.distributed import collective_torch as ct
    from rtp_llm.models_py.distributed.collective_torch import Group
    from rtp_llm.models_py.model_desc.qwen3_next import (
        Qwen3NextGatedDeltaNet,
        Qwen3NextMetadata,
    )
    from rtp_llm.ops import DataType, LinearAttentionConfig, ParallelismConfig

    try:
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = str(nccl_port)
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(rank)

        par = ParallelismConfig()
        par.world_rank = rank
        par.world_size = world_size
        par.local_rank = rank
        par.tp_size = world_size
        par.dp_size = 1

        ct._parallelism_config = par
        ct._initialized = True
        ct._group_map[Group.DP_AND_TP] = dist.group.WORLD

        device = torch.device(f"cuda:{rank}")
        cp_size = world_size
        hidden_size = num_v_heads * head_v_dim
        dtype = torch.bfloat16
        qkv_dim = head_k_dim * num_k_heads * 2 + head_v_dim * num_v_heads

        # Mimic C++ padding: align each sequence to (cp_size * 2 * chunk_size).
        chunk_size = 64
        align = cp_size * 2 * chunk_size
        padded_seq_lengths = [
            ((sl + align - 1) // align) * align for sl in sequence_lengths
        ]
        cp_chunk_lengths = [psl // cp_size for psl in padded_seq_lengths]
        total_orig = sum(sequence_lengths)
        total_padded = sum(padded_seq_lengths)

        linear_cfg = LinearAttentionConfig()
        linear_cfg.linear_num_key_heads = num_k_heads
        linear_cfg.linear_num_value_heads = num_v_heads
        linear_cfg.linear_key_head_dim = head_k_dim
        linear_cfg.linear_value_head_dim = head_v_dim
        linear_cfg.linear_conv_kernel_dim = conv_kernel_dim
        linear_cfg.ssm_state_dtype = DataType.TYPE_BF16
        linear_cfg.conv_state_dtype = DataType.TYPE_BF16

        torch.manual_seed(123)
        layers_cp = []
        layers_nocp = []
        for _ in range(num_layers):
            weights = _build_layer_weights(
                hidden_size,
                num_k_heads,
                num_v_heads,
                head_k_dim,
                head_v_dim,
                conv_kernel_dim,
                device,
            )
            m_cp = Qwen3NextGatedDeltaNet(
                linear_cfg, ParallelismConfig(), weights, layernorm_eps=1e-6
            ).to(device)
            m_nocp = Qwen3NextGatedDeltaNet(
                linear_cfg, ParallelismConfig(), weights, layernorm_eps=1e-6
            ).to(device)
            layers_cp.append(m_cp)
            layers_nocp.append(m_nocp)

        for m in layers_cp:
            m.parallelism_config.tp_size = cp_size
            m.parallelism_config.tp_rank = rank

        torch.manual_seed(42)
        # Original (un-padded) hidden for the non-CP reference.
        full_hidden = torch.randn(total_orig, hidden_size, device=device, dtype=dtype)
        # Padded hidden for CP path: zeros at pad positions (mimics C++).
        padded_full_hidden = torch.zeros(
            total_padded, hidden_size, device=device, dtype=dtype
        )
        src_off = dst_off = 0
        for sl, psl in zip(sequence_lengths, padded_seq_lengths):
            padded_full_hidden[dst_off : dst_off + sl] = full_hidden[
                src_off : src_off + sl
            ]
            src_off += sl
            dst_off += psl

        # ---------------------------------------------------------------
        # 1. Non-CP reference (un-padded full sequence)
        # ---------------------------------------------------------------
        nocp_inputs = _build_nocp_attn_inputs(
            sequence_lengths, tokens_per_block, device
        )
        nocp_inputs = _add_device_tensors(nocp_inputs, device)

        nocp_conv_meta = prepare_causal_conv1d_metadata(
            query_start_loc=nocp_inputs.cu_seqlens, device=device
        )
        nocp_meta = Qwen3NextMetadata(prefill_conv1d_meta=nocp_conv_meta)

        with torch.no_grad(), _noop_write_cache_store():
            ref_h = full_hidden
            for layer in layers_nocp:
                kv = MockLayerKVCache(
                    allocate_kv_cache(
                        sequence_lengths,
                        tokens_per_block,
                        num_v_heads,
                        head_v_dim,
                        head_k_dim,
                        conv_kernel_dim,
                        qkv_dim,
                        dtype,
                        device,
                    ),
                    tokens_per_block,
                )
                ref_h = layer(ref_h, None, kv, nocp_inputs, nocp_meta)
        ref_output = ref_h

        # ---------------------------------------------------------------
        # 2. CP path on padded layout
        # ---------------------------------------------------------------
        # rank's local input = zigzag slice of *padded* full hidden
        all_rank_pos = compute_rank_positions(padded_seq_lengths, cp_size)
        rank_idx = torch.tensor(all_rank_pos[rank], device=device)
        local_hidden = padded_full_hidden[rank_idx].contiguous()

        cp_attn_inputs = _build_cp_attn_inputs_padded(
            sequence_lengths,
            padded_seq_lengths,
            cp_size,
            tokens_per_block,
            device=device,
        )
        cp_attn_inputs = _add_device_tensors(cp_attn_inputs, device)

        cp_meta = _build_cp_metadata(
            sequence_lengths,
            cp_chunk_lengths,
            cp_size,
            rank,
            cp_attn_inputs,
            device,
        )

        with torch.no_grad(), _noop_write_cache_store():
            cp_h = local_hidden
            for layer in layers_cp:
                kv = MockLayerKVCache(
                    allocate_kv_cache(
                        sequence_lengths,
                        tokens_per_block,
                        num_v_heads,
                        head_v_dim,
                        head_k_dim,
                        conv_kernel_dim,
                        qkv_dim,
                        dtype,
                        device,
                    ),
                    tokens_per_block,
                )
                cp_h = layer(cp_h, None, kv, cp_attn_inputs, cp_meta)
        cp_output = cp_h

        # ---------------------------------------------------------------
        # 3. Compare only at real (non-pad) token positions
        # ---------------------------------------------------------------
        real_local_idx, ref_orig_pos = _compute_real_local_info(
            sequence_lengths, padded_seq_lengths, cp_size, rank
        )
        real_local_idx = real_local_idx.to(device)
        ref_orig_pos = ref_orig_pos.to(device)

        # A rank may legitimately hold zero real tokens (e.g. cp=2 + seq=[12]
        # → rank 0 gets all 12 reals as front-front + back-back; rank 1's two
        # halves are entirely in the pad region). Treat empty as trivially OK
        # — output at pad positions is unused by the model anyway.
        if real_local_idx.numel() == 0:
            max_diff = 0.0
            mean_diff = 0.0
            passed = True
        else:
            cp_real = cp_output[real_local_idx].float()
            ref_real = ref_output[ref_orig_pos].float()
            diff = (cp_real - ref_real).abs()
            max_diff = diff.max().item()
            mean_diff = diff.mean().item()
            passed = torch.allclose(cp_real, ref_real, rtol=1e-2, atol=1e-2)

        dist.barrier()
        logging.info(
            f"  rank {rank} unaligned: max_diff={max_diff:.6f} "
            f"mean_diff={mean_diff:.6f} {'PASS' if passed else 'FAIL'}"
        )
        dist.barrier()
        torch.cuda.synchronize()
        dist.destroy_process_group()

        assert passed, (
            f"rank {rank} unaligned failed: max_diff={max_diff:.6f} "
            f"mean_diff={mean_diff:.6f}"
        )

    except Exception as e:
        print(f"Rank {rank} error: {e}")
        import traceback

        traceback.print_exc()
        raise


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

# exchange_conv_context only supports cp_size==2; the production scenario for
# the zigzag path is also cp=2, so default GPU_COUNT to 2.
GPU_COUNT = int(os.environ.get("GPU_COUNT", "2"))


@unittest.skipUnless(
    torch.cuda.is_available() and torch.cuda.device_count() >= GPU_COUNT,
    f"Requires >= {GPU_COUNT} GPUs",
)
class TestGDNCPZigzagPrefill(unittest.TestCase):
    """Verify CP zigzag GatedDeltaNet prefill matches non-CP reference (real NCCL)."""

    def setUp(self):
        try:
            mp.set_start_method("spawn", force=True)
        except RuntimeError:
            pass
        self.port_manager = PortManager()

    def _run_test(
        self,
        cp_size: int,
        sequence_lengths: List[int],
        num_layers: int = 1,
        num_k_heads: int = 16,
        num_v_heads: int = 32,
        head_k_dim: int = 128,
        head_v_dim: int = 128,
        conv_kernel_dim: int = 4,
        tokens_per_block: int = 16,
    ):
        # zigzag splits each sequence into 2*cp halves, so seq_len must be
        # divisible by 2*cp. C++ would pad otherwise; tests use clean inputs.
        assert all(
            sl % (cp_size * 2) == 0 for sl in sequence_lengths
        ), f"Seq lengths must be divisible by cp_size*2={cp_size * 2}"

        ports, locks = self.port_manager.get_consecutive_ports(1)
        nccl_port = ports[0]

        try:
            processes = []
            for rank in range(cp_size):
                p = mp.Process(
                    target=_worker,
                    args=(
                        rank,
                        cp_size,
                        nccl_port,
                        sequence_lengths,
                        num_layers,
                        num_k_heads,
                        num_v_heads,
                        head_k_dim,
                        head_v_dim,
                        conv_kernel_dim,
                        tokens_per_block,
                    ),
                    name=f"rank-{rank}",
                )
                p.start()
                processes.append(p)

            for p in processes:
                p.join(timeout=300)
                if p.exitcode != 0:
                    raise RuntimeError(
                        f"Process {p.name} exited with code {p.exitcode}"
                    )
        finally:
            for lock in locks:
                lock.__exit__(None, None, None)

    # --- Single layer ---

    def test_single_layer_single_seq(self):
        self._run_test(cp_size=GPU_COUNT, sequence_lengths=[256], num_layers=1)

    def test_single_layer_multi_batch(self):
        self._run_test(cp_size=GPU_COUNT, sequence_lengths=[256, 512], num_layers=1)

    # --- Multi layer ---

    def test_multi_layer_single_seq(self):
        self._run_test(cp_size=GPU_COUNT, sequence_lengths=[256], num_layers=4)

    def test_multi_layer_multi_batch(self):
        self._run_test(cp_size=GPU_COUNT, sequence_lengths=[256, 512], num_layers=4)

    # --- Larger sequence (production-like) ---

    def test_single_layer_8k(self):
        self._run_test(cp_size=GPU_COUNT, sequence_lengths=[8192], num_layers=1)

    def test_single_layer_64k(self):
        self._run_test(cp_size=GPU_COUNT, sequence_lengths=[64 * 1024], num_layers=1)

    # --- Non chunk-aligned (padding/mask path; production-faithful) ---

    def _run_test_unaligned(
        self,
        cp_size: int,
        sequence_lengths: List[int],
        num_layers: int = 1,
        num_k_heads: int = 16,
        num_v_heads: int = 32,
        head_k_dim: int = 128,
        head_v_dim: int = 128,
        conv_kernel_dim: int = 4,
        tokens_per_block: int = 16,
    ):
        # Only require cp_size*2 divisibility (zigzag pairing); sequences may be
        # smaller than cp_size*2*chunk_size — the worker will pad like C++.
        assert all(
            sl % (cp_size * 2) == 0 for sl in sequence_lengths
        ), f"Seq lengths must be divisible by cp_size*2={cp_size * 2}"

        ports, locks = self.port_manager.get_consecutive_ports(1)
        nccl_port = ports[0]

        try:
            processes = []
            for rank in range(cp_size):
                p = mp.Process(
                    target=_worker_unaligned,
                    args=(
                        rank,
                        cp_size,
                        nccl_port,
                        sequence_lengths,
                        num_layers,
                        num_k_heads,
                        num_v_heads,
                        head_k_dim,
                        head_v_dim,
                        conv_kernel_dim,
                        tokens_per_block,
                    ),
                    name=f"rank-{rank}",
                )
                p.start()
                processes.append(p)

            for p in processes:
                p.join(timeout=300)
                if p.exitcode != 0:
                    raise RuntimeError(
                        f"Process {p.name} exited with code {p.exitcode}"
                    )
        finally:
            for lock in locks:
                lock.__exit__(None, None, None)

    # cp=2: align=256. Below cases force C++-style padding + need_pad=True path.

    def test_unaligned_short_nocrash(self):

        self._run_test_unaligned(cp_size=2, sequence_lengths=[12], num_layers=1)

    def test_unaligned_medium(self):
        # 260 → padded to 512. Both ranks hold real tokens — full numeric
        # correctness coverage of the need_pad path.
        self._run_test_unaligned(cp_size=2, sequence_lengths=[260], num_layers=1)

    def test_unaligned_mixed_batch(self):
        # Mixed pad ratios in one batch (heavy + light pad).
        self._run_test_unaligned(cp_size=2, sequence_lengths=[12, 260], num_layers=1)

    def test_unaligned_multi_layer(self):
        # Cross-layer state propagation in unaligned regime.
        self._run_test_unaligned(cp_size=2, sequence_lengths=[260], num_layers=4)


if __name__ == "__main__":
    unittest.main()
