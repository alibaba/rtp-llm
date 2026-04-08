# -*- coding: utf-8 -*-
"""
Tests for Qwen3.5 GDN optimized kernels:
  1. fused_recurrent_gated_delta_rule_with_gating vs split (gating + recurrent)
  2. FlashInfer decode vs FLA Triton decode (SM90+ only)
  3. FlashInfer prefill vs FLA Triton prefill (SM90+ only)
  4. store_final_state_only_to_block_map correctness
  5. FlashInfer decode cross-block boundary state copy
"""

import logging
import math
import random

import pytest
import torch
import torch.nn.functional as F

pytestmark = [pytest.mark.gpu(type="H20")]

logging.basicConfig(
    level="INFO",
    format="[%(asctime)s][%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    force=True,
)
log = logging.getLogger("test_gdn_opt")

# Qwen3.5 config
H = 16      # Q/K heads
HV = 32     # V heads
D = 128     # head dim (K=V=128)
CONV_KERNEL = 4
CHUNK_SIZE = 64


def _assert_close(name: str, ref: torch.Tensor, tri: torch.Tensor, atol: float = 0.005):
    diff = (ref - tri).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    passed = max_diff < atol
    status = "PASS" if passed else "FAIL"
    log.info(f"  {name}: max_diff={max_diff:.6f} mean_diff={mean_diff:.6f} [{status}]")
    assert passed, f"{name}: max_diff={max_diff:.6f} > atol={atol}"


def _recurrent_gated_delta_rule_ref(q, k, v, g, beta, scale=None, initial_state=None):
    """Python reference: loop-based GDN recurrence."""
    q, k, v, beta, g = map(
        lambda x: x.transpose(1, 2).contiguous().to(torch.float32), [q, k, v, beta, g]
    )
    B, HH, T, K = q.shape
    V = v.shape[-1]
    o = torch.zeros(B, HH, T, V, device=v.device, dtype=v.dtype)
    h = torch.zeros(B, HH, K, V, device=v.device, dtype=v.dtype)
    if initial_state is not None:
        h = initial_state.clone()
    if scale is None:
        scale = K ** -0.5
    q = q * scale
    for i in range(T):
        h = h.clone() * g[:, :, i].exp()[..., None, None]
        b_v = v[:, :, i].clone() - (h.clone() * k[:, :, i, :, None]).sum(-2)
        b_v = b_v * beta[:, :, i, ..., None]
        h = h.clone() + k[:, :, i, :, None] * b_v[:, :, None, :]
        o[:, :, i] = torch.einsum("bhd,bhdm->bhm", q[:, :, i], h)
    return o.transpose(1, 2).contiguous(), h


def _compute_gating_ref(A_log, a, dt_bias, b):
    """Reference gating computation matching fused_gdn_gating."""
    x = a.float() + dt_bias.float()
    softplus_x = torch.where(x > 20.0, x, torch.log1p(torch.exp(x)))
    g = -A_log.float().exp() * softplus_x
    beta = b.float().sigmoid()
    return g, beta


def _make_block_map_and_cache(B, seq_lengths, spb=128, device="cuda"):
    """Create block_map and ssm_cache for testing."""
    block_nums = [math.ceil(s / spb) + 1 for s in seq_lengths]
    total_blocks = sum(block_nums) + 1
    block_map = torch.zeros(B, max(block_nums) + 2, dtype=torch.int32, device=device)
    offset = 1
    for i in range(B):
        block_map[i, :block_nums[i]] = torch.arange(offset, offset + block_nums[i], dtype=torch.int32)
        offset += block_nums[i]
    ssm_cache = torch.randn(total_blocks, HV, D, D, dtype=torch.float32, device=device) * 0.01
    return block_map, ssm_cache, total_blocks


# ============================================================
# Test 1: Fused gating + recurrent vs split (gating then recurrent)
# ============================================================
@pytest.mark.parametrize("B,S", [
    (1, 1), (4, 1), (16, 1), (4, 2), (8, 4),
])
def test_fused_gating_recurrent_vs_split(B, S):
    """Verify fused_recurrent_gated_delta_rule_with_gating matches split gating+recurrent."""
    from rtp_llm.models_py.triton_kernels.fla import fused_recurrent_gated_delta_rule
    from rtp_llm.models_py.triton_kernels.fla.fused_recurrent import (
        fused_recurrent_gated_delta_rule_with_gating,
    )
    from rtp_llm.models_py.triton_kernels.fla.gdn_gating import fused_gdn_gating

    log.info(f"test_fused_gating_recurrent_vs_split: B={B} S={S}")
    device = "cuda"
    torch.manual_seed(42)

    q = torch.randn(B, S, H, D, device=device, dtype=torch.bfloat16)
    k = torch.randn(B, S, H, D, device=device, dtype=torch.bfloat16)
    v = torch.randn(B, S, HV, D, device=device, dtype=torch.bfloat16)
    A_log = torch.randn(HV, device=device, dtype=torch.float32) * 0.1
    dt_bias = torch.randn(HV, device=device, dtype=torch.bfloat16)
    a_raw = torch.randn(B, S, HV, device=device, dtype=torch.bfloat16)
    b_raw = torch.randn(B, S, HV, device=device, dtype=torch.bfloat16)

    spb = 128
    seq_lengths = [random.randint(10, 512) for _ in range(B)]
    block_map, ssm_cache, total_blocks = _make_block_map_and_cache(B, seq_lengths, spb, device)
    h0 = torch.randn(B, HV, D, D, device=device, dtype=torch.float32) * 0.01
    load_offsets = [(s - 2) // spb for s in seq_lengths]
    for i in range(B):
        ssm_cache[int(block_map[i, load_offsets[i]])] = h0[i]
    ssm_cache_ref = ssm_cache.clone()
    seq_lengths_t = torch.tensor(seq_lengths, dtype=torch.int32, device=device)

    # Split path: gating then recurrent
    a_2d = a_raw[:, 0, :] if S == 1 else a_raw.reshape(B * S, HV)
    b_2d = b_raw[:, 0, :] if S == 1 else b_raw.reshape(B * S, HV)
    if S == 1:
        g, beta = fused_gdn_gating(A_log, a_2d, b_2d, dt_bias)
        g = g.view(B, S, HV)
        beta = beta.view(B, S, HV)
    else:
        # For multi-token, compute gating per token
        gs, betas = [], []
        for t in range(S):
            g_t, beta_t = fused_gdn_gating(A_log, a_raw[:, t, :], b_raw[:, t, :], dt_bias)
            gs.append(g_t.squeeze(0))
            betas.append(beta_t.squeeze(0))
        g = torch.stack(gs, dim=1)
        beta = torch.stack(betas, dim=1)

    ref_out, _ = fused_recurrent_gated_delta_rule(
        q=q.clone(), k=k.clone(), v=v.clone(),
        g=g, beta=beta, scale=None,
        initial_state=ssm_cache_ref,
        inplace_final_state=True,
        block_map=block_map,
        seq_size_per_block=spb,
        sequence_lengths=seq_lengths_t,
        use_qk_l2norm_in_kernel=True,
    )

    # Fused path
    tri_out, _ = fused_recurrent_gated_delta_rule_with_gating(
        q=q.clone(), k=k.clone(), v=v.clone(),
        A_log=A_log, a=a_raw.clone(), dt_bias=dt_bias, b=b_raw.clone(),
        scale=None,
        initial_state=ssm_cache,
        inplace_final_state=True,
        block_map=block_map,
        seq_size_per_block=spb,
        sequence_lengths=seq_lengths_t,
        use_qk_l2norm_in_kernel=True,
    )

    _assert_close("output", ref_out, tri_out, atol=0.01)


# ============================================================
# Test 2: FlashInfer decode vs FLA Triton decode
# ============================================================
def _skip_if_no_flashinfer():
    try:
        from rtp_llm.models_py.utils.arch import is_flashinfer_gdn_available
        if not is_flashinfer_gdn_available():
            pytest.skip("FlashInfer GDN not available (requires SM90+)")
    except Exception:
        pytest.skip("FlashInfer GDN import failed")


@pytest.mark.parametrize("B", [1, 4, 16, 64])
def test_flashinfer_decode_vs_fla_triton(B):
    """Verify FlashInfer decode matches FLA Triton decode output."""
    _skip_if_no_flashinfer()
    from rtp_llm.models_py.triton_kernels.fla import fused_recurrent_gated_delta_rule
    from rtp_llm.models_py.triton_kernels.fla.gdn_gating import fused_gdn_gating
    from rtp_llm.models_py.triton_kernels.fla.flashinfer_gdn import flashinfer_gdn_decode

    log.info(f"test_flashinfer_decode_vs_fla_triton: B={B}")
    device = "cuda"
    torch.manual_seed(42)
    S = 1

    q = torch.randn(B, S, H, D, device=device, dtype=torch.bfloat16)
    k = torch.randn(B, S, H, D, device=device, dtype=torch.bfloat16)
    v = torch.randn(B, S, HV, D, device=device, dtype=torch.bfloat16)
    A_log = torch.randn(HV, device=device, dtype=torch.float32) * 0.1
    dt_bias = torch.randn(HV, device=device, dtype=torch.bfloat16)
    a = torch.randn(B, HV, device=device, dtype=torch.bfloat16)
    b = torch.randn(B, HV, device=device, dtype=torch.bfloat16)

    spb = 128
    seq_lengths = [random.randint(10, 512) for _ in range(B)]
    block_map, ssm_cache, total_blocks = _make_block_map_and_cache(B, seq_lengths, spb, device)
    h0 = torch.randn(B, HV, D, D, device=device, dtype=torch.float32) * 0.01
    load_offsets = [(s - 2) // spb for s in seq_lengths]
    for i in range(B):
        ssm_cache[int(block_map[i, load_offsets[i]])] = h0[i]
    ssm_cache_fi = ssm_cache.clone()
    seq_lengths_t = torch.tensor(seq_lengths, dtype=torch.int32, device=device)

    # FLA Triton path
    g, beta = fused_gdn_gating(A_log, a.clone(), b.clone(), dt_bias)
    g = g.view(B, S, HV)
    beta = beta.view(B, S, HV)
    ref_out, _ = fused_recurrent_gated_delta_rule(
        q=q.clone(), k=k.clone(), v=v.clone(),
        g=g, beta=beta, scale=None,
        initial_state=ssm_cache,
        inplace_final_state=True,
        block_map=block_map,
        seq_size_per_block=spb,
        sequence_lengths=seq_lengths_t,
        use_qk_l2norm_in_kernel=True,
    )

    # FlashInfer path
    fi_out = flashinfer_gdn_decode(
        q=q.clone(), k=k.clone(), v=v.clone(),
        A_log=A_log, a=a.clone(), dt_bias=dt_bias, b=b.clone(),
        ssm_states=ssm_cache_fi,
        block_map=block_map,
        sequence_lengths=seq_lengths_t,
        seq_size_per_block=spb,
    )

    ref_flat = ref_out.reshape(B, HV, D)
    fi_flat = fi_out.reshape(B, HV, D)
    _assert_close("decode_output", ref_flat, fi_flat, atol=0.01)


# ============================================================
# Test 3: FlashInfer prefill vs FLA Triton prefill
# ============================================================
@pytest.mark.parametrize("B,L", [(1, 128), (2, 256), (4, 512)])
def test_flashinfer_prefill_vs_fla_triton(B, L):
    """Verify FlashInfer prefill matches FLA Triton prefill output."""
    _skip_if_no_flashinfer()
    from rtp_llm.models_py.triton_kernels.fla import chunk_gated_delta_rule
    from rtp_llm.models_py.triton_kernels.fla.flashinfer_gdn import flashinfer_gdn_prefill

    log.info(f"test_flashinfer_prefill_vs_fla_triton: B={B} L={L}")
    device = "cuda"
    torch.manual_seed(42)
    total_tokens = B * L

    q = torch.randn(1, total_tokens, H, D, device=device, dtype=torch.bfloat16)
    k = torch.randn(1, total_tokens, H, D, device=device, dtype=torch.bfloat16)
    v = torch.randn(1, total_tokens, HV, D, device=device, dtype=torch.bfloat16)
    g = torch.randn(1, total_tokens, HV, device=device, dtype=torch.float32) * 0.1
    beta = torch.sigmoid(torch.randn(1, total_tokens, HV, device=device, dtype=torch.float32))
    initial_state = torch.randn(B, HV, D, D, device=device, dtype=torch.float32) * 0.01
    cu_seqlens = torch.arange(0, (B + 1) * L, L, device=device, dtype=torch.long)

    # FLA Triton
    ref_out, _, ref_final = chunk_gated_delta_rule(
        q.clone(), k.clone(), v.clone(), g.clone(), beta.clone(),
        initial_state=initial_state.clone(),
        output_final_state=True,
        cu_seqlens=cu_seqlens,
        use_qk_l2norm_in_kernel=True,
    )

    # FlashInfer
    fi_out, fi_final = flashinfer_gdn_prefill(
        q=q.clone().squeeze(0),
        k=k.clone().squeeze(0),
        v=v.clone().squeeze(0),
        g=g.clone().squeeze(0),
        beta=beta.clone().squeeze(0),
        initial_state=initial_state.clone(),
        cu_seqlens=cu_seqlens,
        use_qk_l2norm=True,
        output_final_state=True,
    )

    ref_flat = ref_out.squeeze(0)  # [total_tokens, HV, D]
    _assert_close("prefill_output", ref_flat, fi_out, atol=0.02)
    if ref_final is not None and fi_final is not None:
        _assert_close("prefill_final_state", ref_final, fi_final, atol=0.02)


# ============================================================
# Test 4: store_final_state_only_to_block_map
# ============================================================
@pytest.mark.parametrize("B", [1, 4, 8])
def test_store_final_state_only(B):
    """Verify store_final_state_only_to_block_map writes correct block."""
    from rtp_llm.models_py.triton_kernels.fla.block import store_final_state_only_to_block_map

    log.info(f"test_store_final_state_only: B={B}")
    device = "cuda"
    torch.manual_seed(42)
    spb = 64

    prefix_lengths = torch.tensor([random.randint(0, 256) for _ in range(B)], dtype=torch.int32, device=device)
    input_lengths = torch.tensor([random.randint(64, 256) for _ in range(B)], dtype=torch.int32, device=device)
    cu_seqlens = torch.zeros(B + 1, dtype=torch.long, device=device)
    for i in range(B):
        cu_seqlens[i + 1] = cu_seqlens[i] + input_lengths[i]

    total_seq_per_batch = prefix_lengths + input_lengths
    max_blocks = max((total_seq_per_batch - 1) // spb + 2).item()
    total_blocks = B * max_blocks + 1
    block_map = torch.zeros(B, max_blocks, dtype=torch.int32, device=device)
    offset = 1
    for i in range(B):
        n_blocks = (total_seq_per_batch[i].item() - 1) // spb + 2
        block_map[i, :n_blocks] = torch.arange(offset, offset + n_blocks, dtype=torch.int32)
        offset += n_blocks

    ssm_states = torch.zeros(total_blocks, HV, D, D, dtype=torch.float32, device=device)
    final_states = torch.randn(B, HV, D, D, dtype=torch.float32, device=device)

    store_final_state_only_to_block_map(
        final_states, prefix_lengths, cu_seqlens, block_map, ssm_states, spb
    )

    # Verify: final_state[i] should be at ssm_states[block_map[i, dest_pos]]
    for i in range(B):
        dest_pos = (prefix_lengths[i].item() + input_lengths[i].item() - 1) // spb
        block_idx = block_map[i, dest_pos].item()
        if block_idx > 0:
            stored = ssm_states[block_idx]
            _assert_close(f"batch_{i}_final_state", final_states[i], stored, atol=1e-6)


# ============================================================
# Test 5: FlashInfer decode cross-block boundary
# ============================================================
def test_flashinfer_decode_cross_block_boundary():
    """Verify cross-block state copy when seq_len % spb == 0."""
    _skip_if_no_flashinfer()
    from rtp_llm.models_py.triton_kernels.fla.flashinfer_gdn import flashinfer_gdn_decode

    log.info("test_flashinfer_decode_cross_block_boundary")
    device = "cuda"
    torch.manual_seed(42)
    B = 4
    spb = 64

    # Force some sequences to be exactly at block boundary
    seq_lengths = [spb, spb * 2, spb + 1, spb * 3]  # first two cross boundary
    seq_lengths_t = torch.tensor(seq_lengths, dtype=torch.int32, device=device)

    q = torch.randn(B, 1, H, D, device=device, dtype=torch.bfloat16)
    k = torch.randn(B, 1, H, D, device=device, dtype=torch.bfloat16)
    v = torch.randn(B, 1, HV, D, device=device, dtype=torch.bfloat16)
    A_log = torch.randn(HV, device=device, dtype=torch.float32) * 0.1
    dt_bias = torch.randn(HV, device=device, dtype=torch.bfloat16)
    a = torch.randn(B, HV, device=device, dtype=torch.bfloat16)
    b = torch.randn(B, HV, device=device, dtype=torch.bfloat16)

    block_map, ssm_cache, total_blocks = _make_block_map_and_cache(B, seq_lengths, spb, device)
    h0 = torch.randn(B, HV, D, D, device=device, dtype=torch.float32) * 0.01
    for i in range(B):
        read_offset = (seq_lengths[i] - 2) // spb
        ssm_cache[int(block_map[i, read_offset])] = h0[i]

    flashinfer_gdn_decode(
        q=q, k=k, v=v,
        A_log=A_log, a=a, dt_bias=dt_bias, b=b,
        ssm_states=ssm_cache,
        block_map=block_map,
        sequence_lengths=seq_lengths_t,
        seq_size_per_block=spb,
    )

    # For boundary-crossing sequences (seq_len % spb == 0):
    # state should be copied from read block to write block
    for i in range(B):
        if seq_lengths[i] % spb == 0:
            read_offset = (seq_lengths[i] - 1) // spb
            write_offset = seq_lengths[i] // spb
            read_block = int(block_map[i, read_offset])
            write_block = int(block_map[i, write_offset])
            if write_block > 0:
                _assert_close(
                    f"cross_block_copy_batch_{i}",
                    ssm_cache[read_block],
                    ssm_cache[write_block],
                    atol=1e-6,
                )
                log.info(f"  batch {i}: cross-block copy verified (block {read_block} → {write_block})")


if __name__ == "__main__":
    torch.manual_seed(42)
    log.info("=== Test 1: Fused Gating + Recurrent ===")
    for B, S in [(1, 1), (4, 1), (16, 1)]:
        test_fused_gating_recurrent_vs_split(B, S)

    log.info("\n=== Test 2: FlashInfer Decode vs FLA Triton ===")
    for B in [1, 4, 16]:
        test_flashinfer_decode_vs_fla_triton(B)

    log.info("\n=== Test 3: FlashInfer Prefill vs FLA Triton ===")
    for B, L in [(1, 128), (2, 256)]:
        test_flashinfer_prefill_vs_fla_triton(B, L)

    log.info("\n=== Test 4: store_final_state_only ===")
    for B in [1, 4, 8]:
        test_store_final_state_only(B)

    log.info("\n=== Test 5: Cross-Block Boundary ===")
    test_flashinfer_decode_cross_block_boundary()

    log.info("\nAll tests passed!")
