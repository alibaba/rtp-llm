"""Regression tests for BF16 vLLM compressor cyclic state-cache lookup."""

from __future__ import annotations

import unittest

import torch

from rtp_llm.models_py.modules.dsv4._compressor_vllm_triton import (
    run_fused_compress_kv_write_bf16,
)


DEVICE = "cuda"
STATE_BLOCK_SIZE = 256
ROPE_HEAD_DIM = 64
RMS_EPS = 1e-6


def _cos_sin_cache(rows: int) -> torch.Tensor:
    cache = torch.zeros(rows, ROPE_HEAD_DIM, dtype=torch.float32, device=DEVICE)
    cache[:, : ROPE_HEAD_DIM // 2] = 1.0
    return cache


def _run_decode_boundary(head_dim: int, compress_ratio: int, overlap: bool) -> None:
    """Run one long-position decode boundary through the BF16 fused writer.

    Production BF16 vLLM state pools keep only two 256-token blocks per
    request. Position 4095 maps to logical block 15, which must wrap to table
    column 1. A direct ``pos // block_size`` lookup reads beyond the two-entry
    table and reproduced Xid 13 online.
    """

    position = 4095
    state_width = (1 + int(overlap)) * head_dim
    state_cache = torch.zeros(
        3,
        STATE_BLOCK_SIZE,
        2 * state_width,
        dtype=torch.float32,
        device=DEVICE,
    )
    block_table = torch.tensor([[1, 2]], dtype=torch.int32, device=DEVICE)

    # Fill the cyclic block for logical block 15 -> table col 1 -> block id 2.
    state_cache[2, :, :state_width] = 1.0
    state_cache[2, :, state_width:] = 0.0

    token_to_req = torch.zeros(1, dtype=torch.int32, device=DEVICE)
    positions = torch.tensor([position], dtype=torch.long, device=DEVICE)
    slot_mapping = torch.zeros(1, dtype=torch.long, device=DEVICE)
    kv_slot_mapping = torch.zeros(1, dtype=torch.long, device=DEVICE)
    boundary_token_indices = torch.zeros(1, dtype=torch.long, device=DEVICE)
    kv_raw = torch.empty(0, state_width, dtype=torch.float32, device=DEVICE)
    score_raw = torch.empty(0, state_width, dtype=torch.float32, device=DEVICE)
    ape = torch.zeros(compress_ratio, state_width, dtype=torch.float32, device=DEVICE)
    rms_norm_weight = torch.ones(head_dim, dtype=torch.bfloat16, device=DEVICE)
    kv_cache = torch.full(
        (1, head_dim), float("nan"), dtype=torch.bfloat16, device=DEVICE
    )

    run_fused_compress_kv_write_bf16(
        state_cache,
        token_to_req,
        positions,
        slot_mapping,
        block_table,
        rms_norm_weight,
        RMS_EPS,
        _cos_sin_cache(position + 1),
        kv_cache,
        kv_slot_mapping,
        kv_raw,
        score_raw,
        ape,
        0,
        disable_raw_path=True,
        boundary_token_indices=boundary_token_indices,
        head_dim=head_dim,
        rope_head_dim=ROPE_HEAD_DIM,
        compress_ratio=compress_ratio,
        overlap=overlap,
    )
    torch.cuda.synchronize()

    out = kv_cache[0].float()
    assert torch.isfinite(out).all().item()
    assert torch.allclose(out, torch.ones_like(out), atol=5e-3, rtol=5e-3)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class BF16CompressorCyclicStateBlocksTest(unittest.TestCase):
    def test_ratio4_decode_uses_cyclic_state_blocks(self) -> None:
        _run_decode_boundary(head_dim=512, compress_ratio=4, overlap=True)

    def test_ratio128_decode_uses_cyclic_state_blocks(self) -> None:
        _run_decode_boundary(head_dim=512, compress_ratio=128, overlap=False)


if __name__ == "__main__":
    unittest.main()
