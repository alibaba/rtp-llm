"""Numerical regression tests for the GLM-5.3-Flash KPool writer."""

from __future__ import annotations

import torch

from rtp_llm.models_py.modules.dsv4.fp8._compressor_vllm_triton import (
    run_fused_compress_kv_write,
    run_save_partial_states,
)
from rtp_llm.models_py.modules.hybrid.indexer_compressor import (
    compress_indexer_projection_reference,
)


HEAD_DIM = 128
COMPRESS_RATIO = 4
STATE_RING_ENTRIES = 4
KV_ENTRIES_PER_BLOCK = 32


def _state_slots(positions: torch.Tensor) -> torch.Tensor:
    # Physical block zero is reserved. This test owns physical block one.
    return STATE_RING_ENTRIES + torch.remainder(positions, STATE_RING_ENTRIES)


def _kv_slots(positions: torch.Tensor) -> torch.Tensor:
    boundary = torch.remainder(positions + 1, COMPRESS_RATIO) == 0
    pooled_id = torch.div(positions, COMPRESS_RATIO, rounding_mode="floor")
    slots = KV_ENTRIES_PER_BLOCK + pooled_id
    return torch.where(boundary, slots, torch.full_like(slots, -1))


def _launch(
    key: torch.Tensor,
    score: torch.Tensor,
    ape: torch.Tensor,
    positions: torch.Tensor,
    *,
    state_cache: torch.Tensor,
    kv_cache: torch.Tensor,
    disable_raw_path: bool,
) -> None:
    state_slots = _state_slots(positions)
    run_save_partial_states(
        key,
        score,
        ape,
        positions,
        state_cache,
        state_slots,
        compress_ratio=COMPRESS_RATIO,
    )
    run_fused_compress_kv_write(
        state_cache,
        torch.zeros_like(positions, dtype=torch.int32),
        positions,
        state_slots,
        torch.tensor([[1]], dtype=torch.int32, device="cuda"),
        torch.ones(HEAD_DIM, dtype=torch.bfloat16, device="cuda"),
        1e-6,
        torch.empty((1, 0), dtype=torch.float32, device="cuda"),
        kv_cache,
        _kv_slots(positions),
        key,
        score,
        ape,
        int(positions[0].item()),
        disable_raw_path=disable_raw_path,
        head_dim=HEAD_DIM,
        rope_head_dim=0,
        compress_ratio=COMPRESS_RATIO,
        overlap=False,
        state_tokens_per_block=128,
        kpool_mode=True,
    )


def _dequantized_entry(kv_cache: torch.Tensor, pooled_id: int) -> torch.Tensor:
    block = kv_cache[1].flatten()
    value_offset = pooled_id * HEAD_DIM
    value = (
        block[value_offset : value_offset + HEAD_DIM].view(torch.float8_e4m3fn).float()
    )
    scale_offset = KV_ENTRIES_PER_BLOCK * HEAD_DIM + pooled_id * 4
    scale = block[scale_offset : scale_offset + 4].view(torch.float32)[0]
    return value * scale


def _entry_scale(kv_cache: torch.Tensor, pooled_id: int) -> torch.Tensor:
    block = kv_cache[1].flatten()
    scale_offset = KV_ENTRIES_PER_BLOCK * HEAD_DIM + pooled_id * 4
    return block[scale_offset : scale_offset + 4].view(torch.float32)[0]


def _assert_matches_reference(
    key: torch.Tensor,
    score: torch.Tensor,
    ape: torch.Tensor,
    kv_cache: torch.Tensor,
) -> None:
    expected, _ = compress_indexer_projection_reference(
        key, score, ape, compress_ratio=COMPRESS_RATIO, overlap=0
    )
    assert expected.shape[0] == key.shape[0] // COMPRESS_RATIO
    for pooled_id in range(expected.shape[0]):
        actual = _dequantized_entry(kv_cache, pooled_id)
        # The writer rounds the Hadamard output to BF16 and stores E4M3 with a
        # power-of-two scale. Validate that scale independently, then allow one
        # E4M3 ULP. Near the largest finite mantissa an ULP is 32 quantized
        # units, rather than one scale unit.
        scale = _entry_scale(kv_cache, pooled_id)
        expected_absmax = expected[pooled_id].float().abs().max().clamp_min(1e-4)
        expected_scale = torch.exp2(torch.ceil(torch.log2(expected_absmax / 448.0)))
        torch.testing.assert_close(scale, expected_scale, rtol=0.0, atol=0.0)
        assert torch.isfinite(scale).item() and scale.item() > 0
        torch.testing.assert_close(
            actual,
            expected[pooled_id].float(),
            rtol=0.0,
            atol=32.0 * float(scale.item()),
        )


def test_glm53_kpool_prefill_math() -> None:
    generator = torch.Generator(device="cuda").manual_seed(20260828)
    key = torch.randn(
        8, HEAD_DIM, generator=generator, dtype=torch.float32, device="cuda"
    )
    score = torch.randn(
        8, HEAD_DIM, generator=generator, dtype=torch.float32, device="cuda"
    )
    ape = torch.randn(
        COMPRESS_RATIO,
        HEAD_DIM,
        generator=generator,
        dtype=torch.float32,
        device="cuda",
    )
    state_cache = torch.zeros(
        2,
        STATE_RING_ENTRIES,
        2 * HEAD_DIM,
        dtype=torch.float32,
        device="cuda",
    )
    kv_cache = torch.zeros(
        2, KV_ENTRIES_PER_BLOCK, HEAD_DIM + 4, dtype=torch.uint8, device="cuda"
    )
    positions = torch.arange(8, dtype=torch.int64, device="cuda")

    _launch(
        key,
        score,
        ape,
        positions,
        state_cache=state_cache,
        kv_cache=kv_cache,
        disable_raw_path=False,
    )
    torch.cuda.synchronize()
    _assert_matches_reference(key, score, ape, kv_cache)
    block = kv_cache[1].flatten()
    unused_values = block[2 * HEAD_DIM : KV_ENTRIES_PER_BLOCK * HEAD_DIM]
    unused_scales = block[KV_ENTRIES_PER_BLOCK * HEAD_DIM + 2 * 4 :]
    assert not unused_values.any().item(), "partial group must not write K bytes"
    assert not unused_scales.any().item(), "partial group must not write scales"


def test_glm53_kpool_decode_reads_ring_state() -> None:
    generator = torch.Generator(device="cuda").manual_seed(53)
    key = torch.randn(
        COMPRESS_RATIO,
        HEAD_DIM,
        generator=generator,
        dtype=torch.float32,
        device="cuda",
    )
    score = torch.randn(
        COMPRESS_RATIO,
        HEAD_DIM,
        generator=generator,
        dtype=torch.float32,
        device="cuda",
    )
    ape = torch.randn(
        COMPRESS_RATIO,
        HEAD_DIM,
        generator=generator,
        dtype=torch.float32,
        device="cuda",
    )
    state_cache = torch.zeros(
        2,
        STATE_RING_ENTRIES,
        2 * HEAD_DIM,
        dtype=torch.float32,
        device="cuda",
    )
    kv_cache = torch.zeros(
        2, KV_ENTRIES_PER_BLOCK, HEAD_DIM + 4, dtype=torch.uint8, device="cuda"
    )

    # Model a three-token prefix followed by one decode token. The final
    # launch disables the raw path, so all four inputs must be reconstructed
    # from the FP32 ring state.
    prefix_positions = torch.arange(3, dtype=torch.int64, device="cuda")
    run_save_partial_states(
        key[:3],
        score[:3],
        ape,
        prefix_positions,
        state_cache,
        _state_slots(prefix_positions),
        compress_ratio=COMPRESS_RATIO,
    )
    decode_position = torch.tensor([3], dtype=torch.int64, device="cuda")
    _launch(
        key[3:],
        score[3:],
        ape,
        decode_position,
        state_cache=state_cache,
        kv_cache=kv_cache,
        disable_raw_path=True,
    )
    torch.cuda.synchronize()
    _assert_matches_reference(key, score, ape, kv_cache)


if __name__ == "__main__":
    test_glm53_kpool_prefill_math()
    test_glm53_kpool_decode_reads_ring_state()
