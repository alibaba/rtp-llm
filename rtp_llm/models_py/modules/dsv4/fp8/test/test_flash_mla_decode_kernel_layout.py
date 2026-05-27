"""FlashMLA sparse FP8 decode layout acceptance tests.

These tests call ``flash_mla_with_kvcache`` directly.  They intentionally
avoid model/framework setup and only verify the kernel accepts the DSV4 FP8
paged-cache tensor layouts that the framework publishes to Python:

* SWA_KV is a small ring per physical block, not a 16K-token row.
* Per-block storage has TMA padding, so FlashMLA sees a 4D as_strided view.
* With kernel_seq_size_per_block=128, compressed pools may expose small
  compressed-entry blocks such as 1, 2, 32, or 64 entries.
"""

from __future__ import annotations

import unittest
from typing import Optional

import torch

ENTRY_BYTES = 584
TOKEN_DATA_BYTES = 576
TMA_STRIDE_BYTES = 576
HEAD_DIM = 512
NOPE_DIM = 448
ROPE_DIM = 64
ROPE_BYTES = ROPE_DIM * 2
NUM_HEADS = 64
TOPK = 128
FP8_MAX = 448.0
COMPRESSOR_ENTRIES = (1, 2, 32, 64)
SWA_ENTRIES = (128, 130, 132, 134)


def _align_up(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment


def _make_padded_fp8_cache_3d(
    num_blocks: int,
    entries_per_block: int,
    *,
    device: str = "cuda",
) -> torch.Tensor:
    """Return ``[num_blocks, entries_per_block, 584]`` uint8 cache.

    The storage row stride is aligned exactly like DSV4KVSpec /
    DSV4StateSpec's FP8 block padding.  The logical entry stride remains 584B.
    """
    stride_bytes = _align_up(entries_per_block * ENTRY_BYTES, TMA_STRIDE_BYTES)
    backing = torch.zeros((num_blocks, stride_bytes), dtype=torch.uint8, device=device)
    cache_3d = backing.as_strided(
        (num_blocks, entries_per_block, ENTRY_BYTES),
        (stride_bytes, ENTRY_BYTES, 1),
    )
    assert cache_3d.stride(0) % TMA_STRIDE_BYTES == 0
    return cache_3d


def _make_padded_fp8_cache(
    num_blocks: int,
    entries_per_block: int,
    *,
    device: str = "cuda",
) -> torch.Tensor:
    return _make_padded_fp8_cache_3d(
        num_blocks,
        entries_per_block,
        device=device,
    ).unsqueeze(-2)


def _pack_model1_fp8_cache(
    kv_bf16: torch.Tensor,
    entries_per_block: int,
) -> torch.Tensor:
    """Pack ``[N, 512]`` BF16 logical slots into FlashMLA's FP8 cache view.

    The logical tensor view is ``[blocks, entries, 1, 584]`` with padded
    block stride.  The bytes inside each block follow fp8_model1_mla:
    ``entries * 576`` token-data bytes first, then ``entries * 8`` scale bytes.
    """
    assert kv_bf16.dim() == 2 and kv_bf16.shape[1] == HEAD_DIM
    assert kv_bf16.dtype == torch.bfloat16
    num_tokens = int(kv_bf16.shape[0])
    num_blocks = (num_tokens + entries_per_block - 1) // entries_per_block
    cache_3d = _make_padded_fp8_cache_3d(
        num_blocks,
        entries_per_block,
        device=str(kv_bf16.device),
    )

    stride_bytes = int(cache_3d.stride(0))
    backing = torch.as_strided(
        cache_3d,
        (num_blocks, stride_bytes),
        (stride_bytes, 1),
    )

    kv_fp32 = kv_bf16.to(torch.float32)
    nope_fp8_chunks = []
    scale_chunks = []
    for start in range(0, NOPE_DIM, 64):
        tile = kv_fp32[:, start : start + 64]
        absmax = torch.clamp(tile.abs().max(dim=-1, keepdim=True).values, min=1e-4)
        exponent = torch.ceil(torch.log2(absmax / FP8_MAX))
        scale = torch.exp2(exponent)
        fp8 = torch.clamp(tile / scale, -FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
        nope_fp8_chunks.append(fp8.view(torch.uint8))
        scale_chunks.append(
            torch.clamp(exponent.squeeze(-1) + 127, 0, 255).to(torch.uint8)
        )

    nope_bytes = torch.cat(nope_fp8_chunks, dim=-1).contiguous()
    scale_bytes = torch.zeros(num_tokens, 8, dtype=torch.uint8, device=kv_bf16.device)
    scale_bytes[:, : len(scale_chunks)] = torch.stack(scale_chunks, dim=-1)
    rope_bytes = (
        kv_bf16[:, NOPE_DIM:]
        .contiguous()
        .view(torch.uint8)
        .reshape(
            num_tokens,
            ROPE_BYTES,
        )
    )

    for slot in range(num_tokens):
        block = slot // entries_per_block
        pos = slot % entries_per_block
        token_offset = pos * TOKEN_DATA_BYTES
        scale_offset = entries_per_block * TOKEN_DATA_BYTES + pos * 8
        backing[block, token_offset : token_offset + NOPE_DIM].copy_(nope_bytes[slot])
        backing[
            block,
            token_offset + NOPE_DIM : token_offset + TOKEN_DATA_BYTES,
        ].copy_(rope_bytes[slot])
        backing[block, scale_offset : scale_offset + 8].copy_(scale_bytes[slot])

    return cache_3d.unsqueeze(-2)


def _make_q(batch_size: int, q_len: int, *, device: str = "cuda") -> torch.Tensor:
    return torch.randn(
        batch_size,
        q_len,
        NUM_HEADS,
        HEAD_DIM,
        dtype=torch.bfloat16,
        device=device,
    )


def _make_indices(
    batch_size: int,
    q_len: int,
    topk: int,
    total_slots: int,
    *,
    device: str = "cuda",
) -> torch.Tensor:
    return torch.randint(
        0,
        total_slots,
        (batch_size, q_len, topk),
        dtype=torch.int32,
        device=device,
    )


def _call_flash_mla(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    indices: torch.Tensor,
    *,
    extra_k_cache: Optional[torch.Tensor] = None,
    extra_indices: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    from flash_mla import (  # type: ignore[import-not-found]
        flash_mla_with_kvcache,
        get_mla_metadata,
    )

    batch_size, q_len, num_heads, _ = q.shape
    sched_meta, _ = get_mla_metadata(
        cache_seqlens=None,
        num_q_tokens_per_head_k=batch_size * q_len * num_heads,
        topk=indices.shape[-1],
        num_heads_q=num_heads,
        num_heads_k=1,
        is_fp8_kvcache=True,
    )
    out, _ = flash_mla_with_kvcache(
        q=q,
        k_cache=k_cache,
        block_table=None,
        head_dim_v=HEAD_DIM,
        cache_seqlens=None,
        tile_scheduler_metadata=sched_meta,
        num_splits=None,
        is_fp8_kvcache=True,
        indices=indices,
        softmax_scale=HEAD_DIM**-0.5,
        topk_length=None,
        attn_sink=torch.zeros(num_heads, dtype=torch.float32, device=q.device),
        extra_k_cache=extra_k_cache,
        extra_indices_in_kvcache=extra_indices,
        extra_topk_length=None,
    )
    torch.cuda.synchronize()
    return out


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class FlashMlaDecodeKernelLayoutTest(unittest.TestCase):
    def setUp(self) -> None:
        try:
            import flash_mla  # noqa: F401
        except Exception as e:  # noqa: BLE001
            self.skipTest(f"flash_mla not importable: {e}")
        torch.manual_seed(20240527)

    def test_swa_padded_stride_entries(self) -> None:
        shapes = (
            (1, 1, 3, 64),
            (2, 1, 4, TOPK),
            (2, 2, 5, 64),
        )
        for ring_entries in SWA_ENTRIES:
            for batch_size, q_len, num_blocks, topk in shapes:
                with self.subTest(
                    ring_entries=ring_entries,
                    batch_size=batch_size,
                    q_len=q_len,
                    num_blocks=num_blocks,
                    topk=topk,
                ):
                    q = _make_q(batch_size, q_len)
                    swa_cache = _make_padded_fp8_cache(num_blocks, ring_entries)
                    swa_indices = _make_indices(
                        batch_size,
                        q_len,
                        topk,
                        num_blocks * ring_entries,
                    )

                    out = _call_flash_mla(q, swa_cache, swa_indices)

                    self.assertEqual(
                        tuple(out.shape),
                        (batch_size, q_len, NUM_HEADS, HEAD_DIM),
                    )
                    self.assertTrue(torch.isfinite(out).all().item())

    def test_dual_pool_layouts_are_result_consistent(self) -> None:
        batch_size = 1
        q_len = 1
        q = _make_q(batch_size, q_len)
        swa_kv = torch.randn(134, HEAD_DIM, dtype=torch.bfloat16, device="cuda") * 0.5
        cmp_kv = torch.randn(TOPK, HEAD_DIM, dtype=torch.bfloat16, device="cuda") * 0.5
        swa_indices = torch.cat(
            [
                torch.arange(0, TOPK - 4, dtype=torch.int32, device="cuda"),
                torch.tensor([128, 129, 132, 133], dtype=torch.int32, device="cuda"),
            ]
        ).view(batch_size, q_len, TOPK)
        cmp_indices = torch.arange(
            TOPK - 1,
            -1,
            -1,
            dtype=torch.int32,
            device="cuda",
        ).view(batch_size, q_len, TOPK)

        def run_layout(compressed_entries: int, ring_entries: int) -> torch.Tensor:
            swa_cache = _pack_model1_fp8_cache(swa_kv, ring_entries)
            cmp_cache = _pack_model1_fp8_cache(cmp_kv, compressed_entries)
            return _call_flash_mla(
                q,
                swa_cache,
                swa_indices,
                extra_k_cache=cmp_cache,
                extra_indices=cmp_indices,
            )

        baseline_key = (32, 128)
        baseline = run_layout(*baseline_key).detach().clone()
        self.assertTrue(torch.isfinite(baseline).all().item())
        self.assertGreater(baseline.float().abs().max().item(), 0.0)

        for compressed_entries in COMPRESSOR_ENTRIES:
            for ring_entries in SWA_ENTRIES:
                with self.subTest(
                    compressed_entries=compressed_entries,
                    ring_entries=ring_entries,
                ):
                    out = run_layout(compressed_entries, ring_entries)

                    self.assertEqual(
                        tuple(out.shape), (batch_size, q_len, NUM_HEADS, HEAD_DIM)
                    )
                    self.assertTrue(torch.isfinite(out).all().item())
                    self.assertGreater(out.float().abs().max().item(), 0.0)
                    torch.testing.assert_close(
                        out,
                        baseline,
                        rtol=1e-2,
                        atol=1e-2,
                        msg=(
                            "FlashMLA output changed for "
                            f"compressed_entries={compressed_entries}, "
                            f"ring_entries={ring_entries}; "
                            f"baseline={baseline_key}"
                        ),
                    )


if __name__ == "__main__":
    unittest.main()
