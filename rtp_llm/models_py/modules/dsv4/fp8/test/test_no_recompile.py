"""Verify pool-size-dependent params don't trigger Triton recompilation.

After converting NUM_STATE_BLOCKS, NUM_KV_BLOCKS, KV_BLOCK_STRIDE,
STATE_RING_ENTRIES, max_blocks_per_seq, and block_stride from
tl.constexpr to regular + do_not_specialize, calling the same kernel
with different pool sizes must NOT recompile.

Run:
  bazelisk test //rtp_llm/models_py/modules/dsv4/fp8/test:test_no_recompile \
    --config=cuda13 --test_output=all --test_env=CUDA_VISIBLE_DEVICES=0
"""

from __future__ import annotations

import unittest

import torch
import triton

from rtp_llm.models_py.modules.dsv4.fp8._compressor_vllm_triton import (
    _fused_kv_compress_norm_rope_insert_indexer_attn,
    _fused_kv_compress_norm_rope_insert_sparse_attn,
    _save_partial_states_kernel,
    run_fused_compress_kv_write,
    run_save_partial_states,
)
from rtp_llm.models_py.modules.dsv4.fp8._swa_dequant_triton import (
    _gather_k_cache_packed_kernel,
    gather_k_cache_packed,
)

DEVICE = "cuda"


def _triton_cache_size(kernel_fn) -> int:
    """Count compiled kernels across all devices."""
    total = 0
    for kernel_cache, *_ in kernel_fn.device_caches.values():
        total += len(kernel_cache)
    return total


class NoRecompileTest(unittest.TestCase):

    def setUp(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA required")

    def test_save_partial_states_no_recompile(self):
        """_save_partial_states_kernel must not recompile when num_state_blocks changes."""
        head_dim = 512
        compress_ratio = 4
        coff = 2
        state_width = coff * head_dim

        for pool_blocks in [10, 20]:
            state_cache = torch.zeros(
                pool_blocks, 256, 2 * state_width, dtype=torch.float32, device=DEVICE
            )
            N = 8
            kv = torch.randn(N, state_width, dtype=torch.float32, device=DEVICE)
            score = torch.randn(N, state_width, dtype=torch.float32, device=DEVICE)
            ape = torch.randn(
                compress_ratio, state_width, dtype=torch.float32, device=DEVICE
            )
            positions = torch.arange(N, dtype=torch.int64, device=DEVICE)
            slots = torch.arange(N, dtype=torch.int64, device=DEVICE) + 256
            run_save_partial_states(
                kv, score, ape, positions, state_cache, slots, compress_ratio
            )
            torch.cuda.synchronize()

            if pool_blocks == 10:
                cache_after_first = _triton_cache_size(_save_partial_states_kernel)

        cache_after_second = _triton_cache_size(_save_partial_states_kernel)
        self.assertEqual(
            cache_after_first,
            cache_after_second,
            f"_save_partial_states_kernel recompiled: cache grew from {cache_after_first} to {cache_after_second}",
        )

    def _run_fused_compress(self, head_dim, pool_blocks):
        compress_ratio = 4
        overlap = head_dim == 512 or True
        coff = 2 if compress_ratio == 4 else 1
        state_width = coff * head_dim

        state_cache = torch.zeros(
            pool_blocks, 256, 2 * state_width, dtype=torch.float32, device=DEVICE
        )
        kv_eb = 256 // compress_ratio
        kv_blocks = max(pool_blocks, 2)
        token_stride = 576 if head_dim == 512 else 128
        scale_dim = 8 if head_dim == 512 else 4
        entry_bytes = token_stride + scale_dim
        kv_cache = torch.zeros(
            kv_blocks, kv_eb, entry_bytes, dtype=torch.uint8, device=DEVICE
        )
        block_table = torch.arange(
            1, pool_blocks, dtype=torch.int32, device=DEVICE
        ).unsqueeze(0)
        N = compress_ratio * 2
        positions = torch.arange(N, dtype=torch.int64, device=DEVICE)
        slot_mapping = torch.arange(N, dtype=torch.int64, device=DEVICE) + 256
        kv_raw = torch.randn(N, state_width, dtype=torch.float32, device=DEVICE)
        score_raw = torch.randn(N, state_width, dtype=torch.float32, device=DEVICE)
        ape = torch.randn(
            compress_ratio, state_width, dtype=torch.float32, device=DEVICE
        )
        rms_w = torch.ones(head_dim, dtype=torch.bfloat16, device=DEVICE)
        rope_dim = 64 if head_dim == 512 else 128
        cos_sin = torch.zeros(N + 16, rope_dim, dtype=torch.float32, device=DEVICE)
        cos_sin[:, : rope_dim // 2] = 1.0
        token_to_req = torch.zeros(N, dtype=torch.int32, device=DEVICE)
        kv_slot_mapping = torch.full((N,), -1, dtype=torch.int64, device=DEVICE)
        boundary_idx = 0
        for i in range(N):
            if (i + 1) % compress_ratio == 0:
                kv_slot_mapping[i] = kv_eb + boundary_idx
                boundary_idx += 1
        run_save_partial_states(
            kv_raw, score_raw, ape, positions, state_cache, slot_mapping, compress_ratio
        )
        run_fused_compress_kv_write(
            state_cache,
            token_to_req,
            positions,
            slot_mapping,
            block_table,
            rms_w,
            1e-6,
            cos_sin,
            kv_cache,
            kv_slot_mapping,
            kv_raw,
            score_raw,
            ape,
            0,
            head_dim=head_dim,
            rope_head_dim=rope_dim,
            compress_ratio=compress_ratio,
            overlap=overlap,
            disable_raw_path=False,
            state_tokens_per_block=256,
        )
        torch.cuda.synchronize()

    def test_fused_sparse_attn_no_recompile(self):
        """Sparse attn (head_dim=512) must not recompile when pool sizes change."""
        self._run_fused_compress(512, pool_blocks=10)
        cache_after_first = _triton_cache_size(
            _fused_kv_compress_norm_rope_insert_sparse_attn
        )

        self._run_fused_compress(512, pool_blocks=20)
        cache_after_second = _triton_cache_size(
            _fused_kv_compress_norm_rope_insert_sparse_attn
        )

        self.assertEqual(
            cache_after_first,
            cache_after_second,
            f"sparse_attn recompiled: cache grew {cache_after_first} -> {cache_after_second}",
        )

    def test_fused_indexer_attn_no_recompile(self):
        """Indexer attn (head_dim=128) must not recompile when pool sizes change."""
        self._run_fused_compress(128, pool_blocks=10)
        cache_after_first = _triton_cache_size(
            _fused_kv_compress_norm_rope_insert_indexer_attn
        )

        self._run_fused_compress(128, pool_blocks=20)
        cache_after_second = _triton_cache_size(
            _fused_kv_compress_norm_rope_insert_indexer_attn
        )

        self.assertEqual(
            cache_after_first,
            cache_after_second,
            f"indexer_attn recompiled: cache grew {cache_after_first} -> {cache_after_second}",
        )

    def test_gather_k_cache_packed_no_recompile(self):
        """gather_k_cache_packed must not recompile when max_blocks_per_seq or block_stride changes."""
        block_size = 64
        entry_bytes = 584

        for num_blocks, max_blocks in [(10, 4), (20, 8)]:
            k_cache = torch.zeros(
                num_blocks, block_size, entry_bytes, dtype=torch.uint8, device=DEVICE
            )
            seq_lens = torch.tensor([block_size], dtype=torch.int32, device=DEVICE)
            block_table = torch.arange(
                1, max_blocks + 1, dtype=torch.int32, device=DEVICE
            ).unsqueeze(0)
            out = torch.zeros(
                1, block_size, entry_bytes, dtype=torch.uint8, device=DEVICE
            )
            gather_k_cache_packed(
                out, k_cache, seq_lens, None, block_table, block_size, 0
            )
            torch.cuda.synchronize()

            if num_blocks == 10:
                cache_after_first = _triton_cache_size(_gather_k_cache_packed_kernel)

        cache_after_second = _triton_cache_size(_gather_k_cache_packed_kernel)
        self.assertEqual(
            cache_after_first,
            cache_after_second,
            f"gather_k_cache_packed recompiled: cache grew {cache_after_first} -> {cache_after_second}",
        )


if __name__ == "__main__":
    unittest.main()
