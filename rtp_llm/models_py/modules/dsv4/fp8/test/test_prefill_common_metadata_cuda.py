"""CUDA exactness coverage for cross-ratio prefill metadata reuse.

The production broadcast path builds ratio-0 metadata first, then lets the
ratio-4/128 representatives reuse its ratio-independent SWA metadata.  This
test exercises the real CUDA/Triton metadata builders under B>1 context
parallelism and compares the reused results with three independent full
builds, field by field.
"""

from __future__ import annotations

import inspect
import textwrap
import unittest
from typing import Any, Optional
from unittest.mock import patch

import torch

from rtp_llm.models_py.modules.dsv4.cp import CPContext
from rtp_llm.models_py.modules.dsv4.fp8._swa_cp_byte_sliced import (
    CPByteSlicedSlotCompaction,
    build_cp_byte_sliced_slot_compaction,
)
from rtp_llm.models_py.modules.dsv4.fp8.attention import (
    AttentionFP8,
    PrefillMeta,
    SwaPrefillMeta,
)
from rtp_llm.models_py.modules.dsv4.kv_cache_utils import SWA_KV

_SWA_FP8_ENTRY_BYTES = 584


class _StubKvCache:
    """Scalar pool geometry consumed by the production SWA meta builder."""

    group_tags = [SWA_KV]
    seq_size_per_block = 16
    kernel_seq_size_per_block = 8

    def get_seq_size_per_block(self, tag: str) -> int:
        return self.seq_size_per_block

    def get_kernel_seq_size_per_block(self, tag: str) -> int:
        return self.kernel_seq_size_per_block


class _MetadataAttention:
    """Minimal owner for the production common/SWA metadata methods."""

    _build_shared_prefill_meta = AttentionFP8._build_shared_prefill_meta
    _build_swa_prefill_meta_varlen = AttentionFP8._build_swa_prefill_meta_varlen
    _validate_reusable_prefill_common = AttentionFP8._validate_reusable_prefill_common

    def __init__(
        self,
        *,
        compress_ratio: int,
        freqs_cis: torch.Tensor,
        cp_ctx: CPContext,
        block_table: torch.Tensor,
        entries_per_block: int,
    ) -> None:
        self.compress_ratio = int(compress_ratio)
        self.freqs_cis = freqs_cis
        self.rope_head_dim = int(freqs_cis.shape[-1]) * 2
        self.window_size = 8
        self._cp_ctx = cp_ctx
        self._kv_cache = _StubKvCache()
        self._block_tables_by_type = {SWA_KV: block_table}
        self._entries_per_block = int(entries_per_block)

        num_blocks = int(block_table.max().item()) + 1
        local_slice_bytes = (
            self._entries_per_block * _SWA_FP8_ENTRY_BYTES // int(cp_ctx.cp_size)
        )
        self._raw_swa_pool = torch.empty(
            (num_blocks, local_slice_bytes),
            dtype=torch.uint8,
            device=block_table.device,
        )

    def _ensure_freqs_cis_bound(self) -> None:
        pass

    def _build_csa_prefill_meta(self, *args: Any, **kwargs: Any) -> None:
        return None

    def _build_hca_prefill_meta(self, *args: Any, **kwargs: Any) -> None:
        return None

    def _pool_entries_per_block(self, tag: str) -> int:
        return self._entries_per_block if tag == SWA_KV else 0

    def _pool_raw_u8(self, tag: str) -> Optional[torch.Tensor]:
        return self._raw_swa_pool if tag == SWA_KV else None

    def _swa_cp_byte_sliced(self) -> bool:
        return int(self._cp_ctx.cp_size) > 1 and bool(self._cp_ctx.kv_cache_sharded)

    def _swa_entries_per_block(self) -> int:
        if not self._swa_cp_byte_sliced():
            return self._entries_per_block
        return (
            int(self._raw_swa_pool.shape[1]) * int(self._cp_ctx.cp_size)
        ) // _SWA_FP8_ENTRY_BYTES

    def _build_swa_cp_byte_compaction(
        self,
        slot_mapping: torch.Tensor,
        full_entries_per_block: int,
        validation_site: str,
        negative_mode: str,
        gather_lens: Optional[torch.Tensor] = None,
    ) -> Optional[CPByteSlicedSlotCompaction]:
        if not self._swa_cp_byte_sliced():
            return None
        return build_cp_byte_sliced_slot_compaction(
            slot_mapping,
            full_entries_per_block=full_entries_per_block,
            num_blocks=int(self._raw_swa_pool.shape[0]),
            validation_site=validation_site,
            negative_mode=negative_mode,
            gather_lens=gather_lens,
        )


def _assert_exact(
    test: unittest.TestCase, actual: Any, expected: Any, path: str
) -> None:
    """Recursively compare every metadata tensor, scalar, and optional field."""

    test.assertIs(type(actual), type(expected), path)
    if isinstance(actual, torch.Tensor):
        test.assertEqual(actual.dtype, expected.dtype, path)
        test.assertEqual(actual.device, expected.device, path)
        test.assertEqual(tuple(actual.shape), tuple(expected.shape), path)
        test.assertTrue(torch.equal(actual, expected), path)
        return
    if isinstance(actual, tuple) and hasattr(actual, "_fields"):
        for name in actual._fields:
            _assert_exact(
                test,
                getattr(actual, name),
                getattr(expected, name),
                f"{path}.{name}",
            )
        return
    if isinstance(actual, CPContext):
        test.assertIs(actual, expected, path)
        return
    test.assertEqual(actual, expected, path)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required for Triton metadata")
class PrefillCommonMetadataCudaTest(unittest.TestCase):
    def setUp(self) -> None:
        self.device = torch.device("cuda")
        self.entries_per_block = 8

        # Two requests after rank-local CP splitting.  The global request
        # lengths are [5, 3], while this rank owns [3, 2] zigzag-selected
        # tokens at the explicit absolute positions below.
        self.input_lengths = torch.tensor([3, 2], dtype=torch.int32, device=self.device)
        self.input_lengths_global = torch.tensor(
            [5, 3], dtype=torch.int32, device=self.device
        )
        self.prefix_lengths = torch.tensor(
            [10, 20], dtype=torch.int32, device=self.device
        )
        self.sp_per_req = self.prefix_lengths.to(torch.int64)
        self.cu_seqlens = torch.tensor([0, 3, 5], dtype=torch.int32, device=self.device)
        self.cu_seqlens_global = torch.tensor(
            [0, 5, 8], dtype=torch.int32, device=self.device
        )
        self.position_ids = torch.tensor(
            [10, 11, 14, 20, 22], dtype=torch.long, device=self.device
        )
        self.req_id_per_token = torch.tensor(
            [0, 0, 0, 1, 1], dtype=torch.int32, device=self.device
        )
        self.cp_ctx = CPContext(
            cp_size=2,
            cp_rank=0,
            chunk_length=5,
            padded_seq_len=10,
            seq_len_full=8,
            relative_positions=torch.tensor(
                [0, 1, 4, 0, 2], dtype=torch.long, device=self.device
            ),
            prefix_length=10,
            global_positions=self.position_ids,
            local_is_real=torch.ones(5, dtype=torch.bool, device=self.device),
            unpad_restore=torch.arange(8, dtype=torch.long, device=self.device),
            seq_len_total=23,
            cp_info=object(),
            req_id_per_token=self.req_id_per_token,
            prefix_lengths=self.sp_per_req,
            input_lengths_global=self.input_lengths_global,
            cu_seqlens_global=self.cu_seqlens_global,
            unpad_restore_is_prefix=True,
            chunk_lengths_per_req=(3, 2),
            kv_cache_sharded=True,
            input_lengths_global_host=(5, 3),
            prefix_lengths_host=(10, 20),
        )
        self.block_table = torch.tensor(
            [[1, 2, 3, 4], [5, 6, 7, 8]],
            dtype=torch.int32,
            device=self.device,
        )
        self.x = torch.empty((5, 16), dtype=torch.bfloat16, device=self.device)

        # Production init_rope_cache memoizes these two tables by RoPE
        # parameters: all compressed layers share one object, while ratio 0
        # points at a distinct base-RoPE table.
        positions = torch.arange(64, dtype=torch.float32, device=self.device)
        self.base_rope = torch.complex(
            positions[:, None].expand(-1, 4),
            torch.zeros((64, 4), dtype=torch.float32, device=self.device),
        )
        self.compressed_rope = torch.complex(
            positions[:, None].expand(-1, 4) + 1000.0,
            torch.ones((64, 4), dtype=torch.float32, device=self.device),
        )

    def _owner(self, ratio: int) -> _MetadataAttention:
        return _MetadataAttention(
            compress_ratio=ratio,
            freqs_cis=self.base_rope if ratio == 0 else self.compressed_rope,
            cp_ctx=self.cp_ctx,
            block_table=self.block_table,
            entries_per_block=self.entries_per_block,
        )

    def _build(
        self,
        owner: _MetadataAttention,
        *,
        reuse_common_meta: Optional[PrefillMeta] = None,
        reuse_freqs_meta: Optional[PrefillMeta] = None,
    ) -> PrefillMeta:
        return owner._build_shared_prefill_meta(
            self.x,
            int(self.sp_per_req[0].item()),
            sp_per_req=self.sp_per_req,
            cu_seqlens=self.cu_seqlens,
            batch_size=2,
            input_lengths=self.input_lengths,
            prefix_lengths=self.prefix_lengths,
            position_ids=self.position_ids,
            req_id_per_token=self.req_id_per_token,
            max_seqlen_q=3,
            reuse_common_meta=reuse_common_meta,
            reuse_freqs_meta=reuse_freqs_meta,
        )

    def _build_references_and_reused(
        self,
    ) -> tuple[dict[int, PrefillMeta], dict[int, PrefillMeta]]:
        references = {ratio: self._build(self._owner(ratio)) for ratio in (0, 4, 128)}

        reused_0 = self._build(self._owner(0))
        # No compressed frequency source exists yet: ratio 4 must gather its
        # own compressed RoPE while still reusing ratio-0 common/SWA metadata.
        reused_4 = self._build(self._owner(4), reuse_common_meta=reused_0)
        # Ratio 128 sees the same memoized compressed table and shares the
        # already-gathered frequency tensor from ratio 4.
        reused_128 = self._build(
            self._owner(128),
            reuse_common_meta=reused_0,
            reuse_freqs_meta=reused_4,
        )
        reused = {0: reused_0, 4: reused_4, 128: reused_128}

        torch.cuda.synchronize(self.device)
        for ratio in (0, 4, 128):
            _assert_exact(self, reused[ratio], references[ratio], f"ratio{ratio}")

        self.assertIsNot(reused_0.freqs_cis, reused_4.freqs_cis)
        self.assertFalse(torch.equal(reused_0.freqs_cis, reused_4.freqs_cis))
        self.assertIs(reused_128.freqs_cis, reused_4.freqs_cis)
        return references, reused

    def test_cp_b2_reused_metadata_matches_three_independent_builds(self) -> None:
        _, reused = self._build_references_and_reused()
        reused_0 = reused[0]
        reused_4 = reused[4]
        reused_128 = reused[128]

        # The tensors intentionally shared across ratio buckets are the exact
        # source objects, including CP byte-sliced compaction metadata.
        for meta in (reused_4, reused_128):
            self.assertIs(meta.topk_idxs, reused_0.topk_idxs)
            self.assertIs(meta.row_seqlens_full, reused_0.row_seqlens_full)
            self.assertIs(meta.swa_meta.slot_mapping, reused_0.swa_meta.slot_mapping)
            self.assertIs(
                meta.swa_meta.slot_compaction, reused_0.swa_meta.slot_compaction
            )

        self.assertIsInstance(reused_0.swa_meta, SwaPrefillMeta)
        self.assertIsNotNone(reused_0.swa_meta.slot_compaction)
        self.assertEqual(tuple(reused_0.swa_meta.slot_mapping.shape), (8,))
        self.assertEqual(tuple(reused_0.topk_idxs.shape), (5, 8))

    def test_swa_only_reuse_is_rejected_with_python_optimization(self) -> None:
        builder = AttentionFP8._build_shared_prefill_meta
        validator = AttentionFP8._validate_reusable_prefill_common
        for optimization in (0, 1):
            with self.subTest(optimization=optimization):
                namespace = dict(builder.__globals__)
                for method in (builder, validator):
                    exec(
                        compile(
                            textwrap.dedent(inspect.getsource(method)),
                            inspect.getsourcefile(method),
                            "exec",
                            optimize=optimization,
                        ),
                        namespace,
                    )
                with patch.object(
                    _MetadataAttention, builder.__name__, namespace[builder.__name__]
                ), patch.object(
                    _MetadataAttention,
                    validator.__name__,
                    namespace[validator.__name__],
                ):
                    source = self._build(self._owner(0))
                    self.assertIsNotNone(source.swa_meta.combined_gather_lens)
                    compressed = self._build(self._owner(4), reuse_common_meta=source)
                    self.assertIs(compressed.topk_idxs, source.topk_idxs)
                    with self.assertRaisesRegex(
                        ValueError, "SWA-only metadata must be the common source"
                    ):
                        self._build(self._owner(0), reuse_common_meta=source)

    def test_cp1_b2_cold_reuse_matches_three_independent_builds(self) -> None:
        self.input_lengths = torch.tensor([4, 3], dtype=torch.int32, device=self.device)
        self.input_lengths_global = self.input_lengths
        self.prefix_lengths = torch.zeros(2, dtype=torch.int32, device=self.device)
        self.sp_per_req = self.prefix_lengths.to(torch.int64)
        self.cu_seqlens = torch.tensor([0, 4, 7], dtype=torch.int32, device=self.device)
        self.cu_seqlens_global = self.cu_seqlens
        self.position_ids = torch.tensor(
            [0, 1, 2, 3, 0, 1, 2], dtype=torch.long, device=self.device
        )
        self.req_id_per_token = torch.tensor(
            [0, 0, 0, 0, 1, 1, 1], dtype=torch.int32, device=self.device
        )
        self.cp_ctx = CPContext(
            cp_size=1,
            cp_rank=0,
            chunk_length=7,
            padded_seq_len=7,
            seq_len_full=7,
            relative_positions=self.position_ids,
            prefix_length=0,
            global_positions=self.position_ids,
            local_is_real=torch.ones(7, dtype=torch.bool, device=self.device),
            unpad_restore=torch.arange(7, dtype=torch.long, device=self.device),
            seq_len_total=7,
            cp_info=object(),
            req_id_per_token=self.req_id_per_token,
            prefix_lengths=self.sp_per_req,
            input_lengths_global=self.input_lengths_global,
            cu_seqlens_global=self.cu_seqlens_global,
            unpad_restore_is_prefix=True,
            chunk_lengths_per_req=(4, 3),
            kv_cache_sharded=False,
            input_lengths_global_host=(4, 3),
            prefix_lengths_host=(0, 0),
        )
        self.x = torch.empty((7, 16), dtype=torch.bfloat16, device=self.device)

        _, reused = self._build_references_and_reused()
        reused_0 = reused[0]
        self.assertFalse(reused_0.cp_on)
        self.assertFalse(reused_0.any_cont)
        self.assertIsNone(reused_0.swa_meta.slot_compaction)
        self.assertIsNone(reused_0.swa_meta.cache_compaction)
        self.assertIsNone(reused_0.swa_meta.combined_indices)
        self.assertEqual(tuple(reused_0.swa_meta.slot_mapping.shape), (7,))
        self.assertEqual(tuple(reused_0.topk_idxs.shape), (7, 8))


if __name__ == "__main__":
    unittest.main()
