"""UT: SWA FP8 KV cache quantize-and-insert ↔ dequantize-and-gather round-trip.

Covers two Triton kernels:
  * ``_swa_fp8_kv_insert_triton.quantize_and_insert_k_cache``
    (vendored from vLLM ``cache_utils.py:quantize_and_insert_k_kernel``)
  * ``_swa_fp8_dequant_triton.dequantize_and_gather_k_cache``
    (vendored from vLLM ``cache_utils.py:_dequantize_and_gather_k_kernel``)

Mirrors vLLM ``tests/kernels/test_compressor_kv_cache.py``:
``test_deepseek_v4_attention_quant_cache_roundtrip`` +
``test_deepseek_v4_quant_magnitude_range``, restructured as
``unittest.TestCase``.

End-to-end validation:
  1. quantize + insert random BF16 K into the FP8 SWA cache (584B/token)
  2. dequantize + gather back into a BF16 workspace
  3. NoPE (first 448 elements): per-token UE8M0 quant noise bound
     (``<= 16 * tile_scale``)
  4. RoPE (last 64 elements): byte-exact (passthrough, no quant)

Run:
  CUDA_VISIBLE_DEVICES=7 /opt/conda310/bin/python3 -m unittest \\
    rtp_llm.models_py.modules.dsv4.test.test_swa_fp8_kv_roundtrip
"""

from __future__ import annotations

import math
import unittest
from contextlib import nullcontext
from unittest import mock

import torch

from rtp_llm.models_py.modules.dsv4.fp8 import _swa_dequant_triton
from rtp_llm.models_py.modules.dsv4.fp8._swa_cp_byte_sliced import (
    build_cp_byte_sliced_slot_compaction,
)
from rtp_llm.models_py.modules.dsv4.fp8._swa_dequant_triton import (
    CPByteSlicedSwaPrefixPending,
    _validate_runtime_gather_lens,
    dequantize_and_gather_k_cache,
    dequantize_and_gather_k_cache_slots,
    dequantize_and_gather_k_cache_slots_cp_byte_sliced,
    dequantize_packed_k_cache_flat,
    discard_dequantize_and_gather_k_cache_slots_cp_byte_sliced,
    gather_k_cache_packed,
    prepare_dequantize_and_gather_k_cache_slots_cp_byte_sliced,
    start_dequantize_and_gather_k_cache_slots_cp_byte_sliced,
)
from rtp_llm.models_py.modules.dsv4.fp8._swa_kv_insert_triton import (
    quantize_and_insert_k_cache,
    quantize_and_insert_k_cache_cp_byte_sliced,
)

HEAD_DIM = 512
NOPE_DIM = 448
HEAD_BYTES = 584  # 448 fp8 + 128 bf16 + 8 uint8 scale
FP8_MAX = 448.0
QUANT_BLOCK = 64


class CPByteSlicedSlotCompactionTest(unittest.TestCase):
    def test_boundary_cases(self) -> None:
        cases = [
            {
                "name": "all_negative_slots",
                "slots": torch.tensor([[-1, -2], [-3, -1]], dtype=torch.int64),
                "gather_lens": torch.tensor([0, 0], dtype=torch.int32),
                "negative_mode": "skip_any",
                "expected_unique": [],
                "expected_compact": [[-1, -1], [-1, -1]],
                "expected_gather_lens": (0, 0),
            },
            {
                "name": "single_block",
                "slots": torch.tensor([[8, 10, -1], [15, 9, 8]], dtype=torch.int64),
                "gather_lens": torch.tensor([3, 3], dtype=torch.int32),
                "negative_mode": "skip_minus_one",
                "expected_unique": [1],
                "expected_compact": [[0, 2, -1], [7, 1, 0]],
                "expected_gather_lens": (3, 3),
            },
            {
                "name": "gather_lens_none",
                "slots": torch.tensor([[16, 17], [24, -1]], dtype=torch.int64),
                "gather_lens": None,
                "negative_mode": "skip_minus_one",
                "expected_unique": [2, 3],
                "expected_compact": [[0, 1], [8, -1]],
                "expected_gather_lens": (),
            },
        ]
        for case in cases:
            with self.subTest(case=case["name"]):
                compaction = build_cp_byte_sliced_slot_compaction(
                    case["slots"],
                    full_entries_per_block=8,
                    num_blocks=4,
                    validation_site=f"test.{case['name']}",
                    negative_mode=case["negative_mode"],
                    gather_lens=case["gather_lens"],
                )
                self.assertEqual(
                    compaction.unique_blocks.detach().cpu().tolist(),
                    case["expected_unique"],
                )
                self.assertEqual(
                    compaction.compact_slots.detach().cpu().tolist(),
                    case["expected_compact"],
                )
                self.assertEqual(
                    compaction.gather_lens_cpu,
                    case["expected_gather_lens"],
                )

    def test_runtime_gather_lens_uses_compaction_snapshot_without_tensor_read(self):
        compaction = build_cp_byte_sliced_slot_compaction(
            torch.tensor([[8, 9]], dtype=torch.int64),
            full_entries_per_block=8,
            num_blocks=2,
            validation_site="test.gather_lens_snapshot",
            negative_mode="skip_any",
            gather_lens=torch.tensor([2], dtype=torch.int32),
        )
        runtime_lens = mock.Mock()
        runtime_lens.dim.return_value = 1
        runtime_lens.numel.return_value = 1
        runtime_lens.shape = (1,)
        runtime_lens.to.side_effect = AssertionError("runtime tensor must not be read")

        saved = _validate_runtime_gather_lens(
            runtime_lens,
            compaction,
            batch_size=1,
            max_gather_len=2,
        )

        self.assertIs(saved, compaction.gather_lens)
        self.assertEqual(compaction.gather_lens_cpu, (2,))
        self.assertEqual(saved.tolist(), [2])
        runtime_lens.to.assert_not_called()

    def test_runtime_gather_lens_shape_mismatch_fails_before_launch(self):
        slots = torch.tensor([[8, 9]], dtype=torch.int64)
        compaction = build_cp_byte_sliced_slot_compaction(
            slots,
            full_entries_per_block=8,
            num_blocks=2,
            validation_site="test.gather_lens_shape_mismatch",
            negative_mode="skip_any",
            gather_lens=torch.tensor([2], dtype=torch.int32),
        )
        runtime_lens = torch.tensor([1, 1], dtype=torch.int32)
        raw = torch.zeros((2, 8 * HEAD_BYTES // 2), dtype=torch.uint8)
        out = torch.zeros((1, 2, HEAD_DIM), dtype=torch.bfloat16)

        with self.assertRaisesRegex(ValueError, "shape does not match"):
            dequantize_and_gather_k_cache_slots_cp_byte_sliced(
                out=out,
                k_cache_raw=raw,
                slot_mapping=slots,
                gather_lens=runtime_lens,
                offset=0,
                full_entries_per_block=8,
                cp_rank=0,
                cp_size=2,
                compaction=compaction,
            )
        with self.assertRaisesRegex(ValueError, "shape does not match"):
            start_dequantize_and_gather_k_cache_slots_cp_byte_sliced(
                k_cache_raw=raw,
                slot_mapping=slots,
                gather_lens=runtime_lens,
                offset=0,
                full_entries_per_block=8,
                cp_rank=0,
                cp_size=2,
                compaction=compaction,
                stream=None,
            )

    def test_async_pending_strongly_owns_producer_and_output_tensors(self):
        local_slices = torch.empty((1, 8), dtype=torch.uint8)
        gathered = torch.empty((2, 8), dtype=torch.uint8)
        out = torch.empty((1, 1, HEAD_DIM), dtype=torch.bfloat16)
        producer_stream = object()
        out_stream = object()
        pending = CPByteSlicedSwaPrefixPending(
            cp_size=2,
            B=1,
            W=1,
            offset=0,
            full_entries_per_block=8,
            gathered=gathered,
            unique_blocks=torch.tensor([0], dtype=torch.int64),
            compact_slots=torch.tensor([[0]], dtype=torch.int64),
            gather_lens=torch.tensor([1], dtype=torch.int32),
            gather_lens_cpu=[1],
            work=object(),
            stream=object(),
            producer_stream=producer_stream,
            completion_event=object(),
            local_slices=local_slices,
            out=out,
            out_stream=out_stream,
        )

        self.assertIs(pending.local_slices, local_slices)
        self.assertIs(pending.gathered, gathered)
        self.assertIs(pending.out, out)
        self.assertIs(pending.producer_stream, producer_stream)
        self.assertIs(pending.out_stream, out_stream)

    def test_async_prepare_exception_publishes_terminal_fence_for_discard(self):
        class FakeStream:
            def __init__(self):
                self.waited_events = []
                self.synchronize_calls = 0

            def wait_stream(self, stream):
                del stream

            def wait_event(self, event):
                self.waited_events.append(event)

            def synchronize(self):
                self.synchronize_calls += 1

        class FakeEvent:
            def __init__(self):
                self.recorded_stream = None

            def record(self, stream):
                self.recorded_stream = stream

        work = mock.Mock()
        producer_stream = FakeStream()
        gather_stream = FakeStream()
        assemble_stream = FakeStream()
        current_stream = FakeStream()
        terminal_event = FakeEvent()
        gathered = torch.empty((2, 8), dtype=torch.uint8)
        out = torch.empty((1, 1, HEAD_DIM), dtype=torch.bfloat16)
        pending = CPByteSlicedSwaPrefixPending(
            cp_size=2,
            B=1,
            W=1,
            offset=0,
            full_entries_per_block=8,
            gathered=gathered,
            unique_blocks=torch.tensor([0], dtype=torch.int64),
            compact_slots=torch.tensor([[0]], dtype=torch.int64),
            gather_lens=torch.tensor([1], dtype=torch.int32),
            gather_lens_cpu=(1,),
            work=work,
            stream=gather_stream,
            producer_stream=producer_stream,
            completion_event=object(),
            local_slices=torch.empty((1, 8), dtype=torch.uint8),
        )

        with mock.patch.object(
            torch.cuda, "current_stream", return_value=current_stream
        ), mock.patch.object(
            torch.cuda, "stream", side_effect=lambda stream: nullcontext()
        ), mock.patch.object(
            torch.cuda, "Event", return_value=terminal_event
        ), mock.patch.object(
            _swa_dequant_triton,
            "cp_swa_direct_dequant_scatter_enabled",
            return_value=True,
        ), mock.patch.object(
            _swa_dequant_triton,
            "direct_triton_fast_path_supported",
            return_value=True,
        ), mock.patch.object(
            _swa_dequant_triton,
            "_launch_dequantize_and_gather_k_slots_cp_rank_major_unchecked",
            side_effect=RuntimeError("injected after enqueue"),
        ):
            with self.assertRaisesRegex(RuntimeError, "injected after enqueue"):
                prepare_dequantize_and_gather_k_cache_slots_cp_byte_sliced(
                    pending,
                    out=out,
                    stream=assemble_stream,
                )
            self.assertIs(pending.ready_event, terminal_event)
            self.assertIs(terminal_event.recorded_stream, assemble_stream)
            self.assertIs(pending.gathered, gathered)
            self.assertIs(pending.out, out)
            self.assertEqual(assemble_stream.synchronize_calls, 0)

            discard_dequantize_and_gather_k_cache_slots_cp_byte_sliced(pending)

        self.assertIn(terminal_event, current_stream.waited_events)
        self.assertIn(terminal_event, producer_stream.waited_events)
        self.assertIn(terminal_event, gather_stream.waited_events)
        work.wait.assert_called_once_with()

    def test_async_prepare_event_failure_synchronizes_assemble_stream(self):
        class FakeStream:
            def __init__(self):
                self.synchronize_calls = 0

            def wait_stream(self, stream):
                del stream

            def wait_event(self, event):
                del event

            def synchronize(self):
                self.synchronize_calls += 1

        class RecordFailureEvent:
            def record(self, stream):
                del stream
                raise RuntimeError("injected event record failure")

        for failure_site in ("create", "record"):
            with self.subTest(failure_site=failure_site):
                assemble_stream = FakeStream()
                pending = CPByteSlicedSwaPrefixPending(
                    cp_size=2,
                    B=1,
                    W=1,
                    offset=0,
                    full_entries_per_block=8,
                    gathered=torch.empty((2, 8), dtype=torch.uint8),
                    unique_blocks=torch.tensor([0], dtype=torch.int64),
                    compact_slots=torch.tensor([[0]], dtype=torch.int64),
                    gather_lens=torch.tensor([1], dtype=torch.int32),
                    gather_lens_cpu=(1,),
                    work=mock.Mock(),
                    stream=FakeStream(),
                    producer_stream=FakeStream(),
                    completion_event=object(),
                    local_slices=torch.empty((1, 8), dtype=torch.uint8),
                )
                out = torch.empty((1, 1, HEAD_DIM), dtype=torch.bfloat16)
                event_effect = (
                    RuntimeError("injected event creation failure")
                    if failure_site == "create"
                    else RecordFailureEvent()
                )
                with mock.patch.object(
                    torch.cuda, "current_stream", return_value=FakeStream()
                ), mock.patch.object(
                    torch.cuda, "stream", side_effect=lambda stream: nullcontext()
                ), mock.patch.object(
                    torch.cuda, "Event", side_effect=[event_effect]
                ), mock.patch.object(
                    _swa_dequant_triton,
                    "cp_swa_direct_dequant_scatter_enabled",
                    return_value=True,
                ), mock.patch.object(
                    _swa_dequant_triton,
                    "direct_triton_fast_path_supported",
                    return_value=True,
                ), mock.patch.object(
                    _swa_dequant_triton,
                    "_launch_dequantize_and_gather_k_slots_cp_rank_major_unchecked",
                ):
                    with self.assertRaisesRegex(RuntimeError, "injected event"):
                        prepare_dequantize_and_gather_k_cache_slots_cp_byte_sliced(
                            pending,
                            out=out,
                            stream=assemble_stream,
                        )

                self.assertEqual(assemble_stream.synchronize_calls, 1)
                self.assertIsNone(pending.ready_event)
                self.assertIs(pending.out, out)


def _ue8m0_reference_max_scale(token_nope_bf16: torch.Tensor) -> float:
    """Compute the max UE8M0 tile-scale a reference impl would assign
    across the 7 NoPE quant tiles (each 64 elements). Used to bound
    expected post-roundtrip error (FP8 e4m3 worst-case = 16 * scale).
    """
    assert token_nope_bf16.dim() == 1 and token_nope_bf16.numel() == NOPE_DIM
    n_tiles = NOPE_DIM // QUANT_BLOCK
    max_scale = 0.0
    for i in range(n_tiles):
        tile = token_nope_bf16[i * QUANT_BLOCK : (i + 1) * QUANT_BLOCK].float()
        amax = max(tile.abs().max().item(), 1e-4)
        exponent = math.ceil(math.log2(amax / FP8_MAX))
        scale = 2.0**exponent
        max_scale = max(max_scale, scale)
    return max_scale


class SwaFp8KvRoundtripTest(unittest.TestCase):

    def setUp(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        self.device = torch.device("cuda")
        torch.manual_seed(0)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _alloc_cache(self, num_blocks: int, block_size: int) -> torch.Tensor:
        """[num_blocks, block_size, 584] uint8 — matches RTP-LLM SWA pool."""
        return torch.zeros(
            num_blocks, block_size, HEAD_BYTES, dtype=torch.uint8, device=self.device
        )

    def _roundtrip(
        self,
        compressed_kv: torch.Tensor,
        block_size: int,
    ) -> torch.Tensor:
        """quantize+insert sequential slots → dequant+gather → return [T, 512] bf16."""
        num_tokens = compressed_kv.shape[0]
        data_blocks = (num_tokens + block_size - 1) // block_size
        num_blocks = data_blocks + 1

        k_cache = self._alloc_cache(num_blocks, block_size)
        # Production block id 0 is invalid; valid physical blocks are positive.
        slot_mapping = (
            torch.arange(num_tokens, dtype=torch.int64, device=self.device) + block_size
        )
        quantize_and_insert_k_cache(compressed_kv, k_cache, slot_mapping)

        out = torch.zeros(
            1, num_tokens, HEAD_DIM, dtype=torch.bfloat16, device=self.device
        )
        seq_lens = torch.tensor([num_tokens], dtype=torch.int32, device=self.device)
        block_table = torch.arange(
            1, data_blocks + 1, dtype=torch.int32, device=self.device
        ).unsqueeze(0)
        dequantize_and_gather_k_cache(
            out=out,
            k_cache=k_cache,
            seq_lens=seq_lens,
            gather_lens=None,
            block_table=block_table,
            block_size=block_size,
            offset=0,
        )
        return out[0, :num_tokens]

    def _roundtrip_packed_gather(
        self,
        compressed_kv: torch.Tensor,
        block_size: int,
    ) -> torch.Tensor:
        """quantize+insert → packed gather → flat dequant."""
        num_tokens = compressed_kv.shape[0]
        num_blocks = (num_tokens + block_size - 1) // block_size + 1

        k_cache = self._alloc_cache(num_blocks, block_size)
        slot_mapping = torch.arange(num_tokens, dtype=torch.int64, device=self.device)
        quantize_and_insert_k_cache(compressed_kv, k_cache, slot_mapping)

        seq_lens = torch.tensor([num_tokens], dtype=torch.int32, device=self.device)
        block_table = torch.arange(
            num_blocks, dtype=torch.int32, device=self.device
        ).unsqueeze(0)

        packed = torch.zeros(
            1, num_tokens, HEAD_BYTES, dtype=torch.uint8, device=self.device
        )
        gather_k_cache_packed(
            out=packed,
            k_cache=k_cache,
            seq_lens=seq_lens,
            gather_lens=None,
            block_table=block_table,
            block_size=block_size,
            offset=0,
        )
        out = torch.empty(
            num_tokens, HEAD_DIM, dtype=torch.bfloat16, device=self.device
        )
        dequantize_packed_k_cache_flat(out, packed[0])
        return out

    def _assert_nope_within_ue8m0_bound(
        self, original: torch.Tensor, recovered: torch.Tensor
    ):
        """Per-token NoPE diff must stay within 16 * max_tile_scale."""
        nope_orig = original[:, :NOPE_DIM]
        nope_recv = recovered[:, :NOPE_DIM]
        diff = (nope_recv.float() - nope_orig.float()).abs()
        for t in range(original.shape[0]):
            scale = _ue8m0_reference_max_scale(nope_orig[t])
            max_allowed = 16.0 * scale
            token_diff = diff[t].max().item()
            self.assertLessEqual(
                token_diff,
                max_allowed,
                msg=(
                    f"NoPE token {t}: diff={token_diff:.4g} exceeds "
                    f"max_allowed={max_allowed:.4g} (tile_scale={scale:.4g})"
                ),
            )

    def _assert_rope_exact(self, original: torch.Tensor, recovered: torch.Tensor):
        """RoPE region is BF16 passthrough — byte-exact."""
        rope_orig = original[:, NOPE_DIM:]
        rope_recv = recovered[:, NOPE_DIM:]
        diff = (rope_recv - rope_orig).abs().max().item()
        self.assertEqual(
            diff,
            0.0,
            msg=f"RoPE should be byte-exact, got max diff {diff}",
        )

    # ------------------------------------------------------------------
    # Tests
    # ------------------------------------------------------------------
    def test_random_roundtrip_block_64(self):
        """Sweep token counts at block_size=64 (vLLM default page size)."""
        for num_tokens in [1, 4, 8, 17, 64, 100]:
            with self.subTest(num_tokens=num_tokens):
                compressed_kv = torch.randn(
                    num_tokens, HEAD_DIM, dtype=torch.bfloat16, device=self.device
                )
                recovered = self._roundtrip(compressed_kv, block_size=64)
                self._assert_nope_within_ue8m0_bound(compressed_kv, recovered)
                self._assert_rope_exact(compressed_kv, recovered)

    def test_random_roundtrip_block_256(self):
        """Sweep token counts at block_size=256 (RTP-LLM eb=256)."""
        for num_tokens in [1, 32, 256, 257, 600]:
            with self.subTest(num_tokens=num_tokens):
                compressed_kv = torch.randn(
                    num_tokens, HEAD_DIM, dtype=torch.bfloat16, device=self.device
                )
                recovered = self._roundtrip(compressed_kv, block_size=256)
                self._assert_nope_within_ue8m0_bound(compressed_kv, recovered)
                self._assert_rope_exact(compressed_kv, recovered)

    def test_packed_gather_matches_direct_dequant(self):
        """Packed-FP8 gather + local dequant must be bitwise-equivalent to
        the original gather+dequant path. This pins the CP all_gather payload
        optimization's byte layout.
        """
        for block_size, num_tokens in [(64, 117), (256, 513)]:
            with self.subTest(block_size=block_size, num_tokens=num_tokens):
                compressed_kv = torch.randn(
                    num_tokens, HEAD_DIM, dtype=torch.bfloat16, device=self.device
                )
                direct = self._roundtrip(compressed_kv, block_size=block_size)
                packed = self._roundtrip_packed_gather(
                    compressed_kv, block_size=block_size
                )
                self.assertTrue(torch.equal(direct, packed))

    def test_magnitude_range(self):
        """Per-token NoPE quant scale must adapt to token magnitude.

        Mirrors vLLM ``test_deepseek_v4_quant_magnitude_range``.
        """
        block_size = 16
        num_tokens = 4
        compressed_kv = torch.zeros(
            num_tokens, HEAD_DIM, dtype=torch.bfloat16, device=self.device
        )
        compressed_kv[0] = 0.001  # very small
        compressed_kv[1] = 1.0  # unit scale
        compressed_kv[2] = 100.0  # large
        compressed_kv[3] = torch.randn(
            HEAD_DIM, dtype=torch.bfloat16, device=self.device
        )

        recovered = self._roundtrip(compressed_kv, block_size=block_size)
        self._assert_nope_within_ue8m0_bound(compressed_kv, recovered)
        self._assert_rope_exact(compressed_kv, recovered)

    def test_skipped_slots_not_overwritten(self):
        """slot_mapping with -1 sentinels must skip those tokens
        without polluting the cache (insert kernel contract)."""
        block_size = 16
        num_tokens = 6
        compressed_kv = torch.randn(
            num_tokens, HEAD_DIM, dtype=torch.bfloat16, device=self.device
        )
        # Token 1 and 4 marked for skip; their slots stay zero.
        slot_mapping = torch.tensor(
            [16, -1, 17, 18, -1, 19], dtype=torch.int64, device=self.device
        )
        num_blocks = 2
        k_cache = self._alloc_cache(num_blocks, block_size)
        # Sentinel pre-fill so we can detect any unexpected write.
        sentinel = 0xAB
        k_cache.fill_(sentinel)

        quantize_and_insert_k_cache(compressed_kv, k_cache, slot_mapping)

        # Read back valid slots via gather and confirm they decode close
        # to original; valid slot positions are 0,1,2,3 in physical block 1.
        out = torch.zeros(1, 4, HEAD_DIM, dtype=torch.bfloat16, device=self.device)
        seq_lens = torch.tensor([4], dtype=torch.int32, device=self.device)
        block_table = torch.tensor([[1, -1]], dtype=torch.int32, device=self.device)
        dequantize_and_gather_k_cache(
            out=out,
            k_cache=k_cache,
            seq_lens=seq_lens,
            gather_lens=None,
            block_table=block_table,
            block_size=block_size,
            offset=0,
        )
        # Map back: out[0,0] ↔ compressed_kv[0]; out[0,1]↔kv[2]; etc.
        kv_valid = compressed_kv[[0, 2, 3, 5]]
        self._assert_rope_exact(kv_valid, out[0])
        self._assert_nope_within_ue8m0_bound(kv_valid, out[0])

        # Slots 4..15 of block 1 (untouched) plus all of block 0 must
        # still be sentinel — confirming -1 skipped tokens didn't
        # accidentally write somewhere.
        # The kernel uses a packed-per-block layout: each block holds
        #   [block_size * 576 token-data bytes || block_size * 8 scale bytes],
        # NOT a per-token contiguous [block_size, 584] view. So inspect the
        # block as a flat byte buffer and check the kernel-aligned untouched
        # regions: data bytes [4*576, 9216) and scales bytes [9216+4*8, 9344).
        TOKEN_DATA_SIZE = 576  # 448 fp8 + 128 bf16 (RoPE)
        TOKEN_SCALE_SIZE = 8
        block1_flat = k_cache[1].reshape(-1)
        data_region_end = block_size * TOKEN_DATA_SIZE  # 9216 for block_size=16
        scales_touched_end = data_region_end + 4 * TOKEN_SCALE_SIZE  # 9248
        untouched_data = block1_flat[4 * TOKEN_DATA_SIZE : data_region_end]
        untouched_scales = block1_flat[scales_touched_end:]
        self.assertTrue(
            torch.all(untouched_data == sentinel),
            msg="Untouched slot-data bytes in block 1 should remain sentinel.",
        )
        self.assertTrue(
            torch.all(untouched_scales == sentinel),
            msg="Untouched scale bytes in block 1 should remain sentinel.",
        )
        self.assertTrue(
            torch.all(k_cache[0] == sentinel),
            msg="Block 0 should remain sentinel (not targeted by this test).",
        )

    def test_sparse_paged_block_table(self):
        """Non-sequential block_table: physical block IDs jumbled relative
        to logical position. Verifies the dequant kernel resolves
        physical block via ``block_table[req, pos // block_size]`` and
        the insert kernel respects the paged formula
        ``slot = block_id * block_size + pos_in_block``.
        """
        block_size = 64
        num_tokens = 200  # 4 logical blocks (last partial)
        n_logical = (num_tokens + block_size - 1) // block_size  # 4
        # Physical blocks intentionally permuted.
        physical_ids = [3, 1, 7, 5]
        num_blocks = max(physical_ids) + 1

        compressed_kv = torch.randn(
            num_tokens, HEAD_DIM, dtype=torch.bfloat16, device=self.device
        )
        k_cache = self._alloc_cache(num_blocks, block_size)

        # Build slot_mapping: token i → physical_ids[i // block_size] * block_size + (i % block_size)
        slot_mapping = torch.tensor(
            [
                physical_ids[i // block_size] * block_size + (i % block_size)
                for i in range(num_tokens)
            ],
            dtype=torch.int64,
            device=self.device,
        )
        quantize_and_insert_k_cache(compressed_kv, k_cache, slot_mapping)

        # Gather back through the same block_table layout.
        out = torch.zeros(
            1, num_tokens, HEAD_DIM, dtype=torch.bfloat16, device=self.device
        )
        seq_lens = torch.tensor([num_tokens], dtype=torch.int32, device=self.device)
        block_table = torch.tensor(
            [physical_ids], dtype=torch.int32, device=self.device
        )
        # Pad the block_table to the max ``ceil(seq_len/block_size)`` via the
        # logical layout — kernel reads ``block_table[req, pos // block_size]``.
        self.assertEqual(block_table.shape, (1, n_logical))
        dequantize_and_gather_k_cache(
            out=out,
            k_cache=k_cache,
            seq_lens=seq_lens,
            gather_lens=None,
            block_table=block_table,
            block_size=block_size,
            offset=0,
        )
        recovered = out[0, :num_tokens]
        self._assert_nope_within_ue8m0_bound(compressed_kv, recovered)
        self._assert_rope_exact(compressed_kv, recovered)

    def test_flat_slot_gather_supports_physical_rows_and_swa_ring(self):
        """Flat-slot SWA read handles physical block-table rows with a small ring.

        New DSV4 layout: block_table rows cover 16K raw tokens, while the SWA
        pool block only has ``128 + step`` entries. The read path must consume
        already-translated flat slots instead of deriving ``pos // ring_entries``.
        """
        ring_entries = 132
        tokens_per_block = 16384
        block_table = torch.tensor(
            [[1, 2], [3, 4]], dtype=torch.int32, device=self.device
        )
        seq_lens = torch.tensor(
            [tokens_per_block, tokens_per_block + 12],
            dtype=torch.int32,
            device=self.device,
        )
        gather_lens = torch.tensor([128, 130], dtype=torch.int32, device=self.device)
        max_gather = int(gather_lens.max().item())

        step = torch.arange(max_gather, dtype=torch.long, device=self.device)
        seq_l = seq_lens.to(torch.long)
        gather_l = gather_lens.to(torch.long)
        abs_pos = (seq_l - gather_l).unsqueeze(1) + step.unsqueeze(0)
        valid = step.unsqueeze(0) < gather_l.unsqueeze(1)
        block_in_seq = abs_pos // tokens_per_block
        in_block = abs_pos % ring_entries
        req = torch.arange(2, dtype=torch.long, device=self.device).unsqueeze(1)
        block_id = block_table.to(torch.long)[req, block_in_seq]
        slot_mapping = torch.where(
            valid,
            block_id * ring_entries + in_block,
            torch.full_like(in_block, -1),
        )
        # Exercise in-window -1 semantics, not only padded rows.
        slot_mapping[0, 5] = -1

        compressed_kv = torch.randn(
            2 * max_gather, HEAD_DIM, dtype=torch.bfloat16, device=self.device
        )
        k_cache = self._alloc_cache(5, ring_entries)
        quantize_and_insert_k_cache(compressed_kv, k_cache, slot_mapping.reshape(-1))

        out = torch.full(
            (2, max_gather, HEAD_DIM),
            -3,
            dtype=torch.bfloat16,
            device=self.device,
        )
        dequantize_and_gather_k_cache_slots(
            out=out,
            k_cache=k_cache,
            slot_mapping=slot_mapping,
            gather_lens=gather_lens,
            offset=0,
        )

        valid = valid & (slot_mapping >= 0)
        recovered = out.reshape(-1, HEAD_DIM)[valid.reshape(-1)]
        expected = compressed_kv[valid.reshape(-1)]
        self._assert_nope_within_ue8m0_bound(expected, recovered)
        self._assert_rope_exact(expected, recovered)
        self.assertTrue(torch.all(out[0, 5] == 0))
        self.assertTrue(torch.all(out[0, 128:] == -3))

    def test_cp_byte_sliced_precomputed_compaction_roundtrip(self):
        """CP byte-sliced read/write must consume precomputed compaction only."""
        cp_size = 2
        full_entries_per_block = 16
        num_blocks = 4
        local_slice_bytes = full_entries_per_block * HEAD_BYTES // cp_size
        write_slots = torch.tensor(
            [16, 17, 18, 19, 32, 33, 34, 35, 48, 49],
            dtype=torch.int64,
            device=self.device,
        )
        compressed_kv = torch.randn(
            write_slots.numel(), HEAD_DIM, dtype=torch.bfloat16, device=self.device
        )
        raw_by_rank = [
            torch.zeros(
                num_blocks,
                local_slice_bytes,
                dtype=torch.uint8,
                device=self.device,
            )
            for _ in range(cp_size)
        ]
        write_compaction = build_cp_byte_sliced_slot_compaction(
            write_slots,
            full_entries_per_block=full_entries_per_block,
            num_blocks=num_blocks,
            validation_site="test.cp_byte.write",
            negative_mode="skip_minus_one",
        )

        for cp_rank, raw in enumerate(raw_by_rank):
            quantize_and_insert_k_cache_cp_byte_sliced(
                compressed_kv,
                raw,
                write_slots,
                full_entries_per_block=full_entries_per_block,
                cp_rank=cp_rank,
                cp_size=cp_size,
                compaction=write_compaction,
            )

        read_slots = torch.tensor(
            [
                [16, 17, -1, 32, 33, 34],
                [35, 48, 49, -1, -1, -1],
            ],
            dtype=torch.int64,
            device=self.device,
        )
        gather_lens = torch.tensor([6, 3], dtype=torch.int32, device=self.device)
        read_compaction = build_cp_byte_sliced_slot_compaction(
            read_slots,
            full_entries_per_block=full_entries_per_block,
            num_blocks=num_blocks,
            validation_site="test.cp_byte.read",
            negative_mode="skip_any",
            gather_lens=gather_lens,
        )

        def fake_all_gather(tensor, group):
            del group
            expected_local = raw_by_rank[0].index_select(
                0, read_compaction.unique_blocks
            )
            self.assertEqual(tuple(tensor.shape), tuple(expected_local.shape))
            return torch.cat(
                [
                    raw.index_select(0, read_compaction.unique_blocks)
                    for raw in raw_by_rank
                ],
                dim=0,
            )

        out = torch.full((2, 6, HEAD_DIM), -7, dtype=torch.bfloat16, device=self.device)
        with mock.patch(
            "rtp_llm.models_py.distributed.collective_torch.all_gather",
            side_effect=fake_all_gather,
        ):
            dequantize_and_gather_k_cache_slots_cp_byte_sliced(
                out=out,
                k_cache_raw=raw_by_rank[0],
                slot_mapping=read_slots,
                gather_lens=gather_lens,
                offset=0,
                full_entries_per_block=full_entries_per_block,
                cp_rank=0,
                cp_size=cp_size,
                compaction=read_compaction,
            )

        slot_to_k = {
            int(slot): compressed_kv[i] for i, slot in enumerate(write_slots.tolist())
        }
        for b in range(read_slots.shape[0]):
            for j in range(int(gather_lens[b].item())):
                slot = int(read_slots[b, j].item())
                if slot < 0:
                    self.assertTrue(torch.all(out[b, j] == 0))
                    continue
                expected = slot_to_k[slot].unsqueeze(0)
                recovered = out[b, j].unsqueeze(0)
                self._assert_rope_exact(expected, recovered)
                self._assert_nope_within_ue8m0_bound(expected, recovered)
        self.assertTrue(torch.all(out[1, 3:] == -7))

    def test_cp_byte_sliced_direct_scatter_matches_fallback_bitwise(self):
        cases = (
            # CP2 exercises chunks that cross a rank boundary and an empty row.
            (2, 16, [2, 3], 9, [9, 0, 4], 800),
            # CP4 exercises non-aligned request lengths and a larger block set.
            (4, 64, [1, 3, 6, 8, 11], 37, [1, 22, 37, 5], 128),
        )
        marker = -29
        for cp_size, entries_per_block, block_id_list, width, lens, padding in cases:
            with self.subTest(cp_size=cp_size, width=width):
                block_ids = torch.tensor(
                    block_id_list, dtype=torch.int64, device=self.device
                )
                num_blocks = max(block_id_list) + 1
                full_stride_bytes = entries_per_block * HEAD_BYTES + padding
                self.assertEqual(full_stride_bytes % cp_size, 0)
                local_slice_bytes = full_stride_bytes // cp_size

                write_slots = (
                    block_ids.unsqueeze(1) * entries_per_block
                    + torch.arange(
                        entries_per_block, dtype=torch.int64, device=self.device
                    ).unsqueeze(0)
                ).reshape(-1)
                values = torch.randn(
                    write_slots.numel(),
                    HEAD_DIM,
                    dtype=torch.bfloat16,
                    device=self.device,
                )
                raw_by_rank = [
                    torch.zeros(
                        num_blocks,
                        local_slice_bytes,
                        dtype=torch.uint8,
                        device=self.device,
                    )
                    for _ in range(cp_size)
                ]
                write_compaction = build_cp_byte_sliced_slot_compaction(
                    write_slots,
                    full_entries_per_block=entries_per_block,
                    num_blocks=num_blocks,
                    validation_site="test.direct_scatter.write",
                    negative_mode="skip_minus_one",
                )
                for cp_rank, raw in enumerate(raw_by_rank):
                    quantize_and_insert_k_cache_cp_byte_sliced(
                        values,
                        raw,
                        write_slots,
                        full_entries_per_block=entries_per_block,
                        cp_rank=cp_rank,
                        cp_size=cp_size,
                        compaction=write_compaction,
                    )

                batch = len(lens)
                linear = torch.arange(batch * width, device=self.device).view(
                    batch, width
                )
                read_slots = (
                    block_ids[linear.remainder(block_ids.numel())]
                    * entries_per_block
                    + (linear * 17 + 3).remainder(entries_per_block)
                )
                gather_lens = torch.tensor(lens, dtype=torch.int32, device=self.device)
                for batch_idx, gather_len in enumerate(lens):
                    if gather_len:
                        read_slots[batch_idx, gather_len // 2] = -1
                read_slots = read_slots.to(torch.int64).contiguous()
                compaction = build_cp_byte_sliced_slot_compaction(
                    read_slots,
                    full_entries_per_block=entries_per_block,
                    num_blocks=num_blocks,
                    validation_site="test.direct_scatter.read",
                    negative_mode="skip_any",
                    gather_lens=gather_lens,
                )

                def fake_all_gather(tensor, group):
                    del group
                    expected_local = raw_by_rank[0].index_select(
                        0, compaction.unique_blocks
                    )
                    self.assertTrue(torch.equal(tensor, expected_local))
                    return torch.cat(
                        [
                            raw.index_select(0, compaction.unique_blocks)
                            for raw in raw_by_rank
                        ],
                        dim=0,
                    )

                def run(enabled):
                    out = torch.full(
                        (batch, width + 6, HEAD_DIM),
                        marker,
                        dtype=torch.bfloat16,
                        device=self.device,
                    )
                    with mock.patch.dict(
                        "os.environ",
                        {"DSV4_CP_SWA_DIRECT_DEQUANT_SCATTER": "1" if enabled else "0"},
                    ), mock.patch(
                        "rtp_llm.models_py.distributed.collective_torch.all_gather",
                        side_effect=fake_all_gather,
                    ), mock.patch.object(
                        _swa_dequant_triton,
                        "dequantize_slots_to_bf16",
                        wraps=_swa_dequant_triton.dequantize_slots_to_bf16,
                    ) as fallback_dequant:
                        dequantize_and_gather_k_cache_slots_cp_byte_sliced(
                            out=out,
                            k_cache_raw=raw_by_rank[0],
                            slot_mapping=read_slots,
                            gather_lens=gather_lens,
                            offset=3,
                            full_entries_per_block=entries_per_block,
                            cp_rank=0,
                            cp_size=cp_size,
                            compaction=compaction,
                        )
                    return out, fallback_dequant.call_count

                fallback, fallback_calls = run(False)
                direct, direct_fallback_calls = run(True)
                self.assertEqual(fallback_calls, 1)
                self.assertEqual(direct_fallback_calls, 0)
                self.assertTrue(
                    torch.equal(direct.view(torch.int16), fallback.view(torch.int16))
                )

    def test_cp_byte_sliced_runtime_requires_compaction(self):
        cp_size = 2
        full_entries_per_block = 16
        num_blocks = 3
        local_slice_bytes = full_entries_per_block * HEAD_BYTES // cp_size
        raw = torch.zeros(
            num_blocks,
            local_slice_bytes,
            dtype=torch.uint8,
            device=self.device,
        )
        k = torch.randn(2, HEAD_DIM, dtype=torch.bfloat16, device=self.device)
        slots = torch.tensor([16, 17], dtype=torch.int64, device=self.device)
        with self.assertRaisesRegex(AssertionError, "metadata-precomputed compaction"):
            quantize_and_insert_k_cache_cp_byte_sliced(
                k,
                raw,
                slots,
                full_entries_per_block=full_entries_per_block,
                cp_rank=0,
                cp_size=cp_size,
                compaction=None,
            )

        out = torch.zeros(1, 2, HEAD_DIM, dtype=torch.bfloat16, device=self.device)
        read_slots = slots.view(1, 2)
        gather_lens = torch.tensor([2], dtype=torch.int32, device=self.device)
        with self.assertRaisesRegex(AssertionError, "metadata-precomputed compaction"):
            dequantize_and_gather_k_cache_slots_cp_byte_sliced(
                out=out,
                k_cache_raw=raw,
                slot_mapping=read_slots,
                gather_lens=gather_lens,
                offset=0,
                full_entries_per_block=full_entries_per_block,
                cp_rank=0,
                cp_size=cp_size,
                compaction=None,
            )


if __name__ == "__main__":
    unittest.main()
