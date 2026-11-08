"""SM120 cases split out of test_swa_fp8_kv_roundtrip.py.
These cases live in this sibling file, leaving the original test file
untouched."""

from __future__ import annotations

from rtp_llm.models_py.modules.dsv4.fp8._swa_dequant_triton import (
    CPByteSlicedSwaPrefixPending,
    _validate_runtime_gather_lens,
    dequantize_and_gather_k_cache,
    dequantize_and_gather_k_cache_slots,
    dequantize_and_gather_k_cache_slots_cp_byte_sliced,
    dequantize_packed_k_cache_flat,
    discard_dequantize_and_gather_k_cache_slots_cp_byte_sliced,
    gather_k_cache_packed,
    gather_packed_k_cache_slots_cp_byte_sliced,
    prepare_dequantize_and_gather_k_cache_slots_cp_byte_sliced,
    prepare_gather_packed_k_cache_slots_cp_byte_sliced,
    start_dequantize_and_gather_k_cache_slots_cp_byte_sliced,
    wait_dequantize_and_gather_k_cache_slots_cp_byte_sliced,
)

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
HEAD_BYTES = 584
FP8_MAX = 448.0
QUANT_BLOCK = 64


class DirectTritonFastPathSupportedTest(unittest.TestCase):
    """Pin ``direct_triton_fast_path_supported`` for known device inputs.

    CUDA cases mock ``torch.cuda.get_device_capability`` so the class runs
    on CPU-only hosts.  The helper caches through
    ``_direct_triton_fast_path_supported_cached`` (``functools.lru_cache``),
    so each mocked case clears that cache first and ``setUp`` schedules a
    cleanup clear to keep mocked capabilities from leaking into later tests.
    """

    def setUp(self):
        cached = _swa_dequant_triton._direct_triton_fast_path_supported_cached
        cached.cache_clear()
        self.addCleanup(cached.cache_clear)

    def test_cpu_device_is_unsupported(self):
        self.assertFalse(
            _swa_dequant_triton.direct_triton_fast_path_supported(torch.device("cpu"))
        )

    def test_cuda_capability_major_gates_support(self):
        device = torch.device("cuda:0")
        cases = (
            ((9, 0), True),
            ((10, 0), True),
            ((12, 0), False),  # SM120 is intentionally unsupported
            ((12, 1), False),
            ((8, 0), False),
        )
        cached = _swa_dequant_triton._direct_triton_fast_path_supported_cached
        for capability, expected in cases:
            with self.subTest(capability=capability):
                cached.cache_clear()
                with mock.patch.object(
                    torch.cuda, "get_device_capability", return_value=capability
                ):
                    self.assertEqual(
                        _swa_dequant_triton.direct_triton_fast_path_supported(device),
                        expected,
                    )

    def test_capability_lookup_failure_is_unsupported(self):
        with mock.patch.object(
            torch.cuda,
            "get_device_capability",
            side_effect=RuntimeError("capability lookup failed"),
        ):
            self.assertFalse(
                _swa_dequant_triton.direct_triton_fast_path_supported(
                    torch.device("cuda:0")
                )
            )


class PackedPrefixWorkspaceTest(unittest.TestCase):
    def setUp(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        torch.manual_seed(37)

    @staticmethod
    def _cache(count, page):
        stride = ((page * HEAD_BYTES + 575) // 576) * 576
        blocks = (count + page - 1) // page
        storage = torch.full((blocks * stride,), 173, dtype=torch.uint8, device="cuda")
        return (
            storage.as_strided((blocks, page, HEAD_BYTES), (stride, HEAD_BYTES, 1)),
            storage,
        )

    def test_packed_prefix_and_fresh_rows(self):
        for batch in (1, 3):
            for width in (257, 4223):
                for source_page in (64, 256):
                    for destination_page in (64, 256):
                        with self.subTest(
                            batch=batch,
                            width=width,
                            source_page=source_page,
                            destination_page=destination_page,
                        ):
                            self._check(batch, width, source_page, destination_page)

    def test_prompt_lengths_reuse_compiled_prefix_kernel(self):
        from rtp_llm.models_py.modules.dsv4.fp8 import _swa_kv_insert_triton

        kernel = _swa_kv_insert_triton._quantize_and_insert_k_kernel
        self._check(1, 257, 64, 64)
        device = torch.cuda.current_device()
        compiled = set(kernel.device_caches[device][0])
        self._check(1, 389, 64, 64)
        self.assertEqual(set(kernel.device_caches[device][0]), compiled)

    def test_byte_sliced_prefix_and_fresh_rows(self):
        for cp_size in (2, 4, 8):
            for destination_page in (64, 256):
                with self.subTest(cp_size=cp_size, destination_page=destination_page):
                    self._check(3, 257, 128, destination_page, cp_size=cp_size)
        self._check(3, 257, 128, 64, cp_size=8, all_invalid=True)

    def test_async_byte_sliced_prefix_and_fresh_rows(self):
        for cp_size in (2, 4, 8):
            for page in (64, 256):
                with self.subTest(cp_size=cp_size, page=page):
                    self._check(3, 257, 128, page, cp_size=cp_size, asynchronous=True)

    def _check(
        self,
        batch,
        width,
        source_page,
        destination_page,
        cp_size=None,
        all_invalid=False,
        asynchronous=False,
    ):
        count = batch * width
        source, source_storage = self._cache(2048, source_page)
        quantize_and_insert_k_cache(
            torch.randn(2048, HEAD_DIM, device="cuda", dtype=torch.bfloat16),
            source,
            torch.arange(2048, device="cuda"),
        )
        original_source = source_storage.clone()
        slots = torch.randint(0, 2048, (batch, 127), device="cuda")
        if cp_size == 4:
            slots = slots % source_page + (slots % 3) * (3 * source_page)
        slots[:, 3] = -1
        if all_invalid:
            slots.fill_(-1)
        lens = torch.tensor(
            [127] if batch == 1 else [0, 63, 127], device="cuda", dtype=torch.int32
        )
        values = torch.randn(count, HEAD_DIM, device="cuda", dtype=torch.bfloat16)
        expected, expected_storage = self._cache(count, destination_page)
        actual, actual_storage = self._cache(count, destination_page)
        destination = torch.arange(count, device="cuda")
        quantize_and_insert_k_cache(values, expected, destination)

        # Independent byte-layout oracle, including missing slots and padding.
        prefix_destinations = (
            torch.arange(127, device="cuda")[None, :]
            + torch.arange(batch, device="cuda")[:, None] * width
        )
        valid_prefix = torch.arange(127, device="cuda")[None, :] < lens[:, None]
        source_slots = slots[valid_prefix]
        target_slots = prefix_destinations[valid_prefix]
        safe_source = source_slots.clamp_min(0)
        for region, size in ((0, 576), (1, 8)):
            offsets = torch.arange(size, device="cuda")[None, :]
            source_offsets = (
                (safe_source // source_page) * source.stride(0)
                + region * source_page * 576
                + (safe_source % source_page) * size
            )[:, None] + offsets
            target_offsets = (
                (target_slots // destination_page) * expected.stride(0)
                + region * destination_page * 576
                + (target_slots % destination_page) * size
            )[:, None] + offsets
            expected_storage[target_offsets] = torch.where(
                (source_slots >= 0)[:, None],
                source_storage[source_offsets],
                0,
            )
        # Serving does not initialize the BF16 rows whose packed bytes are copied.
        values[target_slots] = float("nan")
        prefix_cache, prefix_slots = source, slots
        if cp_size is not None:
            compaction = build_cp_byte_sliced_slot_compaction(
                slots,
                full_entries_per_block=source_page,
                num_blocks=source.shape[0],
                validation_site="test.packed_prefix.cp",
                negative_mode="skip_minus_one",
                gather_lens=lens,
            )
            stride = source.stride(0)
            self.assertEqual(stride % cp_size, 0)
            rank_bytes = stride // cp_size
            full_blocks = source_storage.view(source.shape[0], stride)
            shards = [
                full_blocks[:, rank * rank_bytes : (rank + 1) * rank_bytes].contiguous()
                for rank in range(cp_size)
            ]

            def all_gather(local, group):
                del group
                expected_local = shards[0].index_select(0, compaction.unique_blocks)
                self.assertTrue(torch.equal(local, expected_local))
                return torch.cat(
                    [
                        shard.index_select(0, compaction.unique_blocks)
                        for shard in shards
                    ]
                )

            def async_gather(out, local, group, async_op):
                self.assertTrue(async_op)
                torch.cuda._sleep(1_000_000)
                out.copy_(all_gather(local, group))
                return mock.Mock()

            with mock.patch(
                "rtp_llm.models_py.distributed.collective_torch.all_gather",
                side_effect=all_gather,
            ), mock.patch(
                "rtp_llm.models_py.distributed.collective_torch._get_group",
                return_value=object(),
            ), mock.patch.object(
                torch.distributed, "is_initialized", return_value=True
            ), mock.patch.object(
                torch.distributed, "get_world_size", return_value=cp_size
            ), mock.patch.object(
                torch.distributed, "all_gather_into_tensor", side_effect=async_gather
            ):
                gather_kwargs = dict(
                    k_cache_raw=shards[0],
                    slot_mapping=slots,
                    gather_lens=lens,
                    full_entries_per_block=source_page,
                    cp_rank=0,
                    cp_size=cp_size,
                    compaction=compaction,
                )
                if asynchronous:
                    pending = start_dequantize_and_gather_k_cache_slots_cp_byte_sliced(
                        **gather_kwargs, offset=0, stream=torch.cuda.Stream()
                    )
                    self.assertIsNotNone(pending)
                    prefix_cache, prefix_slots = (
                        prepare_gather_packed_k_cache_slots_cp_byte_sliced(
                            pending, stream=torch.cuda.Stream()
                        )
                    )
                    wait_dequantize_and_gather_k_cache_slots_cp_byte_sliced(pending)
                    pending.work.wait.assert_called_once_with()
                    del pending
                else:
                    prefix_cache, prefix_slots = (
                        gather_packed_k_cache_slots_cp_byte_sliced(**gather_kwargs)
                    )

        def launch():
            quantize_and_insert_k_cache(
                values,
                actual,
                destination,
                prefix_cache=prefix_cache,
                prefix_slot_mapping=prefix_slots,
                prefix_gather_lens=lens,
            )

        for _ in range(3):
            launch()
        self.assertTrue(torch.equal(actual_storage, expected_storage))
        self.assertTrue(torch.equal(source_storage, original_source))
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            launch()
        actual_storage.fill_(173)
        graph.replay()
        torch.cuda.synchronize()
        self.assertTrue(torch.equal(actual_storage, expected_storage))
        self.assertTrue(torch.equal(source_storage, original_source))

    def test_prefix_source_must_not_alias_destination(self):
        cache, _ = self._cache(256, 64)
        with self.assertRaisesRegex(AssertionError, "must not alias"):
            quantize_and_insert_k_cache(
                torch.zeros(256, HEAD_DIM, device="cuda", dtype=torch.bfloat16),
                cache,
                torch.arange(256, device="cuda"),
                prefix_cache=cache,
                prefix_slot_mapping=torch.arange(127, device="cuda")[None, :],
                prefix_gather_lens=torch.tensor([127], device="cuda"),
            )


if __name__ == "__main__":
    import unittest

    unittest.main()
