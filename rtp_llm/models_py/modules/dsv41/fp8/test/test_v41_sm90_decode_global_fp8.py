"""SM90 FP8-primary global producer: exact Torch-owner cache/state contracts.

Fixtures retain GLOBAL FP4 fake quantization before 584-byte V4 storage and
INDEXER 64-entry planar 132-byte pages. The old owner with fusion disabled
is the independent oracle; all byte, norm-backend and graph gates are exact.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from rtp_llm.models_py.modules.dsv41.attn_type import CSA_STATE, INDEXER_KV
from rtp_llm.models_py.modules.dsv41.fp8 import _v41_decode_global as fused
from rtp_llm.models_py.modules.dsv41.fp8 import (
    _v41_sm90_decode_global_kernels as kernels,
)
from rtp_llm.models_py.modules.dsv41.fp8 import _v41_sm90_v4_cache as v4_codec
from rtp_llm.models_py.modules.dsv41.fp8._v41_sm90_fp8_indexer_cache import (
    dg_pages,
    initialize_zero_page,
)
from rtp_llm.models_py.modules.dsv41.fp8.attention_v41 import AttentionV41FP8
from rtp_llm.models_py.modules.dsv41.fp8.test import test_v41_decode_global as fixture
from rtp_llm.models_py.modules.dsv41.fp8.test.test_v41_sm90_v4_cache import _pool


def _make_fixture(ratio, batch, span, *, norm_dtype=torch.bfloat16):
    case = fixture._case(
        "cuda", ratio, batch, span, norm_dtype=norm_dtype, compact_indexer=ratio == 2
    )
    owner = case.owner
    main_region = owner._global_region()
    main = owner._test_pools[main_region]
    case.main_backing, owner._test_pools[main_region] = _pool(
        main.shape[0], main.shape[1]
    )
    index = owner._test_pools[INDEXER_KV]
    owner._test_pools[INDEXER_KV] = torch.full(
        (*index.shape[:2], 132), 0x7F, device="cuda", dtype=torch.uint8
    )
    # The verify spans fit a sixteen-entry physical state ring. Keep
    # the same deterministic poison and original slot/read implementations.
    ring = 16
    owner._test_ring_entries = ring
    owner._test_pools[CSA_STATE] = (
        torch.arange((batch * 2 + 1) * ring * 1024, device="cuda", dtype=torch.float32)
        .remainder(97)
        .reshape(-1, 1024)
        - 48
    ) * 0.03125
    owner._test_pools[CSA_STATE][:ring].fill_(1234.0)
    owner._source_entries = lambda region, pool: (
        ring if region == CSA_STATE else pool.shape[1]
    )
    owner._kv_cache.get_entries_per_block = lambda region: (
        ring
        if region == CSA_STATE
        else owner._test_pools[
            INDEXER_KV if region == "indexer_kv_c2" else region
        ].shape[1]
    )
    owner._indexer_mode = "fp8_dg"
    owner._use_v4_kv_cache = True
    # This exact module preserves GLOBAL E2M1 fake quant BEFORE V4 storage.
    # A bare SWA writer would silently change the model's GLOBAL semantics.
    owner._swa_codec = v4_codec
    pages = dg_pages(owner._test_pools[INDEXER_KV])
    initialize_zero_page(pages)
    owner._fp8_zero_page_marker = (id(owner._kv_cache), pages.untyped_storage()._cdata)
    return case


def _reference(case):
    with patch.object(fused, "is_supported", return_value=False):
        AttentionV41FP8._produce_global_decode(
            case.owner, case.x, case.positions, case.req_ids, case.starts
        )


def _snapshot(case):
    return {
        "main_backing": case.main_backing.clone(),
        INDEXER_KV: case.owner._test_pools[INDEXER_KV].clone(),
        CSA_STATE: case.owner._test_pools[CSA_STATE].clone(),
    }


def _restore(case, saved):
    case.main_backing.copy_(saved["main_backing"])
    for region in (INDEXER_KV, CSA_STATE):
        case.owner._test_pools[region].copy_(saved[region])


def _physical_write_masks(case):
    owner = case.owner
    result = {
        name: torch.zeros_like(value, dtype=torch.bool)
        for name, value in _snapshot(case).items()
    }
    for region, name in (
        (owner._global_region(), "main_backing"),
        (INDEXER_KV, INDEXER_KV),
    ):
        pool = owner._test_pools[region]
        slots = owner._slots(region, case.positions, case.req_ids).cpu()
        mask = result[name].view(-1)
        for slot in slots[slots >= 0].tolist():
            if region == INDEXER_KV:
                page, offset = divmod(slot, 64)
                base = page * (64 * 132)
                mask[base + offset * 128 : base + (offset + 1) * 128] = True
                mask[
                    base + 64 * 128 + offset * 4 : base + 64 * 128 + (offset + 1) * 4
                ] = True
            else:
                block, offset = divmod(slot, pool.shape[1])
                base = block * pool.stride(0)
                mask[base + offset * 576 : base + (offset + 1) * 576] = True
                start = base + pool.shape[1] * 576 + offset * 8
                mask[start : start + 8] = True
    if owner.compress_ratio == 2:
        lengths = owner._shared_attention.get("decode_input_lengths", case.x.shape[1])
        slots = owner._slots(
            CSA_STATE, case.positions, case.req_ids, state_end=case.starts + lengths
        )
        result[CSA_STATE][slots[slots >= 0].long()] = True
        result[CSA_STATE][0] = True  # old Torch's allocator sentinel zero write
    return result


def _assert_equal(test, old, new, before):
    expected, actual = _snapshot(old), _snapshot(new)
    masks = _physical_write_masks(new)
    for name in expected:
        if name == CSA_STATE:
            # Original contract excludes only sentinel row zero; all live
            # state rows retain exact FP32 values and rejected-tail semantics.
            torch.testing.assert_close(
                actual[name][1:], expected[name][1:], rtol=0, atol=0
            )
        else:
            test.assertTrue(torch.equal(actual[name], expected[name]), name)
        torch.testing.assert_close(
            actual[name][~masks[name]], before[name][~masks[name]], rtol=0, atol=0
        )


def _candidate(case):
    if not fused.is_supported(
        case.owner, case.x, case.positions, case.req_ids, case.starts
    ):
        raise AssertionError("FP8-primary fixture unexpectedly selected fallback")
    AttentionV41FP8._produce_global_decode(
        case.owner, case.x, case.positions, case.req_ids, case.starts
    )


class _KernelLaunchSpy:
    def __init__(self, kernel):
        self.kernel = kernel
        self.grids = []

    def __getitem__(self, grid):
        self.grids.append(grid)
        return self.kernel[grid]


class SM90FP8GlobalProducerTest(unittest.TestCase):
    def setUp(self):
        if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (
            9,
            0,
        ):
            self.skipTest("SM90 required")

    @torch.inference_mode()
    def test_all_worker_batches_gamma5_and7_exact_physical_bytes(self):
        for ratio in (1, 2):
            for norm_dtype in (torch.bfloat16, torch.float32):
                batches = range(1, 17) if norm_dtype == torch.bfloat16 else (1, 8, 16)
                for batch in batches:
                    for span in (6, 8):
                        with self.subTest(
                            ratio=ratio, norm_dtype=norm_dtype, batch=batch, span=span
                        ):
                            old = _make_fixture(
                                ratio, batch, span, norm_dtype=norm_dtype
                            )
                            new = _make_fixture(
                                ratio, batch, span, norm_dtype=norm_dtype
                            )
                            before = _snapshot(old)
                            original_input = new.x.clone()
                            _reference(old)
                            spy = _KernelLaunchSpy(
                                kernels._index_fp8_rope_store_state_kernel
                            )
                            with patch.object(
                                kernels, "_index_fp8_rope_store_state_kernel", spy
                            ):
                                _candidate(new)
                            self.assertEqual(spy.grids, [(batch * span,)])
                            _assert_equal(self, old, new, before)
                            torch.testing.assert_close(
                                new.x, original_input, rtol=0, atol=0
                            )

    @torch.inference_mode()
    def test_graph_observes_changing_inputs_tables_and_rejection_history(self):
        for ratio in (1, 2):
            old = _make_fixture(ratio, 4, 6)
            new = _make_fixture(ratio, 4, 6)
            initial = _snapshot(new)
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):
                    _candidate(new)
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                _candidate(new)
            _restore(new, initial)
            for round_id, starts in enumerate(
                ([127, 255, 7, 508], [128, 256, 8, 509], [129, 258, 9, 510])
            ):
                before = _snapshot(old)
                for case in (old, new):
                    fixture._set_starts(case, starts)
                    if round_id == 1:
                        for table in case.owner._block_tables_by_type.values():
                            table[2, 0] = 0
                            table[3, -1] = -1
                changed = torch.randn_like(old.x)
                old.x.copy_(changed)
                new.x.copy_(changed)
                _reference(old)
                graph.replay()
                torch.cuda.synchronize()
                _assert_equal(self, old, new, before)

    @torch.inference_mode()
    def test_graph_observes_device_packed_lengths(self):
        old = _make_fixture(2, 2, 6)
        new = _make_fixture(2, 2, 6)
        for case in (old, new):
            lengths = torch.tensor([4, 8], device="cuda", dtype=torch.int32)
            case.owner._shared_attention["decode_input_lengths"] = lengths
            case.req_ids = torch.repeat_interleave(
                torch.arange(2, device="cuda"), lengths
            )
            case.positions = torch.cat(
                (
                    case.starts[0] + torch.arange(4, device="cuda"),
                    case.starts[1] + torch.arange(8, device="cuda"),
                )
            )
        before = _snapshot(old)
        _reference(old)
        _candidate(new)
        _assert_equal(self, old, new, before)
        initial = _snapshot(new)
        for _ in range(3):
            _candidate(new)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            _candidate(new)
        _restore(new, initial)
        for case in (old, new):
            lengths = case.owner._shared_attention["decode_input_lengths"]
            lengths.copy_(torch.tensor([5, 7], device="cuda", dtype=torch.int32))
            case.req_ids.copy_(
                torch.repeat_interleave(torch.arange(2, device="cuda"), lengths)
            )
            case.positions.copy_(
                torch.cat(
                    (
                        case.starts[0] + torch.arange(5, device="cuda"),
                        case.starts[1] + torch.arange(7, device="cuda"),
                    )
                )
            )
        before = _snapshot(old)
        _reference(old)
        graph.replay()
        torch.cuda.synchronize()
        _assert_equal(self, old, new, before)

    @torch.inference_mode()
    def test_fp8_layout_rejections_and_unchanged_fp4_architecture_gate(self):
        case = _make_fixture(2, 2, 6)
        owner = case.owner
        args = (owner, case.x, case.positions, case.req_ids, case.starts)
        self.assertTrue(fused.is_supported(*args))
        with patch.object(owner, "_use_v4_kv_cache", False):
            self.assertFalse(fused.is_supported(*args))
        with patch.object(owner, "_indexer_mode", "fp4"):
            self.assertFalse(fused.is_supported(*args))
        with patch.object(owner, "_cp_ctx", SimpleNamespace(cp_size=2)):
            self.assertFalse(fused.is_supported(*args))
        with patch.object(
            fused, "invalid_kv_access_validation_enabled", return_value=True
        ):
            self.assertFalse(fused.is_supported(*args))
        index = owner._test_pools[INDEXER_KV]
        owner._test_pools[INDEXER_KV] = torch.empty(
            (index.shape[0], 96, 132), dtype=torch.uint8, device="cuda"
        )
        self.assertFalse(fused.is_supported(*args))
        owner._test_pools[INDEXER_KV] = index
        region = owner._global_region()
        main = owner._test_pools[region]
        owner._test_pools[region] = torch.empty(
            (*main.shape[:2], 288), dtype=torch.uint8, device="cuda"
        )
        self.assertFalse(fused.is_supported(*args))
        owner._test_pools[region] = main
        # Check SM100's owner invariant: no V4 cache, main288, index68.
        # The new 132-byte route is rejected; the old 68-byte route is eligible.
        for ratio in (1, 2):
            legacy = fixture._case("cuda", ratio, 2, 6)
            legacy.owner._use_v4_kv_cache = False
            legacy.owner._indexer_mode = "fp4"
            args = (
                legacy.owner,
                legacy.x,
                legacy.positions,
                legacy.req_ids,
                legacy.starts,
            )
            self.assertTrue(fused.is_supported(*args))
            with patch.object(
                torch.cuda, "get_device_capability", return_value=(10, 0)
            ):
                self.assertTrue(fused.is_supported(*args))
                pool = legacy.owner._test_pools[INDEXER_KV]
                legacy.owner._test_pools[INDEXER_KV] = torch.empty(
                    (*pool.shape[:2], 132), dtype=torch.uint8, device="cuda"
                )
                self.assertFalse(fused.is_supported(*args))

    @torch.inference_mode()
    def test_sm90_executes_unchanged_fp4_cache_path(self):
        for ratio in (1, 2):
            with self.subTest(ratio=ratio):
                old = fixture._case("cuda", ratio, 2, 6)
                new = fixture._clone_case(old)
                for case in (old, new):
                    case.owner._indexer_mode = "fp4"
                    case.owner._use_v4_kv_cache = False
                initial = fixture._snapshot(old)
                fixture._reference(old)
                spy = _KernelLaunchSpy(kernels._index_fp8_rope_store_state_kernel)
                with patch.object(kernels, "_index_fp8_rope_store_state_kernel", spy):
                    fixture._candidate(new)
                self.assertEqual(spy.grids, [])
                fixture._assert_result(self, old, new, initial)


if __name__ == "__main__":
    unittest.main()
