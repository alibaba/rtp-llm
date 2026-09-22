"""V4.1 global decode producer: old owner path versus fused cache writes.

These tests build only an attention owner and its typed pools, not a model.
The old reference calls AttentionV41FP8._produce_global_decode with fusion
explicitly disabled. Cache comparisons use the physical block layouts, with
poison in block zero, unallocated slots and ratio2 indexer padding.
"""

from __future__ import annotations

import importlib
import os
import unittest
from pathlib import Path
from types import MethodType, SimpleNamespace
from unittest.mock import patch

import torch

from rtp_llm.models_py.modules.dsv4.attn_type import (
    CSA_KV,
    CSA_STATE,
    HCA_KV,
    INDEXER_KV,
)
from rtp_llm.models_py.modules.dsv4.fp8.attention_v41 import AttentionV41FP8

_FUSED_MODULE = "rtp_llm.models_py.modules.dsv4.fp8._v41_decode_global"
_KV_TPB = 128
_STATE_TPB = 256
_STATE_EB = 8
_MAX_SEQ_LEN = 512


def _fused():
    return importlib.import_module(_FUSED_MODULE)


def _layout_owner(device, ratio, batch, *, ring_entries=_STATE_EB):
    main_region = CSA_KV if ratio == 2 else HCA_KV
    pages = _MAX_SEQ_LEN // _KV_TPB
    # Permute physical ids independently across regions to catch pool mixups.
    main_table = torch.arange(1, batch * pages + 1, device=device).view(batch, pages)
    index_table = main_table.flip(1).contiguous()
    state_table = torch.arange(1, batch * 2 + 1, device=device).view(batch, 2)
    tables = {
        main_region: main_table.int(),
        INDEXER_KV: index_table.int(),
        CSA_STATE: state_table.int(),
    }
    pools = {
        main_region: torch.full(
            (batch * pages + 1, _KV_TPB // ratio, 288),
            0x5A,
            device=device,
            dtype=torch.uint8,
        ),
        INDEXER_KV: torch.full(
            (batch * pages + 1, _KV_TPB, 68),
            0x7F,
            device=device,
            dtype=torch.uint8,
        ),
        CSA_STATE: (
            torch.arange(
                (batch * 2 + 1) * ring_entries * 1024,
                device=device,
                dtype=torch.float32,
            )
            .remainder(97)
            .reshape(-1, 1024)
            - 48
        )
        * 0.03125,
    }
    # Any accidental read of the unallocated block will visibly change output.
    pools[CSA_STATE][:ring_entries].fill_(1234.0)
    owner = SimpleNamespace(
        compress_ratio=ratio,
        head_dim=512,
        index_head_dim=128,
        index_n_heads=32,
        rope_head_dim=64,
        eps=1e-6,
        layer_id=2 if ratio == 2 else 20,
        kv_source_layer_id=2 if ratio == 2 else 20,
        _rope_max_seq_len=_MAX_SEQ_LEN,
        _cp_ctx=None,
        _kv_cache=SimpleNamespace(
            kernel_seq_size_per_block=_KV_TPB,
            seq_size_per_block=_STATE_TPB,
        ),
        _block_tables_by_type=tables,
        _shared_attention={"global": {}, "topk": {}},
        _test_pools=pools,
        _test_ring_entries=ring_entries,
    )
    owner._source_pool = lambda region: pools[region]
    owner._source_entries = lambda region, pool: (
        ring_entries if region == CSA_STATE else pool.shape[1]
    )
    for method in ("_slots", "_read_state", "_gather_shards", "_global_region"):
        setattr(owner, method, MethodType(getattr(AttentionV41FP8, method), owner))
    return owner


def _case(device, ratio, batch, span, *, seed=4141, norm_dtype=torch.float32):
    # CPU-only construction keeps fixture randomness independent of capture.
    generator = torch.Generator(device="cpu").manual_seed(seed)
    owner = _layout_owner(device, ratio, batch)
    for name in ("global_wkv", "global_wgate"):
        setattr(
            owner,
            name,
            (torch.randn(512, 5120, generator=generator) / 5120**0.5).to(
                device=device, dtype=torch.bfloat16
            ),
        )
    owner.index_wk = (torch.randn(128, 512, generator=generator) / 512**0.5).to(
        device=device, dtype=torch.bfloat16
    )
    owner.global_norm = (torch.rand(512, generator=generator) * 0.5 + 0.75).to(
        device=device, dtype=norm_dtype
    )
    owner.index_k_norm = (torch.rand(128, generator=generator) * 0.5 + 0.75).to(
        device=device, dtype=norm_dtype
    )
    # Extra frequency rows allow explicit out-of-table positions without an
    # unrelated RoPE indexing error. Allocated KV capacity remains 512 tokens.
    angles = torch.outer(torch.arange(1024).float(), torch.linspace(0.001, 0.1, 32))
    owner.freqs_cis = torch.polar(torch.ones_like(angles), angles).to(device)
    x = torch.randn(batch, span, 5120, generator=generator).to(
        device=device, dtype=torch.bfloat16
    )
    starts = torch.tensor([127, 255, 7, 508][:batch], device=device, dtype=torch.int64)
    req_ids = torch.arange(batch, device=device).repeat_interleave(span)
    positions = (starts[:, None] + torch.arange(span, device=device)).flatten()
    return SimpleNamespace(
        owner=owner, x=x, starts=starts, req_ids=req_ids, positions=positions
    )


def _clone_case(case):
    owner = case.owner
    cloned = _layout_owner(
        case.x.device,
        owner.compress_ratio,
        case.x.shape[0],
        ring_entries=owner._test_ring_entries,
    )
    for region, pool in owner._test_pools.items():
        cloned._test_pools[region].copy_(pool)
        cloned._block_tables_by_type[region] = owner._block_tables_by_type[
            region
        ].clone()
    for name in (
        "global_wkv",
        "global_wgate",
        "global_norm",
        "index_wk",
        "index_k_norm",
        "freqs_cis",
    ):
        setattr(cloned, name, getattr(owner, name))
    return SimpleNamespace(
        owner=cloned,
        x=case.x.clone(),
        starts=case.starts.clone(),
        req_ids=case.req_ids.clone(),
        positions=case.positions.clone(),
    )


def _reference(case):
    with patch.object(_fused(), "is_supported", return_value=False):
        AttentionV41FP8._produce_global_decode(
            case.owner, case.x, case.positions, case.req_ids, case.starts
        )


def _candidate(case):
    ok = _fused().try_produce_global(
        case.owner, case.x, case.positions, case.req_ids, case.starts
    )
    if not ok:
        raise AssertionError("Supported GPU fixture unexpectedly used fallback")


def _snapshot(case):
    return {region: pool.clone() for region, pool in case.owner._test_pools.items()}


def _write_mask(case, region):
    """Independent byte selection, not a slot-major interpretation of the pool."""
    owner = case.owner
    pool = owner._test_pools[region]
    state = region == CSA_STATE
    slots = owner._slots(
        region,
        case.positions,
        case.req_ids,
        **({"state_end": case.starts + case.x.shape[1]} if state else {}),
    ).cpu()
    if state:
        mask = torch.zeros(pool.shape[0], dtype=torch.bool)
        mask[slots[slots >= 0].long()] = True
        # The old index_copy deliberately writes zeros into sentinel slot0.
        mask[0] = True
        return mask.to(pool.device)
    entries = pool.shape[1]
    mask = torch.zeros(pool.shape[0], entries * pool.shape[2], dtype=torch.bool)
    for slot in slots[slots >= 0].tolist():
        block, offset = divmod(slot, entries)
        if region == INDEXER_KV:
            # Planar per block: payload plane then packed-scale plane.
            mask[block, offset * 64 : (offset + 1) * 64] = True
            start = entries * 64 + offset * 4
            mask[block, start : start + 4] = True
        else:
            # GLOBAL is also planar, with group-16 E4M3 scales.
            mask[block, offset * 256 : (offset + 1) * 256] = True
            start = entries * 256 + offset * 32
            mask[block, start : start + 32] = True
    return mask.reshape_as(pool).to(pool.device)


def _canonical_cache_zero_signs(pool, region):
    """Only the zero-input test permits semantically equal signed-zero data.

    Main payload bytes, indexer payload bytes, padding, and untouched
    regions remain exact. Differences are permitted only for e2m1 ±0
    nibbles (byte & 0x77 == 0). E4M3 group scales never round to zero
    (the codec floors the group maximum at 6 * 2**-9). This does not
    admit a one-ULP difference in nonzero data.
    """
    canonical = pool.contiguous().clone()
    blocks, entries, _ = canonical.shape
    raw = canonical.view(blocks, -1)
    if region == INDEXER_KV:
        payload = raw[:, : entries * 64]
        payload.masked_fill_((payload & 0x77) == 0, 0)
    else:
        data = raw[:, : entries * 256]
        data.masked_fill_((data & 0x77) == 0, 0)
    return canonical


def _assert_result(
    test, old, new, initial, *, check_untouched=True, allow_signed_zero=False
):
    for region, expected in old.owner._test_pools.items():
        actual = new.owner._test_pools[region]
        if region == CSA_STATE:
            # Zero is an allocator sentinel, never a valid state read. The old
            # path may write zero there; avoiding that write is also valid.
            torch.testing.assert_close(actual[1:], expected[1:], rtol=0, atol=0)
        else:
            raw_different = actual != expected
            if allow_signed_zero:
                different = _canonical_cache_zero_signs(
                    actual, region
                ) != _canonical_cache_zero_signs(expected, region)
                print(
                    f"zero-input ratio={old.owner.compress_ratio} region={region}: signed-zero data bytes={int(raw_different.sum())}, remaining different bytes={int(different.sum())}",
                    flush=True,
                )
            else:
                different = raw_different
            dump_path = os.environ.get("DSV41_DECODE_GLOBAL_DUMP_FAILURE")
            if dump_path and bool(different.any()):
                torch.save(
                    {
                        "ratio": old.owner.compress_ratio,
                        "x": old.x,
                        "starts": old.starts,
                        "positions": old.positions,
                        "req_ids": old.req_ids,
                        "tables": old.owner._block_tables_by_type,
                        "initial_pools": initial,
                        "old_pools": old.owner._test_pools,
                        "new_pools": new.owner._test_pools,
                        "weights": {
                            name: getattr(old.owner, name)
                            for name in (
                                "global_wkv",
                                "global_wgate",
                                "global_norm",
                                "index_wk",
                                "index_k_norm",
                                "freqs_cis",
                            )
                        },
                    },
                    Path(dump_path),
                )
            test.assertFalse(
                bool(different.any()),
                f"region={region}: {int(different.sum())} physical cache bytes differ",
            )
        if check_untouched:
            mask = _write_mask(new, region)
            torch.testing.assert_close(
                actual[~mask], initial[region][~mask], rtol=0, atol=0
            )


def _set_starts(case, starts):
    case.starts.copy_(
        torch.as_tensor(starts, device=case.x.device, dtype=case.starts.dtype)
    )
    span = case.x.shape[1]
    case.positions.copy_(
        (case.starts[:, None] + torch.arange(span, device=case.x.device)).flatten()
    )


class V41DecodeGlobalCPU(unittest.TestCase):
    def test_slot_contract_separates_uniform_indexer_and_compressed_main(self):
        for ratio in (1, 2):
            owner = _layout_owner("cpu", ratio, 2)
            positions = torch.tensor([-1, 0, 1, 126, 127, 128, 129, 511, 512])
            requests = torch.zeros_like(positions)
            for region in (owner._global_region(), INDEXER_KV):
                entries = owner._test_pools[region].shape[1]
                actual = owner._slots(region, positions, requests)
                expected = []
                for position in positions.tolist():
                    column = position // _KV_TPB
                    valid = 0 <= column < 4 and (position + 1) % ratio == 0
                    block = (
                        int(owner._block_tables_by_type[region][0, column])
                        if valid
                        else 0
                    )
                    expected.append(
                        block * entries + position % _KV_TPB // ratio
                        if valid and block > 0
                        else -1
                    )
                torch.testing.assert_close(
                    actual, torch.tensor(expected), rtol=0, atol=0
                )
                owner._block_tables_by_type[region][0, 0] = 0
                owner._block_tables_by_type[region][0, 1] = -1
                slots = owner._slots(
                    region, torch.tensor([127, 129]), torch.tensor([0, 0])
                )
                self.assertEqual(slots.tolist(), [-1, -1])
            if ratio == 2:
                self.assertEqual(owner._test_pools[CSA_KV].shape[1], 64)
                self.assertEqual(owner._test_pools[INDEXER_KV].shape[1], 128)

    def test_state_ring_uses_physical_coverage_and_reads_zero_for_unallocated(self):
        owner = _layout_owner("cpu", 2, 2)
        positions = torch.tensor([7, 8, 255, 256, 511, 512])
        requests = torch.zeros_like(positions)
        expected = []
        for position in positions.tolist():
            block = int(
                owner._block_tables_by_type[CSA_STATE][0, (position // 256) % 2]
            )
            expected.append(block * 8 + position % 8)
        torch.testing.assert_close(
            owner._slots(CSA_STATE, positions, requests),
            torch.tensor(expected),
            rtol=0,
            atol=0,
        )
        # Only the final ring tail of each physical block may be committed.
        positions = torch.arange(20)
        slots = owner._slots(
            CSA_STATE,
            positions,
            torch.zeros_like(positions),
            state_end=torch.tensor([20, 0]),
        )
        self.assertTrue((slots[:12] == -1).all())
        self.assertEqual(len(set(slots[12:].tolist())), 8)
        owner._block_tables_by_type[CSA_STATE][0, 0] = 0
        owner._block_tables_by_type[CSA_STATE][1, 1] = -1
        got = owner._read_state(torch.tensor([7, 257]), torch.tensor([0, 1]))
        torch.testing.assert_close(got, torch.zeros(2, 1024), rtol=0, atol=0)

    def test_cpu_gate_returns_false_without_mutating_pools(self):
        case = _case("cpu", 2, 1, 6)
        before = _snapshot(case)
        self.assertFalse(
            _fused().try_produce_global(
                case.owner, case.x, case.positions, case.req_ids, case.starts
            )
        )
        for region, pool in before.items():
            torch.testing.assert_close(
                case.owner._test_pools[region], pool, rtol=0, atol=0
            )


class V41DecodeGlobalCUDA(unittest.TestCase):
    def setUp(self):
        if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
            self.skipTest("SM100 CUDA required")
        torch.manual_seed(41041)
        environment = patch.dict(
            os.environ,
            {
                "DSV4_TRAP_INVALID_KV_ACCESS": "1",
                "DSV4_VALIDATE_INVALID_KV_ACCESS": "0",
            },
        )
        environment.start()
        self.addCleanup(environment.stop)

    @torch.inference_mode()
    def test_ratio_batch_span_matrix_all_cache_bytes(self):
        for ratio in (1, 2):
            for batch in (1, 2, 4):
                for span in (1, 5, 6):
                    with self.subTest(ratio=ratio, batch=batch, span=span):
                        old = _case("cuda", ratio, batch, span)
                        if span == 5:
                            old.positions = old.positions.int()
                            old.req_ids = old.req_ids.int()
                            old.starts = old.starts.int()
                        new = _clone_case(old)
                        initial = _snapshot(old)
                        _reference(old)
                        _candidate(new)
                        _assert_result(self, old, new, initial)

    @torch.inference_mode()
    def test_padded_main_and_table_row_strides_both_trap_modes(self):
        for ratio in (1, 2):
            for trap in ("0", "1"):
                with self.subTest(ratio=ratio, trap=trap):
                    old = _case("cuda", ratio, 2, 6)
                    new = _clone_case(old)
                    backings = []
                    for case in (old, new):
                        region = case.owner._global_region()
                        pool = case.owner._test_pools[region]
                        width = pool.shape[1] * 288
                        backing = torch.full(
                            (pool.shape[0], width + 256),
                            0xA5,
                            dtype=torch.uint8,
                            device="cuda",
                        )
                        padded = backing.as_strided(pool.shape, (width + 256, 288, 1))
                        padded.copy_(pool)
                        case.owner._test_pools[region] = padded
                        backings.append((backing[:, width:], 0xA5))
                        for region, table in case.owner._block_tables_by_type.items():
                            backing = torch.full(
                                (table.shape[0], table.shape[1] + 3),
                                -999,
                                dtype=table.dtype,
                                device="cuda",
                            )
                            padded = backing[:, : table.shape[1]]
                            padded.copy_(table)
                            case.owner._block_tables_by_type[region] = padded
                            backings.append((backing[:, table.shape[1] :], -999))
                    initial = _snapshot(old)
                    with patch.dict(os.environ, {"DSV4_TRAP_INVALID_KV_ACCESS": trap}):
                        _reference(old)
                        _candidate(new)
                    _assert_result(self, old, new, initial)
                    for padding, sentinel in backings:
                        self.assertTrue(bool((padding == sentinel).all()))

    @torch.inference_mode()
    def test_unallocated_pages_out_of_table_and_norm_dtype(self):
        for ratio in (1, 2):
            for dtype in (torch.float32, torch.bfloat16):
                with self.subTest(ratio=ratio, norm_dtype=dtype):
                    old = _case("cuda", ratio, 4, 6, norm_dtype=dtype)
                    _set_starts(old, [127, 255, 0, 511])
                    if dtype == torch.bfloat16:
                        old.owner._block_tables_by_type = {
                            region: table.long()
                            for region, table in old.owner._block_tables_by_type.items()
                        }
                    for table in old.owner._block_tables_by_type.values():
                        table[0, 0] = 0
                        table[1, 1] = -1
                        table[2].zero_()  # fake/unallocated request
                    new = _clone_case(old)
                    initial = _snapshot(old)
                    _reference(old)
                    _candidate(new)
                    _assert_result(self, old, new, initial)

    @torch.inference_mode()
    def test_zero_quantization_and_large_pair_scores(self):
        for ratio in (1, 2):
            with self.subTest(ratio=ratio):
                old = _case("cuda", ratio, 2, 6)
                old.owner.global_wkv.zero_()
                old.owner.global_wgate.zero_()
                old.owner.global_wgate[:, 0] = 2048
                old.owner.index_wk.zero_()
                old.owner._test_pools[CSA_STATE][8:].zero_()
                old.x[:, ::2, 0] = 1
                old.x[:, 1::2, 0] = -1
                new = _clone_case(old)
                initial = _snapshot(old)
                _reference(old)
                _candidate(new)
                _assert_result(self, old, new, initial, allow_signed_zero=True)

    @torch.inference_mode()
    def test_speculative_rejection_reuses_committed_prefix_and_overwrites_tail(self):
        for accepted in (0, 1, 3, 5):
            with self.subTest(accepted=accepted):
                old = _case("cuda", 2, 2, 6)
                _set_starts(old, [7, 253])
                new = _clone_case(old)
                initial = _snapshot(old)
                _reference(old)
                _candidate(new)
                _assert_result(self, old, new, initial)
                initial = _snapshot(old)
                next_starts = [7 + accepted, 253 + accepted]
                for case in (old, new):
                    _set_starts(case, next_starts)
                replacement = torch.randn_like(old.x)
                old.x.copy_(replacement)
                new.x.copy_(replacement)
                _reference(old)
                _candidate(new)
                _assert_result(self, old, new, initial)

    @torch.inference_mode()
    def test_graph_updates_device_inputs_page_tables_and_rejected_positions(self):
        for ratio in (1, 2):
            with self.subTest(ratio=ratio):
                old = _case("cuda", ratio, 4, 6)
                new = _clone_case(old)
                initial = _snapshot(old)
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    for _ in range(3):
                        _candidate(new)
                torch.cuda.current_stream().wait_stream(stream)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    _candidate(new)
                # Capture/warmup wrote state; restore only once, then preserve
                # state across real accepted/rejected replay sequences.
                for region, pool in initial.items():
                    new.owner._test_pools[region].copy_(pool)
                rounds = (
                    [127, 255, 7, 508],
                    [128, 256, 8, 509],
                    [129, 258, 9, 510],
                    [130, 259, 10, 511],
                )
                for round_id, starts in enumerate(rounds):
                    before = _snapshot(old)
                    for case in (old, new):
                        _set_starts(case, starts)
                        if round_id == 1:
                            # Device table changes must be read on replay,
                            # including zero/negative unallocated sentinels.
                            for table in case.owner._block_tables_by_type.values():
                                table[2, 0] = 0
                                table[3, -1] = -1
                        if round_id == 2:
                            for (
                                region,
                                table,
                            ) in case.owner._block_tables_by_type.items():
                                table[2, 0] = (
                                    old.owner._test_pools[region].shape[0]
                                    // (
                                        old.owner._test_ring_entries
                                        if region == CSA_STATE
                                        else 1
                                    )
                                    - 1
                                )
                    value = torch.randn_like(old.x)
                    old.x.copy_(value)
                    new.x.copy_(value)
                    _reference(old)
                    graph.replay()
                    torch.cuda.synchronize()
                    _assert_result(self, old, new, before)
                del graph

    @torch.inference_mode()
    def test_unsupported_inputs_do_not_write_pools(self):
        base = _case("cuda", 2, 1, 6)
        for label in (
            "missing_cache",
            "validation_debug",
            "long_span",
            "sharded",
            "wrong_dtype",
            "wrong_width",
            "too_many_tokens",
        ):
            with self.subTest(label=label):
                case = _clone_case(base)
                if label == "missing_cache":
                    case.owner._kv_cache = None
                elif label == "long_span":
                    case.x = case.x[:, :1].expand(1, 9, 5120).contiguous()
                    case.req_ids = torch.zeros(9, device="cuda", dtype=torch.int64)
                    case.positions = case.starts + torch.arange(9, device="cuda")
                elif label == "sharded":
                    case.owner._cp_ctx = SimpleNamespace(
                        cp_size=4, cp_rank=0, kv_cache_sharded=True
                    )
                elif label == "wrong_dtype":
                    case.x = case.x.float()
                elif label == "wrong_width":
                    case.x = case.x[..., :256].contiguous()
                elif label == "too_many_tokens":
                    case.x = case.x[:, :1].expand(65, 1, 5120).contiguous()
                before = _snapshot(case)
                helper = _fused()
                with patch.dict(
                    os.environ,
                    {
                        "DSV4_TRAP_INVALID_KV_ACCESS": (
                            "0" if label == "validation_debug" else "1"
                        ),
                        "DSV4_VALIDATE_INVALID_KV_ACCESS": (
                            "1" if label == "validation_debug" else "0"
                        ),
                    },
                ), patch.object(
                    helper, "_compress_norm_main_store_kernel"
                ) as stage_a, patch.object(
                    helper, "_index_norm_store_state_kernel"
                ) as stage_b:
                    self.assertFalse(
                        _fused().try_produce_global(
                            case.owner,
                            case.x,
                            case.positions,
                            case.req_ids,
                            case.starts,
                        )
                    )
                self.assertEqual(stage_a.mock_calls, [])
                self.assertEqual(stage_b.mock_calls, [])
                for region, pool in before.items():
                    torch.testing.assert_close(
                        case.owner._test_pools[region], pool, rtol=0, atol=0
                    )


if __name__ == "__main__":
    unittest.main()
