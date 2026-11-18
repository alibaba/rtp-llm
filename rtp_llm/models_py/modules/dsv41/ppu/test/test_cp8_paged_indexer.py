"""Shared-page CP Indexer ABI contracts and explicit PPU arithmetic probes.

CPU tests double the provider, not its arithmetic. Device tests use real
MXFP4 quantization, both real DeepGEMM scoring paths, token selection and
RTP sparse remapping. The synthetic cache publication is test-only; these
tests do not claim WORLD8 cache transport or model performance evidence.
"""

import os
import unittest
import weakref
from contextlib import ExitStack
from dataclasses import replace
from types import SimpleNamespace as NS
from unittest.mock import call, patch

import torch

from rtp_llm.models_py.modules.dsv41.cp import CPContext
from rtp_llm.models_py.modules.dsv41.fp8 import _v41_sparse_prefill_indexer as sparse
from rtp_llm.models_py.modules.dsv41.fp8._indexer_score import fp8_fp4_mqa_indexer_score
from rtp_llm.models_py.modules.dsv41.fp8._v41_prefill_indexer import PrefillIndexerKeys
from rtp_llm.models_py.modules.dsv41.fp8.test.test_v41_prefill_topk import (
    assert_equivalent,
    reference,
)
from rtp_llm.models_py.modules.dsv41.ppu import _quant
from rtp_llm.models_py.modules.dsv41.ppu import cp_prefill_paged_indexer as paged
from rtp_llm.models_py.modules.dsv41.ppu import prefill_token_topk as token
from rtp_llm.models_py.modules.dsv41.ppu.cp_prefill_shared_indexer import (
    IndexerReadbackPages,
    SharedIndexerBank,
    page_layout,
)
from rtp_llm.models_py.modules.dsv41.ppu.native_cache_adapter import (
    SharedNativeCache,
    _source_tensor_key,
)


UPLOAD = "rtp_llm.models_py.modules.dsv41.fp8._v41_swa_metadata._upload"


def _cpu_upload(values, device, dtype):
    """CPU double for the pinned asynchronous host-upload boundary only."""
    return torch.tensor(values, device=device, dtype=dtype)


def _fixture(
    counts,
    prefix,
    chunks,
    *,
    ratio=1,
    device="cpu",
    source=20,
    padding=False,
    values=None,
):
    """Real bank type/layout, with only cache placement admission doubled."""
    counts, chunks = tuple(counts), tuple(chunks)
    cache = SharedNativeCache.__new__(SharedNativeCache)
    cache._is_prefill, cache.sharded, cache.lengths = (
        True,
        True,
        tuple(count * ratio for count in counts),
    )
    cache.geometry = NS(ratio=lambda layer: ratio)
    layout = page_layout(counts, prefix)
    storage = torch.zeros(
        layout.pages, paged.PAGE * 68, dtype=torch.uint8, device=device
    )
    payload = storage[:, : paged.PAGE * 64].view(layout.pages, paged.PAGE, 64)
    scales = (
        storage[:, paged.PAGE * 64 :].view(torch.int32).view(layout.pages, paged.PAGE)
    )
    scales.fill_(0x7F7F7F7F)
    if values is None:
        payload.copy_(
            (torch.arange(layout.pages * paged.PAGE, device=device).view(-1, 1) % 251)
            .expand(-1, 64)
            .reshape_as(payload)
            .to(torch.uint8)
        )
    else:
        # Prefix bytes are shared, and each tail is restored only once. Unused
        # physical tail rows remain zero, including their complete SF plane.
        prefix_pages = prefix // paged.PAGE
        payload[:prefix_pages].copy_(
            values[0][0][:prefix].view(torch.uint8).view(prefix_pages, paged.PAGE, 64)
        )
        scales[:prefix_pages].copy_(
            values[0][1][:prefix].view(prefix_pages, paged.PAGE)
        )
        for request, count in enumerate(counts):
            tail = count - prefix
            first = layout.tail_page_bases[request]
            pages = (tail + paged.PAGE - 1) // paged.PAGE
            if tail:
                payload[first : first + pages].reshape(-1, 64)[:tail].copy_(
                    values[request][0][prefix:].view(torch.uint8)
                )
                scales[first : first + pages].reshape(-1)[:tail].copy_(
                    values[request][1][prefix:]
                )
    readback = IndexerReadbackPages.create(storage, counts, prefix)
    table = torch.tensor(layout.request_table_host, dtype=torch.int32, device=device)
    bank = SharedIndexerBank.from_readback(readback, cache, source, 7, table)
    stamps = tuple(
        _source_tensor_key(value) for value in (bank.pages, bank.request_table)
    )
    cache._valid_shared_indexer = lambda value: (
        value is bank
        and value.epoch == 7
        and stamps
        == tuple(
            _source_tensor_key(item) for item in (value.pages, value.request_table)
        )
    )
    rows = sum(chunks)
    request_ids = torch.repeat_interleave(
        torch.arange(8, device=device, dtype=torch.int32),
        torch.tensor(chunks, device=device),
    )
    positions = torch.tensor(
        tuple(
            counts[request] * ratio - 1
            for request in range(8)
            for _ in range(chunks[request])
        ),
        device=device,
        dtype=torch.int64,
    )
    valid = torch.ones(rows, dtype=torch.bool, device=device)
    if padding:
        valid[-1] = False
    prefixes = (prefix * ratio,) * 8
    lengths = tuple((count - prefix) * ratio for count in counts)
    cp = CPContext(
        8,
        0,
        rows,
        rows * 8,
        sum(lengths),
        torch.arange(rows, device=device),
        prefixes[0],
        positions,
        valid,
        torch.arange(sum(lengths), device=device),
        max(cache.lengths),
        NS(),
        req_id_per_token=request_ids,
        prefix_lengths=torch.tensor(prefixes, device=device),
        chunk_lengths_per_req=chunks,
        kv_cache_sharded=True,
        input_lengths_global_host=lengths,
        prefix_lengths_host=prefixes,
    )
    state = NS(
        context=cp,
        cache=cache,
        prefill=True,
        idle=False,
        compacted=True,
        lengths=lengths,
        query_prefixes=prefixes,
        attn=NS(is_cuda_graph=False),
        runtime=NS(
            model=NS(_prefill_cp_size=8, parallelism_config=NS(tp_size=8)),
            policy=NS(
                mode="compact",
                keep_tokens=140,
                window=128,
                max_rollback_tokens=0,
                encoder_replay=False,
            ),
        ),
    )
    slices, start = [], 0
    for chunk in chunks:
        slices.append(slice(start, start + chunk))
        start += chunk
    return (
        state,
        bank,
        dict(
            request_slices=tuple(slices), positions=positions, request_ids=request_ids
        ),
    )


def _query(rows, *, device="cpu", weights_dtype=torch.bfloat16):
    return (
        torch.zeros(rows, 32, 64, dtype=torch.int8, device=device),
        torch.full((rows, 32), 0x7F7F7F7F, dtype=torch.int32, device=device),
        torch.ones(rows, 32, dtype=weights_dtype, device=device),
    )


def _plan(rows, key_count):
    selected = torch.tensor([0, 31, 32, 63], dtype=torch.int32).expand(rows, -1)
    indices = torch.full((rows, 2048), 63, dtype=torch.int32)
    indices[:, :4] = selected
    return sparse.SparsePrefillPlan(
        torch.zeros(rows, dtype=torch.int32),
        torch.full((rows,), key_count, dtype=torch.int32),
        indices,
        torch.full((rows,), 32, dtype=torch.int32),
        torch.zeros(16, dtype=torch.uint8),
        key_count,
        8,
    )


class _ProviderDouble:
    """Records provider boundary arguments; deliberately supplies no math oracle."""

    def __init__(self):
        self.dense_metadata, self.dense_calls = [], []
        self.sparse_metadata, self.sparse_calls = [], []

    def get_num_sms(self):
        return 120

    def get_paged_mqa_logits_metadata(self, *args, **kwargs):
        self.dense_metadata.append((args, kwargs))
        return torch.zeros((121, 2), dtype=torch.int32)

    def fp8_fp4_paged_mqa_logits(self, *args, **kwargs):
        self.dense_calls.append((args, kwargs))
        return torch.zeros(args[0][0].shape[0], args[6], dtype=torch.bfloat16)

    def get_paged_sparse_mqa_logits_metadata(self, *args):
        self.sparse_metadata.append(args)
        return torch.zeros(32, dtype=torch.uint8)

    def fp8_fp4_paged_sparse_mqa_logits(self, *args):
        self.sparse_calls.append(args)
        return torch.zeros(args[0][0].shape[0], args[4] * args[5], dtype=torch.bfloat16)


class CP8PagedIndexerContracts(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(torch.no_grad())
        self.stack.enter_context(
            patch.object(paged, "_device_supported", return_value=True)
        )
        self.stack.enter_context(
            patch(
                "rtp_llm.models_py.modules.dsv41.ppu.layers.cp8_yoco_operator_eligible",
                return_value=True,
            )
        )
        self.provider = _ProviderDouble()
        self.stack.enter_context(
            patch.object(paged, "_deep_gemm", return_value=self.provider)
        )
        self.upload = self.stack.enter_context(patch(UPLOAD, side_effect=_cpu_upload))

    def fixture(self):
        state, bank, kwargs = _fixture(
            (768, 769, 800, 1024, 1025, 768, 900, 1100),
            256,
            (1, 3, 0, 2, 1, 1, 0, 2),
            padding=True,
        )
        rows = paged.prepare_rows(state, bank, ratio=1, **kwargs)
        self.assertIsNotNone(rows)
        return state, bank, kwargs, rows

    def test_ragged_request_mapping_uses_real_ids_and_valid_padding(self):
        state, bank, _, rows = self.fixture()
        self.assertEqual(rows.request_ids_host, (0, 1, 1, 1, 3, 3, 4, 5, 7, 7))
        self.assertEqual(rows.request_ids.tolist(), list(rows.request_ids_host))
        self.assertEqual(
            rows.tables.tolist(),
            [list(bank.request_table_host[r]) for r in rows.request_ids_host],
        )
        # The existing selector sees clamped padding positions too; its final
        # CP valid-row mask remains responsible for discarding that output.
        self.assertEqual(
            rows.visible.tolist(),
            [768, 769, 769, 769, 1024, 1024, 1025, 768, 1100, 1100],
        )
        tile = rows.slice(slice(1, 5))
        self.assertEqual(tile.request_ids_host, (1, 1, 1, 3))
        self.assertEqual(tile.tables.tolist()[-1], list(bank.request_table_host[3]))
        with self.assertRaises(ValueError):
            rows.slice(slice(0, 0))
        state.context = replace(state.context)
        self.assertFalse(rows.valid())

    def test_host_upload_boundary_preserves_values_owners_and_mutation_guard(self):
        state, bank, kwargs = _fixture(
            (768, 769, 800, 1024, 1025, 768, 900, 1100),
            256,
            (1, 3, 0, 2, 1, 1, 0, 2),
            padding=True,
        )
        source_ids = kwargs["request_ids"].clone()
        source_positions = kwargs["positions"].clone()
        source_pages, source_table = bank.pages.clone(), bank.request_table.clone()
        references = []

        def upload(values, device, dtype):
            value = _cpu_upload(values, device, dtype)
            references.append(weakref.ref(value))
            return value

        with patch(UPLOAD, side_effect=upload) as boundary:
            rows = paged.prepare_rows(state, bank, ratio=1, **kwargs)
        expected_ids = (0, 1, 1, 1, 3, 3, 4, 5, 7, 7)
        self.assertIsNotNone(rows)
        self.assertEqual(
            boundary.call_args_list,
            [
                call(expected_ids, kwargs["positions"].device, torch.int32),
                call(bank.request_counts, kwargs["positions"].device, torch.int64),
            ],
        )
        # The published rows retain their uploaded ID owner after the upload
        # callback returns. CPU checks do not qualify pinned-copy completion.
        self.assertIs(references[0](), rows.request_ids)
        self.assertIsNot(rows.request_ids, kwargs["request_ids"])
        self.assertEqual(rows.request_ids.tolist(), list(expected_ids))
        self.assertEqual(
            rows.visible.tolist(),
            [bank.request_counts[request] for request in expected_ids],
        )
        self.assertTrue(rows.valid())
        tile = rows.slice(slice(1, 5))
        rows.request_ids[1] = 7
        self.assertFalse(rows.valid())
        self.assertFalse(tile.valid())
        for got, before in (
            (kwargs["request_ids"], source_ids),
            (kwargs["positions"], source_positions),
            (bank.pages, source_pages),
            (bank.request_table, source_table),
        ):
            self.assertTrue(torch.equal(got, before))
        self.assertFalse(self.provider.dense_metadata or self.provider.sparse_metadata)

    def test_capability_handshake_is_prelaunch_and_rejects_unbound_or_other_roles(self):
        state, bank, kwargs, _ = self.fixture()
        self.assertTrue(paged.can_prepare_rows(state, **kwargs))
        before = bank.pages.clone()
        with patch.object(
            torch, "tensor", side_effect=AssertionError("capability allocation")
        ):
            self.assertTrue(paged.can_prepare_rows(state, **kwargs))
        bad = dict(kwargs, positions=kwargs["positions"].clone())
        self.assertFalse(paged.can_prepare_rows(state, **bad))
        for owner, field, value in (
            (state, "prefill", False),
            (state, "idle", True),
            (state.runtime.model, "_prefill_cp_size", 1),
            (state.runtime.model, "_is_decode_role", True),
            (state.runtime.policy, "keep_tokens", 128),
        ):
            previous = getattr(owner, field, None)
            setattr(owner, field, value)
            self.assertFalse(paged.can_prepare_rows(state, **kwargs))
            setattr(owner, field, previous)
        with patch.object(paged, "_deep_gemm", return_value=None):
            self.assertFalse(paged.can_prepare_rows(state, **kwargs))
        with patch.object(paged, "_device_supported", return_value=False):
            self.assertFalse(paged.can_prepare_rows(state, **kwargs))
        self.assertTrue(torch.equal(before, bank.pages))
        self.assertFalse(self.provider.dense_metadata or self.provider.sparse_metadata)

    def test_dense_keeps_packed_queries_and_bf16_weights_with_mandatory_metadata_hint(
        self,
    ):
        _, _, _, rows = self.fixture()
        q, sf, weights = _query(rows.count, weights_dtype=torch.float32)
        original = tuple(value.clone() for value in (q, sf, weights))
        for _ in range(2):
            self.assertEqual(
                paged.score_dense(q, sf, weights, rows).shape, (rows.count, rows.width)
            )
        self.assertEqual(len(self.provider.dense_metadata), 1)
        args, kwargs = self.provider.dense_metadata[0]
        self.assertEqual(args[0].data_ptr(), rows.visible.data_ptr())
        self.assertEqual(args[1:], (256, 120))
        self.assertEqual(kwargs["metadata_extra"], (1, 32, 128, torch.int8))
        self.assertIs(kwargs["indices"], rows.request_ids)
        args, kwargs = self.provider.dense_calls[0]
        self.assertEqual(args[0][0].shape, (rows.count, 1, 32, 64))
        self.assertEqual(args[0][0].data_ptr(), q.data_ptr())
        self.assertEqual(args[0][1].data_ptr(), sf.data_ptr())
        self.assertEqual(args[2].dtype, torch.bfloat16)
        self.assertIs(kwargs["indices"], rows.request_ids)
        for got, expected in zip((q, sf, weights), original):
            self.assertTrue(torch.equal(got, expected))

    def test_sparse_consumes_existing_compact_end_and_retains_logical_causal_plan(self):
        _, _, _, rows = self.fixture()
        plan = _plan(rows.count, rows.width)
        before = tuple(
            value.clone() for value in (plan.sparse_indices, plan.row_ke, plan.end)
        )
        q, sf, weights = _query(rows.count)
        for _ in range(2):
            self.assertEqual(
                paged.score_sparse(q, sf, weights, rows, plan).shape,
                (rows.count, 16384),
            )
        self.assertEqual(len(self.provider.sparse_metadata), 1)
        args = self.provider.sparse_metadata[0]
        self.assertIs(args[0], plan.end)
        self.assertIsNot(args[0], plan.row_ke)
        self.assertIs(args[1], rows.tables)
        self.assertEqual(args[2].tolist(), list(rows.request_ids_host))
        self.assertEqual(args[3], 256)
        self.assertIs(args[4], plan.sparse_indices)
        self.assertEqual(args[5:], (torch.int8, 8))
        for got, expected in zip((plan.sparse_indices, plan.row_ke, plan.end), before):
            self.assertTrue(torch.equal(got, expected))
        plan.sparse_indices[0, 0] = 2
        with self.assertRaisesRegex(RuntimeError, "stale CP paged sparse"):
            paged.score_sparse(q, sf, weights, rows, plan)

    def test_unsupported_inputs_partial_prefix_and_stale_bank_do_not_launch_or_mutate(
        self,
    ):
        state, bank, kwargs, rows = self.fixture()
        uploads = self.upload.call_count
        q, sf, weights = _query(rows.count)
        before = bank.pages.clone()
        self.assertIsNone(paged.score_dense(q.float(), sf, weights, rows))
        plan = replace(
            _plan(rows.count, rows.width),
            row_key_offsets=torch.zeros(rows.count, dtype=torch.int32),
        )
        self.assertIsNone(paged.score_sparse(q, sf, weights, rows, plan))
        self.assertIsNone(
            paged.prepare_rows(state, replace(bank, prefix_rows=257), ratio=1, **kwargs)
        )
        bank.request_table[0, 0] = 1
        self.assertIsNone(paged.score_dense(q, sf, weights, rows))
        self.assertTrue(torch.equal(before, bank.pages))
        self.assertFalse(self.provider.dense_calls or self.provider.sparse_calls)
        self.assertEqual(self.upload.call_count, uploads)


def _device_values(counts, prefix, *, margin=False, nonfinite=False):
    """Quantize once with the same PPU query/K MXFP4 codec used by CP."""
    common = torch.randn(prefix, 128, device="cuda", dtype=torch.bfloat16) * 0.1
    if margin:
        common.zero_()
        common[-256:] = 2
    values = []
    for request, count in enumerate(counts):
        tail = (
            torch.randn(count - prefix, 128, device="cuda", dtype=torch.bfloat16) * 0.1
        )
        if margin:
            tail.zero_()
            first = request * 17 % max(1, tail.shape[0] - 256)
            tail[first : first + 256] = 2
        packed, sf = _quant.quantize_indexer_q(
            torch.cat((common, tail)).view(count, 1, 128)
        )
        packed, sf = packed[:, 0].contiguous(), sf[:, 0].contiguous()
        if nonfinite:
            sf[prefix - 2] = -1  # UE8M0 NaN bytes (255).
            sf[prefix - 1] = -16843010  # UE8M0 maximum finite bytes (254).
        values.append((packed, sf))
    return values


def _legacy_dense(q, sf, weights, rows):
    expected = torch.empty(rows.count, rows.width, device="cuda", dtype=torch.bfloat16)
    for request in range(8):
        indices = [
            row for row, owner in enumerate(rows.request_ids_host) if owner == request
        ]
        if not indices:
            continue
        chunk = slice(indices[0], indices[-1] + 1)
        keys, key_sf = rows.bank.materialize(
            request, rows.bank.request_counts[request], split=True
        )
        scores = fp8_fp4_mqa_indexer_score(
            q[chunk],
            sf[chunk],
            keys,
            key_sf,
            weights[chunk],
            torch.zeros_like(rows.visible[chunk]),
            rows.visible[chunk],
            max_seqlen_k=rows.bank.request_counts[request],
            return_bf16=True,
        )
        expected[chunk, : scores.shape[1]] = scores
    return expected


@unittest.skipUnless(
    os.environ.get("RTP_V41_TEST_LOCAL_WORKSPACE_DEVICE") == "1",
    "explicit PPU device test",
)
class CP8PagedIndexerDevice(unittest.TestCase):
    def setUp(self):
        self.assertTrue(torch.cuda.is_available())
        self.assertTrue(paged._device_supported(torch.empty(1, device="cuda")))
        self.assertIsNotNone(paged._deep_gemm())
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(torch.no_grad())
        # Placement/role admission is synthetic; native scorer and selector are real.
        self.stack.enter_context(
            patch(
                "rtp_llm.models_py.modules.dsv41.ppu.layers.cp8_yoco_operator_eligible",
                return_value=True,
            )
        )
        self.stack.enter_context(patch.dict(os.environ, {token.ENV: "1"}))

    def test_async_upload_lifetime_survives_pinned_allocator_reuse(self):
        counts = (768, 769, 800, 1024, 1025, 768, 900, 1100)
        state, bank, kwargs = _fixture(
            counts, 256, (1, 3, 0, 2, 1, 1, 0, 2), device="cuda", padding=True
        )
        rows = paged.prepare_rows(state, bank, ratio=1, **kwargs)
        self.assertIsNotNone(rows)
        # The real helper releases its pinned temporary after enqueueing H2D.
        # Reusing similarly sized host allocations must not overwrite that
        # outstanding transfer. The allocator owns its completion lifetime.
        for _ in range(32):
            scratch_ids = torch.empty(rows.count, dtype=torch.int32, pin_memory=True)
            scratch_counts = torch.empty(8, dtype=torch.int64, pin_memory=True)
            scratch_ids.fill_(-1)
            scratch_counts.fill_(-1)
            del scratch_ids, scratch_counts
        torch.cuda.synchronize()
        expected_ids = (0, 1, 1, 1, 3, 3, 4, 5, 7, 7)
        self.assertEqual(rows.request_ids.tolist(), list(expected_ids))
        self.assertEqual(rows.visible.tolist(), [counts[r] for r in expected_ids])
        self.assertEqual(
            rows.tables.tolist(),
            [list(bank.request_table_host[r]) for r in expected_ids],
        )
        self.assertTrue(rows.valid())

    def rows(
        self,
        counts,
        prefix,
        chunks,
        *,
        ratio=1,
        margin=False,
        nonfinite=False,
        padding=True,
        empty_last=False,
        varying_windows=False,
    ):
        values = _device_values(counts, prefix, margin=margin, nonfinite=nonfinite)
        state, bank, kwargs = _fixture(
            counts,
            prefix,
            chunks,
            ratio=ratio,
            device="cuda",
            padding=padding,
            values=values,
        )
        if varying_windows:
            for request, chunk in enumerate(kwargs["request_slices"]):
                count = chunk.stop - chunk.start
                if count:
                    begin = prefix * ratio
                    span = counts[request] * ratio - 1 - begin
                    kwargs["positions"][chunk] = begin + torch.arange(
                        count, device="cuda"
                    ) * span // max(1, count - 1)
        if empty_last:
            kwargs["positions"][-1] = -1
        rows = paged.prepare_rows(state, bank, ratio=ratio, **kwargs)
        self.assertIsNotNone(rows)
        return rows

    def compare_logits(self, actual, expected, visible):
        mask = torch.arange(actual.shape[1], device="cuda")[None] < visible[:, None]
        torch.testing.assert_close(
            actual[mask], expected[mask], rtol=0, atol=0, equal_nan=True
        )

    def select(self, actual, expected, visible, *, exact_ids=False):
        self.compare_logits(actual, expected, visible)
        original = expected.masked_fill(
            torch.arange(expected.shape[1], device="cuda")[None] >= visible[:, None],
            -torch.inf,
        )
        selected = token.try_select_tokens(actual, visible, disposable=True)
        self.assertIsNotNone(selected)
        control = reference(original, visible)
        assert_equivalent(self, selected, control, original, visible)
        if exact_ids:
            self.assertTrue(
                torch.equal(selected.sort(-1).values, control.sort(-1).values)
            )
        return selected, control, original

    def test_actual_dense_full_and_compact_cp_shapes_match_legacy_bf16_logits(self):
        for width, prefix, chunk, ratio in (
            (65536, 63488, 512, 2),
            (131072, 126976, 18, 1),
        ):
            with self.subTest(width=width, rows_per_request=chunk):
                rows = self.rows(
                    (width,) * 8,
                    prefix,
                    (chunk,) * 8,
                    ratio=ratio,
                    varying_windows=True,
                )
                query = (
                    torch.randn(
                        rows.count, 32, 128, device="cuda", dtype=torch.bfloat16
                    )
                    * 0.1
                )
                q, sf = _quant.quantize_indexer_q(query)
                weights = (
                    torch.randn(rows.count, 32, device="cuda", dtype=torch.bfloat16)
                    * 0.01
                )
                actual = paged.score_dense(q, sf, weights, rows)
                expected = _legacy_dense(q, sf, weights, rows)
                self.compare_logits(actual, expected, rows.visible)
                del actual, expected, q, sf, weights, query, rows

    def test_actual_dense_selection_positive_margin_ties_and_nonfinite_slot_contract(
        self,
    ):
        for margin, nonfinite in ((True, False), (False, False), (False, True)):
            with self.subTest(margin=margin, nonfinite=nonfinite):
                rows = self.rows(
                    (3072,) * 8,
                    1024,
                    (2,) * 8,
                    margin=margin,
                    nonfinite=nonfinite,
                    empty_last=True,
                )
                query = torch.ones(
                    rows.count, 32, 128, device="cuda", dtype=torch.bfloat16
                )
                q, sf = _quant.quantize_indexer_q(query)
                weights = (
                    torch.ones(rows.count, 32, device="cuda", dtype=torch.bfloat16) / 32
                )
                actual = paged.score_dense(q, sf, weights, rows)
                expected = _legacy_dense(q, sf, weights, rows)
                self.select(actual, expected, rows.visible, exact_ids=margin)
                # Explicit all-zero scores exercise unspecified cutoff ties.
                if not margin and not nonfinite:
                    weights.zero_()
                    self.select(
                        paged.score_dense(q, sf, weights, rows),
                        _legacy_dense(q, sf, weights, rows),
                        rows.visible,
                    )

    def test_actual_sparse_ragged_request_pairs_compact_counts_and_page_boundaries(
        self,
    ):
        counts, chunks, prefix = (
            tuple(16643 + request * 17 for request in range(8)),
            (1, 3, 0, 2, 1, 1, 0, 2),
            16384,
        )
        rows = self.rows(counts, prefix, chunks, nonfinite=True, empty_last=True)
        candidates = torch.full(
            (rows.count, 2048), -1, dtype=torch.int32, device="cuda"
        )
        # Duplicate, invalid, shared-prefix/tail-boundary and partial-final blocks.
        candidates[:, :8] = torch.tensor(
            [0, 31, 32, prefix // 8 - 1, prefix // 8, prefix // 8, -2, 1 << 20],
            device="cuda",
        )
        tail_blocks = (
            torch.tensor(
                [counts[owner] for owner in rows.request_ids_host], device="cuda"
            )
            - 1
        ) // 8
        candidates[:, 8] = tail_blocks
        self.compare_sparse(rows, candidates)

    def test_actual_sparse_full_2048_candidates_in_eight_compact_request_domains(self):
        rows = self.rows((131072,) * 8, 126976, (18,) * 8, empty_last=True)
        chosen = torch.cat(
            (
                torch.arange(2045, device="cuda"),
                torch.tensor([15871, 15872, 16383], device="cuda"),
            )
        )
        candidates = chosen.to(torch.int32).expand(rows.count, -1).clone()
        candidates[0, 0] = candidates[0, 1]  # 2047 unique blocks.
        candidates[1].fill_(-1)  # Empty selected set with a nonempty logical KV window.
        self.compare_sparse(rows, candidates)

    def compare_sparse(self, rows, candidates):
        counts = rows.bank.request_counts
        query = (
            torch.randn(rows.count, 32, 128, device="cuda", dtype=torch.bfloat16) * 0.1
        )
        q, sf = _quant.quantize_indexer_q(query)
        weights = (
            torch.randn(rows.count, 32, device="cuda", dtype=torch.bfloat16) * 0.01
        )
        plan = sparse.prepare_plan(candidates, rows.visible, max(counts), ppu=True)
        self.assertIsNotNone(plan)
        actual = paged.score_sparse(q, sf, weights, rows, plan)
        expected = torch.empty_like(actual)
        for request, chunk in enumerate(rows.request_slices):
            if chunk.start == chunk.stop:
                continue
            keys, key_sf = rows.bank.materialize(request, counts[request], split=True)
            local_plan = sparse.prepare_plan(
                candidates[chunk], rows.visible[chunk], counts[request], ppu=True
            )
            expected[chunk] = sparse.score(
                q[chunk],
                sf[chunk],
                PrefillIndexerKeys(keys, key_sf),
                weights[chunk],
                local_plan,
            )
            self.assertTrue(torch.equal(local_plan.end, plan.end[chunk]))
        selected, control, original = self.select(actual, expected, plan.end)
        mapped = sparse.remap(selected, plan, logits=original)
        expected_mapped = sparse.remap(control, plan, logits=original)
        self.assertIsNotNone(mapped)
        # Ties may choose different members; both outputs retain the original
        # finite count and cannot escape this row's causal request interval.
        torch.testing.assert_close(
            (mapped >= 0).sum(-1), (expected_mapped >= 0).sum(-1), rtol=0, atol=0
        )
        self.assertTrue(
            ((mapped == -1) | ((mapped >= 0) & (mapped < plan.row_ke[:, None])))
            .all()
            .item()
        )
        self.assertTrue((mapped[-1] == -1).all().item())


if __name__ == "__main__":
    unittest.main()
