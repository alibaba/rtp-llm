"""Real GPU compact-reader probes; these are not model or 1M generation tests.

Expected KV is produced independently from scalar format values before packing
bytes. Attention uses a dense float64 formula over only the selected <=640 rows.
No CPU fallback or GPU skip is permitted in this target.
"""

import json
import math
import os
import unittest
from dataclasses import replace

import torch

from rtp_llm.models_py.modules.dsv41.cache_layout import CacheRegion
from rtp_llm.models_py.modules.dsv41.compact_reader import (
    CompactPages,
    GlobalBinding,
    SwaBinding,
    compact_attention,
    gather_compact,
)

_FP4_VALUES = torch.tensor(
    [
        0.0,
        0.5,
        1.0,
        1.5,
        2.0,
        3.0,
        4.0,
        6.0,
        -0.0,
        -0.5,
        -1.0,
        -1.5,
        -2.0,
        -3.0,
        -4.0,
        -6.0,
    ],
    dtype=torch.float32,
)


def _fixture(region, pages=4, entries=128):
    dimension = 128 if region == CacheRegion.INDEX_K else 512
    group = 16 if region == CacheRegion.GLOBAL else 32
    row_bytes = {
        CacheRegion.SWA: 528,
        CacheRegion.GLOBAL: 288,
        CacheRegion.INDEX_K: 68,
    }[region]
    count = pages * entries
    channel = torch.arange(dimension).unsqueeze(0)
    row = torch.arange(count).unsqueeze(1)
    if region == CacheRegion.SWA:
        # Finite E4M3 values between 0.125 and 1.875, with alternating signs.
        codes = ((channel * 7 + row * 3) % 32 + 32).to(torch.uint8)
        codes |= (((channel + row) % 2) * 128).to(torch.uint8)
        payload = codes
        values = codes.contiguous().view(torch.float8_e4m3fn).float()
    else:
        codes = ((channel * 5 + row * 3) % 16).to(torch.uint8)
        payload = codes[:, 0::2] | (codes[:, 1::2] << 4)
        values = _FP4_VALUES[codes.long()]
    groups = dimension // group
    selector = (torch.arange(groups).unsqueeze(0) + row) % 4
    if region == CacheRegion.GLOBAL:
        scale_values = torch.tensor([0.25, 0.5, 1.5, 2.0])[selector]
        scales = scale_values.to(torch.float8_e4m3fn).view(torch.uint8)
    else:
        scales = (selector + 125).to(torch.uint8)
        scale_values = torch.tensor([0.25, 0.5, 1.0, 2.0])[selector]
    expected = (values * scale_values.repeat_interleave(group, dim=1)).reshape(
        pages, entries, dimension
    )
    encoded = torch.cat((payload, scales), dim=1).reshape(pages, entries * row_bytes)
    stride = (entries * row_bytes + 511) // 512 * 512
    storage = torch.full((pages, stride), 255, dtype=torch.uint8)
    storage[:, : entries * row_bytes] = encoded
    storage[0].fill_(255)
    # Keep a non-contiguous visible shape when physical alignment adds padding.
    data = storage.cuda()[:, : entries * row_bytes]
    return CompactPages(data, region, entries), expected


def _ints(values):
    return torch.tensor(values, dtype=torch.int32, device="cuda")


def _attention_oracle(
    q,
    requests,
    positions,
    floors,
    swa,
    swa_values,
    sinks,
    global_kv=None,
    global_values=None,
    indices=None,
):
    q = q.cpu().double()
    requests, positions, floors = (
        requests.cpu().tolist(),
        positions.cpu().tolist(),
        floors.cpu().tolist(),
    )
    swa_ids = swa.page_ids.cpu().tolist()
    sinks = sinks.cpu().double()
    output = torch.zeros_like(q)
    lse = torch.full(q.shape[:2], -torch.inf, dtype=torch.float64)
    table = global_kv.page_table.cpu().tolist() if global_kv is not None else None
    selected = indices.cpu().tolist() if indices is not None else None
    for row, (request, position, floor) in enumerate(zip(requests, positions, floors)):
        if position == -1:
            continue
        rows = [
            swa_values[swa_ids[request], token % swa.pages.entries_per_page]
            for token in range(max(0, floor, position - 127), position + 1)
        ]
        if global_kv is not None:
            visible = (position + 1) // global_kv.compress_ratio
            for index in selected[row]:
                if 0 <= index < visible:
                    page, offset = divmod(index, global_kv.pages.entries_per_page)
                    rows.append(global_values[table[request][page], offset])
        kv = torch.stack(rows).double()
        scores = q[row] @ kv.T / math.sqrt(512)
        combined = torch.cat((scores, sinks.unsqueeze(1)), dim=1)
        weights = torch.softmax(combined, dim=1)[:, :-1]
        output[row] = weights @ kv
        lse[row] = torch.logsumexp(combined, dim=1)
    return output.float(), lse.float()


def _probe_attention_output_buffers(test, reader, *, planar=False):
    swa_pages, _ = _fixture(CacheRegion.SWA, pages=3, entries=136)
    global_pages, _ = _fixture(CacheRegion.GLOBAL, pages=3)
    swa = SwaBinding(swa_pages, _ints([1]), _ints([0]), _ints([16]))
    global_kv = GlobalBinding(global_pages, _ints([[1, 2]]), 1)
    if planar:
        from rtp_llm.models_py.modules.dsv41.flashmla import (
            PlanarGlobalBinding,
            PlanarSwaBinding,
        )

        swa = PlanarSwaBinding.from_compact(swa)
        global_kv = PlanarGlobalBinding.from_compact(global_kv)
    query = torch.zeros((1, 64, 512), dtype=torch.bfloat16, device="cuda")
    request_storage = torch.zeros(64, dtype=torch.int32, device="cuda")
    request = request_storage[:1]
    position, floor = _ints([7]), _ints([7])
    indices = _ints([[0] + [-1] * 63])
    sinks = torch.linspace(-2, 2, 64, device="cuda")
    shared_lse = torch.empty((1, 64), device="cuda")
    aliases = {
        "query-output": {"output": query},
        "cache-output": {
            "output": swa.pages.data[1, : query.numel() * 2]
            .view(torch.bfloat16)
            .view_as(query)
        },
        "sink-lse": {"lse": sinks[None, :]},
        "request-status": {"status": request_storage[None, :]},
        "indices-status": {"status": indices},
        "lse-status": {"lse": shared_lse, "status": shared_lse.view(torch.int32)},
    }
    inputs = (
        query,
        swa.pages.data,
        global_kv.pages.data,
        request_storage,
        indices,
        sinks,
    )
    original = [tensor.clone() for tensor in inputs]

    def run(q=query, **outputs):
        return reader(
            q,
            request,
            position,
            floor,
            swa,
            sinks,
            global_kv=global_kv,
            global_indices=indices,
            **outputs,
        )

    for name, buffers in aliases.items():
        with test.subTest(alias=name):
            with test.assertRaisesRegex(ValueError, "output buffers must not alias"):
                run(**buffers)
            for value, before in zip(inputs, original):
                torch.testing.assert_close(value, before, rtol=0, atol=0)

    query_bytes = query.numel() * 2
    vector_bytes = 64 * 4
    backing = torch.empty(
        2 * query_bytes + 2 * vector_bytes, dtype=torch.uint8, device="cuda"
    )
    q = backing[:query_bytes].view(torch.bfloat16).view_as(query)
    output = backing[query_bytes : 2 * query_bytes].view(torch.bfloat16).view_as(query)
    lse = backing[2 * query_bytes : 2 * query_bytes + vector_bytes].view(torch.float32)
    lse = lse[None]
    status = backing[-vector_bytes:].view(torch.int32)[None]
    q.copy_(query)
    buffers = {"output": output, "lse": lse, "status": status}
    pointers = tuple(tensor.data_ptr() for tensor in buffers.values())
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        run(q, **buffers)
        run(q, **buffers)
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        captured = run(q, **buffers)
    for step in range(3):
        position.fill_(7 + step)
        floor.copy_(position)
        q.add_(0.001953125)
        graph.replay()
        torch.cuda.synchronize()
        captured.check()
        direct = run(q)
        direct.check()
        for actual, expected in (
            (captured.output, direct.output),
            (captured.lse, direct.lse),
            (captured.status, direct.status),
        ):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        test.assertEqual(
            pointers, tuple(tensor.data_ptr() for tensor in buffers.values())
        )
    print(
        json.dumps(
            {
                "test": test.id(),
                "rejected_before_write": list(aliases),
                "shared_allocation_disjoint_buffers": True,
                "captures": 1,
                "replays": 3,
            }
        ),
        flush=True,
    )


class CompactReaderGpuTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
            raise RuntimeError(
                "this required GPU target needs an actual Blackwell device"
            )
        if os.environ.get("DSV41_NATIVE_COMPACT_READER") != "1":
            raise RuntimeError(
                "set DSV41_NATIVE_COMPACT_READER=1 for this explicit candidate probe"
            )

    def test_attention_output_aliases_rejected_and_disjoint_graph_buffers_reused(self):
        _probe_attention_output_buffers(self, compact_attention)

    def test_gather_rejects_output_aliases_without_overwriting_pages_or_metadata(self):
        pages, _ = _fixture(CacheRegion.INDEX_K)
        table, requests, positions, lengths = (
            _ints([[1, 2]]),
            _ints([0]),
            _ints([[0, 1]]),
            _ints([256]),
        )
        shared = torch.empty((1, 2, 128), dtype=torch.float32, device="cuda")
        aliases = {
            "page-output": {
                "output": pages.data[1, :1024].view(torch.float32).view(1, 2, 128)
            },
            "positions-status": {"status": positions},
            "table-status": {"status": table},
            "output-status": {
                "output": shared,
                "status": shared.view(torch.int32).view(-1)[:2].view(1, 2),
            },
        }
        before = [tensor.clone() for tensor in (pages.data, table, positions)]
        for name, outputs in aliases.items():
            with self.subTest(alias=name):
                with self.assertRaisesRegex(
                    ValueError, "output buffers must not alias"
                ):
                    gather_compact(
                        pages,
                        table,
                        requests,
                        positions,
                        lengths,
                        output_dtype=torch.float32,
                        **outputs,
                    )
                for value, original in zip((pages.data, table, positions), before):
                    torch.testing.assert_close(value, original, rtol=0, atol=0)

    def test_all_three_formats_decode_exactly_from_padded_pages(self):
        for region in (CacheRegion.SWA, CacheRegion.GLOBAL, CacheRegion.INDEX_K):
            with self.subTest(region=region):
                pages, reference = _fixture(region)
                result = gather_compact(
                    pages,
                    _ints([[3, 1, 2]]),
                    _ints([0, 0]),
                    _ints([[0, 127, 128, 255, 256, -1], [383, 384, -1, -1, -1, -1]]),
                    _ints([384, 384]),
                    output_dtype=torch.float32,
                )
                result.check()
                expected = torch.stack(
                    (
                        torch.stack(
                            (
                                reference[3, 0],
                                reference[3, 127],
                                reference[1, 0],
                                reference[1, 127],
                                reference[2, 0],
                                torch.zeros(reference.shape[-1]),
                            )
                        ),
                        torch.stack(
                            (
                                reference[2, 127],
                                *[torch.zeros(reference.shape[-1]) for _ in range(5)],
                            )
                        ),
                    )
                )
                torch.testing.assert_close(
                    result.output.cpu(), expected, rtol=0, atol=0
                )

    def test_fp4_decoder_covers_all_sixteen_nibble_encodings(self):
        for region in (CacheRegion.GLOBAL, CacheRegion.INDEX_K):
            pages, reference = _fixture(region, entries=128)
            result = gather_compact(
                pages,
                _ints([[1]]),
                _ints([0]),
                _ints([[0]]),
                _ints([1]),
                output_dtype=torch.float32,
            )
            result.check()
            torch.testing.assert_close(
                result.output[0, 0].cpu(), reference[1, 0], rtol=0, atol=0
            )

    def test_missing_visible_pages_report_error_but_future_rows_are_masked(self):
        pages, _ = _fixture(CacheRegion.GLOBAL)
        result = gather_compact(
            pages, _ints([[1, 0]]), _ints([0]), _ints([[128, 256, -1]]), _ints([256])
        )
        self.assertEqual(result.status.cpu().tolist(), [[1, 0, 0]])
        with self.assertRaisesRegex(RuntimeError, "missing KV"):
            result.check()

    def test_high_logical_positions_use_sparse_table_without_dense_history(self):
        pages, reference = _fixture(CacheRegion.GLOBAL)
        table = torch.zeros((1, 8192), dtype=torch.int32, device="cuda")
        table[0, 8189] = 2
        result = gather_compact(
            pages,
            table,
            _ints([0]),
            _ints([[1048320 - 1]]),
            _ints([1048320]),
            output_dtype=torch.float32,
        )
        result.check()
        torch.testing.assert_close(
            result.output[0, 0].cpu(), reference[2, 127], rtol=0, atol=0
        )

    def test_global_e4m3_scales_include_smallest_subnormal_and_largest_finite(self):
        pages, _ = _fixture(CacheRegion.GLOBAL)
        encoded = pages.data[1, :288]
        encoded[:256].fill_(0x76)
        scale_codes = [1, 0x38, 0x7E, 0x38] * 8
        encoded[256:].copy_(torch.tensor(scale_codes, dtype=torch.uint8, device="cuda"))
        result = gather_compact(
            pages,
            _ints([[1]]),
            _ints([0]),
            _ints([[0]]),
            _ints([1]),
            output_dtype=torch.float32,
        )
        result.check()
        expected = torch.tensor([4.0, 6.0] * 256) * torch.tensor(
            [2**-9, 1.0, 448.0, 1.0] * 8
        ).repeat_interleave(16)
        torch.testing.assert_close(result.output[0, 0].cpu(), expected, rtol=0, atol=0)

    def test_index_ue8m0_small_scales_preserve_float32_subnormals(self):
        pages, _ = _fixture(CacheRegion.INDEX_K)
        encoded = pages.data[1, :68]
        encoded[:64].fill_(0x21)
        encoded[64:].copy_(
            torch.tensor([0, 1, 126, 127], dtype=torch.uint8, device="cuda")
        )
        result = gather_compact(
            pages,
            _ints([[1]]),
            _ints([0]),
            _ints([[0]]),
            _ints([1]),
            output_dtype=torch.float32,
        )
        result.check()
        expected = torch.tensor([0.5, 1.0] * 64, dtype=torch.float64) * torch.tensor(
            [2**-127, 2**-126, 0.5, 1.0], dtype=torch.float64
        ).repeat_interleave(32)
        torch.testing.assert_close(
            result.output[0, 0].cpu(), expected.float(), rtol=0, atol=0
        )

    def test_gather_workspace_is_bounded_before_allocation(self):
        pages, _ = _fixture(CacheRegion.SWA)
        with self.assertRaisesRegex(ValueError, "64 MiB"):
            gather_compact(
                pages,
                _ints([[1]]),
                _ints([0] * 129),
                torch.zeros((129, 512), dtype=torch.int32, device="cuda"),
                _ints([128] * 129),
                output_dtype=torch.float32,
            )

    def test_planar_pages_cannot_enter_native_gather_or_bindings(self):
        from rtp_llm.models_py.modules.dsv41.flashmla import (
            PlanarGlobalBinding,
            PlanarPages,
            PlanarSwaBinding,
        )

        for region in (CacheRegion.SWA, CacheRegion.GLOBAL):
            pages, _ = _fixture(
                region, entries=136 if region == CacheRegion.SWA else 128
            )
            planar = PlanarPages(pages.data, region, pages.entries_per_page)
            with self.assertRaisesRegex(TypeError, "row-interleaved"):
                gather_compact(
                    planar, _ints([[1]]), _ints([0]), _ints([[0]]), _ints([1])
                )
            with self.assertRaisesRegex(TypeError, "row-interleaved"):
                if region == CacheRegion.SWA:
                    SwaBinding(planar, _ints([1]), _ints([0]), _ints([1])).validate(
                        pages.data.device
                    )
                else:
                    GlobalBinding(planar, _ints([[1]]), 1).validate(
                        1, pages.data.device
                    )
        swa_pages, _ = _fixture(CacheRegion.SWA, entries=136)
        swa = SwaBinding(swa_pages, _ints([1]), _ints([0]), _ints([1]))
        planar_swa = PlanarSwaBinding(
            PlanarPages(swa_pages.data, CacheRegion.SWA, 136),
            swa.page_ids,
            swa.valid_starts,
            swa.valid_ends,
        )
        planar_global = PlanarGlobalBinding(planar, _ints([[1]]), 1)
        for candidate_swa, candidate_global in (
            (planar_swa, None),
            (swa, planar_global),
        ):
            with self.assertRaisesRegex(TypeError, "row-interleaved"):
                compact_attention(
                    torch.zeros((1, 64, 512), dtype=torch.bfloat16, device="cuda"),
                    _ints([0]),
                    _ints([0]),
                    _ints([0]),
                    candidate_swa,
                    torch.zeros(64, device="cuda"),
                    global_kv=candidate_global,
                )

    def test_gather_rejects_out_of_context_positions_lengths_and_active_requests(self):
        pages, _ = _fixture(CacheRegion.INDEX_K)
        for requests, positions, lengths in (
            ([0], [[1048576]], [1048576]),
            ([0], [[0]], [1048577]),
            ([0], [[-1]], [-1]),
            ([-1], [[0]], [1]),
            ([1], [[-1]], [1]),
        ):
            result = gather_compact(
                pages, _ints([[1]]), _ints(requests), _ints(positions), _ints(lengths)
            )
            if positions == [[-1]] and lengths == [1]:
                result.check()
            else:
                self.assertTrue(torch.all((result.status & 2) != 0).item())
                with self.assertRaisesRegex(RuntimeError, "rejected metadata"):
                    result.check()
            torch.testing.assert_close(
                result.output, torch.zeros_like(result.output), rtol=0, atol=0
            )

    def test_attention_context_boundary_graph_rejects_then_recovers_without_recapture(
        self,
    ):
        pages, reference = _fixture(CacheRegion.SWA, entries=136)
        positions, floors, requests = _ints([1048575]), _ints([0]), _ints([0])
        swa = SwaBinding(pages, _ints([1]), _ints([1048440]), _ints([1048576]))
        query = torch.zeros((1, 64, 512), dtype=torch.bfloat16, device="cuda")
        sinks = torch.zeros(64, dtype=torch.float32, device="cuda")
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())

        def operation():
            return compact_attention(
                query,
                requests,
                positions,
                floors,
                swa,
                sinks,
                output_dtype=torch.float32,
            )

        with torch.cuda.stream(stream):
            operation()
            operation()
        stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            result = operation()
        pointers = (
            result.output.data_ptr(),
            result.status.data_ptr(),
            result.lse.data_ptr(),
        )
        for position in (1048575, 1048576, 1048575):
            positions.fill_(position)
            swa.valid_starts.fill_(position - 135)
            swa.valid_ends.fill_(position + 1)
            graph.replay()
            torch.cuda.synchronize()
            self.assertEqual(
                pointers,
                (
                    result.output.data_ptr(),
                    result.status.data_ptr(),
                    result.lse.data_ptr(),
                ),
            )
            if position == 1048576:
                self.assertTrue(torch.all(result.status == 2).item())
                with self.assertRaisesRegex(RuntimeError, "rejected metadata"):
                    result.check()
                torch.testing.assert_close(
                    result.output, torch.zeros_like(result.output), rtol=0, atol=0
                )
                self.assertTrue(torch.all(result.lse == -torch.inf).item())
                continue
            result.check()
            expected, expected_lse = _attention_oracle(
                query, requests, positions, floors, swa, reference, sinks
            )
            torch.testing.assert_close(
                result.output.cpu(), expected, rtol=2e-5, atol=2e-5
            )
            torch.testing.assert_close(
                result.lse.cpu(), expected_lse, rtol=2e-5, atol=2e-5
            )

    def test_gather_graph_reads_updated_page_table_and_indices(self):
        pages, reference = _fixture(CacheRegion.INDEX_K)
        table, requests, positions, lengths = (
            _ints([[1, 2]]),
            _ints([0]),
            _ints([[0, 128]]),
            _ints([256]),
        )
        output = torch.empty((1, 2, 128), dtype=torch.float32, device="cuda")
        status = torch.empty((1, 2), dtype=torch.int32, device="cuda")
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                gather_compact(
                    pages,
                    table,
                    requests,
                    positions,
                    lengths,
                    output=output,
                    status=status,
                    output_dtype=torch.float32,
                )
        stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            result = gather_compact(
                pages,
                table,
                requests,
                positions,
                lengths,
                output=output,
                status=status,
                output_dtype=torch.float32,
            )
        table.copy_(_ints([[3, 1]]))
        positions.copy_(_ints([[127, 255]]))
        graph.replay()
        result.check()
        torch.testing.assert_close(
            result.output[0].cpu(),
            torch.stack((reference[3, 127], reference[1, 127])),
            rtol=0,
            atol=0,
        )

    def _attention_case(self, ratio=2, positions=None, floors=None, indices=None):
        positions = positions or [127, 128, 129, 130, 131, 132]
        q = (
            torch.randn(
                (len(positions), 64, 512),
                generator=torch.Generator().manual_seed(812),
                dtype=torch.float32,
            )
            .mul_(0.125)
            .bfloat16()
            .cuda()
        )
        requests = _ints([i % 2 for i in range(len(positions))])
        floors = _ints(floors or [0] * len(positions))
        position_tensor = _ints(positions)
        swa_pages, swa_values = _fixture(CacheRegion.SWA, entries=136)
        swa = SwaBinding(
            swa_pages,
            _ints([1, 2]),
            _ints([max(0, min(positions) - 127)] * 2),
            _ints([max(positions) + 1] * 2),
        )
        global_pages, global_values = _fixture(CacheRegion.GLOBAL, pages=4, entries=64)
        global_kv = GlobalBinding(global_pages, _ints([[1, 2, 3], [3, 2, 1]]), ratio)
        indices = _ints(indices or [[0, 31, 62, 63, 64, 65, 66, -1]] * len(positions))
        sinks = torch.linspace(-2, 2, 64, dtype=torch.float32, device="cuda")
        return (
            q,
            requests,
            position_tensor,
            floors,
            swa,
            swa_values,
            sinks,
            global_kv,
            global_values,
            indices,
        )

    def _assert_attention(self, case, **options):
        (
            q,
            requests,
            positions,
            floors,
            swa,
            swa_values,
            sinks,
            global_kv,
            global_values,
            indices,
        ) = case
        result = compact_attention(
            q,
            requests,
            positions,
            floors,
            swa,
            sinks,
            global_kv=global_kv,
            global_indices=indices,
            output_dtype=torch.float32,
            **options,
        )
        result.check()
        expected, expected_lse = _attention_oracle(
            q,
            requests,
            positions,
            floors,
            swa,
            swa_values,
            sinks,
            global_kv,
            global_values,
            indices,
        )
        torch.testing.assert_close(result.output.cpu(), expected, rtol=2e-5, atol=2e-5)
        torch.testing.assert_close(result.lse.cpu(), expected_lse, rtol=2e-5, atol=2e-5)
        return result

    def test_ratio2_six_verify_rows_enforce_per_query_pair_visibility(self):
        self._assert_attention(self._attention_case())

    def test_ratio1_reads_source_kv_with_independent_per_query_candidates(self):
        self._assert_attention(
            self._attention_case(
                ratio=1, indices=[[0, 1, 126, 127, 128, 129, 130, -1]] * 6
            )
        )

    def test_bounded_replay_floor_and_short_suffix_history_are_distinct(self):
        for floors in ([100] * 6, [0] * 6):
            self._assert_attention(self._attention_case(floors=floors))

    def test_swa_only_uses_sink_once_and_returns_bf16(self):
        pages, _ = _fixture(CacheRegion.SWA, entries=136)
        # Every real SWA element is exactly 1, encoded by E4M3 0x38 with scale 2**0.
        raw = pages.data
        for page in range(1, 4):
            rows = raw[page, : 136 * 528].view(136, 528)
            rows[:, :512].fill_(0x38)
            rows[:, 512:].fill_(127)
        swa = SwaBinding(pages, _ints([1]), _ints([0]), _ints([129]))
        q = torch.zeros((3, 64, 512), dtype=torch.bfloat16, device="cuda")
        result = compact_attention(
            q,
            _ints([0, 0, 0]),
            _ints([0, 127, 128]),
            _ints([0, 0, 0]),
            swa,
            torch.zeros(64, device="cuda"),
        )
        result.check()
        expected = (
            torch.tensor([1 / 2, 128 / 129, 128 / 129], dtype=torch.bfloat16)
            .reshape(3, 1, 1)
            .expand(3, 64, 512)
        )
        torch.testing.assert_close(result.output.cpu(), expected, rtol=0, atol=0)

    def test_full_512_global_candidates_and_64_heads(self):
        q = (
            torch.randn((1, 64, 512), generator=torch.Generator().manual_seed(19))
            .mul_(0.125)
            .bfloat16()
            .cuda()
        )
        swa_pages, swa_values = _fixture(CacheRegion.SWA, entries=136)
        swa = SwaBinding(swa_pages, _ints([1]), _ints([1912]), _ints([2048]))
        global_pages, global_values = _fixture(
            CacheRegion.GLOBAL, pages=17, entries=128
        )
        global_kv = GlobalBinding(global_pages, _ints([list(range(1, 17))]), 1)
        self._assert_attention(
            (
                q,
                _ints([0]),
                _ints([2047]),
                _ints([0]),
                swa,
                swa_values,
                torch.zeros(64, device="cuda"),
                global_kv,
                global_values,
                _ints([list(range(0, 2048, 4))]),
            )
        )

    def test_dominant_sink_and_padding_produce_zero_without_nan(self):
        case = list(self._attention_case())
        case[2][-1] = -1
        case[6].fill_(10000)
        result = self._assert_attention(tuple(case))
        self.assertTrue(torch.equal(result.output, torch.zeros_like(result.output)))

    def test_missing_swa_and_duplicate_global_indices_are_explicit_errors(self):
        case = self._attention_case()
        q, requests, positions, floors, swa, _, sinks, global_kv, _, indices = case
        for bad_swa, bad_indices in (
            (replace(swa, valid_starts=_ints([10, 10])), indices),
            (swa, _ints([[0, 0, -1, -1, -1, -1, -1, -1]] * 6)),
        ):
            result = compact_attention(
                q,
                requests,
                positions,
                floors,
                bad_swa,
                sinks,
                global_kv=global_kv,
                global_indices=bad_indices,
            )
            with self.assertRaisesRegex(RuntimeError, "rejected metadata"):
                result.check()

    def test_empty_local_rank_returns_no_rows_without_touching_kv(self):
        pages, _ = _fixture(CacheRegion.SWA, entries=136)
        swa = SwaBinding(pages, _ints([0]), _ints([0]), _ints([0]))
        empty = _ints([])
        result = compact_attention(
            torch.empty((0, 64, 512), dtype=torch.bfloat16, device="cuda"),
            empty,
            empty,
            empty,
            swa,
            torch.zeros(64, device="cuda"),
        )
        result.check()
        self.assertEqual(tuple(result.output.shape), (0, 64, 512))
        self.assertEqual(tuple(result.status.shape), (0, 64))

    def test_attention_graph_refreshes_all_page_and_query_metadata(self):
        case = self._attention_case()
        (
            q,
            requests,
            positions,
            floors,
            swa,
            swa_values,
            sinks,
            global_kv,
            global_values,
            indices,
        ) = case
        output = torch.empty_like(q, dtype=torch.float32)
        status = torch.empty(q.shape[:2], dtype=torch.int32, device="cuda")
        lse = torch.empty(q.shape[:2], dtype=torch.float32, device="cuda")
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                compact_attention(
                    q,
                    requests,
                    positions,
                    floors,
                    swa,
                    sinks,
                    global_kv=global_kv,
                    global_indices=indices,
                    output=output,
                    status=status,
                    lse=lse,
                    output_dtype=torch.float32,
                )
        stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            result = compact_attention(
                q,
                requests,
                positions,
                floors,
                swa,
                sinks,
                global_kv=global_kv,
                global_indices=indices,
                output=output,
                status=status,
                lse=lse,
                output_dtype=torch.float32,
            )
        swa.page_ids.copy_(_ints([3, 1]))
        global_kv.page_table.copy_(_ints([[3, 2, 1], [1, 2, 3]]))
        positions.copy_(_ints([126, 127, 128, 129, 130, 131]))
        floors.fill_(100)
        indices.copy_(_ints([[1, 2, 3, 4, 5, -1, -1, -1]] * 6))
        graph.replay()
        result.check()
        expected, expected_lse = _attention_oracle(
            q,
            requests,
            positions,
            floors,
            swa,
            swa_values,
            sinks,
            global_kv,
            global_values,
            indices,
        )
        torch.testing.assert_close(result.output.cpu(), expected, rtol=2e-5, atol=2e-5)
        torch.testing.assert_close(result.lse.cpu(), expected_lse, rtol=2e-5, atol=2e-5)


if __name__ == "__main__":
    unittest.main()
