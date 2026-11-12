"""Independent byte and changed-input Graph checks for Verify journal reads."""

import unittest

import torch

from rtp_llm.models_py.modules.dsv41.ppu.graph_cache_kernels import (
    position_slots,
    read_packed_rows,
)
from rtp_llm.models_py.modules.dsv41.ppu.verify_ops import (
    try_journal_slots,
    try_prepare_verify_positions,
    try_read_journal_rows,
    try_read_verify_indexer_entries,
)


def _legacy_verify_positions(positions, valid):
    entry_positions = positions - 1
    commit = valid & ((positions & 1) == 1)
    current_raw = torch.where(valid, positions, -1)
    previous_raw = torch.where(commit, entry_positions, -1)
    return entry_positions, commit, current_raw, previous_raw


def _legacy_slots(raw, state):
    native = position_slots(raw, state["row_tables"], **state["layout"])
    return torch.where(native >= 0, raw, -1)


def _legacy_read(raw, state):
    layout = state["layout"]
    slots = position_slots(raw, state["row_tables"], **layout)
    native = read_packed_rows(
        state["pool"], slots, width=state["width"],
        entries=layout["entries"], planar=state["planar"],
    )
    shape = (state["requests"].numel(),) + (1,) * (raw.ndim - 1)
    starts = state["starts"][state["requests"]].reshape(shape)
    prefixes = state["prefixes"][state["requests"]].reshape(shape)
    lag = int(layout["full"] and layout["ratio"] == 2)
    offset = raw + lag - prefixes
    index = (starts + offset).clamp(0, state["requests"].numel() - 1)
    live = (
        (raw >= 0) & (offset >= 0) & state["valid"][index]
        & (state["requests"][index] == state["requests"].reshape(shape))
        & (state["generations"][index] == raw)
    )
    return torch.where(live[..., None], state["payloads"][index], native)


def _legacy_verify_indexer_entries(entries, positions, valid_mask, state):
    rows, columns = positions.numel(), entries.shape[-1]
    ratio = state["layout"]["ratio"]
    raw_positions = entries.expand(rows, columns) * ratio
    visible = (positions + 1) // ratio
    query_valid = valid_mask[:, None] & (entries < visible[:, None])
    slots = _legacy_slots(torch.where(query_valid, raw_positions, -1), state)
    return _legacy_read(slots, state), query_valid & (slots >= 0)


def _indexer_entry_candidate(entries, positions, valid_mask, state):
    return try_read_verify_indexer_entries(
        entries, positions, valid_mask,
        **{k: state[k] for k in (
            "row_tables", "pool", "payloads", "generations", "requests",
            "starts", "prefixes", "valid",
        )},
        span=state["layout"]["span"], physical_entries=state["layout"]["entries"],
        ratio=state["layout"]["ratio"],
    )


def _candidate(raw, state):
    layout = state["layout"]
    return (
        try_journal_slots(raw, state["row_tables"], **layout),
        try_read_journal_rows(
            raw, **{k: state[k] for k in (
                "row_tables", "pool", "payloads", "generations", "requests",
                "starts", "prefixes", "valid", "width", "planar",
            )}, **{k: layout[k] for k in ("span", "entries", "ratio", "full")},
        ),
    )


def _fixture(device, *, ratio=1, full=True, planar=True, width=68, dtype=torch.uint8):
    rows, requests, span, entries = 32, 8, 256, 256
    if not full:
        entries = 2 if dtype == torch.float32 else 130
    prefixes = torch.tensor(
        [131064, 131067, 131070, 131071, 131072, 131073, 131075, 131080],
        dtype=torch.int64, device=device,
    )
    ids = torch.arange(requests, dtype=torch.int64, device=device).repeat_interleave(4)
    starts = torch.arange(requests, dtype=torch.int64, device=device) * 4
    table = (torch.arange(rows * 520, device=device).reshape(rows, 520) % 8 + 1).int()
    table[1, 511] = 0
    table[4, 512] = -1
    table[7, 512] = 99
    pool = torch.arange(9 * entries * width, dtype=torch.int64, device=device)
    pool = (pool % 251).to(torch.uint8).reshape(9, entries * width)
    payloads = ((torch.arange(rows * width, device=device) * 17 + 3) % 251).to(torch.uint8)
    payloads = payloads.reshape(rows, width)
    if dtype == torch.float32:
        # Raw bit patterns cover NaN payloads, infinities, signed zero and
        # subnormals. These reads must not normalize any FP32 representation.
        pool = torch.arange(9 * entries * width, dtype=torch.int32, device=device)
        pool = (pool * 65537).reshape(9, entries * width).view(torch.float32)
        payloads = torch.arange(rows * width, dtype=torch.int32, device=device)
        payloads = (payloads * 131071).reshape(rows, width).view(torch.float32)
        pool.view(torch.int32)[1, :4] = torch.tensor(
            [0x7FC00123, 0x7F800000, -2147483648, 1], dtype=torch.int32, device=device,
        )
        payloads.view(torch.int32)[0, :4] = torch.tensor(
            [0x7FC00142, -8388608, -2147483648, 1], dtype=torch.int32, device=device,
        )
    lag = int(full and ratio == 2)
    generations = prefixes[ids] + torch.arange(rows, device=device) % 4 - lag
    valid = torch.ones(rows, dtype=torch.bool, device=device)
    valid[3::5] = False
    raw = prefixes[ids, None] + torch.tensor([-1, 0, 1, 4], device=device)[None] - lag
    raw[1, 0] = -1
    raw[2, 3] = 256 * 520
    state = dict(
        row_tables=table, pool=pool, payloads=payloads, generations=generations,
        requests=ids, starts=starts, prefixes=prefixes, valid=valid,
        width=width, planar=planar,
        layout=dict(span=span, entries=entries, ratio=ratio, full=full, pool_blocks=9),
    )
    return raw, state


class VerifyOpsContracts(unittest.TestCase):
    def test_cpu_explicit_indexer_entry_reader_keeps_legacy_route(self):
        _, state = _fixture(torch.device("cpu"), ratio=2)
        positions = state["prefixes"][state["requests"]] + 3
        entries = positions[:, None] // 2 + torch.arange(4)[None]
        valid_mask = torch.ones(32, dtype=torch.bool)
        packed, effective = _legacy_verify_indexer_entries(entries, positions, valid_mask, state)
        self.assertEqual(packed.shape, (32, 4, 68))
        self.assertEqual(effective.shape, (32, 4))
        self.assertIsNone(_indexer_entry_candidate(entries, positions, valid_mask, state))

    def test_cpu_position_metadata_stays_on_legacy_path(self):
        for dtype in (torch.int32, torch.int64):
            positions = torch.tensor([-3, -2, -1, 0, 1, 2, 131071], dtype=dtype)
            valid = torch.tensor([True, False, True, False, True, True, True])
            reference = _legacy_verify_positions(positions, valid)
            self.assertEqual([t.shape for t in reference], [positions.shape] * 4)
            self.assertIsNone(try_prepare_verify_positions(positions, valid))

    def test_cpu_routes_remain_on_independent_legacy_path(self):
        raw, state = _fixture(torch.device("cpu"))
        self.assertEqual(_legacy_slots(raw, state).shape, raw.shape)
        self.assertEqual(_legacy_read(raw, state).shape, (*raw.shape, 68))
        self.assertEqual(_candidate(raw, state), (None, None))

    def test_cpu_fp32_oracle_preserves_payload_bits(self):
        raw, state = _fixture(
            torch.device("cpu"), full=False, planar=False, width=1024, dtype=torch.float32,
        )
        value = _legacy_read(raw, state)
        self.assertEqual(value.dtype, torch.float32)
        self.assertEqual(value.shape, (32, 4, 1024))
        self.assertEqual(_candidate(raw, state), (None, None))


@unittest.skipUnless(torch.cuda.is_available(), "CUDA-like accelerator required")
class VerifyOpsDevice(unittest.TestCase):
    def _assert_bytes(self, actual, expected):
        self.assertIsNotNone(actual)
        self.assertEqual(actual.shape, expected.shape)
        self.assertEqual(actual.dtype, expected.dtype)
        torch.testing.assert_close(
            actual.contiguous().view(torch.uint8),
            expected.contiguous().view(torch.uint8), rtol=0, atol=0,
        )

    def _check(self, raw, state, actual=None):
        output = _candidate(raw, state) if actual is None else actual
        self._assert_bytes(output[0], _legacy_slots(raw, state))
        self._assert_bytes(output[1], _legacy_read(raw, state))

    def test_native_and_journal_layouts_are_byte_exact(self):
        device = torch.device("cuda", torch.cuda.current_device())
        cases = (
            dict(ratio=1), dict(ratio=2),
            dict(full=False, planar=False, width=528),
            dict(full=False, planar=False, width=1024, dtype=torch.float32),
            dict(full=True, planar=False, width=288, ratio=2),
        )
        for options in cases:
            with self.subTest(**options):
                raw, state = _fixture(device, **options)
                pool_before, payload_before = state["pool"].clone(), state["payloads"].clone()
                self._check(raw, state)
                self._check(raw.int(), state)
                self._assert_bytes(state["pool"], pool_before)
                self._assert_bytes(state["payloads"], payload_before)
                empty = raw[:, :0].contiguous()
                self._check(empty, state)

    def test_missing_native_page_can_still_read_matching_journal(self):
        device = torch.device("cuda", torch.cuda.current_device())
        raw, state = _fixture(device, ratio=2)
        raw = state["generations"][:, None].clone()
        state["row_tables"].zero_()
        state["valid"].fill_(True)
        actual = _candidate(raw, state)
        self.assertTrue((actual[0] == -1).all().item())
        self._assert_bytes(actual[1][:, 0], state["payloads"])
        self._check(raw, state, actual)

    def test_graph_replay_changes_every_live_input(self):
        device = torch.device("cuda", torch.cuda.current_device())
        for options in (dict(ratio=1), dict(ratio=2), dict(full=False, planar=False, width=1024, dtype=torch.float32)):
            with self.subTest(**options):
                raw, state = _fixture(device, **options)
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    self._check(raw, state)
                stream.synchronize()
                baseline, candidate = torch.cuda.CUDAGraph(), torch.cuda.CUDAGraph()
                with torch.cuda.graph(baseline, stream=stream):
                    expected = (_legacy_slots(raw, state), _legacy_read(raw, state))
                with torch.cuda.graph(candidate, stream=stream):
                    actual = _candidate(raw, state)

                def check():
                    baseline.replay()
                    candidate.replay()
                    for got, want in zip(actual, expected):
                        self._assert_bytes(got, want)

                with torch.cuda.stream(stream):
                    check()
                    raw.add_(1)
                    check()
                    state["valid"].logical_not_()
                    check()
                    state["row_tables"][:, 511:513].zero_()
                    check()
                    state["requests"].copy_(state["requests"].roll(4))
                    check()
                    state["starts"].add_(1)
                    check()
                    state["prefixes"].add_(2)
                    check()
                    state["generations"].add_(3)
                    check()
                    state["payloads"].view(torch.uint8).bitwise_xor_(37)
                    check()
                    state["pool"].view(torch.uint8).bitwise_xor_(19)
                    check()
                stream.synchronize()

    def test_unsupported_layout_keeps_fallback(self):
        raw, state = _fixture(torch.device("cuda", torch.cuda.current_device()))
        self.assertIsNone(try_journal_slots(raw[:, ::2], state["row_tables"], **state["layout"]))
        bad = dict(state)
        bad["payloads"] = state["payloads"].to(torch.int16)
        self.assertEqual(_candidate(raw, bad)[1], None)

    def test_position_metadata_integer_extremes_and_arbitrary_rows(self):
        device = torch.device("cuda", torch.cuda.current_device())
        for dtype in (torch.int32, torch.int64):
            for rows in (0, 1, 3, 24, 32, 65, 257, 1025):
                with self.subTest(dtype=dtype, rows=rows):
                    positions = torch.arange(rows, device=device, dtype=dtype) * 3 - 7
                    valid = torch.arange(rows, device=device) % 4 != 0
                    if rows:
                        positions[0] = torch.iinfo(dtype).min
                        valid[0] = True
                    if rows > 1:
                        positions[1] = torch.iinfo(dtype).max
                        valid[1] = True
                    before_positions, before_valid = positions.clone(), valid.clone()
                    actual = try_prepare_verify_positions(positions, valid)
                    self.assertIsNotNone(actual)
                    for got, expected in zip(actual, _legacy_verify_positions(positions, valid)):
                        self._assert_bytes(got, expected)
                        if rows:
                            self.assertNotIn(got.data_ptr(), (positions.data_ptr(), valid.data_ptr()))
                    self._assert_bytes(positions, before_positions)
                    self._assert_bytes(valid, before_valid)

    def test_position_metadata_graph_replay_reads_changed_live_inputs(self):
        device = torch.device("cuda", torch.cuda.current_device())
        for dtype, rows in ((torch.int64, 32), (torch.int32, 32), (torch.int64, 65)):
            with self.subTest(dtype=dtype, rows=rows):
                positions = torch.arange(rows, device=device, dtype=dtype) + 131064
                valid = torch.arange(rows, device=device) % 3 != 1
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    try_prepare_verify_positions(positions, valid)
                    _legacy_verify_positions(positions, valid)
                stream.synchronize()
                baseline, candidate = torch.cuda.CUDAGraph(), torch.cuda.CUDAGraph()
                with torch.cuda.graph(baseline, stream=stream):
                    expected = _legacy_verify_positions(positions, valid)
                with torch.cuda.graph(candidate, stream=stream):
                    actual = try_prepare_verify_positions(positions, valid)
                pointers = tuple(t.data_ptr() for t in actual)

                def check():
                    baseline.replay()
                    candidate.replay()
                    self.assertEqual(tuple(t.data_ptr() for t in actual), pointers)
                    for got, want in zip(actual, expected):
                        self._assert_bytes(got, want)

                with torch.cuda.stream(stream):
                    check()
                    positions.add_(1)
                    check()
                    valid.logical_not_()
                    check()
                    positions.fill_(torch.iinfo(dtype).min)
                    check()
                    positions.fill_(torch.iinfo(dtype).max)
                    check()
                    valid.zero_()
                    check()
                    positions.copy_(torch.arange(rows, device=device, dtype=dtype) * 7 - 23)
                    valid[::2] = True
                    check()
                stream.synchronize()

    def test_position_metadata_unsupported_layout_preserves_fallback(self):
        device = torch.device("cuda", torch.cuda.current_device())
        positions = torch.arange(32, device=device, dtype=torch.int64)
        valid = torch.ones(32, device=device, dtype=torch.bool)
        self.assertIsNone(try_prepare_verify_positions(positions, valid, ratio=1))
        self.assertIsNone(try_prepare_verify_positions(positions[:, None], valid[:, None]))
        self.assertIsNone(try_prepare_verify_positions(positions[::2], valid[::2]))
        self.assertIsNone(try_prepare_verify_positions(positions, valid.to(torch.uint8)))
        self.assertIsNone(try_prepare_verify_positions(positions.float(), valid))
        self.assertIsNone(try_prepare_verify_positions(positions, valid.cpu()))
        self.assertIsNone(try_prepare_verify_positions(positions, valid[:-1]))

    def test_indexer_entries_match_integer_chain_and_planar_bytes(self):
        device = torch.device("cuda", torch.cuda.current_device())
        for ratio in (1, 2):
            for width in (1, 2, 3, 4):
                with self.subTest(ratio=ratio, width=width):
                    _, state = _fixture(device, ratio=ratio)
                    positions = state["prefixes"][state["requests"]] + 3
                    valid_mask = torch.arange(32, device=device) % 5 != 1
                    entries = positions[:, None] // ratio + torch.arange(width, device=device)[None]
                    variants = (
                        entries,
                        entries[:1],
                        entries[0],
                        entries[:1].expand(32, width),
                        entries[0].expand(32, width),
                        torch.stack((entries, entries + 7), -1)[..., 0],
                    )
                    before_pool, before_payloads = state["pool"].clone(), state["payloads"].clone()
                    for values in variants:
                        actual = _indexer_entry_candidate(values, positions, valid_mask, state)
                        self.assertIsNotNone(actual)
                        for got, expected in zip(actual, _legacy_verify_indexer_entries(
                            values, positions, valid_mask, state,
                        )):
                            self._assert_bytes(got, expected)
                    self._assert_bytes(state["pool"], before_pool)
                    self._assert_bytes(state["payloads"], before_payloads)
                    positions[:5].copy_(torch.tensor([
                        torch.iinfo(torch.int64).max, torch.iinfo(torch.int64).min,
                        -3, -2, -1,
                    ], device=device))
                    extreme_entries = torch.tensor([
                        torch.iinfo(torch.int64).min, torch.iinfo(torch.int64).max, -1, 0,
                    ], device=device)[:width]
                    valid_mask.fill_(True)
                    actual = _indexer_entry_candidate(extreme_entries, positions, valid_mask, state)
                    for got, expected in zip(actual, _legacy_verify_indexer_entries(
                        extreme_entries, positions, valid_mask, state,
                    )):
                        self._assert_bytes(got, expected)

    def test_indexer_entries_native_allocation_gates_journal_override(self):
        device = torch.device("cuda", torch.cuda.current_device())
        for ratio in (1, 2):
            with self.subTest(ratio=ratio):
                _, state = _fixture(device, ratio=ratio)
                positions = torch.full((32,), 3, dtype=torch.int64, device=device)
                entries = torch.zeros((32, 1), dtype=torch.int64, device=device)
                valid_mask = torch.ones(32, dtype=torch.bool, device=device)
                state["prefixes"].zero_()
                state["valid"].fill_(True)
                state["generations"].zero_()
                state["payloads"].fill_(123)
                # Without the earlier slots gate, matching journal generations
                # would return123 despite no native allocation.
                state["row_tables"].zero_()
                packed, effective = _indexer_entry_candidate(entries, positions, valid_mask, state)
                self.assertTrue((packed == 0).all().item())
                self.assertFalse(effective.any().item())
                for got, expected in zip((packed, effective), _legacy_verify_indexer_entries(
                    entries, positions, valid_mask, state,
                )):
                    self._assert_bytes(got, expected)
                state["row_tables"].fill_(1)
                packed, effective = _indexer_entry_candidate(entries, positions, valid_mask, state)
                self.assertTrue((packed == 123).all().item())
                self.assertTrue(effective.all().item())

    def test_indexer_entries_graph_replay_changes_every_live_input(self):
        device = torch.device("cuda", torch.cuda.current_device())
        for ratio in (1, 2):
            for width in (1, 2, 3, 4):
                with self.subTest(ratio=ratio, width=width):
                    _, state = _fixture(device, ratio=ratio)
                    positions = state["prefixes"][state["requests"]] + 3
                    valid_mask = torch.arange(32, device=device) % 3 != 1
                    entries = positions[:, None] // ratio + torch.arange(width, device=device)[None]
                    stream = torch.cuda.Stream()
                    stream.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(stream):
                        _indexer_entry_candidate(entries, positions, valid_mask, state)
                        _legacy_verify_indexer_entries(entries, positions, valid_mask, state)
                    stream.synchronize()
                    baseline, candidate = torch.cuda.CUDAGraph(), torch.cuda.CUDAGraph()
                    with torch.cuda.graph(baseline, stream=stream):
                        expected = _legacy_verify_indexer_entries(entries, positions, valid_mask, state)
                    with torch.cuda.graph(candidate, stream=stream):
                        actual = _indexer_entry_candidate(entries, positions, valid_mask, state)
                    pointers = tuple(t.data_ptr() for t in actual)
                    targets = {
                        "entries": entries, "positions": positions,
                        "query_valid": valid_mask,
                        **{key: state[key] for key in (
                            "row_tables", "valid", "requests", "starts", "prefixes",
                            "generations", "payloads", "pool",
                        )},
                    }
                    originals = {key: value.clone() for key, value in targets.items()}
                    stream.wait_stream(torch.cuda.current_stream())

                    def check():
                        baseline.replay()
                        candidate.replay()
                        self.assertEqual(tuple(t.data_ptr() for t in actual), pointers)
                        for got, want in zip(actual, expected):
                            self._assert_bytes(got, want)

                    with torch.cuda.stream(stream):
                        # Reset between independent mutations. An all-missing
                        # page table must not hide subsequent metadata reads.
                        mutations = (
                            lambda: entries.add_(1),
                            lambda: positions.add_(2),
                            lambda: valid_mask.logical_not_(),
                            lambda: state["row_tables"][:, 511:513].zero_(),
                            lambda: state["valid"].logical_not_(),
                            lambda: state["requests"].copy_(state["requests"].roll(4)),
                            lambda: state["starts"].add_(1),
                            lambda: state["prefixes"].add_(2),
                            lambda: state["generations"].add_(3),
                            lambda: state["payloads"].bitwise_xor_(37),
                            lambda: state["pool"].bitwise_xor_(19),
                        )
                        for mutation in mutations:
                            for key, target in targets.items():
                                target.copy_(originals[key])
                            check()
                            mutation()
                            check()
                    stream.synchronize()

    def test_indexer_entries_unsupported_layout_preserves_fallback(self):
        device = torch.device("cuda", torch.cuda.current_device())
        _, state = _fixture(device)
        positions = state["prefixes"][state["requests"]] + 3
        valid_mask = torch.ones(32, dtype=torch.bool, device=device)
        entries = positions[:, None].clone()
        self.assertIsNone(_indexer_entry_candidate(entries.int(), positions, valid_mask, state))
        self.assertIsNone(_indexer_entry_candidate(entries, positions.int(), valid_mask, state))
        self.assertIsNone(_indexer_entry_candidate(entries, positions, valid_mask.int(), state))
        self.assertIsNone(_indexer_entry_candidate(entries.expand(32, 5), positions, valid_mask, state))
        bad = dict(state)
        bad["starts"] = None
        self.assertIsNone(_indexer_entry_candidate(entries, positions, valid_mask, bad))


if __name__ == "__main__":
    unittest.main()
