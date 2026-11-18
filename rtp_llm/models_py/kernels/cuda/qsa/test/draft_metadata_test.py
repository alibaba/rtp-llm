import unittest

import torch

from rtp_llm.models_py.kernels.cuda.qsa.draft_metadata_triton import (
    draft_rows,
    replay_invalid,
    replay_invalid_from_sources,
)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class DraftMetadataTest(unittest.TestCase):
    def mapping_case(self, batch, lanes, generation):
        lengths = [(i + generation) % lanes + 1 for i in range(batch)]
        if batch > 1:
            lengths[-1] = 0
        prefixes = [125 + i * 127 + generation for i in range(batch)]
        starts = [0]
        for n in lengths:
            starts.append(starts[-1] + n)
        inputs = tuple(
            torch.tensor(x, device="cuda", dtype=torch.int32)
            for x in (lengths, prefixes, starts)
        )
        sources, inverse, valid, positions, visible = [], [], [], [], []
        for i, n in enumerate(lengths):
            sources.extend(starts[i] + j if j < n else 0 for j in range(lanes))
            visible.append([prefixes[i] + j + 1 if j < n else 0 for j in range(lanes)])
            inverse.extend(i * lanes + j for j in range(n))
            positions.extend(prefixes[i] + j for j in range(n))
        live = starts[-1]
        valid = [True] * live + [False] * (batch * lanes - live)
        inverse += [0] * (batch * lanes - live)
        positions += [0] * (batch * lanes - live)
        expected = (sources, inverse, valid, positions, visible)
        return inputs, expected

    def test_mapping_all_capacities_and_dynamic_graph_replay(self):
        for batch in (1, 3, 8, 17, 32):
            for lanes in (1, 2, 3, 4):
                with self.subTest(batch=batch, lanes=lanes):
                    inputs, expected = self.mapping_case(batch, lanes, 0)
                    output = draft_rows(*inputs, batch * lanes)
                    self.assertIsNotNone(output)
                    for actual, wanted in zip(output, expected):
                        self.assertEqual(actual.tolist(), wanted)
                    stream = torch.cuda.Stream()
                    stream.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(stream):
                        draft_rows(*inputs, batch * lanes)
                    torch.cuda.current_stream().wait_stream(stream)
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        captured = draft_rows(*inputs, batch * lanes)
                    addresses = [x.data_ptr() for x in captured]
                    for generation in (1, 7, 0):
                        next_inputs, wanted = self.mapping_case(
                            batch, lanes, generation
                        )
                        for dst, src in zip(inputs, next_inputs):
                            dst.copy_(src)
                        graph.replay()
                        self.assertEqual([x.data_ptr() for x in captured], addresses)
                        for actual, reference in zip(captured, wanted):
                            self.assertEqual(actual.tolist(), reference)

    def validation_case(self, factor=3):
        lengths = torch.tensor([1, 4, 0], device="cuda", dtype=torch.int32)
        prefixes = torch.tensor([7, 127, 0], device="cuda", dtype=torch.int32)
        table = torch.tensor([[1, 2], [3, 4], [0, 0]], device="cuda", dtype=torch.int32)
        expected = torch.tensor(
            [7, 127, 128, 129, 130], device="cuda", dtype=torch.int64
        )
        transported = torch.zeros(12, factor, device="cuda", dtype=torch.int64)
        transported[:5].copy_(expected[:, None])
        return (lengths, prefixes, table, transported.flatten(), expected), dict(
            pool_size=12, page_size=128, token_capacity=12, position_factor=factor
        )

    def test_validation_pages_axes_padding_and_negative_device_prefix(self):
        for factor in (1, 3):
            for defect in (
                None,
                "negative",
                "last_column",
                "zero_page",
                "large_page",
                "axis",
                "padding",
            ):
                with self.subTest(factor=factor, defect=defect):
                    inputs, kwargs = self.validation_case(factor)
                    lengths, prefixes, table, transported, expected = inputs
                    if defect == "negative":
                        prefixes[0] = -1
                    elif defect == "last_column":
                        prefixes[1] = 255
                    elif defect == "zero_page":
                        table[1, 1] = 0
                    elif defect == "large_page":
                        table[0, 0] = 12
                    elif defect == "axis":
                        transported[4 * factor + factor - 1] += 1
                    elif defect == "padding":
                        prefixes[2] = -999
                        table[2] = -100
                        transported[5 * factor :] = -100
                    flag = replay_invalid(*inputs, **kwargs)
                    self.assertIsNotNone(flag)
                    self.assertEqual(bool(flag.item()), defect not in (None, "padding"))

    def test_validation_matches_int32_wrap_and_floor_semantics(self):
        for prefix, length in (
            (-1, 1),
            (-2147483648, 4),
            (2147483647, 4),
            (127, 4),
            (128, 1),
        ):
            inputs, kwargs = self.validation_case()
            lengths, prefixes, table, transported, expected = inputs
            prefixes[0] = prefix
            lengths[0] = length
            first = (prefixes - prefixes.remainder(4)) // 128
            last = (prefixes + lengths - 1) // 128
            active = lengths > 0
            columns = torch.stack((first, last), 1).clamp(0, 1).long()
            physical = table.gather(1, columns)
            reference = (active & ((first < 0) | (last >= 2))).any() | (
                active[:, None] & ((physical <= 0) | (physical >= 12))
            ).any()
            self.assertEqual(replay_invalid(*inputs, **kwargs).item(), reference.item())

    def test_validator_graph_refreshes_failure_flag(self):
        inputs, kwargs = self.validation_case()
        replay_invalid(*inputs, **kwargs)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            flag = replay_invalid(*inputs, **kwargs)
        for page, expected_invalid in ((4, False), (0, True), (4, False), (12, True)):
            inputs[2][1, 1] = page
            graph.replay()
            self.assertEqual(flag.item(), expected_invalid)

    def test_unsupported_geometry_falls_back(self):
        inputs, kwargs = self.validation_case()
        self.assertIsNone(replay_invalid(*inputs, **dict(kwargs, page_size=127)))
        self.assertIsNone(replay_invalid(*inputs, **dict(kwargs, position_factor=2)))
        self.assertIsNone(replay_invalid(*(x.cpu() for x in inputs), **kwargs))
        rows, _ = self.mapping_case(3, 4, 0)
        self.assertIsNone(draft_rows(*rows, 13))
        self.assertIsNone(draft_rows(*(x.cpu() for x in rows), 12))


class SourceReplayVerdictFixture:
    @staticmethod
    def source_case(
        batch=3,
        lanes=4,
        factor=3,
        *,
        lengths=None,
        prefixes=None,
        page_size=128,
        columns=8,
        generation=0,
        device="cuda",
    ):
        # These host producer inputs build the fixture, never an oracle derived
        # from the captured CU or positions under test.
        if lengths is None:
            lengths = [(i + generation) % lanes + 1 for i in range(batch)]
            if batch > 1:
                lengths[-1] = 0
        if prefixes is None:
            prefixes = [
                page_size * (1 + (i + generation) % 4)
                + (3 * i + 5 * generation) % page_size
                for i in range(batch)
            ]
        starts = [0]
        packed_positions = []
        for length, prefix in zip(lengths, prefixes):
            starts.append(starts[-1] + length)
            packed_positions.extend(
                (prefix + lane + (1 << 31)) % (1 << 32) - (1 << 31)
                for lane in range(max(length, 0))
            )
        capacity = batch * lanes
        source_lengths = torch.tensor(lengths, device=device, dtype=torch.int32)
        source_prefixes = torch.tensor(prefixes, device=device, dtype=torch.int32)
        table = torch.tensor(
            [
                [1 + (i * columns + j + generation) % 30 for j in range(columns)]
                for i in range(batch)
            ],
            device=device,
            dtype=torch.int32,
        ).reshape(batch, columns)
        transported = torch.full(
            (capacity, factor), -100, device=device, dtype=torch.int32
        )
        if packed_positions:
            transported[: len(packed_positions)].copy_(
                torch.tensor(packed_positions, device=device, dtype=torch.int32)[
                    :, None
                ]
            )
        args = (
            source_lengths,
            source_prefixes,
            source_lengths.clone(),
            source_prefixes.clone(),
            table,
            transported.flatten(),
            torch.tensor(starts, device=device, dtype=torch.int32),
        )
        return args, dict(
            pool_size=31,
            page_size=page_size,
            token_capacity=capacity,
            position_factor=factor,
        )

    @staticmethod
    def source_reference(args, kwargs):
        (
            lengths,
            prefixes,
            actual_lengths,
            actual_prefixes,
            table,
            positions,
            actual_cu,
        ) = [x.cpu().tolist() for x in args]
        cu, expected, padded = [0], [], False
        lanes = kwargs["token_capacity"] // len(lengths)
        for row, (length, prefix) in enumerate(zip(lengths, prefixes)):
            if length < 0 or length > lanes or (length and (padded or prefix <= 0)):
                return True
            padded = padded or length == 0
            if length != actual_lengths[row] or (
                length and prefix != actual_prefixes[row]
            ):
                return True
            cu.append(cu[-1] + length)
            if not length:
                continue
            if prefix + length - 1 > (1 << 31) - 1:
                return True
            first = prefix // 4 * 4 // kwargs["page_size"]
            last = (prefix + length - 1) // kwargs["page_size"]
            if first < 0 or last >= len(table[row]):
                return True
            if any(
                not 0 < table[row][col] < kwargs["pool_size"]
                for col in range(first, last + 1)
            ):
                return True
            for position in range(prefix, prefix + length):
                expected.extend([position] * kwargs["position_factor"])
        return actual_cu != cu or positions[: len(expected)] != expected

    def assert_source_verdict(self, args, kwargs, invalid):
        reference = self.source_reference(args, kwargs)
        self.assertIsNotNone(reference)
        self.assertEqual(reference, invalid)
        flag = replay_invalid_from_sources(*args, **kwargs)
        self.assertIsNotNone(flag)
        self.assertEqual(flag.shape, torch.Size([]))
        self.assertEqual(flag.dtype, torch.bool)
        self.assertEqual(flag.device, args[0].device)
        self.assertEqual(bool(flag.item()), invalid)
        return flag


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class SourceReplayVerdictTest(SourceReplayVerdictFixture, unittest.TestCase):
    def test_valid_empty_and_poisoned_padding_match_independent_reference(self):
        for batch in (1, 3, 32):
            for lanes in (1, 4):
                for factor in (1, 3):
                    with self.subTest(batch=batch, lanes=lanes, factor=factor):
                        args, kwargs = self.source_case(batch, lanes, factor)
                        self.assert_source_verdict(args, kwargs, False)
                        live = sum((i % lanes) + 1 for i in range(batch - 1))
                        if batch > 1:
                            args[1][-1] = (1 << 31) - 1
                            args[3][-1] = -(1 << 31)
                            args[4][-1].fill_(-(1 << 31))
                            args[5][live * factor :].fill_((1 << 31) - 1)
                            self.assert_source_verdict(args, kwargs, False)
                        args[0].zero_()
                        args[2].zero_()
                        args[1].fill_(-(1 << 31))
                        args[3].fill_((1 << 31) - 1)
                        args[4].fill_(-(1 << 31))
                        args[5].fill_((1 << 31) - 1)
                        args[6].zero_()
                        self.assert_source_verdict(args, kwargs, False)

    def test_self_consistent_capture_cannot_replace_independent_sources(self):
        for factor in (1, 3):
            for lengths, prefixes in (
                ([2, 3, 0], [9, 126, -999]),
                ([1, 4, 0], [8, 128, -999]),
                ([1, 0, 0], [7, -999, -999]),
            ):
                with self.subTest(factor=factor, lengths=lengths, prefixes=prefixes):
                    args, kwargs = self.source_case(
                        factor=factor,
                        lengths=[1, 4, 0],
                        prefixes=[7, 127, -999],
                    )
                    changed, _ = self.source_case(
                        factor=factor, lengths=lengths, prefixes=prefixes
                    )
                    # A self-certifying check would accept this other valid
                    # partition. The unchanged independent sources reject it.
                    self.assert_source_verdict(changed, kwargs, False)
                    for index in (2, 3, 5, 6):
                        args[index].copy_(changed[index])
                    self.assert_source_verdict(args, kwargs, True)

    def test_small_pages_validate_the_entire_raw_state_interval(self):
        # align4(7)..(7+4-1) spans seven page1 pages, four page2
        # pages, and two page4 pages. With <=4 lanes page4 has no
        # strictly interior page, so both of its visible pages are tested.
        for page_size, first, last in ((1, 4, 10), (2, 2, 5), (4, 1, 2)):
            args, kwargs = self.source_case(
                1,
                4,
                3,
                lengths=[4],
                prefixes=[7],
                page_size=page_size,
                columns=12,
            )
            original_table = args[4].clone()
            self.assert_source_verdict(args, kwargs, False)
            args[4][0, 0] = 0
            args[4][0, 11] = (1 << 31) - 1
            self.assert_source_verdict(args, kwargs, False)
            for column in range(first, last + 1):
                with self.subTest(page_size=page_size, column=column):
                    args[4].copy_(original_table)
                    args[4][0, column] = 0
                    self.assert_source_verdict(args, kwargs, True)
            middle = (first + last) // 2
            for bad_page in (-1, -(1 << 31), 31, (1 << 31) - 1):
                with self.subTest(page_size=page_size, physical_page=bad_page):
                    args[4].copy_(original_table)
                    args[4][0, middle] = bad_page
                    self.assert_source_verdict(args, kwargs, True)

    def test_int32_boundaries_and_large_invalid_indices_are_bounded(self):
        maximum = (1 << 31) - 1
        for prefix, length, page_size, invalid in (
            (1, 1, 128, False),
            (maximum, 1, 1 << 30, False),
            (maximum - 3, 4, 1 << 30, False),
            (maximum - 2, 4, 1 << 30, True),
            (maximum, 4, 1 << 30, True),
            (maximum, 1, 1, True),
            (-(1 << 31), 1, 1, True),
            (-1, 4, 1 << 30, True),
            (0, 1, 128, True),
        ):
            with self.subTest(prefix=prefix, length=length, page_size=page_size):
                args, kwargs = self.source_case(
                    1,
                    4,
                    3,
                    lengths=[length],
                    prefixes=[prefix],
                    page_size=page_size,
                    columns=2,
                )
                self.assert_source_verdict(args, kwargs, invalid)
        args, kwargs = self.source_case(lengths=[1, 1, 1])
        args[0].fill_(maximum)
        args[2].fill_(maximum)
        # Match int32 mirrors of a source cumsum that requires int64.
        args[6].copy_(
            torch.tensor(
                [0, maximum, -2, maximum - 2], device="cuda", dtype=torch.int32
            )
        )
        self.assert_source_verdict(args, kwargs, True)


if __name__ == "__main__":
    unittest.main()
