import unittest

import torch

from rtp_llm.models_py.modules.qwen4_exp.qsa.draft_metadata_triton import (
    draft_rows,
    replay_invalid,
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


if __name__ == "__main__":
    unittest.main()
