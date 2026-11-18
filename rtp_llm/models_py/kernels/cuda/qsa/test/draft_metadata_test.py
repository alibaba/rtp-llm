import unittest

import torch

from rtp_llm.models_py.kernels.cuda.qsa.draft_metadata_triton import (
    draft_rows,
    prepare_replay_metadata,
    supports_replay_metadata,
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

    def test_draft_rows_rejects_unsupported_layouts(self):
        rows, _ = self.mapping_case(3, 4, 0)
        self.assertIsNone(draft_rows(*rows, 13))
        self.assertIsNone(draft_rows(*(x.cpu() for x in rows), 12))

    def prepare_case(self, batch, lanes, factor, generation=0, dtype=torch.int64):
        host_lengths = [(i + generation) % lanes + 1 for i in range(batch)]
        if batch > 1:
            host_lengths[-1] = 0
        host_prefixes = [125 + i * 127 + generation for i in range(batch)]
        starts = [0]
        positions = []
        for length, prefix in zip(host_lengths, host_prefixes):
            starts.append(starts[-1] + length)
            positions.extend(prefix + lane for lane in range(length))
        positions.extend([0] * (batch * lanes - len(positions)))
        expected = [position for position in positions for _ in range(factor)]
        lengths, prefixes, cu = (
            torch.tensor(values, device="cuda", dtype=torch.int32)
            for values in (host_lengths, host_prefixes, starts)
        )
        table = torch.arange(
            1, batch * 40 + 1, device="cuda", dtype=torch.int32
        ).reshape(batch, 40)
        transported = torch.full(
            (batch * lanes * factor,), -123, device="cuda", dtype=dtype
        )
        reference = torch.tensor(
            [host_lengths, host_prefixes], device="cuda", dtype=torch.int64
        )
        invalid = torch.ones((), device="cuda", dtype=torch.bool)
        return (
            (lengths, prefixes, cu, table, transported, reference, invalid),
            dict(
                pool_size=batch * 40 + 1,
                page_size=128,
                token_capacity=batch * lanes,
                position_factor=factor,
            ),
            expected,
        )

    def test_prepare_all_capacities_and_axes_from_independent_scalar_reference(self):
        for batch in (1, 3, 8, 17, 32):
            for lanes in (1, 2, 3, 4):
                for factor in (1, 3):
                    with self.subTest(batch=batch, lanes=lanes, factor=factor):
                        inputs, kwargs, expected = self.prepare_case(
                            batch, lanes, factor
                        )
                        self.assertTrue(supports_replay_metadata(*inputs[:5], **kwargs))
                        self.assertIs(
                            prepare_replay_metadata(*inputs, **kwargs), inputs[-1]
                        )
                        self.assertFalse(inputs[-1].item())
                        self.assertEqual(inputs[4].tolist(), expected)

    def test_prepare_dynamic_graph_replay_refreshes_positions_and_flag(self):
        for factor in (1, 3):
            for dtype in (torch.int32, torch.int64):
                with self.subTest(factor=factor, dtype=dtype):
                    inputs, kwargs, _ = self.prepare_case(3, 4, factor, dtype=dtype)
                    stream = torch.cuda.Stream()
                    stream.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(stream):
                        prepare_replay_metadata(*inputs, **kwargs)
                    torch.cuda.current_stream().wait_stream(stream)
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        flag = prepare_replay_metadata(*inputs, **kwargs)
                    addresses = (inputs[4].data_ptr(), flag.data_ptr())
                    for generation, corrupt in ((1, False), (7, True), (0, False)):
                        next_inputs, _, expected = self.prepare_case(
                            3, 4, factor, generation, dtype
                        )
                        for destination, source in zip(inputs, next_inputs):
                            destination.copy_(source)
                        if corrupt:
                            inputs[2][-1] += 1
                        graph.replay()
                        self.assertEqual(
                            (inputs[4].data_ptr(), flag.data_ptr()), addresses
                        )
                        self.assertEqual(flag.item(), corrupt)
                        if not corrupt:
                            self.assertEqual(inputs[4].tolist(), expected)

    def prepare_validation_case(self, factor=3, dtype=torch.int64):
        inputs, kwargs, _ = self.prepare_case(3, 4, factor, dtype=dtype)
        lengths, prefixes, cu, table, transported, reference, invalid = inputs
        lengths.copy_(torch.tensor([1, 4, 0], device="cuda", dtype=torch.int32))
        prefixes.copy_(torch.tensor([7, 127, -999], device="cuda", dtype=torch.int32))
        cu.copy_(torch.tensor([0, 1, 5, 5], device="cuda", dtype=torch.int32))
        table = torch.tensor(
            [[1, 2], [3, 4], [-100, -100]], device="cuda", dtype=torch.int32
        )
        reference.copy_(
            torch.tensor([[1, 4, 0], [7, 127, 999]], device="cuda", dtype=torch.int64)
        )
        invalid.fill_(False)
        kwargs["pool_size"] = 12
        return (lengths, prefixes, cu, table, transported, reference, invalid), kwargs

    def test_prepare_rejects_corrupt_device_metadata_before_writer(self):
        for defect in (
            "cu_start",
            "cu_boundary",
            "cu_padding",
            "cu_negative",
            "cu_large",
            "length_same_total",
            "length_negative",
            "length_too_large",
            "prefix_mismatch",
            "prefix_negative",
            "state_first_zero",
            "state_last_large",
            "state_last_negative",
            "state_first_outside",
            "state_last_outside",
        ):
            with self.subTest(defect=defect):
                inputs, kwargs = self.prepare_validation_case()
                lengths, prefixes, cu, table, _, reference, invalid = inputs
                if defect == "cu_start":
                    cu[0] = 1
                elif defect == "cu_boundary":
                    cu[1] = 2
                elif defect == "cu_padding":
                    cu[-1] = 6
                elif defect == "cu_negative":
                    cu[0] = -2147483648
                elif defect == "cu_large":
                    cu[1] = 2147483647
                elif defect == "length_same_total":
                    lengths[0] = 2
                    lengths[1] = 3
                elif defect == "length_negative":
                    lengths[0] = -1
                elif defect == "length_too_large":
                    lengths[0] = 5
                elif defect == "prefix_mismatch":
                    prefixes[0] = 8
                elif defect == "prefix_negative":
                    prefixes[0] = -2147483648
                    reference[1, 0] = -2147483648
                elif defect == "state_first_zero":
                    table[0, 0] = 0
                elif defect == "state_last_large":
                    table[1, 1] = 12
                elif defect == "state_last_negative":
                    table[1, 1] = -1
                elif defect == "state_first_outside":
                    prefixes[0] = 384
                    reference[1, 0] = 384
                elif defect == "state_last_outside":
                    prefixes[1] = 255
                    reference[1, 1] = 255
                self.assertIs(prepare_replay_metadata(*inputs, **kwargs), invalid)
                self.assertTrue(invalid.item())

    def test_prepare_clears_capacity_padding_and_ignores_inactive_prefixes(self):
        for all_padding in (False, True):
            for factor in (1, 3):
                with self.subTest(all_padding=all_padding, factor=factor):
                    inputs, kwargs = self.prepare_validation_case(factor)
                    lengths, prefixes, cu, table, transported, reference, invalid = (
                        inputs
                    )
                    if all_padding:
                        lengths.zero_()
                        cu.zero_()
                        prefixes.fill_(-2147483648)
                        reference[0].zero_()
                        table.fill_(-1)
                    prepare_replay_metadata(*inputs, **kwargs)
                    self.assertFalse(invalid.item())
                    positions = [] if all_padding else [7, 127, 128, 129, 130]
                    positions += [0] * (12 - len(positions))
                    self.assertEqual(
                        transported.tolist(),
                        [position for position in positions for _ in range(factor)],
                    )

    def test_prepare_int64_page_math_and_int32_output_overflow(self):
        for dtype in (torch.int32, torch.int64):
            for factor in (1, 3):
                with self.subTest(dtype=dtype, factor=factor):
                    inputs, kwargs, _ = self.prepare_case(1, 4, factor, dtype=dtype)
                    lengths, prefixes, cu, _, transported, reference, invalid = inputs
                    lengths.fill_(4)
                    prefixes.fill_(2147483647)
                    cu.copy_(torch.tensor([0, 4], device="cuda", dtype=torch.int32))
                    table = torch.tensor([[1, 2, 3]], device="cuda", dtype=torch.int32)
                    reference.copy_(
                        torch.tensor(
                            [[4], [2147483647]], device="cuda", dtype=torch.int64
                        )
                    )
                    kwargs.update(page_size=1 << 30, pool_size=4)
                    prepare_replay_metadata(
                        lengths,
                        prefixes,
                        cu,
                        table,
                        transported,
                        reference,
                        invalid,
                        **kwargs
                    )
                    self.assertEqual(invalid.item(), dtype == torch.int32)
                    positions = (
                        [2147483647, 0, 0, 0]
                        if dtype == torch.int32
                        else [2147483647 + lane for lane in range(4)]
                    )
                    self.assertEqual(
                        transported.tolist(),
                        [position for position in positions for _ in range(factor)],
                    )

    def test_prepare_unsupported_geometry_leaves_outputs_untouched(self):
        for defect in (
            "cpu",
            "noncontiguous",
            "length_dtype",
            "cu_dtype",
            "table_dtype",
            "reference_dtype",
            "reference_shape",
            "output_dtype",
            "output_size",
            "invalid_dtype",
            "invalid_shape",
            "page_size",
            "small_page",
            "factor",
            "capacity",
        ):
            with self.subTest(defect=defect):
                inputs, kwargs = self.prepare_validation_case()
                inputs = list(inputs)
                if defect == "cpu":
                    inputs = [tensor.cpu() for tensor in inputs]
                elif defect == "noncontiguous":
                    inputs[3] = inputs[3].T.contiguous().T
                elif defect == "length_dtype":
                    inputs[0] = inputs[0].long()
                elif defect == "cu_dtype":
                    inputs[2] = inputs[2].long()
                elif defect == "table_dtype":
                    inputs[3] = inputs[3].long()
                elif defect == "reference_dtype":
                    inputs[5] = inputs[5].int()
                elif defect == "reference_shape":
                    inputs[5] = inputs[5].flatten()
                elif defect == "output_dtype":
                    inputs[4] = inputs[4].float()
                elif defect == "output_size":
                    inputs[4] = inputs[4][:-1]
                elif defect == "invalid_dtype":
                    inputs[6] = inputs[6].int()
                elif defect == "invalid_shape":
                    inputs[6] = inputs[6].reshape(1)
                elif defect == "page_size":
                    kwargs["page_size"] = 127
                elif defect == "small_page":
                    kwargs["page_size"] = 2
                elif defect == "factor":
                    kwargs["position_factor"] = 2
                    inputs[4] = inputs[4][:24].clone()
                elif defect == "capacity":
                    kwargs["token_capacity"] = 13
                    inputs[4] = inputs[4].new_full((39,), -400)
                transported, invalid = inputs[4].clone(), inputs[6].clone()
                self.assertEqual(
                    supports_replay_metadata(*inputs[:5], **kwargs),
                    defect
                    in (
                        "reference_dtype",
                        "reference_shape",
                        "invalid_dtype",
                        "invalid_shape",
                    ),
                )
                self.assertIsNone(prepare_replay_metadata(*inputs, **kwargs))
                self.assertTrue(torch.equal(inputs[4], transported))
                self.assertTrue(torch.equal(inputs[6], invalid))


if __name__ == "__main__":
    unittest.main()
