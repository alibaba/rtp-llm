"""Resident physical-index regression for ordinary and MTP TRT sparse decode.

CPU signature checks never import service/CUDA extensions. GPU cases require
an explicitly reserved SM10x device and reuse PackedFixture's independent
RTP656 bytes plus the real PinnedMlaWorkingSet begin/write/prefetch path.
"""

import ast
import unittest
from pathlib import Path

import torch
from trtllm_sparse_decode_test import PAGE, TOPK, PackedFixture, load_backend_class


class PinnedSignatureCpuTest(unittest.TestCase):
    def test_optional_input_is_distinct_from_compacted_output(self):
        directory = Path(__file__).resolve().parents[1]
        tree = ast.parse((directory / "trtllm_sparse_impl.py").read_text())
        cls = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "TrtllmSparseMlaFp8Op"
        )
        forward = next(
            node
            for node in cls.body
            if isinstance(node, ast.FunctionDef) and node.name == "forward"
        )
        defaults = dict(
            zip(
                [arg.arg for arg in forward.args.args][-len(forward.args.defaults) :],
                forward.args.defaults,
            )
        )
        self.assertIsNone(ast.literal_eval(defaults["physical_indices"]))
        call = next(
            node
            for node in ast.walk(cls)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "convert_selected_kv"
        )
        kwargs = {item.arg: item.value for item in call.keywords}
        self.assertEqual(
            ast.dump(kwargs["physical_indices"]),
            ast.dump(ast.Name(id="physical_indices", ctx=ast.Load())),
        )
        self.assertIsInstance(kwargs["indices_out"], ast.Attribute)
        self.assertEqual(kwargs["indices_out"].attr, "physical_indices")
        self.assertEqual(kwargs["indices_out"].value.id, "self")


def assert_bytes(test, actual, expected, name):
    test.assertEqual(actual.shape, expected.shape, name)
    test.assertEqual(actual.dtype, expected.dtype, name)
    test.assertTrue(
        torch.equal(
            actual.contiguous().view(torch.uint8),
            expected.contiguous().view(torch.uint8),
        ),
        name,
    )


def backing_ids(fixture):
    """Independent backing-page map, deliberately retaining future selections."""
    logical = fixture.topk.long()
    request = fixture.request_ids[:, None].long()
    valid = (
        (logical >= 0)
        & (logical < fixture.length)
        & (request >= 0)
        & (request < fixture.batch)
    )
    page = fixture.table[
        request.clamp(0, fixture.batch - 1),
        logical.clamp(0, fixture.length - 1) // PAGE,
    ].long()
    return (
        torch.where(valid, page * PAGE + logical.remainder(PAGE), -1).int().contiguous()
    )


class TrtllmPinnedIndicesGpuTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if (
            not torch.cuda.is_available()
            or torch.cuda.get_device_capability(0)[0] != 10
        ):
            raise unittest.SkipTest("An explicitly reserved SM10x GPU is required")
        load_backend_class()

    def setUp(self):
        self.stream = torch.cuda.Stream()
        self.stream.wait_stream(torch.cuda.current_stream())
        self.context = torch.cuda.stream(self.stream)
        self.context.__enter__()

    def tearDown(self):
        self.stream.synchronize()
        self.context.__exit__(None, None, None)

    def assert_pair(self, ordinary, resident, expected, actual):
        assert_bytes(self, actual, expected, "attention output")
        for name in ("q_fp8", "physical_indices", "valid_counts", "trt_seq_lens"):
            assert_bytes(self, getattr(resident, name), getattr(ordinary, name), name)
        for row, count in enumerate(ordinary.valid_counts.cpu().tolist()):
            assert_bytes(
                self,
                resident.selected_kv[row, : max(count, 1)],
                ordinary.selected_kv[row, : max(count, 1)],
                f"selected KV row{row}",
            )

    def test_slot_zero_bounds_causal_duplicates_and_no_second_table_mapping(self):
        fixture = PackedFixture(batch=2)
        fixture.topk.fill_(-1)
        fixture.seq_lens.fill_(66)
        fixture.q[1].fill_(float("nan"))
        fixture.topk[0, :11] = torch.tensor(
            [0, 1, 1, -1, 66, 64, 65, 0, 0, 0, 0],
            device=fixture.device,
            dtype=torch.int32,
        )
        physical = torch.zeros_like(fixture.topk)
        physical[0, :11] = torch.tensor(
            [0, 1, 1, 0, 0, -1, -2, 64, 2, 63, 2**31 - 1],
            device=fixture.device,
            dtype=torch.int32,
        )
        resident_cache = torch.zeros(
            (1, PAGE, 656), dtype=torch.uint8, device=fixture.device
        )
        original_sources = backing_ids(fixture)
        src0, src1 = original_sources[0, 0].long(), original_sources[0, 1].long()
        flat = fixture.cache.view(-1, 656)
        for slot, source in ((0, src0), (1, src1), (2, src0), (63, src0)):
            resident_cache[0, slot].copy_(flat[source])
        self.assertGreater(int(resident_cache[0, 0].count_nonzero()), 0)
        ordinary, resident = fixture.new_op(), fixture.new_op()
        # The original logical-domain width is retained, but table VALUES
        # cannot participate in a resident address that is already remapped.
        resident.plan(
            fixture.params, torch.full_like(fixture.table, 2**30), fixture.attn_inputs
        )
        effective_topk = fixture.topk.masked_fill(
            (physical < 0) | (physical >= PAGE), -1
        )
        expected = ordinary.forward(fixture.q, fixture.cache, effective_topk).clone()
        saved = physical.clone()
        actual = resident.forward(
            fixture.q,
            resident_cache.view(torch.float8_e4m3fn).unsqueeze(2),
            fixture.topk.unsqueeze(1),
            physical_indices=physical.unsqueeze(1),
        )
        self.assert_pair(ordinary, resident, expected, actual)
        self.assertEqual(resident.valid_counts.cpu().tolist(), [5, 0])
        self.assertEqual(
            resident.source_indices[0, :5].cpu().tolist(), [0, 1, 1, 2, 63]
        )
        assert_bytes(
            self, physical, saved, "physical input must not become indices_out"
        )
        self.assertEqual(int(actual[1].count_nonzero()), 0)
        for invalid in (physical.long(), physical[:, None].expand(-1, 2, -1)):
            with self.assertRaisesRegex(ValueError, "physical_indices"):
                resident.forward(
                    fixture.q, resident_cache, fixture.topk, physical_indices=invalid
                )

    def test_logical_request_and_table_width_guards_still_apply(self):
        fixture = PackedFixture(batch=4)
        fixture.q.fill_(float("nan"))
        fixture.topk.fill_(-1)
        fixture.topk[:3, 0] = 0
        fixture.request_ids[0] = -1
        fixture.request_ids[1] = fixture.batch
        fixture.topk[2, 0] = fixture.table.shape[1] * PAGE
        fixture.seq_lens.fill_(fixture.length + PAGE)
        op = fixture.new_op()
        cache = fixture.cache[:1].clone()
        cache[0, 0].copy_(fixture.cache[1, 0])
        actual = op.forward(
            fixture.q,
            cache,
            fixture.topk,
            physical_indices=torch.zeros_like(fixture.topk),
        )
        self.assertEqual(op.valid_counts.cpu().tolist(), [0, 0, 0, 0])
        self.assertEqual(int(actual.count_nonzero()), 0)

    def test_real_working_set_write_mixed_hbm_and_graph_live_empty_remap(self):
        from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.pinned_mla_cache import (
            PinnedMlaWorkingSet,
        )

        for queries in (1, 4, 6):
            for hbm_tokens in (0, 128):
                with self.subTest(queries=queries, hbm_tokens=hbm_tokens):
                    fixture = PackedFixture(batch=1, queries=queries)
                    table = fixture.table.flatten()
                    position = int((table == 1).nonzero()[0, 0])
                    first = table[0].clone()
                    table[0], table[position] = 1, first
                    fixture.topk.fill_(-1)
                    columns = torch.tensor(
                        [0, 7, 15, 31, 63, 95, 127, 511], device=fixture.device
                    )
                    for row in range(fixture.rows):
                        fixture.topk[row, columns] = torch.tensor(
                            [
                                0,
                                1,
                                1,
                                63,
                                64,
                                65,
                                fixture.length - 1,
                                int(fixture.seq_lens[row]),
                            ],
                            device=fixture.device,
                            dtype=torch.int32,
                        )
                    ids = backing_ids(fixture)
                    capacity = fixture.rows * TOPK
                    host = fixture.cache[hbm_tokens // PAGE :].cpu().pin_memory()
                    resident_cache = torch.empty(
                        ((hbm_tokens + capacity) // PAGE, PAGE, 656),
                        dtype=torch.uint8,
                        device=fixture.device,
                    )
                    if hbm_tokens:
                        resident_cache[: hbm_tokens // PAGE].copy_(
                            fixture.cache[: hbm_tokens // PAGE]
                        )
                    working = PinnedMlaWorkingSet(
                        [host],
                        capacity,
                        PAGE,
                        fixture.device,
                        hbm_tokens=hbm_tokens,
                        hbm_cache=[resident_cache],
                    )
                    ordinary, resident = fixture.new_op(
                        cuda_graph=True
                    ), fixture.new_op(cuda_graph=True)
                    writes = torch.stack((ids[0, 0], ids[0, 63])).long()
                    values = fixture.cache.view(-1, 656)[writes].clone()

                    def run():
                        physical = working.begin(ids)
                        working.write(0, writes, values)
                        cache = working.layer_cache(0)
                        if queries == 4:
                            cache = cache.view(torch.float8_e4m3fn).unsqueeze(2)
                            physical = physical.unsqueeze(1)
                        return resident.forward(
                            fixture.q, cache, fixture.topk, physical_indices=physical
                        )

                    for _ in range(3):
                        output = run()
                    expected = fixture.forward(ordinary).clone()
                    self.assert_pair(ordinary, resident, expected, output)
                    selected = ids >= 0
                    physical = working.physical_indices
                    if hbm_tokens:
                        self.assertTrue(bool(((ids < hbm_tokens) & selected).any()))
                        self.assertTrue(bool((ids >= hbm_tokens).any()))
                        self.assertTrue(bool((physical[selected] >= hbm_tokens).any()))
                    else:
                        self.assertTrue(bool((physical[selected] == 0).any()))
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph, stream=self.stream):
                        output = run()
                    before = output.clone()
                    values.copy_(fixture.cache.view(-1, 656)[writes + 1])
                    fixture.cache.view(-1, 656)[writes] = values
                    graph.replay()
                    expected = fixture.forward(ordinary).clone()
                    self.assert_pair(ordinary, resident, expected, output)
                    self.assertFalse(
                        torch.equal(before, output),
                        "working.write update was not observed",
                    )
                    old_q, old_topk = fixture.q.clone(), fixture.topk.clone()
                    old_physical = working.physical_indices.clone()
                    fixture.q.fill_(float("nan"))
                    fixture.topk.fill_(-1)
                    ids.fill_(-1)
                    graph.replay()
                    self.assertEqual(int(output.count_nonzero()), 0)
                    self.assertEqual(
                        int(resident.q_fp8.view(torch.uint8).count_nonzero()), 0
                    )
                    fixture.q.copy_(old_q)
                    fixture.topk.copy_(old_topk)
                    fixture.table.copy_(fixture.table.roll(1, dims=1))
                    ids.copy_(backing_ids(fixture))
                    graph.replay()
                    expected = fixture.forward(ordinary).clone()
                    self.assert_pair(ordinary, resident, expected, output)
                    self.assertFalse(
                        torch.equal(old_physical, working.physical_indices),
                        "same-graph resident remapping was not observed",
                    )
                    torch.cuda.synchronize()


if __name__ == "__main__":
    unittest.main()
