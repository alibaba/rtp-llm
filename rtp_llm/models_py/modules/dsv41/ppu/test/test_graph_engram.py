"""CPU contracts for graph Engram windows, ENP padding and owned row reads.

Collective mocks exercise the fixed-capacity protocol; actual PCCL capture and
multi-rank execution are verified by the PPU integration target.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from rtp_llm.models_py.modules.dsv41.ppu.engram import (
    EngramHashParams,
    EngramShardLayout,
    HostShard,
    compute_engram_hash_ids,
)
from rtp_llm.models_py.modules.dsv41.ppu.graph_engram import GraphEngram


def _geometry(*, layers=(3,), embeddings=(300,)):
    return SimpleNamespace(
        engram_layer_ids=layers,
        engram_num_embeddings=embeddings,
        engram_max_ngram_size=4,
        engram_n_heads=2,
        engram_head_dim=64,
        engram_vocab_size=13,
        engram_compressed_vocab_size=32,
        engram_token_map=tuple(range(32)),
        engram_pad_token_id=2,
        image_token_id=31,
    )


def _table(rows=300):
    generator = torch.Generator().manual_seed(419)
    payload = (torch.randn(rows, 64, generator=generator) * 0.2).to(torch.float8_e4m3fn)
    scales = torch.randint(124, 131, (rows, 2), generator=generator).to(torch.uint8)
    return payload, scales


def _rows_reference(payload, scales, ids):
    # Evaluate the full table independently of the candidate's ownership map.
    valid = (ids >= 0) & (ids < payload.shape[0])
    safe = ids.clamp(0, payload.shape[0] - 1)
    values = payload.view(torch.uint8)[safe].view(torch.float8_e4m3fn).float()
    powers = torch.pow(2.0, scales[safe].float() - 127).repeat_interleave(32, dim=-1)
    return (values * powers).to(torch.bfloat16).masked_fill(~valid[..., None], 0)


class GraphEngramTest(unittest.TestCase):
    def _single(self, capacity=4):
        geometry = _geometry()
        payload, scales = _table()
        layout = EngramShardLayout.from_table(300, 1)
        shard = HostShard.from_tensors(payload, scales, layout, 0)
        ops = SimpleNamespace(_engram={3: (shard, layout, None)})
        graph = GraphEngram(ops, geometry, device="cpu", max_batch_size=capacity)
        return graph, payload, scales

    def test_scheduler_window_hash_matches_existing_independent_path(self):
        graph, _, _ = self._single()
        windows = torch.tensor([[8, 7, 6, 5], [4, 3, -1, -1], [9, 31, 2, 1]])
        geometry = _geometry()
        params = EngramHashParams.build(
            layer_ids=geometry.engram_layer_ids,
            max_ngram_size=4,
            n_heads=2,
            vocab_size=13,
            compressed_vocab_size=32,
            num_embeddings=(300,),
            token_map=list(range(32)),
            pad_token_id=2,
        )
        expected = compute_engram_hash_ids(windows, params, windows == 31)
        torch.testing.assert_close(
            graph.hash(windows.to(torch.int32)), expected, rtol=0, atol=0
        )
        changed = windows.clone()
        changed[0, 0] = 11
        self.assertFalse(torch.equal(graph.hash(changed)[0], expected[0]))

    def test_padding_image_and_out_of_range_current_tokens_have_no_lookup(self):
        graph, _, _ = self._single()
        windows = torch.tensor([[-1, -1, -1, -1], [31, 1, 2, 3], [999, 2, 1, -1]])
        self.assertTrue(torch.equal(graph.hash(windows), torch.full((3, 1, 6), -1)))

    def test_bucket_changes_update_rows_without_changing_workspace_addresses(self):
        graph, payload, scales = self._single()
        binding = graph._bindings[3]
        buffers = (binding.ids, binding.gathered, binding.contribution)
        pointers = [tensor.data_ptr() for tensor in buffers]
        first = torch.tensor([[0, 1, 299, 300, -1, 11], [5, 8, 9, 10, 12, 15]])
        actual = graph.lookup(
            first, layer_id=3, valid_mask=torch.tensor([True, False])
        ).clone()
        expected = _rows_reference(payload, scales, first)
        expected[1].zero_()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertTrue(torch.equal(binding.ids[6:], torch.full((18,), -1)))
        second = torch.tensor([[7, 8, 9, 10, 11, 12]])
        actual = graph.lookup(second, layer_id=3, valid_mask=torch.tensor([True]))
        torch.testing.assert_close(
            actual, _rows_reference(payload, scales, second), rtol=0, atol=0
        )
        self.assertTrue(torch.equal(binding.ids[6:], torch.full((18,), -1)))
        self.assertEqual(pointers, [tensor.data_ptr() for tensor in buffers])

    def test_cross_owner_rows_and_idle_rank_keep_fixed_collective_shapes(self):
        geometry = _geometry()
        payload, scales = _table()
        layout = EngramShardLayout.from_table(300, 2)
        shard = HostShard.from_tensors(payload, scales, layout, 1)
        process_group = object()
        group = SimpleNamespace(pg=process_group)
        ops = SimpleNamespace(_engram={3: (shard, layout, group)})
        capacity = 3 * 6
        peer = torch.full((capacity,), -1, dtype=torch.int64)
        peer[:6] = torch.tensor([1, 256, 299, 300, -1, 8])
        shapes = []

        def gather(out, local, *, group):
            self.assertIs(group, process_group)
            shapes.append((out.shape, local.shape))
            out[:capacity].copy_(peer)
            out[capacity:].copy_(local)

        def reduce(values, *, op, group):
            self.assertIs(group, process_group)
            # The other owner's independently evaluated contribution.
            gathered = graph._bindings[3].gathered
            other_ids = gathered.masked_fill(gathered >= 256, -1)
            values.add_(_rows_reference(payload, scales, other_ids))

        with (
            patch("torch.distributed.is_initialized", return_value=True),
            patch("torch.distributed.get_world_size", return_value=2),
            patch("torch.distributed.get_rank", return_value=1),
            patch(
                "torch.distributed.all_gather_into_tensor", side_effect=gather
            ) as gathered,
            patch("torch.distributed.all_reduce", side_effect=reduce) as reduced,
        ):
            graph = GraphEngram(ops, geometry, device="cpu", max_batch_size=3)
            queries = torch.tensor([[1, 256, 299, 300, -1, 8]])
            actual = graph.lookup(
                queries, layer_id=3, valid_mask=torch.tensor([True])
            ).clone()
            torch.testing.assert_close(
                actual, _rows_reference(payload, scales, queries), rtol=0, atol=0
            )
            idle = graph.lookup(queries, layer_id=3, valid_mask=torch.tensor([False]))
            self.assertTrue(torch.equal(idle, torch.zeros_like(idle)))
            self.assertEqual(gathered.call_count, 2)
            self.assertEqual(reduced.call_count, 2)
            self.assertEqual(shapes, [(torch.Size([36]), torch.Size([18]))] * 2)

    def test_replacing_host_backing_is_rejected_while_original_storage_is_held(self):
        graph, _, _ = self._single()
        binding = graph._bindings[3]
        pointer = binding.payload_pointer
        graph.validate()
        binding.shard.payload = (
            binding.payload.view(torch.uint8).clone().view(torch.float8_e4m3fn)
        )
        self.assertEqual(binding.payload.data_ptr(), pointer)
        with self.assertRaisesRegex(RuntimeError, "backing storage changed"):
            graph.validate()

    def test_capacity_width_and_validity_contracts(self):
        graph, _, _ = self._single(capacity=1)
        with self.assertRaisesRegex(ValueError, "exceed capacity"):
            graph.hash(torch.zeros(2, 4, dtype=torch.int64))
        with self.assertRaisesRegex(ValueError, "n-gram width"):
            graph.hash(torch.zeros(1, 3, dtype=torch.int64))
        with self.assertRaisesRegex(ValueError, "one bool"):
            graph.lookup(
                torch.zeros(1, 6, dtype=torch.int64),
                layer_id=3,
                valid_mask=torch.ones(1),
            )


if __name__ == "__main__":
    unittest.main()
