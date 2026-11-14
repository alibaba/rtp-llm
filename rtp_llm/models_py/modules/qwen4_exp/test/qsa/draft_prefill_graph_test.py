import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from rtp_llm.models.qwen4_exp.qwen4_exp_kv_cache import (
    INDEXER_KV_TAG,
    INDEXER_STATE_TAG,
)
from rtp_llm.models_py.model_desc.qwen4_exp_runtime.draft_prefill_graph import (
    DraftPrefillGraphReplay,
    draft_graph_lanes,
    draft_graph_rows,
)


class DraftPrefillGraphTest(unittest.TestCase):
    def test_fixed_capacity_mapping_handles_padding_and_changed_lengths(self):
        for lengths, prefixes in (([1, 4, 0], [7, 127, 0]), ([3, 2, 1], [125, 5, 9])):
            cu = [0]
            for length in lengths:
                cu.append(cu[-1] + length)
            inputs = SimpleNamespace(
                input_lengths_device=torch.tensor(lengths, dtype=torch.int32),
                prefix_lengths_device=torch.tensor(prefixes, dtype=torch.int32),
                cu_seqlens_device=torch.tensor(cu, dtype=torch.int32),
            )
            sources, inverse, valid, positions, visible = draft_graph_rows(inputs, 12)
            packed = torch.arange(12)
            recovered = packed.index_select(0, sources).index_select(0, inverse)
            torch.testing.assert_close(recovered[valid], packed[: cu[-1]])
            expected_positions = [
                p + i for p, n in zip(prefixes, lengths) for i in range(n)
            ]
            self.assertEqual(positions[valid].tolist(), expected_positions)
            self.assertTrue(torch.all(visible[inputs.input_lengths_device == 0] == 0))

    def test_short_capacity_mapping_and_prewrite_bounds(self):
        for lanes in (1, 2, 3):
            lengths = [lanes, 1, 0]
            inputs = SimpleNamespace(
                input_lengths_device=torch.tensor(lengths, dtype=torch.int32),
                prefix_lengths_device=torch.tensor([127, 5, 0], dtype=torch.int32),
                cu_seqlens_device=torch.tensor(
                    [0, lanes, lanes + 1, lanes + 1], dtype=torch.int32
                ),
            )
            sources, inverse, valid, _, visible = draft_graph_rows(inputs, 3 * lanes)
            packed = torch.arange(3 * lanes)
            recovered = packed.index_select(0, sources).index_select(0, inverse)
            torch.testing.assert_close(recovered[valid], packed[: lanes + 1])
            self.assertEqual(visible.shape, (3, lanes))
        inputs.input_lengths_device = torch.zeros(0, dtype=torch.int32)
        with self.assertRaisesRegex(RuntimeError, "capacity"):
            draft_graph_lanes(inputs, 4)
        inputs.input_lengths_device = torch.ones(3, dtype=torch.int32)
        for bad in (0, 5, 15):
            with self.assertRaisesRegex(RuntimeError, "capacity"):
                draft_graph_lanes(inputs, bad)
        hook, groups, delegate = self._replay()
        hook.token_capacity = 6
        with self.assertRaisesRegex(RuntimeError, "capacity"):
            hook.prepare_cuda_graph(groups)
        delegate.prepare_cuda_graph.assert_not_called()

    def _replay(self, *, device="cpu"):
        lengths = torch.tensor([1, 4, 0], dtype=torch.int32)
        prefixes = torch.tensor([7, 127, 0], dtype=torch.int32)
        cu = torch.tensor([0, 1, 5, 5], dtype=torch.int32)
        positions = (
            torch.tensor(
                [7, 127, 128, 129, 130] + [0] * 7, device=device, dtype=torch.int32
            )
            .unsqueeze(1)
            .expand(-1, 3)
            .contiguous()
            .flatten()
        )
        tables = torch.tensor([[1, 2], [3, 4], [0, 0]], dtype=torch.int32)
        groups = {}
        for tag in ("full", INDEXER_KV_TAG, INDEXER_STATE_TAG):
            groups[tag] = SimpleNamespace(
                input_lengths=lengths,
                prefix_lengths=prefixes,
                cu_seqlens=cu,
                cu_seqlens_device=cu.to(device).clone(),
                input_lengths_device=lengths.to(device),
                prefix_lengths_device=prefixes.to(device),
                combo_position_ids=positions,
                kv_cache_kernel_block_id=tables.clone(),
                kv_cache_block_id_device=tables.to(device).clone(),
            )
        delegate = Mock()
        hook = DraftPrefillGraphReplay(
            {"full": delegate},
            [
                ("full", 12, 128, False),
                (INDEXER_KV_TAG, 12, 32, True),
                (INDEXER_STATE_TAG, 12, 128, False),
            ],
            3,
            12,
        )
        return hook, groups, delegate

    def test_replay_checks_all_pools_before_delegate(self):
        hook, groups, delegate = self._replay()
        self.assertNotIsInstance(hook, dict)
        hook.prepare_cuda_graph(groups)
        delegate.prepare_cuda_graph.assert_called_once_with(groups["full"])
        delegate.reset_mock()
        groups[INDEXER_KV_TAG].kv_cache_kernel_block_id[1, 0] = 0
        with self.assertRaisesRegex(RuntimeError, "unallocated"):
            hook.prepare_cuda_graph(groups)
        delegate.prepare_cuda_graph.assert_not_called()

    def test_replay_resolves_state_pages_from_refreshed_kernel_table(self):
        hook, groups, delegate = self._replay()
        state = groups[INDEXER_STATE_TAG]
        physical = state.kv_cache_block_id_device
        expanded = physical[..., None] * 2 + torch.arange(2, dtype=torch.int32)
        state.kv_cache_kernel_block_id_device = torch.where(
            physical[..., None] > 0, expanded, 0
        ).flatten(1)
        state.kv_cache_block_id_device = None
        state.kv_cache_block_id = torch.zeros_like(physical)
        state.is_cuda_graph = True
        hook.state_cache = SimpleNamespace(
            get_seq_size_per_block=lambda tag: 256,
            get_kernel_seq_size_per_block=lambda tag: 128,
        )
        hook.bounds[-1] = (INDEXER_STATE_TAG, 12, 256, False)
        hook.prepare_cuda_graph(groups)
        delegate.prepare_cuda_graph.assert_called_once_with(groups["full"])
        delegate.reset_mock()
        state.kv_cache_kernel_block_id_device[1, :2].zero_()
        with self.assertRaisesRegex(RuntimeError, "raw-state pages"):
            hook.prepare_cuda_graph(groups)
        delegate.prepare_cuda_graph.assert_not_called()

    def test_replay_ignores_unused_host_cu_padding_tail(self):
        hook, groups, delegate = self._replay()
        groups["full"].cu_seqlens[-1] = 12
        hook.prepare_cuda_graph(groups)
        delegate.prepare_cuda_graph.assert_called_once()

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA replay mirrors are required")
    def test_device_cu_validation_uses_existing_verdict_without_host_copy(self):
        hook, groups, delegate = self._replay(device="cuda")
        groups["full"].cu_seqlens[1] = 999  # Public host mirror can be stale.
        reads = []
        original_item, original_tolist = torch.Tensor.item, torch.Tensor.tolist

        def item(tensor, *args, **kwargs):
            if tensor.is_cuda:
                reads.append(1)
            return original_item(tensor, *args, **kwargs)

        def tolist(tensor, *args, **kwargs):
            self.assertFalse(tensor.is_cuda, "draft graph copied device CU to host")
            return original_tolist(tensor, *args, **kwargs)

        with (
            patch.object(torch.Tensor, "item", item),
            patch.object(torch.Tensor, "tolist", tolist),
        ):
            hook.prepare_cuda_graph(groups)
        self.assertEqual(len(reads), 1)
        delegate.prepare_cuda_graph.assert_called_once()
        delegate.reset_mock()
        groups["full"].cu_seqlens_device[1] = 999
        with self.assertRaisesRegex(RuntimeError, "cu_seqlens"):
            hook.prepare_cuda_graph(groups)
        delegate.prepare_cuda_graph.assert_not_called()

    def test_replay_canonicalizes_transported_media_positions_in_place(self):
        hook, groups, delegate = self._replay()
        anchor = groups["full"]
        storage = anchor.combo_position_ids.data_ptr()
        for prefixes in ([7, 127, 0], [5, 125, 0]):
            anchor.prefix_lengths.copy_(torch.tensor(prefixes, dtype=torch.int32))
            anchor.combo_position_ids.fill_(999)
            hook.prepare_cuda_graph(groups)
            expected = [prefixes[0], *range(prefixes[1], prefixes[1] + 4)] + [0] * 7
            torch.testing.assert_close(
                anchor.combo_position_ids,
                torch.tensor(expected, dtype=torch.int32).repeat_interleave(3),
            )
            self.assertEqual(anchor.combo_position_ids.data_ptr(), storage)
            delegate.prepare_cuda_graph.assert_called_once_with(anchor)
            delegate.reset_mock()

    def test_replay_rejects_state_pages_positions_and_length_changes(self):
        for defect in ("state", "position_capacity", "length", "cu"):
            with self.subTest(defect=defect):
                hook, groups, delegate = self._replay()
                anchor = groups["full"]
                if defect == "state":
                    groups[INDEXER_STATE_TAG].kv_cache_block_id_device[1, 1] = 12
                elif defect == "position_capacity":
                    anchor.combo_position_ids = anchor.combo_position_ids[:-1]
                elif defect == "length":
                    anchor.input_lengths[2] = 5
                else:
                    anchor.cu_seqlens_device[2] = 6
                with self.assertRaises(RuntimeError):
                    hook.prepare_cuda_graph(groups)
                delegate.prepare_cuda_graph.assert_not_called()

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA replay mirrors are required")
    def test_replay_uses_device_state_table_with_stale_host_table(self):
        hook, groups, delegate = self._replay(device="cuda")
        groups[INDEXER_STATE_TAG].kv_cache_block_id = torch.zeros(
            3, 2, dtype=torch.int32
        )
        hook.prepare_cuda_graph(groups)
        delegate.prepare_cuda_graph.assert_called_once()
        groups[INDEXER_STATE_TAG].kv_cache_block_id_device[1, 1] = 0
        with self.assertRaisesRegex(RuntimeError, "raw-state pages"):
            hook.prepare_cuda_graph(groups)


if __name__ == "__main__":
    unittest.main()
