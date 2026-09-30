import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import torch

from rtp_llm.models.qwen4_exp.qwen4_exp_kv_cache import (
    INDEXER_KV_TAG,
    INDEXER_STATE_TAG,
)
from rtp_llm.models_py.modules.qwen4_exp.draft_prefill_graph import (
    DraftPrefillGraphReplay,
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

    def test_replay_ignores_unused_host_cu_padding_tail(self):
        hook, groups, delegate = self._replay()
        groups["full"].cu_seqlens[-1] = 12
        hook.prepare_cuda_graph(groups)
        delegate.prepare_cuda_graph.assert_called_once()

    def test_replay_rejects_state_pages_positions_and_length_changes(self):
        for defect in ("state", "position", "length", "cu"):
            with self.subTest(defect=defect):
                hook, groups, delegate = self._replay()
                anchor = groups["full"]
                if defect == "state":
                    groups[INDEXER_STATE_TAG].kv_cache_block_id_device[1, 1] = 12
                elif defect == "position":
                    anchor.combo_position_ids[0] = 8
                elif defect == "length":
                    anchor.input_lengths[2] = 5
                else:
                    anchor.cu_seqlens[2] = 6
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
