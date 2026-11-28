"""CPU cache geometry/graph-metadata contract, not a captured CUDA Graph."""

import unittest
from types import SimpleNamespace

import torch

from rtp_llm.config.qwen4_cache_tags import INDEXER_KV_TAG, INDEXER_STATE_TAG
from rtp_llm.models_py.model_desc.qwen4_exp_runtime.draft_prefill_graph import (
    make_mtp_draft_replay,
)


class _CpuCache:
    def get_layer_cache(self, layer_idx, tag):
        assert layer_idx == 0
        pools = {"full": (16, 8), INDEXER_KV_TAG: (24, 8), INDEXER_STATE_TAG: (12, 8)}
        blocks, page = pools[tag]
        return SimpleNamespace(
            kv_cache_base=torch.empty(blocks), seq_size_per_block=page
        )

    def get_seq_size_per_block(self, tag):
        return 8

    def get_kernel_seq_size_per_block(self, tag):
        return 4


class _Plan:
    def __init__(self):
        self.attn_configs = SimpleNamespace(kernel_tokens_per_block=4)
        self.draft = False
        self.seen = []

    def set_mtp_draft_mode(self, mode):
        self.draft = mode

    def prepare_cuda_graph(self, inputs):
        self.seen.append(inputs)


def _groups():
    def make():
        def ints(*rows):
            return torch.tensor(rows, dtype=torch.int32)

        return SimpleNamespace(
            is_cuda_graph=True,
            input_lengths=ints(2, 0),
            input_lengths_device=ints(2, 0),
            prefix_lengths=ints(8, 0),
            prefix_lengths_device=ints(8, 0),
            cu_seqlens_device=ints(0, 2, 2),
            combo_position_ids=torch.full((8,), -99, dtype=torch.int32),
            kv_cache_kernel_block_id=ints([1, 2, 3, 0], [0, 0, 0, 0]),
            kv_cache_kernel_block_id_device=ints([2, 3, 4, 5], [0, 0, 0, 0]),
        )

    return {tag: make() for tag in ("full", INDEXER_KV_TAG, INDEXER_STATE_TAG)}


class MTPDraftReplayTest(unittest.TestCase):
    def replay(self):
        plan = _Plan()
        model = SimpleNamespace(
            layers=[object()],
            config=SimpleNamespace(_qwen4_indexer_compress_ratio=4),
            kv_cache=_CpuCache(),
            _attention_tag=lambda layer: "full",
            cuda_graph_position_id_len_factor=lambda: 2,
        )
        result = make_mtp_draft_replay(model, {"full": plan}, token_capacity=4)
        self.assertTrue(plan.draft)
        self.assertEqual(
            result.bounds,
            [
                ("full", 16, 4, False),
                (INDEXER_KV_TAG, 24, 2, True),
                (INDEXER_STATE_TAG, 12, 8, False),
            ],
        )
        return result, plan

    def test_padded_replay_updates_main_and_side_positions(self):
        replay, plan = self.replay()
        groups = _groups()
        replay.prepare_cuda_graph(groups)
        expected = torch.tensor([8, 8, 9, 9, 0, 0, 0, 0], dtype=torch.int32)
        for values in groups.values():
            torch.testing.assert_close(values.combo_position_ids, expected)
        self.assertEqual(plan.seen, [groups["full"]])

    def test_invalid_state_mapping_fails_before_plan_callback(self):
        replay, plan = self.replay()
        groups = _groups()
        groups[INDEXER_STATE_TAG].kv_cache_kernel_block_id_device.zero_()
        with self.assertRaisesRegex(RuntimeError, "invalid raw-state pages"):
            replay.prepare_cuda_graph(groups)
        self.assertEqual(plan.seen, [])


if __name__ == "__main__":
    unittest.main()
