import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from rtp_llm.models_py.modules.dsv4.moe import mega_buf
from rtp_llm.models_py.modules.dsv4.moe.strategies import mega as mega_strategy


class _Strategy:
    pass


class MegaSymmLifecycleTest(unittest.TestCase):
    def setUp(self) -> None:
        self.saved_graph_baked = mega_buf._MEGA_BUFFERS_GRAPH_BAKED
        self.saved_generation = mega_buf._MEGA_CUDA_GRAPH_GENERATION
        self.saved_invalidated = mega_buf._MEGA_CUDA_GRAPH_INVALIDATED
        self.saved_owners = set(mega_buf._MEGA_CUDA_GRAPH_OWNERS)
        self.saved_invalidated_owners = set(
            mega_buf._MEGA_CUDA_GRAPH_INVALIDATED_OWNERS
        )
        mega_buf._MEGA_BUF_CACHE.clear()
        mega_buf._MEGA_OUTPUT_CACHE.clear()
        mega_buf._MEGA_CUDA_GRAPH_OWNERS.clear()
        mega_buf._MEGA_CUDA_GRAPH_INVALIDATED_OWNERS.clear()
        mega_buf._MEGA_CUDA_GRAPH_INVALIDATED = False
        mega_buf.set_mega_buffers_graph_baked(True)
        self.strategy = _Strategy()
        mega_buf._MEGA_STRATEGY_REGISTRY.add(self.strategy)

    def tearDown(self) -> None:
        mega_buf._MEGA_STRATEGY_REGISTRY.discard(self.strategy)
        mega_buf._MEGA_BUF_CACHE.clear()
        mega_buf._MEGA_OUTPUT_CACHE.clear()
        mega_buf._MEGA_CUDA_GRAPH_OWNERS.clear()
        mega_buf._MEGA_CUDA_GRAPH_OWNERS.update(self.saved_owners)
        mega_buf._MEGA_CUDA_GRAPH_INVALIDATED_OWNERS.clear()
        mega_buf._MEGA_CUDA_GRAPH_INVALIDATED_OWNERS.update(
            self.saved_invalidated_owners
        )
        mega_buf._MEGA_CUDA_GRAPH_GENERATION = self.saved_generation
        mega_buf._MEGA_CUDA_GRAPH_INVALIDATED = self.saved_invalidated
        mega_buf._MEGA_BUFFERS_GRAPH_BAKED = self.saved_graph_baked

    def test_three_invalidate_rebuild_recapture_cycles_are_idempotent(self) -> None:
        expected_generation = mega_buf.mega_buffer_generation()

        for cycle in range(3):
            old_buffer = SimpleNamespace(destroy=Mock())
            old_output = object()
            self.strategy._mega_buf = old_buffer
            self.strategy._mega_y = old_output
            self.strategy._mega_group = object()
            self.strategy._mega_buffer_generation = expected_generation
            self.strategy._mega_buf_kwargs = {}
            self.strategy._mega_out_capacity_tokens = 4
            self.strategy._mega_out_hidden = 8
            self.strategy._mega_out_device = torch.device("cpu")
            mega_buf._MEGA_BUF_CACHE[("cycle", cycle)] = old_buffer
            mega_buf._MEGA_OUTPUT_CACHE[("cycle", cycle)] = old_output

            generation = mega_buf.invalidate_mega_cuda_graph_resources()
            expected_generation += 1
            self.assertEqual(generation, expected_generation)
            self.assertEqual(
                mega_buf.invalidate_mega_cuda_graph_resources(), generation
            )
            old_buffer.destroy.assert_called_once_with()
            self.assertIsNone(self.strategy._mega_buf)
            self.assertIsNone(self.strategy._mega_y)

            new_group = object()
            new_buffer = SimpleNamespace(num_max_tokens_per_rank=8)
            new_output = object()
            with patch.object(
                mega_strategy, "_get_or_create_mega_buf", return_value=new_buffer
            ) as create_buffer, patch.object(
                mega_strategy, "_get_or_create_mega_output", return_value=new_output
            ), patch.object(
                mega_strategy.torch.distributed, "is_initialized", return_value=True
            ), patch.object(
                mega_strategy.torch.distributed,
                "group",
                SimpleNamespace(WORLD=new_group),
            ), patch.object(
                self.strategy,
                "_ensure_mega_buffers",
                side_effect=lambda: mega_strategy.MegaMoEStrategy._ensure_mega_buffers(
                    self.strategy
                ),
                create=True,
            ) as ensure_buffers:
                self.assertEqual(mega_buf.rebuild_mega_symm_buffers(), 1)

            ensure_buffers.assert_called_once_with()
            self.assertIs(create_buffer.call_args.kwargs["group"], new_group)
            self.assertIs(self.strategy._mega_group, new_group)
            self.assertIs(self.strategy._mega_buf, new_buffer)
            self.assertIs(self.strategy._mega_y, new_output)
            self.assertEqual(self.strategy._mega_buffer_generation, generation)

            mega_buf.mark_mega_cuda_graph_resources_recaptured()
            mega_buf.mark_mega_cuda_graph_resources_recaptured()
            self.assertFalse(mega_buf.mega_cuda_graph_resources_invalidated())


if __name__ == "__main__":
    unittest.main()
