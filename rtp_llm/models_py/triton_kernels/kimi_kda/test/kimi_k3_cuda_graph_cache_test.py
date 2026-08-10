import math
import unittest
from types import SimpleNamespace

import torch

from rtp_llm.models_py.model_desc.kimi_k3_cuda_graph_cache import (
    load_cuda_graph_decode_tensors,
    store_cuda_graph_decode_state,
)


class KimiK3CudaGraphCacheTest(unittest.TestCase):
    def setUp(self) -> None:
        if not torch.cuda.is_available():
            self.skipTest("CUDA is required")
        self.projection_size = 4
        self.batch = 2
        self.heads = 1
        self.head_dim = 2
        self.history = 3
        self.page_size = 4
        self.block_count = 4

    def _attention_inputs(self, lengths_plus_one: torch.Tensor):
        return SimpleNamespace(
            sequence_lengths_plus_1_d=lengths_plus_one,
            kv_cache_kernel_block_id_device=torch.tensor(
                [[1, 2, 3], [0, 0, 0]], dtype=torch.int32, device="cuda"
            ),
        )

    def _state(self, offset: float):
        conv_shape = (self.batch, self.projection_size, self.history)
        recurrent_shape = (
            self.batch,
            self.heads,
            self.head_dim,
            self.head_dim,
        )

        def values(shape, start, dtype):
            return (
                torch.arange(math.prod(shape), device="cuda")
                .reshape(shape)
                .add_(start + offset)
                .to(dtype)
            )

        return SimpleNamespace(
            q_conv_state=values(conv_shape, 10, torch.bfloat16),
            k_conv_state=values(conv_shape, 20, torch.bfloat16),
            v_conv_state=values(conv_shape, 30, torch.bfloat16),
            recurrent_state=values(recurrent_shape, 40, torch.float32),
        )

    def test_fallback_store_replay_uses_live_page_and_skips_padding(self) -> None:
        lengths_plus_one = torch.tensor(
            [4, 0], dtype=torch.int32, device="cuda"
        )
        attention_inputs = self._attention_inputs(lengths_plus_one)
        state = self._state(0)
        ssm_cache = torch.zeros(
            self.block_count,
            self.heads,
            self.head_dim,
            self.head_dim,
            dtype=torch.float32,
            device="cuda",
        )
        conv_cache = torch.zeros(
            self.block_count,
            self.history,
            3 * self.projection_size,
            dtype=torch.bfloat16,
            device="cuda",
        )

        warmup_stream = torch.cuda.Stream()
        warmup_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(warmup_stream):
            store_cuda_graph_decode_state(
                state,
                ssm_cache.clone(),
                conv_cache.clone(),
                attention_inputs.sequence_lengths_plus_1_d,
                attention_inputs.kv_cache_kernel_block_id_device,
                self.page_size,
            )
        torch.cuda.current_stream().wait_stream(warmup_stream)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            store_cuda_graph_decode_state(
                state,
                ssm_cache,
                conv_cache,
                attention_inputs.sequence_lengths_plus_1_d,
                attention_inputs.kv_cache_kernel_block_id_device,
                self.page_size,
            )

        replay_state = self._state(100)
        for field in (
            "q_conv_state",
            "k_conv_state",
            "v_conv_state",
            "recurrent_state",
        ):
            getattr(state, field).copy_(getattr(replay_state, field))
        ssm_cache.zero_()
        conv_cache.zero_()
        ssm_cache[0].fill_(7)
        conv_cache[0].fill_(8)
        lengths_plus_one.copy_(
            torch.tensor([5, 0], dtype=torch.int32, device="cuda")
        )

        graph.replay()
        torch.cuda.synchronize()

        expected_conv = torch.cat(
            (
                replay_state.q_conv_state[0],
                replay_state.k_conv_state[0],
                replay_state.v_conv_state[0],
            ),
            dim=0,
        ).transpose(0, 1)
        torch.testing.assert_close(
            ssm_cache[2], replay_state.recurrent_state[0].transpose(-1, -2)
        )
        torch.testing.assert_close(conv_cache[2], expected_conv)
        torch.testing.assert_close(ssm_cache[0], torch.full_like(ssm_cache[0], 7))
        torch.testing.assert_close(conv_cache[0], torch.full_like(conv_cache[0], 8))
        torch.testing.assert_close(ssm_cache[1], torch.zeros_like(ssm_cache[1]))
        torch.testing.assert_close(conv_cache[1], torch.zeros_like(conv_cache[1]))

    def test_fallback_load_replay_uses_live_previous_page(self) -> None:
        lengths_plus_one = torch.tensor(
            [5, 0], dtype=torch.int32, device="cuda"
        )
        attention_inputs = self._attention_inputs(lengths_plus_one)
        ssm_cache = torch.zeros(
            self.block_count,
            self.heads,
            self.head_dim,
            self.head_dim,
            dtype=torch.float32,
            device="cuda",
        )
        conv_cache = torch.zeros(
            self.block_count,
            self.history,
            3 * self.projection_size,
            dtype=torch.bfloat16,
            device="cuda",
        )
        ssm_cache[1].fill_(1)
        ssm_cache[2].fill_(2)
        conv_cache[1].fill_(3)
        conv_cache[2].fill_(4)

        warmup_stream = torch.cuda.Stream()
        warmup_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(warmup_stream):
            load_cuda_graph_decode_tensors(
                ssm_cache,
                conv_cache,
                attention_inputs.sequence_lengths_plus_1_d,
                attention_inputs.kv_cache_kernel_block_id_device,
                self.page_size,
                self.projection_size,
            )
        torch.cuda.current_stream().wait_stream(warmup_stream)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            loaded = load_cuda_graph_decode_tensors(
                ssm_cache,
                conv_cache,
                attention_inputs.sequence_lengths_plus_1_d,
                attention_inputs.kv_cache_kernel_block_id_device,
                self.page_size,
                self.projection_size,
            )

        lengths_plus_one.copy_(
            torch.tensor([6, 0], dtype=torch.int32, device="cuda")
        )
        graph.replay()
        torch.cuda.synchronize()

        q_state, _, _, recurrent_state = loaded
        torch.testing.assert_close(
            recurrent_state[0], ssm_cache[2].transpose(-1, -2)
        )
        torch.testing.assert_close(
            q_state[0],
            conv_cache[2, :, : self.projection_size].transpose(0, 1),
        )
        torch.testing.assert_close(
            recurrent_state[1], torch.zeros_like(recurrent_state[1])
        )
        torch.testing.assert_close(q_state[1], torch.zeros_like(q_state[1]))


if __name__ == "__main__":
    unittest.main()
