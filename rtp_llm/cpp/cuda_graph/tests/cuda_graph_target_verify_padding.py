import unittest

import torch

from rtp_llm.cpp.cuda_graph.tests.libtest_cuda_graph_runner import CudaGraphRunner
from rtp_llm.ops.compute_ops import (
    PyAttentionInputs,
    PyModelInputs,
    PyModelOutputs,
    get_typemeta,
)


class _RecordingAttention:
    def __init__(self, owner):
        self.owner = owner

    def prepare_cuda_graph(self, attn_inputs):
        torch.cuda.synchronize()
        self.owner.replay_metadata = {
            "input_lengths": attn_inputs.input_lengths.tolist(),
            "input_lengths_device": attn_inputs.input_lengths_device.cpu().tolist(),
            "prefix_lengths": attn_inputs.prefix_lengths.tolist(),
            "prefix_lengths_device": attn_inputs.prefix_lengths_device.cpu().tolist(),
            "cu_seqlens": attn_inputs.cu_seqlens.tolist(),
            "cu_seqlens_device": attn_inputs.cu_seqlens_device.cpu().tolist(),
            "cu_kv_seqlens_device": attn_inputs.cu_kv_seqlens_device.cpu().tolist(),
        }


class _DummyModel:
    def __init__(self):
        self.replay_metadata = None

    def prepare_fmha_impl(self, inputs, is_cuda_graph=False):
        return _RecordingAttention(self)

    def forward(self, inputs, fmha_impl=None):
        outputs = PyModelOutputs()
        outputs.hidden_states = inputs.input_hiddens + 1
        return outputs


class TestCudaGraphTargetVerifyPadding(unittest.TestCase):
    def setUp(self):
        self.hidden_size = 8
        self.max_seq_len = 40960
        self.tokens_per_block = 1024
        self.kernel_tokens_per_block = 16
        self.num_tokens_per_bs = 4
        self.model = _DummyModel()
        self.runner = CudaGraphRunner()
        self.runner.init_target_verify(
            self.model,
            self.hidden_size,
            self.max_seq_len,
            self.tokens_per_block,
            self.kernel_tokens_per_block,
            self.num_tokens_per_bs,
            [8],
        )

    def tearDown(self):
        # Release graph-owned Python objects before interpreter shutdown.
        self.runner.reset()
        self.runner = None
        self.model = None
        torch.cuda.synchronize()

    def _build_replay_inputs(self):
        batch_size = 6
        token_num = batch_size * self.num_tokens_per_bs
        q_lengths = torch.full((batch_size,), 4, dtype=torch.int32)
        kv_lengths = torch.tensor(
            [8000, 8000, 8000, 8000, 8000, 13504], dtype=torch.int32
        )
        prefix_lengths = kv_lengths - q_lengths

        inputs = PyModelInputs()
        inputs.input_ids = torch.arange(token_num, dtype=torch.int32, device="cuda")
        inputs.input_hiddens = torch.zeros(
            (token_num, self.hidden_size), dtype=torch.bfloat16, device="cuda"
        )

        attn = PyAttentionInputs()
        attn.is_prefill = True
        attn.is_target_verify = True
        attn.is_s_padded = True
        attn.dtype = get_typemeta(torch.zeros(1, dtype=torch.bfloat16))
        attn.input_lengths = q_lengths.pin_memory()
        attn.prefix_lengths = prefix_lengths.pin_memory()
        attn.sequence_lengths = (kv_lengths - 1).pin_memory()
        attn.sequence_lengths_plus_1_device = kv_lengths.cuda()

        cu_q = torch.zeros(batch_size + 1, dtype=torch.int32)
        cu_q[1:] = torch.cumsum(q_lengths, dim=0)
        cu_kv = torch.zeros(batch_size + 1, dtype=torch.int32)
        cu_kv[1:] = torch.cumsum(kv_lengths, dim=0)
        attn.cu_seqlens = cu_q.pin_memory()
        attn.cu_seqlens_device = cu_q.cuda()
        attn.cu_kv_seqlens_device = cu_kv.cuda()
        attn.decode_cu_seqlens_device = torch.arange(
            batch_size + 1, dtype=torch.int32, device="cuda"
        )

        block_num = (
            self.max_seq_len + self.kernel_tokens_per_block - 1
        ) // self.kernel_tokens_per_block
        block_ids = torch.zeros(
            (batch_size, block_num), dtype=torch.int32, device="cuda"
        )
        attn.kv_cache_kernel_block_id_device = block_ids
        attn.kv_cache_kernel_block_id = block_ids.cpu().pin_memory()
        attn.kv_cache_block_id_device = block_ids
        attn.kv_cache_block_id = attn.kv_cache_kernel_block_id
        attn.padding_offset = torch.zeros(token_num, dtype=torch.int32, device="cuda")
        attn.context_total_kv_length = int(kv_lengths.sum().item())
        attn.total_tokens = token_num
        inputs.attention_inputs = attn
        self.runner.initDeviceLengthInputs(inputs)
        return inputs

    def test_batch_6_replaying_graph_8_clears_cumulative_tail(self):
        inputs = self._build_replay_inputs()
        self.assertTrue(self.runner.canRun(inputs))

        outputs = self.runner.forward(inputs)
        torch.cuda.synchronize()

        self.assertEqual(self.runner.getCurrentRealGraphSize(), 8)
        self.assertEqual(tuple(outputs.hidden_states.shape), (24, self.hidden_size))
        torch.testing.assert_close(
            outputs.hidden_states,
            torch.ones_like(outputs.hidden_states),
        )
        self.assertEqual(
            self.model.replay_metadata,
            {
                "input_lengths": [4, 4, 4, 4, 4, 4, 0, 0],
                "input_lengths_device": [4, 4, 4, 4, 4, 4, 0, 0],
                "prefix_lengths": [7996, 7996, 7996, 7996, 7996, 13500, 0, 0],
                "prefix_lengths_device": [
                    7996,
                    7996,
                    7996,
                    7996,
                    7996,
                    13500,
                    0,
                    0,
                ],
                "cu_seqlens": [0, 4, 8, 12, 16, 20, 24, 24, 24],
                "cu_seqlens_device": [0, 4, 8, 12, 16, 20, 24, 24, 24],
                "cu_kv_seqlens_device": [
                    0,
                    8000,
                    16000,
                    24000,
                    32000,
                    40000,
                    53504,
                    53504,
                    53504,
                ],
            },
        )


if __name__ == "__main__":
    unittest.main()
