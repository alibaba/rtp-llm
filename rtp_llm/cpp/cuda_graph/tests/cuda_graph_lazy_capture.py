import os
import sys
import unittest
from types import SimpleNamespace

import torch

import rtp_llm.models
from rtp_llm.cpp.cuda_graph.tests.libtest_cuda_graph_runner import CudaGraphRunner
from rtp_llm.models_py.model_desc.generic_moe import GraphPaddingMask
from rtp_llm.models_py.triton_kernels.causal_conv1d import (
    causal_conv1d_fn,
    prepare_causal_conv1d_graph_metadata,
    prepare_causal_conv1d_metadata,
)
from rtp_llm.models_py.triton_kernels.fla import store_ssm_state_to_block_map
from rtp_llm.models_py.triton_kernels.fla.chunk import chunk_gated_delta_rule
from rtp_llm.models_py.triton_kernels.fla.index import prepare_chunk_graph_metadata
from rtp_llm.ops.compute_ops import (
    PyAttentionInputs,
    PyModelInputs,
    PyModelOutputs,
    get_typemeta,
)


class _Attention:
    def prepare_cuda_graph(self, _inputs):
        pass


class _InspectingAttention:
    def __init__(self, model):
        self.model = model

    def prepare_cuda_graph(self, inputs):
        self.model.last_graph_inputs = {
            "input_lengths": inputs.input_lengths.clone(),
            "prefix_lengths": inputs.prefix_lengths.clone(),
            "cu_seqlens": inputs.cu_seqlens.clone(),
            "cu_kv_seqlens": inputs.cu_kv_seqlens_device.cpu(),
            "padding_offset": inputs.padding_offset.clone(),
            "block_ids_host": inputs.kv_cache_kernel_block_id.clone(),
            "block_ids_device": inputs.kv_cache_kernel_block_id_device.cpu(),
        }


class _Model:
    def prepare_fmha_impl(self, _inputs, _capture):
        return _Attention()

    def forward(self, inputs, _attention):
        token_sum = inputs.input_hiddens.sum(dim=0, keepdim=True)
        return PyModelOutputs(token_sum.expand_as(inputs.input_hiddens))


class _FailingModel(_Model):
    def __init__(self):
        self.forward_calls = 0

    def forward(self, inputs, attention):
        self.forward_calls += 1
        if self.forward_calls == 3:
            raise RuntimeError("intentional capture failure")
        return super().forward(inputs, attention)


class _InspectingModel(_Model):
    def __init__(self):
        self.last_graph_inputs = None

    def prepare_fmha_impl(self, _inputs, _capture):
        return _InspectingAttention(self)


class TestCudaGraphLazyCapture(unittest.TestCase):
    hidden_size = 8
    max_seq_len = 16
    tokens_per_block = 16
    buckets = [1, 4, 8]

    def _runner(self, model=None):
        runner = CudaGraphRunner()
        runner.init_decode(
            model or _Model(),
            self.hidden_size,
            self.max_seq_len,
            self.tokens_per_block,
            self.tokens_per_block,
            self.buckets,
            True,
        )
        self.addCleanup(self._close_runner, runner)
        return runner

    def _prefill_runner(self, model=None, mori_max_tokens=0):
        runner = CudaGraphRunner()
        runner.init_generation_prefill(
            model or _Model(),
            self.hidden_size,
            4,
            self.max_seq_len,
            self.tokens_per_block,
            self.tokens_per_block,
            [4, 8],
            mori_max_tokens,
        )
        self.addCleanup(self._close_runner, runner)
        return runner

    def _close_runner(self, runner):
        torch.cuda.synchronize()
        runner.close()
        torch.cuda.empty_cache()

    def _inputs(self, batch_size, value=1.0):
        inputs = PyModelInputs()
        attention = PyAttentionInputs()

        inputs.input_ids = torch.arange(batch_size, dtype=torch.int32, device="cuda")
        inputs.input_hiddens = torch.full(
            (batch_size, self.hidden_size),
            value,
            dtype=torch.bfloat16,
            device="cuda",
        )

        attention.input_lengths = torch.ones(
            batch_size, dtype=torch.int32, pin_memory=True
        )
        attention.sequence_lengths = torch.ones(
            batch_size, dtype=torch.int32, pin_memory=True
        )
        attention.prefix_lengths = torch.empty(0, dtype=torch.int32, pin_memory=True)
        attention.sequence_lengths_plus_1_device = torch.full(
            (batch_size,), 2, dtype=torch.int32, device="cuda"
        )
        attention.decode_cu_seqlens_device = torch.arange(
            batch_size + 1, dtype=torch.int32, device="cuda"
        )

        block_ids = torch.zeros((batch_size, 1), dtype=torch.int32, device="cuda")
        attention.kv_cache_kernel_block_id_device = block_ids
        attention.kv_cache_kernel_block_id = block_ids.cpu().pin_memory()
        attention.kv_cache_block_id_device = block_ids
        attention.kv_cache_block_id = attention.kv_cache_kernel_block_id

        attention.cu_seqlens = torch.arange(
            batch_size + 1, dtype=torch.int32, pin_memory=True
        )
        attention.cu_seqlens_device = attention.cu_seqlens.cuda()
        attention.cu_kv_seqlens_device = attention.cu_seqlens_device.clone()
        attention.padding_offset = torch.zeros(
            self.max_seq_len, dtype=torch.int32, pin_memory=True
        )
        attention.is_prefill = False
        attention.dtype = get_typemeta(torch.empty(1, dtype=torch.bfloat16))
        attention.context_total_kv_length = batch_size
        attention.total_tokens = batch_size
        inputs.attention_inputs = attention
        return inputs

    def _prefill_inputs(self, input_lengths, prefix_lengths=None, value=1.0):
        input_lengths = list(input_lengths)
        prefix_lengths = (
            [0] * len(input_lengths) if prefix_lengths is None else list(prefix_lengths)
        )
        token_num = sum(input_lengths)
        inputs = PyModelInputs()
        attention = PyAttentionInputs()

        inputs.input_ids = torch.arange(token_num, dtype=torch.int32, device="cuda")
        inputs.input_hiddens = torch.full(
            (token_num, self.hidden_size),
            value,
            dtype=torch.float16,
            device="cuda",
        )

        attention.input_lengths = torch.tensor(
            input_lengths, dtype=torch.int32, pin_memory=True
        )
        attention.sequence_lengths = torch.zeros(
            len(input_lengths), dtype=torch.int32, pin_memory=True
        )
        attention.prefix_lengths = torch.tensor(
            prefix_lengths, dtype=torch.int32, pin_memory=True
        )

        cu_seqlens = [0]
        cu_kv_seqlens = [0]
        for input_len, prefix_len in zip(input_lengths, prefix_lengths):
            cu_seqlens.append(cu_seqlens[-1] + input_len)
            cu_kv_seqlens.append(cu_kv_seqlens[-1] + input_len + prefix_len)
        attention.cu_seqlens = torch.tensor(
            cu_seqlens, dtype=torch.int32, pin_memory=True
        )
        attention.cu_seqlens_device = attention.cu_seqlens.cuda()
        attention.cu_kv_seqlens_device = torch.tensor(
            cu_kv_seqlens, dtype=torch.int32, device="cuda"
        )

        block_ids_host = torch.arange(
            1, len(input_lengths) + 1, dtype=torch.int32
        ).reshape(-1, 1)
        attention.kv_cache_kernel_block_id = block_ids_host.pin_memory()
        attention.kv_cache_kernel_block_id_device = block_ids_host.cuda()
        attention.kv_cache_block_id = attention.kv_cache_kernel_block_id
        attention.kv_cache_block_id_device = attention.kv_cache_kernel_block_id_device
        attention.padding_offset = torch.arange(
            token_num, dtype=torch.int32, pin_memory=True
        )
        attention.is_prefill = True
        attention.dtype = get_typemeta(torch.empty(1, dtype=torch.float16))
        attention.context_total_kv_length = token_num + sum(prefix_lengths)
        attention.total_tokens = token_num
        inputs.attention_inputs = attention
        return inputs

    def test_bucket_selection_and_out_of_range_fallback(self):
        runner = self._runner()
        expected = {1: 1, 2: 4, 4: 4, 5: 8, 8: 8}

        self.assertFalse(runner.captureCurrentBucket())
        for batch_size, bucket in expected.items():
            self.assertEqual(runner.plan(self._inputs(batch_size)), "CaptureAfterEager")
            self.assertEqual(runner.getCurrentRealGraphSize(), bucket)

        self.assertEqual(runner.plan(self._inputs(9)), "Eager")
        self.assertFalse(runner.captureCurrentBucket())

    def test_first_hit_captures_and_later_hits_replay(self):
        runner = self._runner()
        inputs = self._inputs(3, value=2.0)

        self.assertEqual(runner.plan(inputs), "CaptureAfterEager")
        self.assertTrue(runner.captureCurrentBucket())
        self.assertEqual(runner.plan(inputs), "Replay")

        outputs = runner.forward(inputs)
        torch.cuda.synchronize()
        torch.testing.assert_close(
            outputs.hidden_states,
            torch.full_like(outputs.hidden_states, 6.0),
        )

    def test_failed_capture_falls_back_without_retry(self):
        runner = self._runner(_FailingModel())
        inputs = self._inputs(2)

        self.assertEqual(runner.plan(inputs), "CaptureAfterEager")
        self.assertFalse(runner.captureCurrentBucket())
        self.assertEqual(runner.plan(inputs), "Eager")
        self.assertFalse(runner.captureCurrentBucket())

    def test_padded_tail_is_cleared_between_replays(self):
        runner = self._runner()
        full_bucket = self._inputs(4, value=1.0)

        self.assertEqual(runner.plan(full_bucket), "CaptureAfterEager")
        self.assertTrue(runner.captureCurrentBucket())
        self.assertEqual(runner.plan(full_bucket), "Replay")
        runner.forward(full_bucket)
        torch.cuda.synchronize()

        partial_bucket = self._inputs(2, value=2.0)
        self.assertEqual(runner.plan(partial_bucket), "Replay")
        outputs = runner.forward(partial_bucket)
        torch.cuda.synchronize()
        torch.testing.assert_close(
            outputs.hidden_states,
            torch.full_like(outputs.hidden_states, 4.0),
        )

    def test_generation_prefill_uses_total_token_buckets(self):
        runner = self._prefill_runner()
        expected = {
            (3,): 4,
            (2, 2): 4,
            (2, 3): 8,
            (4, 4): 8,
        }
        for lengths, bucket in expected.items():
            self.assertEqual(
                runner.plan(self._prefill_inputs(lengths)), "CaptureAfterEager"
            )
            self.assertEqual(runner.getCurrentRealGraphSize(), bucket)

        self.assertEqual(runner.plan(self._prefill_inputs([9])), "Eager")
        self.assertEqual(runner.plan(self._prefill_inputs([1, 1, 1, 1, 1])), "Eager")
        self.assertEqual(runner.plan(self._prefill_inputs([2], [15])), "Eager")

    def test_generation_prefill_first_hit_then_replay(self):
        runner = self._prefill_runner()
        inputs = self._prefill_inputs([2, 1], value=2.0)

        self.assertEqual(runner.plan(inputs), "CaptureAfterEager")
        self.assertTrue(runner.captureCurrentBucket())
        self.assertEqual(runner.plan(inputs), "Replay")

        outputs = runner.forward(inputs)
        torch.cuda.synchronize()
        torch.testing.assert_close(
            outputs.hidden_states,
            torch.full_like(outputs.hidden_states, 6.0),
        )

    def test_generation_prefill_clears_all_metadata_tails(self):
        model = _InspectingModel()
        runner = self._prefill_runner(model)
        bucket_input = self._prefill_inputs([4], value=1.0)
        self.assertEqual(runner.plan(bucket_input), "CaptureAfterEager")
        self.assertTrue(runner.captureCurrentBucket())

        poison = self._prefill_inputs([2, 2], [1, 1], value=3.0)
        self.assertEqual(runner.plan(poison), "Replay")
        runner.forward(poison)
        torch.cuda.synchronize()

        partial = self._prefill_inputs([1, 2], [1, 0], value=2.0)
        self.assertEqual(runner.plan(partial), "Replay")
        outputs = runner.forward(partial)
        torch.cuda.synchronize()
        torch.testing.assert_close(
            outputs.hidden_states,
            torch.full_like(outputs.hidden_states, 6.0),
        )

        snapshot = model.last_graph_inputs
        self.assertIsNotNone(snapshot)
        torch.testing.assert_close(
            snapshot["input_lengths"], torch.tensor([1, 2, 0, 0], dtype=torch.int32)
        )
        torch.testing.assert_close(
            snapshot["prefix_lengths"], torch.tensor([1, 0, 0, 0], dtype=torch.int32)
        )
        torch.testing.assert_close(
            snapshot["cu_seqlens"], torch.tensor([0, 1, 3, 3, 3], dtype=torch.int32)
        )
        torch.testing.assert_close(
            snapshot["cu_kv_seqlens"],
            torch.tensor([0, 2, 4, 4, 4], dtype=torch.int32),
        )
        torch.testing.assert_close(
            snapshot["padding_offset"], torch.tensor([0, 1, 2, 0], dtype=torch.int32)
        )
        expected_blocks = torch.tensor([[1], [2], [0], [0]], dtype=torch.int32)
        torch.testing.assert_close(snapshot["block_ids_host"], expected_blocks)
        torch.testing.assert_close(snapshot["block_ids_device"], expected_blocks)

    def test_graph_padding_mask_updates_in_place(self):
        cu_seqlens = torch.tensor([0, 3, 7], dtype=torch.int32, pin_memory=True)
        cu_seqlens_device = cu_seqlens.cuda()
        attention_inputs = SimpleNamespace(
            is_cuda_graph=True,
            is_prefill=True,
            cu_seqlens=cu_seqlens,
            cu_seqlens_device=cu_seqlens_device,
        )
        graph_padding_mask = GraphPaddingMask()
        padding_mask = graph_padding_mask.get(
            attention_inputs, 8, cu_seqlens_device.device
        )
        original_ptr = padding_mask.data_ptr()

        graph = torch.cuda.CUDAGraph()
        capture_stream = torch.cuda.Stream()
        capture_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(capture_stream):
            graph.capture_begin()
            graph_padding_mask.get(attention_inputs, 8, cu_seqlens_device.device)
            graph.capture_end()
        torch.cuda.current_stream().wait_stream(capture_stream)

        cu_seqlens[-1].fill_(4)
        cu_seqlens_device[-1].fill_(5)
        graph.replay()
        torch.cuda.synchronize()

        self.assertEqual(padding_mask.data_ptr(), original_ptr)
        torch.testing.assert_close(
            padding_mask,
            torch.tensor(
                [False, False, False, False, False, True, True, True],
                device="cuda",
            ),
        )

    def test_causal_conv_metadata_updates_during_graph_replay(self):
        query_start_loc = torch.tensor(
            [0, 4, 8, 12, 16], dtype=torch.int32, device="cuda"
        )
        x = torch.randn(16, 32, dtype=torch.float16, device="cuda").T
        weight = torch.randn(32, 4, dtype=torch.float16, device="cuda")
        prefix_lengths = torch.zeros(4, dtype=torch.int32, device="cuda")
        metadata = prepare_causal_conv1d_graph_metadata(
            query_start_loc, query_start_loc.device, 16
        )
        causal_conv1d_fn(
            x,
            weight,
            None,
            None,
            query_start_loc,
            None,
            prefix_lengths,
            16,
            metadata=metadata,
        )
        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()
        capture_stream = torch.cuda.Stream()
        capture_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(capture_stream):
            graph.capture_begin()
            prepare_causal_conv1d_graph_metadata(
                query_start_loc, query_start_loc.device, 16, metadata
            )
            graph_output = causal_conv1d_fn(
                x,
                weight,
                None,
                None,
                query_start_loc,
                None,
                prefix_lengths,
                16,
                metadata=metadata,
            )
            graph.capture_end()
        torch.cuda.current_stream().wait_stream(capture_stream)

        query_start_loc.copy_(
            torch.tensor([0, 1, 9, 10, 14], dtype=torch.int32, device="cuda")
        )
        graph.replay()
        torch.cuda.synchronize()
        expected = causal_conv1d_fn(
            x,
            weight,
            None,
            None,
            query_start_loc,
            None,
            prefix_lengths,
            16,
            metadata=prepare_causal_conv1d_metadata(
                query_start_loc, query_start_loc.device
            ),
        )
        torch.cuda.synchronize()
        torch.testing.assert_close(graph_output[:, :14], expected[:, :14])

    def test_causal_conv_state_store_uses_replay_block_ids(self):
        query_start_loc = torch.tensor(
            [0, 21, 21, 21, 21], dtype=torch.int32, device="cuda"
        )
        x = torch.randn(32, 21, dtype=torch.float16, device="cuda")
        weight = torch.randn(32, 4, dtype=torch.float16, device="cuda")
        prefix_lengths = torch.zeros(4, dtype=torch.int32, device="cuda")
        block_map = torch.zeros((4, 8), dtype=torch.int32, device="cuda")
        conv_states = torch.zeros(
            (5, 3, 32), dtype=torch.float16, device="cuda"
        ).transpose(1, 2)
        metadata = prepare_causal_conv1d_graph_metadata(
            query_start_loc, query_start_loc.device, 21
        )

        causal_conv1d_fn(
            x,
            weight,
            None,
            conv_states,
            query_start_loc,
            block_map,
            prefix_lengths,
            1024,
            metadata=metadata,
        )
        torch.cuda.synchronize()
        conv_states.zero_()

        graph = torch.cuda.CUDAGraph()
        capture_stream = torch.cuda.Stream()
        capture_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(capture_stream):
            graph.capture_begin()
            prepare_causal_conv1d_graph_metadata(
                query_start_loc, query_start_loc.device, 21, metadata
            )
            causal_conv1d_fn(
                x,
                weight,
                None,
                conv_states,
                query_start_loc,
                block_map,
                prefix_lengths,
                1024,
                metadata=metadata,
            )
            graph.capture_end()
        torch.cuda.current_stream().wait_stream(capture_stream)

        block_map[0, 0] = 2
        conv_states.zero_()
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(conv_states[2], x[:, -3:])

    def test_ssm_state_store_uses_replay_block_ids(self):
        query_start_loc = torch.tensor(
            [0, 21, 21, 21, 21], dtype=torch.int32, device="cuda"
        )
        prefix_lengths = torch.zeros(4, dtype=torch.int32, device="cuda")
        block_map = torch.zeros((4, 8), dtype=torch.int32, device="cuda")
        metadata = prepare_chunk_graph_metadata(query_start_loc, 21, 64)
        h = torch.randn(5, 1, 16, 16, dtype=torch.float32, device="cuda")
        final_states = torch.randn(4, 1, 16, 16, dtype=torch.float32, device="cuda")
        ssm_states = torch.zeros(5, 1, 16, 16, dtype=torch.float32, device="cuda")

        store_ssm_state_to_block_map(
            h,
            final_states,
            prefix_lengths,
            query_start_loc,
            block_map,
            ssm_states,
            1024,
            64,
            block_v=16,
            chunk_indices=metadata.chunk_indices,
        )
        torch.cuda.synchronize()
        ssm_states.zero_()

        graph = torch.cuda.CUDAGraph()
        capture_stream = torch.cuda.Stream()
        capture_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(capture_stream):
            graph.capture_begin()
            prepare_chunk_graph_metadata(query_start_loc, 21, 64, metadata)
            store_ssm_state_to_block_map(
                h,
                final_states,
                prefix_lengths,
                query_start_loc,
                block_map,
                ssm_states,
                1024,
                64,
                block_v=16,
                chunk_indices=metadata.chunk_indices,
            )
            graph.capture_end()
        torch.cuda.current_stream().wait_stream(capture_stream)

        block_map[0, 0] = 2
        ssm_states.zero_()
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(ssm_states[2], final_states[0])

    def test_fla_final_state_and_store_match_eager_on_replay(self):
        torch.manual_seed(7)
        query_start_loc = torch.tensor(
            [0, 21, 21, 21, 21], dtype=torch.int32, device="cuda"
        )
        prefix_lengths = torch.zeros(4, dtype=torch.int32, device="cuda")
        block_map = torch.zeros((4, 8), dtype=torch.int32, device="cuda")
        q = torch.randn((1, 21, 4, 128), dtype=torch.bfloat16, device="cuda")
        k = torch.randn_like(q)
        v = torch.randn_like(q)
        g = torch.nn.functional.logsigmoid(
            torch.randn((1, 21, 4), dtype=torch.float32, device="cuda")
        )
        beta = torch.rand((1, 21, 4), dtype=torch.bfloat16, device="cuda")
        initial_state = torch.zeros(
            (4, 4, 128, 128), dtype=torch.float32, device="cuda"
        )
        ssm_states = torch.zeros((5, 4, 128, 128), dtype=torch.float32, device="cuda")
        metadata = prepare_chunk_graph_metadata(query_start_loc, 21, 64)

        chunk_gated_delta_rule(
            q,
            k,
            v,
            g,
            beta,
            initial_state=initial_state,
            output_final_state=True,
            cu_seqlens=query_start_loc,
            use_qk_l2norm_in_kernel=True,
            chunk_metadata=metadata,
        )
        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()
        capture_stream = torch.cuda.Stream()
        capture_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(capture_stream):
            graph.capture_begin()
            prepare_chunk_graph_metadata(query_start_loc, 21, 64, metadata)
            graph_output, graph_h, graph_final_state = chunk_gated_delta_rule(
                q,
                k,
                v,
                g,
                beta,
                initial_state=initial_state,
                output_final_state=True,
                cu_seqlens=query_start_loc,
                use_qk_l2norm_in_kernel=True,
                chunk_metadata=metadata,
            )
            store_ssm_state_to_block_map(
                graph_h,
                graph_final_state,
                prefix_lengths,
                query_start_loc,
                block_map,
                ssm_states,
                1024,
                64,
                chunk_indices=metadata.chunk_indices,
            )
            graph.capture_end()
        torch.cuda.current_stream().wait_stream(capture_stream)

        block_map[0, 0] = 2
        ssm_states.zero_()
        graph.replay()
        torch.cuda.synchronize()
        expected_output, _, expected_final_state = chunk_gated_delta_rule(
            q,
            k,
            v,
            g,
            beta,
            initial_state=initial_state,
            output_final_state=True,
            cu_seqlens=query_start_loc,
            use_qk_l2norm_in_kernel=True,
        )
        torch.cuda.synchronize()

        torch.testing.assert_close(graph_output, expected_output)
        torch.testing.assert_close(graph_final_state, expected_final_state)
        torch.testing.assert_close(ssm_states[2], expected_final_state[0])

    def test_generation_prefill_disables_buckets_above_mori_capacity(self):
        runner = self._prefill_runner(mori_max_tokens=4)
        self.assertEqual(runner.plan(self._prefill_inputs([4])), "CaptureAfterEager")
        self.assertEqual(runner.plan(self._prefill_inputs([5])), "Eager")
        self.assertFalse(runner.captureCurrentBucket())


if __name__ == "__main__":
    program = unittest.main(exit=False)
    sys.stdout.flush()
    sys.stderr.flush()
    # ROCm may segfault during Python global teardown after valid HIP Graph use.
    os._exit(0 if program.result.wasSuccessful() else 1)
