import logging
import os
import unittest

import torch

import rtp_llm.models
from rtp_llm.cpp.cuda_graph.tests.cuda_graph_test_utils import (
    CudaGraphTestModelBuilder,
    ModelBuildConfig,
)
from rtp_llm.cpp.cuda_graph.tests.libtest_cuda_graph_runner import CudaGraphRunner
from rtp_llm.ops.compute_ops import PyAttentionInputs, PyModelInputs, get_typemeta


class ScratchCudaGraphTestModelBuilder(CudaGraphTestModelBuilder):
    """Keep this test's loader choice local instead of changing the shared builder."""

    def _set_configs(self) -> None:
        super()._set_configs()
        assert self.py_env_configs is not None
        self.py_env_configs.load_config.load_method = "scratch"


class TestCudaGraphTargetVerifyPadding(unittest.TestCase):
    def setUp(self) -> None:
        os.environ["RESERVER_RUNTIME_MEM_MB"] = "10240"
        self.device = "cuda:0"
        self.max_seq_len = 64
        self.tokens_per_block = 64
        self.num_tokens_per_bs = 2
        self.decode_capture_batch_sizes = [1, 8]

        self.model_builder = ScratchCudaGraphTestModelBuilder(
            ModelBuildConfig(
                model_path="/mnt/nas1/hf/Qwen2.5-0.5B-Instruct",
                max_seq_len=self.max_seq_len,
                tokens_per_block=self.tokens_per_block,
                device=self.device,
            )
        )

        graph_build = self.model_builder.build_model(init_kv_cache=True)
        self.graph_model = graph_build.model
        self.graph_cache = graph_build.kv_cache
        self.compute_dtype = graph_build.compute_dtype
        self.hidden_size = graph_build.hidden_size
        self.kernel_tokens_per_block = int(
            graph_build.model_config.attn_config.kernel_tokens_per_block
        )

        self.graph_runner = CudaGraphRunner()
        self.graph_runner.init_target_verify(
            self.graph_model,
            self.hidden_size,
            self.max_seq_len,
            self.tokens_per_block,
            self.kernel_tokens_per_block,
            self.num_tokens_per_bs,
            self.decode_capture_batch_sizes,
        )

        reference_build = self.model_builder.build_model(init_kv_cache=True)
        self.reference_model = reference_build.model
        self.reference_cache = reference_build.kv_cache

        # Graph capture and its warmups can touch the dummy cache block. Start
        # the non-Graph reference from exactly the same post-capture cache state
        # so the full cache comparison below detects replay-only pollution.
        self._copy_cache(self.graph_cache, self.reference_cache)

    @staticmethod
    def _copy_cache(source, destination) -> None:
        torch.cuda.synchronize()
        for source_tensor, destination_tensor in zip(
            source.kv_cache_base_by_layer, destination.kv_cache_base_by_layer
        ):
            destination_tensor.copy_(source_tensor)
        for source_tensor, destination_tensor in zip(
            source.kv_scale_base_by_layer, destination.kv_scale_base_by_layer
        ):
            destination_tensor.copy_(source_tensor)
        torch.cuda.synchronize()

    def _build_inputs(self, batch_size: int) -> PyModelInputs:
        token_count = batch_size * self.num_tokens_per_bs
        inputs = PyModelInputs()
        attention_inputs = PyAttentionInputs()

        inputs.input_ids = (
            torch.arange(token_count, dtype=torch.int32, device=self.device) % 32
        ) + 1
        inputs.input_hiddens = torch.zeros(
            (token_count, self.hidden_size),
            dtype=self.compute_dtype,
            device=self.device,
        )

        input_lengths = torch.full(
            (batch_size,), self.num_tokens_per_bs, dtype=torch.int32
        ).pin_memory()
        # input_lengths_device and prefix_lengths_device are intentionally
        # read-only in the Python binding. Match the values used to initialize
        # the capture buffers so active rows already contain the right device
        # metadata; prepareInputs still has to clear rows [batch_size, 8).
        prefix_lengths = torch.full(
            (batch_size,),
            self.max_seq_len - self.num_tokens_per_bs,
            dtype=torch.int32,
        ).pin_memory()
        sequence_lengths = torch.empty(0, dtype=torch.int32).pin_memory()

        cu_seqlens = torch.zeros(batch_size + 1, dtype=torch.int32).pin_memory()
        cu_seqlens[1:] = input_lengths.cumsum(0)
        cu_kv_seqlens = torch.zeros(
            batch_size + 1, dtype=torch.int32
        ).pin_memory()
        cu_kv_seqlens[1:] = (input_lengths + prefix_lengths).cumsum(0)
        decode_cu_seqlens = torch.arange(
            batch_size + 1, dtype=torch.int32
        ).pin_memory()

        sp_steps = self.num_tokens_per_bs - 1
        physical_blocks = (
            (self.max_seq_len + self.tokens_per_block - 1)
            // self.tokens_per_block
        ) + sp_steps
        kernel_blocks = (
            physical_blocks
            * self.tokens_per_block
            // self.kernel_tokens_per_block
        )
        block_ids = torch.arange(
            1, batch_size * kernel_blocks + 1, dtype=torch.int32
        ).reshape(batch_size, kernel_blocks)
        block_ids = block_ids.pin_memory()
        block_ids_device = block_ids.to(self.device, non_blocking=True)

        attention_inputs.is_prefill = True
        attention_inputs.is_target_verify = True
        attention_inputs.is_s_padded = True
        attention_inputs.dtype = get_typemeta(
            torch.zeros(1, dtype=self.compute_dtype)
        )
        attention_inputs.input_lengths = input_lengths
        attention_inputs.prefix_lengths = prefix_lengths
        attention_inputs.sequence_lengths = sequence_lengths
        attention_inputs.sequence_lengths_plus_1_device = (
            prefix_lengths + 1
        ).to(self.device, non_blocking=True)
        attention_inputs.cu_seqlens = cu_seqlens
        attention_inputs.cu_seqlens_device = cu_seqlens.to(
            self.device, non_blocking=True
        )
        attention_inputs.cu_kv_seqlens_device = cu_kv_seqlens.to(
            self.device, non_blocking=True
        )
        attention_inputs.decode_cu_seqlens = decode_cu_seqlens
        attention_inputs.decode_cu_seqlens_device = decode_cu_seqlens.to(
            self.device, non_blocking=True
        )
        attention_inputs.kv_cache_kernel_block_id = block_ids
        attention_inputs.kv_cache_kernel_block_id_device = block_ids_device
        attention_inputs.kv_cache_block_id = block_ids
        attention_inputs.kv_cache_block_id_device = block_ids_device
        attention_inputs.padding_offset = torch.zeros(
            token_count, dtype=torch.int32, device=self.device
        )
        attention_inputs.context_total_kv_length = int(cu_kv_seqlens[-1].item())
        attention_inputs.total_tokens = token_count

        inputs.attention_inputs = attention_inputs
        return inputs

    def _assert_cache_matches_reference(self) -> None:
        for layer_id, (graph_tensor, reference_tensor) in enumerate(
            zip(
                self.graph_cache.kv_cache_base_by_layer,
                self.reference_cache.kv_cache_base_by_layer,
            )
        ):
            # BlockPool reserves physical block 0 for internal/dummy use. A padded
            # graph replay may write its fixed-launch tail there; every allocatable
            # cache block must still match the non-Graph reference exactly.
            torch.testing.assert_close(
                graph_tensor[1:],
                reference_tensor[1:],
                rtol=1e-2,
                atol=1e-2,
                msg=lambda msg: f"KV cache mismatch at layer {layer_id}: {msg}",
            )

    def _run_and_compare(self, batch_size: int, expected_graph_size: int) -> None:
        graph_inputs = self._build_inputs(batch_size)
        reference_inputs = self._build_inputs(batch_size)

        self.assertTrue(self.graph_runner.canRun(graph_inputs))
        graph_outputs = self.graph_runner.forward(graph_inputs)
        torch.cuda.synchronize()
        reference_outputs = self.reference_model.forward(reference_inputs)
        torch.cuda.synchronize()

        self.assertEqual(
            self.graph_runner.getCurrentRealGraphSize(), expected_graph_size
        )
        torch.testing.assert_close(
            graph_outputs.hidden_states,
            reference_outputs.hidden_states.to(graph_outputs.hidden_states.dtype),
            rtol=1e-2,
            atol=1e-2,
        )
        self._assert_cache_matches_reference()

    def test_smaller_target_verify_batch_reuses_larger_graph(self) -> None:
        # Populate every row of the batch-8 graph, then replay batch 3 through
        # the same graph key. The second replay exercises all target-verify
        # metadata tails and the seq_len_sum cumulative-length boundary.
        self._run_and_compare(batch_size=8, expected_graph_size=8)
        self._run_and_compare(batch_size=3, expected_graph_size=8)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    unittest.main()
