import logging
import os
import unittest
from typing import List

import torch

import rtp_llm.models
from rtp_llm.cpp.devices.cuda_impl.tests.cuda_graph_test_utils import (
    CudaGraphTestModelBuilder,
    ModelBuildConfig,
)
from rtp_llm.cpp.devices.cuda_impl.tests.libtest_cuda_graph_runner import (
    CudaGraphRunner,
)
from rtp_llm.models_py.model_desc.module_base import GptModelBase
from rtp_llm.ops.compute_ops import PyAttentionInputs, PyModelInputs, get_typemeta


class TestCudaGraphDecodePadding(unittest.TestCase):
    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)
        os.environ["RESERVER_RUNTIME_MEM_MB"] = "10240"

        # Test parameters (can be configured)
        self.max_seq_len = 64
        self.tokens_per_block = 64
        self.max_batch_size = 64
        self.device = "cuda:0"

        # Generate decode_capture_batch_sizes from 1 to max_batch_size, excluding some values
        excluded_batch_sizes = {3, 4, 6, 10, 15, 19, 27}
        self.decode_capture_batch_sizes = [
            bs
            for bs in range(1, self.max_batch_size + 1)
            if bs not in excluded_batch_sizes
        ]

        # Build model using shared model builder
        self.model_builder = CudaGraphTestModelBuilder(
            ModelBuildConfig(
                model_path="/mnt/nas1/hf/Qwen2.5-0.5B-Instruct",
                tokens_per_block=self.tokens_per_block,
                device=self.device,
            )
        )
        build_result = self.model_builder.build_model(init_kv_cache=True)
        model = build_result.model
        self.model_config = build_result.model_config
        self.compute_dtype = build_result.compute_dtype
        self.layer_num = build_result.layer_num
        self.block_nums = build_result.block_nums
        self.kv_cache = build_result.kv_cache
        print("build model successfully")

        # hidden_size from engine build path (ModelBuildResult.hidden_size from model_config)
        hidden_size = build_result.hidden_size
        assert (
            hidden_size > 0
        ), "hidden_size must be set for CudaGraphRunner decode (from model_config in engine build path)"
        self.hidden_size = hidden_size

        self.op = CudaGraphRunner()
        self.op.init_decode(
            model,
            hidden_size,
            self.max_seq_len,
            self.tokens_per_block,
            self.decode_capture_batch_sizes,
        )
        print(f"CUDA Graph initialized with batch sizes: 1 to {self.max_batch_size}")

        # Build a second model for comparison (normal forward)
        normal_build_result = self.model_builder.build_model(init_kv_cache=True)
        self.normal_model = normal_build_result.model

    def build_inputs(self, batch_size: int, max_seq_len: int, seq_size_per_block: int):
        """Build inputs in Python, similar to CudaGraphPrefill.py"""
        num_tokens_per_bs = 1  # decode mode: 1 token per batch
        max_num_token = batch_size * num_tokens_per_bs

        inputs = PyModelInputs()
        attention_inputs = PyAttentionInputs()

        # input_ids [tokens_nums] = [batch_size * num_tokens_per_bs]
        inputs.input_ids = torch.arange(max_num_token, dtype=torch.int32, device="cuda")

        # input_hiddens [max_num_token, hidden_size] (required by CudaGraphRunner.prepareInputs decode path)
        inputs.input_hiddens = torch.zeros(
            (max_num_token, self.hidden_size),
            dtype=self.compute_dtype,
            device="cuda",
        )

        # prefix_lengths [batch_size, int32] (for attention `prepare`)
        attention_inputs.prefix_lengths = torch.empty(0)

        # input_lengths [batch_size, int32] (decode only)
        attention_inputs.input_lengths = torch.ones(batch_size, dtype=torch.int32)

        # sequence_lengths [batch_size, int32] (decode only), with pin_memory
        attention_inputs.sequence_lengths = torch.ones(
            batch_size, dtype=torch.int32
        ).pin_memory()

        # sequence_lengths_plus_1_d [batch_size] int32 on cuda (decode: 1 token per batch -> 2)
        attention_inputs.sequence_lengths_plus_1_d = torch.full(
            (batch_size,), 2, dtype=torch.int32, device="cuda"
        )

        # decode_cu_seqlens_d [batch_size + 1] int32 on cuda, cumulative [0, 1, ..., batch_size]
        attention_inputs.decode_cu_seqlens_d = torch.arange(
            batch_size + 1, dtype=torch.int32, device="cuda"
        )

        # kv_cache_block_id_device [batch_size, block_num], ids from 1 to batch_size*block_num
        block_num = (max_seq_len + seq_size_per_block - 1) // seq_size_per_block
        num_blocks = batch_size * block_num
        block_ids = torch.arange(
            1, num_blocks + 1, dtype=torch.int32, device="cuda"
        ).view(batch_size, block_num)
        attention_inputs.kv_cache_block_id_device = block_ids
        attention_inputs.kv_cache_block_id_host = block_ids.cpu()

        # padding_offset
        attention_inputs.padding_offset = torch.zeros(
            max_seq_len, dtype=torch.int32, device="cuda"
        )

        # Set attention parameters
        attention_inputs.is_prefill = False
        attention_inputs.dtype = get_typemeta(torch.zeros(1, dtype=torch.float16))

        # cu_seqlens
        cu_len = batch_size + 1
        cu_seqlens = torch.zeros(cu_len, dtype=torch.int32, device="cpu").pin_memory()

        attention_inputs.cu_seqlens = cu_seqlens
        attention_inputs.cu_kv_seqlens = cu_seqlens.clone()
        # For decode mode: each batch has 1 token, so total = batch_size
        attention_inputs.context_total_kv_length = batch_size * num_tokens_per_bs
        attention_inputs.total_tokens = batch_size * num_tokens_per_bs
        inputs.attention_inputs = attention_inputs
        return inputs

    def _test_single(self, batch_size: int):
        inputs = self.build_inputs(
            batch_size,
            self.max_seq_len,
            self.tokens_per_block,
        )
        inputs2 = self.build_inputs(
            batch_size,
            self.max_seq_len,
            self.tokens_per_block,
        )

        # Ensure inputs can run with CUDA graph before calling forward
        can_run = self.op.canRun(inputs)
        assert can_run, (
            "Expected canRun(inputs) to be True so that forward uses CUDA graph; "
            "check inputs (e.g. batch_size vs capture range) if this fails."
        )

        outputs1 = self.op.forward(inputs)
        torch.cuda.synchronize()
        outputs2 = self.normal_model.forward(inputs2)
        torch.cuda.synchronize()

        current_real_graph_size = self.op.getCurrentRealGraphSize()
        print(
            f"current_real_graph_size: {current_real_graph_size}, batch_size: {batch_size}"
        )

        # With continuous capture from 1 to max_batch_size, real graph size should equal batch_size
        assert (
            current_real_graph_size >= batch_size
        ), f"Expected real graph size {batch_size}, got {current_real_graph_size}"

        print(f"outputs1.hidden_states: {outputs1.hidden_states}")
        print(f"outputs2.hidden_states: {outputs2.hidden_states}")

        outputs2.hidden_states = outputs2.hidden_states.type(
            outputs1.hidden_states.dtype
        )
        close_mask = torch.isclose(
            outputs1.hidden_states[:batch_size],
            outputs2.hidden_states,
            rtol=1e-2,
            atol=1e-2,
        )
        pass_ratio = close_mask.float().mean().item()
        assert (
            pass_ratio >= 0.999
        ), f"Only {pass_ratio*100:.2f}% elements pass, expected >= 99.9%"

    def test_batch_decode(self):
        batch_range = [
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
            19,
            27,
            48,
            64,
        ]

        for bs in batch_range:
            self._test_single(bs)
            print(f"success for batch size: {bs}")


if __name__ == "__main__":
    logging.basicConfig(level=print)
    unittest.main()
