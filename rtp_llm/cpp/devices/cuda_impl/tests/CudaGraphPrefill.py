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
from rtp_llm.cpp.devices.cuda_impl.tests.libtest_cuda_graph_prefill_ops import (
    CudaGraphPrefillOp,
)
from rtp_llm.models_py.model_desc.module_base import GptModelBase
from rtp_llm.ops.compute_ops import PyAttentionInputs, PyModelInputs, get_typemeta


class TestCudaGraphPrefill(unittest.TestCase):
    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)
        os.environ["RESERVER_RUNTIME_MEM_MB"] = "10240"

        # Set device first to ensure all tensors are created on the correct device
        self.device = "cuda:0"
        torch.cuda.set_device(self.device)

        # Test parameters (can be configured)
        self.max_seq_len = 64
        self.tokens_per_block = 64
        self.max_context_batch_size = 64
        self.max_prefill_cuda_graph_len = 960

        # Generate prefill_capture_seq_lens
        self.prefill_capture_seq_lens = self._generate_prefill_capture_seq_lens()

        # Build model using shared model builder
        self.model_builder = CudaGraphTestModelBuilder(
            ModelBuildConfig(
                model_path="/mnt/nas1/hf/gte-Qwen2-7B-instruct/",
                tokens_per_block=self.tokens_per_block,
                device=self.device,
                act_type="BF16",
                device_reserve_memory_bytes=-4294967296,
            )
        )
        build_result = self.model_builder.build_model(init_kv_cache=False)
        model = build_result.model
        self.model_config = build_result.model_config
        self.compute_dtype = build_result.compute_dtype
        print("build model successfully")

        self.op = CudaGraphPrefillOp()
        self.op.init(
            model,
            self.max_context_batch_size,
            self.max_seq_len,
            self.tokens_per_block,
            self.max_prefill_cuda_graph_len,
            self.prefill_capture_seq_lens,
        )
        torch.cuda.synchronize()
        print(
            f"CudaGraphPrefillOp initialized with max_prefill_cuda_graph_len={self.max_prefill_cuda_graph_len}"
        )

        self.normal_model = model

    def _generate_prefill_capture_seq_lens(self) -> list:
        """Generate prefill capture sequence lengths"""
        # Default sequence lengths for prefill capture
        seq_lens = [
            1,
            3,
            6,
            30,
            60,
            100,
            125,
            128,
            466,
            448,
            512,
            786,
        ]
        return seq_lens

    def _calculate_padding_offset(
        self, input_lengths: torch.Tensor, cu_seqlens: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculate padding offset for FusedRopeKVCache kernel.

        inputs_length:  [1,2,1,1], total_tokens = 5
        padding_offsets: [0,1,1,1,2]
        """
        batch_size = input_lengths.size(0)
        total_tokens = cu_seqlens[batch_size].item()
        max_seq_len = input_lengths.max().item()

        padding_offset = torch.zeros(total_tokens, dtype=torch.int32, device="cpu")
        cum_offset = 0
        index = 0
        for i in range(batch_size):
            seq_len = input_lengths[i].item()
            for j in range(seq_len):
                padding_offset[index] = cum_offset
                index += 1
            cum_offset += max_seq_len - seq_len

        return padding_offset.cuda()

    def build_inputs(
        self,
        batch_size: int,
        max_seq_len: int,
        seq_size_per_block: int,
        use_max_padded_mode: bool = False,
    ) -> PyModelInputs:
        """
        Build model inputs for prefill test.

        This is the Python implementation of the C++ buildInputs function.
        """
        inputs = PyModelInputs()
        attention_inputs = PyAttentionInputs()
        attention_inputs.is_prefill = True

        # Create input_lengths: 10, 20, 30, ..., 10 * batch_size (capped at max_seq_len)
        input_lengths_data: List[int] = []
        total_tokens: int = 0
        for i in range(batch_size):
            if use_max_padded_mode:
                # When using max_padded_mode, all sequences are padded to max_seq_len
                input_lengths_data.append(max_seq_len)
            else:
                # Otherwise use incremental lengths: 10, 20, 30, ..., 10 * batch_size
                input_lengths_data.append(min(max_seq_len, 10 * (i + 1)))
            total_tokens += input_lengths_data[i]

        # input_ids [total_tokens]
        total_tokens = int(total_tokens)  # Ensure total_tokens is int
        if use_max_padded_mode:
            # When using max_padded_mode, input_ids need to be generated in padded manner
            input_ids = torch.ones(total_tokens, dtype=torch.int32, device="cuda")
            current_pos = 0
            current_id = 1
            for i in range(batch_size):
                # First 10*(i+1) positions of each batch contain meaningful data
                actual_length = min(max_seq_len, 10 * (i + 1))
                for j in range(actual_length):
                    input_ids[current_pos + j] = (current_id % 10) + 10
                    current_id += 1
                # Remaining positions stay as 1 (padding)
                current_pos += max_seq_len
        else:
            # Otherwise use continuous incrementing method
            input_ids = torch.zeros(total_tokens, dtype=torch.int32, device="cuda")
            for i in range(total_tokens):
                input_ids[i] = (i + 1) % 10 + 10

        inputs.input_ids = input_ids
        # input_lengths [batch_size, int32]
        attention_inputs.input_lengths = torch.tensor(
            input_lengths_data, dtype=torch.int32, device="cpu"
        )

        # sequence_lengths [batch_size, int32] - same as input_lengths, with pin_memory
        # attention_inputs.sequence_lengths = (
        #     attention_inputs.input_lengths.clone().pin_memory()
        # )

        # kv_cache_block_id_device [batch_size, block_num]
        # need_block_nums = (max_seq_len + seq_size_per_block - 1) // seq_size_per_block
        # attention_inputs.kv_cache_block_id_device = torch.zeros(
        #     batch_size, need_block_nums, dtype=torch.int32, device="cuda"
        # )
        # attention_inputs.kv_cache_block_id_host = torch.zeros(
        #     batch_size, need_block_nums, dtype=torch.int32, device="cpu"
        # )

        # prefix_lengths [batch_size, int32]
        attention_inputs.prefix_lengths = torch.zeros(
            batch_size, dtype=torch.int32, device="cpu"
        ).pin_memory()

        attention_inputs.is_prefill = True
        attention_inputs.dtype = get_typemeta(torch.zeros(1, dtype=torch.bfloat16))
        attention_inputs.is_s_padded = use_max_padded_mode

        cu_len = batch_size + 1
        cu_seqlens = torch.zeros(cu_len, dtype=torch.int32, device="cuda")

        total_seq_len = 0
        total_seq_len_without_prefix = 0
        for i in range(batch_size):
            cu_seqlens[i] = total_seq_len
            if use_max_padded_mode:
                # When using max_padded_mode, cu_seqlens records actual effective length
                # i.e., 10*(i+1), not padded max_seq_len
                actual_length = min(max_seq_len, 10 * (i + 1))
                total_seq_len += actual_length
                total_seq_len_without_prefix += actual_length
            else:
                total_seq_len += input_lengths_data[i]
                total_seq_len_without_prefix += input_lengths_data[i]

        cu_seqlens[batch_size] = total_seq_len
        attention_inputs.cu_seqlens = cu_seqlens
        attention_inputs.cu_kv_seqlens = cu_seqlens.clone()
        attention_inputs.context_total_kv_length = total_seq_len
        attention_inputs.total_tokens = total_seq_len
        # Calculate padding_offset if not using max_padded_mode
        if not use_max_padded_mode:
            attention_inputs.padding_offset = self._calculate_padding_offset(
                attention_inputs.input_lengths, cu_seqlens
            )

        inputs.attention_inputs = attention_inputs
        return inputs

    def check_pos(self, outputs1: torch.Tensor, outputs2: torch.Tensor):
        # 精确匹配版本
        search_values = outputs1[10]
        target_tensor = outputs2[64:]
        tolerance = 1e-3
        print("Exact matching:")
        for i, value in enumerate(search_values):
            matches = torch.abs(target_tensor - value) < tolerance
            if len(search_values.shape) > 0:
                full_matches = torch.all(matches, dim=1)
            else:
                full_matches = matches

            positions = torch.nonzero(full_matches).squeeze()

            if len(positions) == 0:  # 或者 positions.numel() == 0
                print(f"Value {i}: {value:.6f} -> NOT FOUND")
            else:
                print(f"Value {i}: {value:.6f} -> found at positions: {positions}")

    def _test_single(self, batch_size: int):
        max_seq_len = self.max_seq_len
        seq_size_per_block = self.tokens_per_block
        # # Use Python build_inputs instead of C++ op.buildInputs
        inputs1 = self.build_inputs(batch_size, max_seq_len, seq_size_per_block, False)

        outputs1 = self.normal_model.forward(inputs1)
        torch.cuda.synchronize()
        print(f"outputs1 success for batch size {batch_size}")

        inputs2 = self.build_inputs(batch_size, max_seq_len, seq_size_per_block, True)
        outputs2 = self.normal_model.forward(inputs2)
        torch.cuda.synchronize()
        print(f"outputs2 success for batch size {batch_size}")
        print(f"inputs1: {inputs1.input_ids}")
        print(f"inputs2: {inputs2.input_ids}")
        print(
            f"outputs1.shape: {outputs1.hidden_states.shape}, outputs2.shape: {outputs2.hidden_states.shape}"
        )

        # 从 padded 的 outputs2 中提取有效位置来与 outputs1 比较
        # 根据 cu_seqlens 来提取每个 batch 的有效输出
        cu_seqlens = inputs2.attention_inputs.cu_seqlens.cpu().numpy()
        print(f"cu_seqlens: {cu_seqlens}")

        # 在 padded 模式下，每个 batch 都有 max_seq_len 个输出，但只有前面的部分是有效的
        # 需要根据实际的有效长度来提取
        valid_outputs2 = []
        for i in range(batch_size):
            # 每个 batch 在 outputs2 中的起始位置
            batch_start = i * max_seq_len
            # 实际的有效长度（10, 20, 30, ...）
            actual_length = min(max_seq_len, 10 * (i + 1))
            # 提取有效部分，跳过 padding
            valid_outputs2.append(
                outputs2.hidden_states[batch_start : batch_start + actual_length]
            )

        # 拼接所有有效输出
        valid_outputs2_tensor = torch.cat(valid_outputs2, dim=0)
        print(f"valid_outputs2.shape: {valid_outputs2_tensor.shape}")
        ## batch invariance: https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/
        # 允许最多 0.1% 的元素不符合精度要求
        close_mask = torch.isclose(
            outputs1.hidden_states, valid_outputs2_tensor, rtol=1e-2, atol=1e-2
        )
        print(f"outputs1: {outputs1}")
        print(f"valid_outputs2_tensor: {valid_outputs2_tensor}")
        pass_ratio = close_mask.float().mean().item()
        assert (
            pass_ratio >= 0.999
        ), f"Only {pass_ratio*100:.2f}% elements pass, expected >= 99.9%"

        print(f"trt padded mode success for batch: {batch_size}!!")

        # Use Python build_inputs instead of C++ op.buildInputs
        inputs3 = self.build_inputs(batch_size, max_seq_len, seq_size_per_block, False)

        outputs3 = self.op.forward(inputs3)

        current_real_graph_size = self.op.getCurrentRealGraphSize()
        print(
            f"current_real_graph_size: {current_real_graph_size}, batch_size: {batch_size}"
        )
        # Explicit CUDA sync to ensure all async operations are complete before accessing tensor data
        torch.cuda.synchronize()

        print(f"outputs1.hidden_states: {outputs1.hidden_states}")
        print(f"outputs3.hidden_states: {outputs3.hidden_states}")

        ## batch invariance: https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/
        # 允许最多 0.001% 的元素不符合精度要求
        close_mask = torch.isclose(
            outputs1.hidden_states, outputs3.hidden_states, rtol=1e-2, atol=1e-2
        )
        pass_ratio = close_mask.float().mean().item()
        assert (
            pass_ratio >= 0.9999
        ), f"Only {pass_ratio*100:.2f}% elements pass, expected >= 99.99%"

    def test_batch_prefill(self):
        # Use fewer prefill tests, otherwise the test case will timeout (capture seqlen cost mainly).
        batch_range = [1, 2, 5, 7, 8, 9]
        for bs in batch_range:
            print(f"start test for batch size: {bs}")
            self._test_single(bs)
            print(f"success for batch size: {bs}")


if __name__ == "__main__":
    logging.basicConfig(level=print)
    unittest.main()
