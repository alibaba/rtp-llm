import math
import os
import sys
import unittest

import torch
from numpy import append

from rtp_llm.async_decoder_engine.base_engine import BaseEngine
from rtp_llm.cpp.devices.cuda_impl.tests.libtest_cuda_graph_prefill_ops import (
    CudaGraphPrefillOp,
)

# from rtp_llm.ops.libth_transformer import PyModelInputs
from rtp_llm.models_py.model_desc.module_base import (  # load libth_transformer by default
    GptModelBase,
)
from rtp_llm.models_py.model_desc.qwen3 import Qwen3Model
from rtp_llm.test.model_test.test_util.fake_model_loader import FakeModelLoader


class TestCudaGraphPrefill(unittest.TestCase):
    def build_model(self) -> GptModelBase:
        loader = FakeModelLoader(
            model_type="qwen_2",
            tokenizer_path="/mnt/nas1/hf/gte-Qwen2-7B-instruct",
            ckpt_path="/mnt/nas1/hf/gte-Qwen2-7B-instruct",
            load_py_model=True,
            device_reserve_memory_bytes=-536870912,
            data_type="fp16",
            is_causal=False,
        )
        engine: BaseEngine = loader.init_engine()
        assert engine.model.py_model is not None
        return engine.model.py_model

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

    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)
        sys.path.append(os.environ["TEST_SRCDIR"] + "/rtp_llm/rtp_llm/cpp/models/test")
        self.normal_model = self.build_model()
        model = self.build_model()
        print("model class type: ", type(model))
        self.op = CudaGraphPrefillOp()
        self.op.init(model)
        self.model = model

    def _test_single(self, batch_size: int):
        max_seq_len = 64
        num_tokens_per_bs = max_seq_len
        seq_size_per_block = 64
        inputs1 = self.op.buildInputs(
            batch_size, max_seq_len, num_tokens_per_bs, seq_size_per_block, False
        )

        self.op.setCufmhaPadded(False)
        outputs1 = self.normal_model.forward(inputs1)
        print(f"outputs1 success for batch size {batch_size}")
        inputs2 = self.op.buildInputs(
            batch_size, max_seq_len, num_tokens_per_bs, seq_size_per_block, True
        )
        self.op.setCufmhaPadded(True)
        outputs2 = self.normal_model.forward(inputs2)
        print(f"outputs2 success for batch size {batch_size}")
        print(f"inputs1: {inputs1.input_ids}")
        print(f"inputs2: {inputs2.input_ids}")
        print(
            f"outputs1.shape: {outputs1.hidden_states.shape}, outputs2.shape: {outputs2.hidden_states.shape}"
        )
        # self.check_pos(outputs1.hidden_states, outputs2.hidden_states)
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

        torch.testing.assert_close(
            outputs1.hidden_states,
            valid_outputs2_tensor,
            rtol=1e-2,  # 相对误差容忍度
            atol=1e-2,  # 绝对误差容忍度
        )

        print(f"trt padded mode success for batch: {batch_size}!!")

        inputs3 = self.op.buildInputs(
            batch_size, max_seq_len, num_tokens_per_bs, seq_size_per_block, False
        )

        outputs3 = self.op.forward(inputs3)
        current_real_graph_size = self.op.getCurrentRealGraphSize()
        print(
            f"current_real_graph_size: {current_real_graph_size}, batch_size: {batch_size}"
        )
        # assert (
        #     current_real_graph_size == batch_size
        #     if batch_size % 2 == 1
        #     else batch_size + 1
        # ) or current_real_graph_size == (int(math.ceil(batch_size / 16)) * 16)
        print(f"outputs1.hidden_states: {outputs1.hidden_states}")
        print(f"outputs3.hidden_states: {outputs3.hidden_states}")
        # slice_index: int = (batch_size + 1) * batch_size * 5
        torch.testing.assert_close(
            outputs1.hidden_states,
            outputs3.hidden_states,
            rtol=1e-2,  # 相对误差容忍度
            atol=1e-2,  # 绝对误差容忍度
        )

    def test_batch_prefill(self):
        # Use fewer prefill tests, otherwise the test case will timeout (capture seqlen cost mainly).
        batch_range = [
            1,
            7,
            8,
            15,
        ]
        for bs in batch_range:
            self._test_single(bs)
            print(f"success for batch size: {bs}")


if __name__ == "__main__":
    unittest.main()
