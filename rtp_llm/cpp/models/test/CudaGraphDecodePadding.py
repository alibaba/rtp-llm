import os
import sys
import unittest

import torch
from numpy import append

from rtp_llm.cpp.models.test.libtest_cuda_graph_ops import CudaGraphDecodePaddingOp

# from rtp_llm.ops.libth_transformer import PyModelInputs
from rtp_llm.models_py.model_desc.module_base import (  # load libth_transformer by default
    GptModelBase,
)
from rtp_llm.models_py.model_desc.qwen3 import Qwen3Model
from rtp_llm.test.model_test.test_util.fake_model_loader import FakeModelLoader


class TestCudaGraphDecodePadding(unittest.TestCase):
    def build_model(self) -> GptModelBase:
        loader = FakeModelLoader(
            model_type="qwen_2",
            tokenizer_path="/mnt/nas1/hf/Qwen2.5-0.5B-Instruct",
            ckpt_path="/mnt/nas1/hf/Qwen2.5-0.5B-Instruct",
            load_py_model=True,
            device_reserve_memory_bytes=-536870912,
        )
        async_model = loader.load_model()
        assert async_model.model.py_model is not None
        return async_model.model.py_model

    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)
        sys.path.append(os.environ["TEST_SRCDIR"] + "/rtp_llm/rtp_llm/cpp/models/test")
        model = self.build_model()
        self.op = CudaGraphDecodePaddingOp()
        self.op.init(model)
        self.model = model

    def _test_single(self, batch_size: int):
        max_seq_len = 64
        num_tokens_per_bs = 1
        seq_size_per_block = 64
        inputs = self.op.build_inputs(
            batch_size, max_seq_len, num_tokens_per_bs, seq_size_per_block
        )
        outputs1 = self.op.forward(inputs)
        outputs2 = self.model.forward(inputs)
        current_real_graph_size = self.op.get_current_real_graph_size()
        assert (
            current_real_graph_size == batch_size
            if batch_size % 2 == 0
            else batch_size + 1
        )
        print(f"outputs1.hidden_states: {outputs1.hidden_states}")
        print(f"outputs2.hidden_states: {outputs2.hidden_states}")
        torch.testing.assert_close(outputs1.hidden_states, outputs2.hidden_states)
        # assert((hidden_states1 == hidden_states2))

    def test_bacth_decode(self):
        # batch_range = append(range(0,33) ,[37, 47, 48, 55, 64, 76, 80, 87, 96, 102, 112, 125, 128])
        # batch_range = [1, 2, 3, 4, 5, 6, 7, 8]
        batch_range = [1]
        for bs in batch_range:
            self._test_single(bs)
            print(f"success for batch size: {bs}")


if __name__ == "__main__":
    unittest.main()
