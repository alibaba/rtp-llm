import math
import os
import sys
import unittest

import torch

from rtp_llm.async_decoder_engine.base_engine import BaseEngine
from rtp_llm.cpp.devices.cuda_impl.tests.libtest_cuda_graph_decode_ops import (
    CudaGraphDecodePaddingOp,
)

# from rtp_llm.ops.libth_transformer import PyModelInputs
from rtp_llm.models_py.model_desc.module_base import (  # load libth_transformer by default
    GptModelBase,
)
from rtp_llm.test.model_test.test_util.fake_model_loader import FakeModelLoader
from rtp_llm.utils.weight_type import WEIGHT_TYPE


class TestCudaGraphDecodePadding(unittest.TestCase):
    def build_model(self) -> GptModelBase:
        loader = FakeModelLoader(
            model_type="qwen_2",
            tokenizer_path="/mnt/nas1/hf/Qwen2.5-0.5B-Instruct",
            ckpt_path="/mnt/nas1/hf/Qwen2.5-0.5B-Instruct",
            load_py_model=True,
            data_type=WEIGHT_TYPE.FP16.to_str(),
            device_reserve_memory_bytes=-536870912,
        )
        engine: BaseEngine = loader.init_engine()
        self.engines.append(engine)
        assert engine.model.py_model is not None
        return engine.model.py_model

    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)
        sys.path.append(os.environ["TEST_SRCDIR"] + "/rtp_llm/rtp_llm/cpp/models/test")
        self.engines = []
        os.environ["RESERVER_RUNTIME_MEM_MB"] = "10240"
        model = self.build_model()
        print("build model successfully")
        self.op = CudaGraphDecodePaddingOp()
        self.op.init(model)
        self.normal_model = model

    def _test_single(self, batch_size: int):
        max_seq_len = 64
        num_tokens_per_bs = 1
        seq_size_per_block = 64
        inputs = self.op.buildInputs(
            batch_size, max_seq_len, num_tokens_per_bs, seq_size_per_block
        )
        inputs2 = self.op.buildInputs(
            batch_size, max_seq_len, num_tokens_per_bs, seq_size_per_block
        )
        outputs1 = self.op.forward(inputs)
        outputs2 = self.normal_model.forward(inputs2)
        current_real_graph_size = self.op.getCurrentRealGraphSize()
        print(
            f"current_real_graph_size: {current_real_graph_size}, batch_size: {batch_size}"
        )
        assert (
            current_real_graph_size == batch_size
            if batch_size % 2 == 1
            else batch_size + 1
        ) or current_real_graph_size == (int(math.ceil(batch_size / 16)) * 16)
        print(f"outputs1.hidden_states: {outputs1.hidden_states[0]}")
        print(f"outputs2.hidden_states: {outputs2.hidden_states[0]}")
        outputs2.hidden_states = outputs2.hidden_states.type(
            outputs1.hidden_states.dtype
        )
        torch.testing.assert_close(
            outputs1.hidden_states[:batch_size], outputs2.hidden_states
        )
        # assert((hidden_states1 == hidden_states2))

    def test_bacth_decode(self):
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
            80,
            87,
            96,
            102,
            112,
            125,
            128,
        ]
        for bs in batch_range:
            self._test_single(bs)
            print(f"success for batch size: {bs}")


if __name__ == "__main__":
    unittest.main()
