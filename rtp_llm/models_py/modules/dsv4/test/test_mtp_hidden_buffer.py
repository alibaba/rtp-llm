import types
import unittest

import torch

from rtp_llm.models_py.model_desc.deepseek_v4_model import DeepSeekV4Model
from rtp_llm.models_py.modules.dsv4.transformer import V4Transformer


class MtpHiddenBufferTest(unittest.TestCase):
    def test_accessor_slices_requested_rows(self) -> None:
        v4 = types.SimpleNamespace()
        v4._mtp_hidden_buffer = torch.empty(5, 3, dtype=torch.bfloat16)

        flat = torch.arange(9, dtype=torch.bfloat16).reshape(3, 3)
        V4Transformer._write_mtp_hidden_buffer(v4, flat, is_cuda_graph=False)

        model = types.SimpleNamespace(v4=v4)
        sliced = DeepSeekV4Model.get_mtp_target_hidden_states(model, 2)
        self.assertTrue(torch.equal(flat[:2], sliced))

    def test_accessor_rejects_requests_beyond_buffer_capacity(self) -> None:
        v4 = types.SimpleNamespace()
        v4._mtp_hidden_buffer = torch.empty(5, 3, dtype=torch.bfloat16)
        model = types.SimpleNamespace(v4=v4)

        with self.assertRaisesRegex(RuntimeError, "requested=6, capacity=5"):
            DeepSeekV4Model.get_mtp_target_hidden_states(model, 6)


if __name__ == "__main__":
    unittest.main()
