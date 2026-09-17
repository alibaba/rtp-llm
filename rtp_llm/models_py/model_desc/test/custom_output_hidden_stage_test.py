import unittest

import torch

from rtp_llm.models_py.model_desc.qwen3 import Qwen3Model
from rtp_llm.ops.compute_ops import PyModelInputs


class AddResidual(torch.nn.Module):
    def forward(self, hidden, fmha_impl, kv_cache=None):
        return hidden + 2


class CustomOutputHiddenStageTest(unittest.TestCase):
    def test_qwen3_captures_selected_rows_without_changing_generation(self):
        model = Qwen3Model.__new__(Qwen3Model)
        torch.nn.Module.__init__(model)
        model.embed_tokens = torch.nn.Embedding(8, 4)
        model.layers = torch.nn.ModuleList([AddResidual()])
        model.layer_num = 1
        model.kv_cache = None
        model.norm = torch.nn.RMSNorm(4, eps=1e-6)
        inputs = PyModelInputs()
        inputs.input_ids = torch.tensor([1, 2, 3, 4, 5])
        with torch.no_grad():
            baseline = model.forward(inputs, fmha_impl=object())
            before = model.embed_tokens(inputs.input_ids) + 2
            inputs.pre_final_norm_output_indexes = torch.tensor([4, 1])
            output = model.forward(inputs, fmha_impl=object())
            selected = output.pre_final_norm_hidden_states
            torch.testing.assert_close(selected, before[[4, 1]], rtol=0, atol=0)
            torch.testing.assert_close(
                output.hidden_states, baseline.hidden_states, rtol=0, atol=0
            )
            self.assertEqual(
                selected.untyped_storage().nbytes(), 8 * selected.element_size()
            )
            next_inputs = PyModelInputs()
            next_inputs.input_ids = inputs.input_ids
            self.assertIsNone(
                model.forward(
                    next_inputs, fmha_impl=object()
                ).pre_final_norm_hidden_states
            )


if __name__ == "__main__":
    unittest.main()
