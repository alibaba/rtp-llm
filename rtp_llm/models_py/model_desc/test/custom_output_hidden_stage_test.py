import unittest
from unittest import mock

import torch

from rtp_llm.models_py.model_desc.module_base import GptModelBase
from rtp_llm.models_py.model_desc.qwen3 import Qwen3Model
from rtp_llm.ops.compute_ops import PyModelInputs


class AddResidual(torch.nn.Module):
    def forward(self, hidden, fmha_impl, kv_cache=None):
        return hidden + 2


class InplaceNorm(torch.nn.Module):
    def forward(self, hidden):
        return hidden.mul_(0.25)


def make_model(dtype=torch.float32):
    # Exercise the real forward/selection path without attention kernels or a
    # checkpoint. The norm and the score head use actual torch tensor math.
    model = Qwen3Model.__new__(Qwen3Model)
    torch.nn.Module.__init__(model)
    model.embed_tokens = torch.nn.Embedding(8, 4, dtype=dtype)
    with torch.no_grad():
        model.embed_tokens.weight.copy_(torch.arange(32).reshape(8, 4))
    model.layers = torch.nn.ModuleList([AddResidual()])
    model.layer_num = 1
    model.kv_cache = None
    model.norm = torch.nn.RMSNorm(4, eps=1e-6, dtype=dtype)
    return model.eval()


def make_inputs(indexes=None):
    inputs = PyModelInputs()
    inputs.input_ids = torch.tensor([1, 2, 3, 4, 5])
    if indexes is not None:
        inputs.pre_final_norm_output_indexes = torch.tensor(indexes, dtype=torch.long)
    return inputs


class CustomOutputHiddenStageTest(unittest.TestCase):
    def test_models_must_explicitly_opt_in(self):
        self.assertFalse(GptModelBase.supports_pre_final_norm)
        self.assertTrue(Qwen3Model.supports_pre_final_norm)

    def test_default_has_no_extra_gather_or_output(self):
        model = make_model()
        inputs = make_inputs()
        expected = model.norm(model.embed_tokens(inputs.input_ids) + 2)
        with mock.patch.object(
            torch, "index_select", wraps=torch.index_select
        ) as gather:
            outputs = model.forward(inputs, fmha_impl=object())
        gather.assert_not_called()
        self.assertIsNone(outputs.pre_final_norm_hidden_states)
        torch.testing.assert_close(outputs.hidden_states, expected, rtol=0, atol=0)

    def test_selected_rows_match_training_head_without_changing_generation(self):
        for dtype in (torch.float32, torch.bfloat16):
            with self.subTest(dtype=dtype), torch.no_grad():
                model = make_model(dtype)
                inputs = make_inputs([4, 1])  # preserve request order, not token order
                before = model.embed_tokens(inputs.input_ids) + 2
                baseline = model.forward(make_inputs(), fmha_impl=object())
                outputs = model.forward(inputs, fmha_impl=object())
                selected = outputs.pre_final_norm_hidden_states
                self.assertEqual(selected.shape, (2, 4))
                self.assertEqual(selected.dtype, dtype)
                # Retained storage must be batch-sized, not a view of all tokens.
                self.assertEqual(
                    selected.untyped_storage().nbytes(), 2 * 4 * selected.element_size()
                )
                torch.testing.assert_close(selected, before[[4, 1]], rtol=0, atol=0)
                torch.testing.assert_close(
                    outputs.hidden_states, baseline.hidden_states, rtol=0, atol=0
                )
                head = (
                    torch.nn.Sequential(
                        torch.nn.Linear(4, 2), torch.nn.SiLU(), torch.nn.Linear(2, 1)
                    )
                    .to(dtype)
                    .eval()
                )
                torch.testing.assert_close(
                    torch.sigmoid(head(selected).float()),
                    torch.sigmoid(head(before[[4, 1]]).float()),
                    rtol=0,
                    atol=0,
                )
                lm_head = torch.arange(12, dtype=dtype).reshape(3, 4)
                torch.testing.assert_close(
                    outputs.hidden_states @ lm_head.T,
                    baseline.hidden_states @ lm_head.T,
                    rtol=0,
                    atol=0,
                )

    def test_capture_precedes_even_an_inplace_norm(self):
        model = make_model()
        model.norm = InplaceNorm()
        inputs = make_inputs([1, 3])
        before = model.embed_tokens(inputs.input_ids) + 2
        outputs = model.forward(inputs, fmha_impl=object())
        torch.testing.assert_close(outputs.pre_final_norm_hidden_states, before[[1, 3]])
        torch.testing.assert_close(outputs.hidden_states, before * 0.25)

    def test_capture_does_not_leak_into_next_decode_or_default_call(self):
        model = make_model()
        first = model.forward(make_inputs([1, 3]), fmha_impl=object())
        saved = first.pre_final_norm_hidden_states.clone()
        second = model.forward(make_inputs(), fmha_impl=object())
        self.assertIsNone(second.pre_final_norm_hidden_states)
        torch.testing.assert_close(first.pre_final_norm_hidden_states, saved)


if __name__ == "__main__":
    unittest.main()
