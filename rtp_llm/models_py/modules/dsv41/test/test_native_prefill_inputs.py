"""V4.1 prefill consumes main's real pybind input fields and ragged bounds."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from rtp_llm.ops.compute_ops import PyAttentionInputs, PyModelInputs


class NativePrefillInputsTest(unittest.TestCase):
    @unittest.skipUnless(
        torch.cuda.is_available(), "native attention factory needs CUDA"
    )
    def test_native_positions_and_missing_host_boundaries(self):
        from rtp_llm.models_py.modules.dsv41.prefill import forward

        for explicit in (False, True):
            for empty_host_bounds in (False, True):
                attention = PyAttentionInputs()
                attention.is_prefill = True
                attention.input_lengths = torch.tensor([2, 3], dtype=torch.int32)
                attention.prefix_lengths = torch.tensor([0, 9], dtype=torch.int32)
                attention.cu_seqlens = torch.tensor(
                    [] if empty_host_bounds else [0, 2, 5], dtype=torch.int32
                )
                attention.cu_seqlens_device = torch.tensor(
                    [0, 2, 5], dtype=torch.int32, device="cuda"
                )
                inputs = PyModelInputs()
                inputs.input_ids = torch.tensor([1, 2, 3, 4, 5], dtype=torch.int32)
                inputs.attention_inputs = {"swa_kv": attention}
                expected = torch.tensor([0, 1, 9, 10, 11])
                if explicit:
                    inputs.combo_position_ids = expected
                model = SimpleNamespace(set_cp_info=Mock())
                with patch.object(
                    forward, "forward_layers", return_value=torch.zeros(5, 4)
                ) as layers:
                    output = forward.forward_prefill(model, None, None, inputs)
                torch.testing.assert_close(layers.call_args.args[3].long(), expected)
                self.assertEqual(layers.call_args.args[4].tolist(), [0, 2, 5])
                self.assertEqual(tuple(output.hidden_states.shape), (5, 4))

    @unittest.skipUnless(torch.cuda.is_available(), "native model imports need CUDA")
    def test_prefill_workspace_uses_local_attention_heads(self):
        from rtp_llm.models_py.model_desc.deepseek_v41_base_model import (
            DeepSeekV4Model,
            Dsv4SharedRuntimeBufferStore,
        )

        transformer = SimpleNamespace(
            layers=[SimpleNamespace(attn=SimpleNamespace(n_heads=16))],
            _bind_prefill_workspace_dims=Mock(),
        )
        owner = SimpleNamespace(
            v4=transformer,
            _v4_args=SimpleNamespace(n_heads=64, head_dim=512),
            _prefill_cp_size=1,
            _resolve_prefill_q_token_capacity=lambda: 128,
        )
        with patch.object(
            Dsv4SharedRuntimeBufferStore, "mtp_hidden_requested", return_value=False
        ), patch.object(
            Dsv4SharedRuntimeBufferStore,
            "get_or_create",
            return_value=SimpleNamespace(bind=Mock()),
        ):
            DeepSeekV4Model._bind_runtime_buffers(owner, torch.device("cpu"))
        transformer._bind_prefill_workspace_dims.assert_called_once_with(
            128, 8192, 0, 0, 0
        )


if __name__ == "__main__":
    unittest.main()
