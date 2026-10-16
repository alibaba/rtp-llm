"""V4.1 prefill consumes main's real pybind input fields and ragged bounds."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from rtp_llm.ops.compute_ops import PyAttentionInputs, PyModelInputs


class NativePrefillInputsTest(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_eager_prefill_keeps_native_device_boundaries(self):
        from rtp_llm.models_py.modules.dsv41.prefill import forward

        attention = PyAttentionInputs()
        attention.is_prefill = True
        attention.input_lengths = torch.tensor([2, 3], dtype=torch.int32)
        attention.prefix_lengths = torch.tensor([0, 9], dtype=torch.int32)
        attention.cu_seqlens = torch.tensor([0, 2, 5], dtype=torch.int32)
        attention.cu_seqlens_device = attention.cu_seqlens.cuda()
        inputs = PyModelInputs()
        inputs.input_ids = torch.tensor(
            [1, 2, 3, 4, 5], dtype=torch.int32, device="cuda"
        )
        inputs.combo_position_ids = torch.tensor([0, 1, 9, 10, 11], device="cuda")
        inputs.attention_inputs = {"swa_kv": attention}
        model = SimpleNamespace(set_cp_info=Mock())
        result = torch.zeros(5, 4, device="cuda")
        with patch.object(forward, "forward_layers", return_value=result) as layers:
            forward.forward_prefill(model, None, None, inputs)
        self.assertEqual(
            layers.call_args.args[4].data_ptr(), attention.cu_seqlens_device.data_ptr()
        )
        self.assertEqual(layers.call_args.args[4].device, inputs.input_ids.device)

    @unittest.skipUnless(torch.cuda.is_available(), "native model imports need CUDA")
    def test_device_metadata_capability_after_v41_subclass_configuration(self):
        from rtp_llm.models_py.model_desc import deepseek_v41_base_model as base
        from rtp_llm.models_py.model_desc.deepseek_v41_dspark_model import (
            DeepSeekV41DSparkModel,
        )
        from rtp_llm.models_py.model_desc.deepseek_v41_model import DeepSeekV41Model

        config = SimpleNamespace(
            num_layers=3,
            vocab_size=17,
            capture_aux_hidden_layer_ids=[],
            gen_num_per_cycle=8,
            inter_size=0,
            ckpt_path="unused",
            dspark_noise_token_id=1,
            dspark_target_layer_ids=[37, 38, 39],
            dspark_markov_rank=2,
            deepseek_v41_config={
                "dspark_n_routed_experts": 4,
                "dspark_num_experts_per_tok": 2,
            },
        )
        for model_class in (
            base.DeepSeekV4Model,
            DeepSeekV41Model,
            DeepSeekV41DSparkModel,
        ):
            for fp8 in (False, True):
                with self.subTest(model=model_class.__name__, fp8=fp8):
                    args = base.V4Args(
                        n_layers=3,
                        compress_ratios=[0, 0, 0],
                        fp8_kv_cache=fp8,
                    )
                    self.assertIsNone(args.v41_config)
                    # Run the real base and subclass constructors. V4.1's
                    # configure_v41 runs after the base constructor returns;
                    # caching the capability in the base therefore loses it.
                    with patch.object(
                        base, "_args_from_model_config", return_value=args
                    ):
                        model = model_class(config, None, None, None, 4)
                    self.assertEqual(
                        model.supports_device_only_attention_metadata,
                        fp8 and model_class is not base.DeepSeekV4Model,
                    )

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
