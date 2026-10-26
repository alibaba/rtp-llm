import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from rtp_llm.models_py.model_desc.module_base import (
    GptModelBase,
    wrap_quantized_lm_head,
)
from rtp_llm.ops.compute_ops import (
    PyAttentionInputs,
    PyContextParallelParams,
    PyModelInputs,
    PyModelOutputs,
)


class HeadOnlyModel(GptModelBase):
    def __init__(self):
        nn.Module.__init__(self)
        self.kv_cache = None
        self.parallelism_config = SimpleNamespace(tp_rank=0, tp_size=1)
        self.config = SimpleNamespace(output_vocab_ids=[], output_vocab_padded_size=0)
        self.nvfp4_lm_head = nn.Linear(8, 16, bias=False, dtype=torch.bfloat16)
        with torch.no_grad():
            self.nvfp4_lm_head.weight.copy_(
                torch.arange(128).reshape(16, 8).to(torch.bfloat16) / 128
            )

    def forward(self, inputs, fmha_impl=None):
        return PyModelOutputs(inputs.input_hiddens)


class QuantizedLmHeadOutputTest(unittest.TestCase):
    @unittest.skipUnless(
        torch.cuda.is_available() and torch.cuda.get_device_capability() == (12, 0),
        "requires SM120 NVFP4",
    )
    def test_real_packed_head_loader_forward_and_cuda_graph(self):
        from rtp_llm.config.modelopt_mixed import ModelOptMixedPrecisionConfig
        from rtp_llm.device.device_impl import CudaImpl
        from rtp_llm.model_loader.weight_module import AtomicWeight, WeightModule
        from rtp_llm.models_py.modules.factory.linear import LinearFactory
        from rtp_llm.utils.model_weight import CkptWeightInfo, W, identity

        config = ModelOptMixedPrecisionConfig(
            {"lm_head": {"quant_algo": "NVFP4", "group_size": 16}}
        )
        loader = WeightModule.create(
            AtomicWeight(W.lm_head, [CkptWeightInfo("lm_head.weight", identity)]),
            config,
        )
        # E2M1 code 7 is exactly 6; each byte packs two weights. Unit block
        # scales give an independent, exact reference for the real FP4 GEMM.
        tensors = {
            "lm_head.weight": torch.full(
                (128, 64), 0x77, dtype=torch.uint8, device="cuda"
            ),
            "lm_head.weight_scale": torch.ones((128, 8), device="cuda").to(
                torch.float8_e4m3fn
            ),
            "lm_head.weight_scale_2": torch.ones((), device="cuda"),
            "lm_head.input_scale": torch.ones((), device="cuda"),
        }
        source = SimpleNamespace(
            load_tensor=lambda name, dtype: [tensors[name].to(dtype)]
        )
        load_config = SimpleNamespace(
            compute_dtype=torch.bfloat16, exported_device=CudaImpl
        )
        model = HeadOnlyModel()
        with patch.dict(os.environ, {"RTP_LLM_FP4_GEMM_BACKEND": "cutlass"}):
            weights = loader._postprocess(
                loader._load_raw_tensor(source, None, "cuda", load_config),
                "cuda",
                load_config,
            )
            model.nvfp4_lm_head = LinearFactory.create_linear(
                weights[W.lm_head],
                None,
                weights[W.lm_head_s],
                config,
                weight_scale_2=weights[W.lm_head_s2],
                input_scale=weights[W.lm_head_i_s],
            )
        model = wrap_quantized_lm_head(model)
        hidden = torch.full((3, 128), 6, dtype=torch.bfloat16, device="cuda")
        inputs = PyModelInputs(input_hiddens=hidden)
        inputs.lm_output_indexes = torch.tensor([2, 0])
        with torch.no_grad():
            torch.testing.assert_close(
                model.forward(inputs).logits,
                torch.full((2, 128), 128 * 36, device="cuda", dtype=torch.float32),
                rtol=0,
                atol=0,
            )
            for _ in range(3):
                model.forward_for_cuda_graph(inputs)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                captured = model.forward_for_cuda_graph(inputs)
            hidden.fill_(3)
            graph.replay()
            output = model.postprocess_outputs(inputs, captured)
            torch.testing.assert_close(
                output.logits,
                torch.full((2, 128), 128 * 18, device="cuda", dtype=torch.float32),
                rtol=0,
                atol=0,
            )

    def test_wrapper_preserves_model_identity_and_runtime_state(self):
        original = HeadOnlyModel()
        original_type = type(original)
        original_modules = list(original.modules())
        original_keys = list(original.state_dict())
        model = wrap_quantized_lm_head(original)
        self.assertIs(model, original)
        self.assertIs(type(model), original_type)
        self.assertEqual(list(model.modules()), original_modules)
        self.assertEqual(list(model.state_dict()), original_keys)
        cache = object()
        model.kv_cache = cache
        self.assertIs(original.kv_cache, cache)
        model.kv_cache = None
        forward = model.forward
        self.assertIs(wrap_quantized_lm_head(model), model)
        self.assertEqual(model.forward, forward)
        self.assertIs(model.nvfp4_lm_head, original.nvfp4_lm_head)
        self.assertIn(original.nvfp4_lm_head, list(model.modules()))
        inputs = PyModelInputs(input_hiddens=torch.ones(2, 8))
        with patch.object(
            model.nvfp4_lm_head, "forward", wraps=model.nvfp4_lm_head.forward
        ) as head, torch.no_grad():
            model.forward(inputs)
            head.assert_called_once()

    def test_forward_selects_rows_and_preserves_hidden_for_mtp(self):
        model = wrap_quantized_lm_head(HeadOnlyModel())
        hidden = torch.arange(48).reshape(6, 8).float() / 16
        inputs = PyModelInputs(input_hiddens=hidden)
        inputs.lm_output_indexes = torch.tensor([5, 1], dtype=torch.int32)
        with torch.no_grad():
            output = model.forward(inputs)
            expected = model.compute_quantized_logits(hidden[[5, 1]])
        self.assertTrue(torch.equal(output.hidden_states, hidden))
        self.assertEqual(output.local_logits.dtype, torch.bfloat16)
        self.assertTrue(torch.equal(output.local_logits.float(), expected))
        self.assertEqual(output.logits.dtype, torch.float32)
        self.assertTrue(torch.equal(output.logits, expected))
        self.assertTrue(torch.equal(output.last_hidden_states, hidden[[5, 1]]))
        self.assertIsNone(output.all_logits)

    def test_forward_all_logits_and_decode(self):
        model = wrap_quantized_lm_head(HeadOnlyModel())
        for token_count, need_all in ((1, False), (6, True)):
            with self.subTest(token_count=token_count):
                hidden = torch.arange(token_count * 8).reshape(token_count, 8).float()
                inputs = PyModelInputs(input_hiddens=hidden)
                if need_all:
                    inputs.lm_output_indexes = torch.tensor([token_count - 1])
                inputs.need_all_logits = need_all
                with torch.no_grad():
                    output = model.forward(inputs)
                    expected = model.compute_quantized_logits(hidden)
                self.assertTrue(torch.equal(output.hidden_states, hidden))
                self.assertTrue(torch.equal(output.local_logits.float(), expected))
                self.assertEqual(output.logits.dtype, torch.float32)
                if need_all:
                    self.assertTrue(torch.equal(output.all_logits, expected))
                    self.assertTrue(torch.equal(output.logits, expected[-1:]))
                else:
                    self.assertTrue(torch.equal(output.logits, expected))

    def test_tp_gathers_selected_bf16_rows_before_vocabulary_crop_and_fp32(self):
        model = wrap_quantized_lm_head(HeadOnlyModel())
        model.parallelism_config.tp_size = 2
        model.config.output_vocab_ids = list(range(29))
        model.config.output_vocab_padded_size = 32
        hidden = torch.arange(48).reshape(6, 8).float() / 16
        for need_all in (False, True):
            inputs = PyModelInputs(input_hiddens=hidden)
            inputs.lm_output_indexes = torch.tensor([5, 1])
            inputs.need_all_logits = need_all
            communicated = []

            def gather(local, group):
                self.assertEqual(local.dtype, torch.bfloat16)
                communicated.append(local.clone())
                return torch.cat((local, local + 16), dim=0)

            with patch(
                "rtp_llm.models_py.distributed.collective_torch.all_gather",
                side_effect=gather,
            ), torch.no_grad():
                outputs = model.forward(inputs)
                local = model._compute_quantized_logits_native(hidden)
            full = torch.cat((local, local + 16), dim=1)[:, :29].float()
            self.assertEqual(len(communicated), 1)
            self.assertEqual(communicated[0].shape[0], 6 if need_all else 2)
            self.assertTrue(torch.equal(outputs.logits, full[[5, 1]]))
            self.assertEqual(outputs.logits.dtype, torch.float32)
            if need_all:
                self.assertTrue(torch.equal(outputs.all_logits, full))

    def test_replay_postprocessing_selects_rows_and_owns_hidden_output(self):
        model = wrap_quantized_lm_head(HeadOnlyModel())
        hidden = torch.arange(48).reshape(6, 8).float() / 16
        inputs = PyModelInputs(input_hiddens=hidden)
        inputs.lm_output_indexes = torch.tensor([5, 1])
        with torch.no_grad():
            raw = model.forward_for_cuda_graph(inputs)
            self.assertEqual(raw.local_logits.shape[0], 6)
            self.assertIsNone(raw.logits)
            self.assertIsNone(raw.last_hidden_states)
            final = model.postprocess_outputs(inputs, raw)
            expected = model.forward(inputs)
        self.assertTrue(torch.equal(final.logits, expected.logits))
        self.assertTrue(torch.equal(final.hidden_states, hidden))
        self.assertTrue(torch.equal(final.last_hidden_states, hidden[[5, 1]]))
        saved_hidden = hidden.clone()
        hidden.zero_()
        self.assertTrue(torch.equal(final.hidden_states, saved_hidden))

    def test_invalid_pruned_vocabulary_layout_fails_before_collective(self):
        model = wrap_quantized_lm_head(HeadOnlyModel())
        model.config.output_vocab_ids = list(range(17))
        model.config.output_vocab_padded_size = 16
        with self.assertRaisesRegex(ValueError, "vocabulary LM head layout"):
            model.forward(PyModelInputs(input_hiddens=torch.ones(2, 8)))

    def test_host_output_indexes_with_cuda_hidden_for_eager_and_replay(self):
        model = wrap_quantized_lm_head(HeadOnlyModel().cuda())
        hidden = torch.arange(48, device="cuda").reshape(6, 8).to(torch.bfloat16) / 16
        indexes = torch.tensor([5, 1], dtype=torch.int32, pin_memory=True)
        with torch.no_grad():
            expected = model.nvfp4_lm_head(hidden).float()
            for need_all in (False, True):
                for replay in (False, True):
                    with self.subTest(need_all=need_all, replay=replay):
                        inputs = PyModelInputs(input_hiddens=hidden)
                        inputs.lm_output_indexes = indexes
                        inputs.need_all_logits = need_all
                        if replay:
                            output = model.postprocess_outputs(
                                inputs, model.forward_for_cuda_graph(inputs)
                            )
                        else:
                            output = model.forward(inputs)
                        torch.testing.assert_close(output.logits, expected[[5, 1]])
                        torch.testing.assert_close(
                            output.last_hidden_states,
                            hidden if need_all else hidden[[5, 1]],
                        )
                        self.assertEqual(output.logits.dtype, torch.float32)
                        self.assertEqual(inputs.lm_output_indexes.device.type, "cpu")
                        self.assertTrue(torch.equal(inputs.lm_output_indexes, indexes))
                        if need_all:
                            torch.testing.assert_close(output.all_logits, expected)

    def test_unquantized_forward_preserves_hidden_only_contract(self):
        model = HeadOnlyModel()
        model.nvfp4_lm_head = None
        forward = model.forward
        self.assertIs(wrap_quantized_lm_head(model), model)
        self.assertEqual(model.forward, forward)
        self.assertIsNone(wrap_quantized_lm_head(None))
        hidden = torch.randn(4, 8)
        output = model.forward(PyModelInputs(input_hiddens=hidden))
        self.assertTrue(torch.equal(output.hidden_states, hidden))
        self.assertIsNone(output.local_logits)
        self.assertIsNone(output.logits)
        self.assertIsNone(output.last_hidden_states)
        self.assertIsNone(output.all_logits)

    def test_cp_reconstructs_selected_hidden_before_projection(self):
        model = wrap_quantized_lm_head(HeadOnlyModel())
        # Two ranks, one padded token: global valid order maps to concatenated
        # rank-local positions [0, 1, 4, 5, 6, 7, 2].
        cp_info = PyContextParallelParams()
        cp_info.prefill_qkv_restore_indice = torch.tensor([0, 1, 4, 5, 6, 7, 2, 3])
        cp_info.prefill_qkv_padding_mask = torch.tensor(
            [True, True, True, True, True, True, True, False]
        )
        attention = PyAttentionInputs()
        attention.context_parallel_info = cp_info
        all_hidden = torch.arange(64).reshape(8, 8).float() / 16
        selected = all_hidden[[7, 2, 0]]
        contributions = []
        for rank in range(2):
            model.parallelism_config.tp_rank = rank
            inputs = PyModelInputs(
                input_hiddens=all_hidden[rank * 4 : (rank + 1) * 4],
                attention_inputs=attention,
            )
            inputs.lm_output_indexes = torch.tensor([5, 6, 0])

            def reduce(local, group):
                contributions.append(local.clone())
                return selected.clone()

            with patch(
                "rtp_llm.models_py.distributed.collective_torch.all_reduce",
                side_effect=reduce,
            ), torch.no_grad():
                output = model.forward(inputs)
                expected = model.compute_quantized_logits(selected)
            self.assertTrue(torch.equal(output.hidden_states, selected))
            self.assertTrue(torch.equal(output.local_logits.float(), expected))
        self.assertTrue(torch.equal(sum(contributions), selected))

    def test_cp_full_outputs_restore_padding_and_token_order(self):
        model = wrap_quantized_lm_head(HeadOnlyModel())
        cp_info = PyContextParallelParams()
        cp_info.prefill_qkv_restore_indice = torch.tensor([0, 1, 4, 5, 6, 7, 2, 3])
        cp_info.prefill_qkv_padding_mask = torch.tensor(
            [True, True, True, True, True, True, True, False]
        )
        attention = PyAttentionInputs()
        attention.context_parallel_info = cp_info
        all_hidden = torch.arange(64).reshape(8, 8).float() / 16
        restored = all_hidden[[0, 1, 4, 5, 6, 7, 2]]
        for need_all_logits in (False, True):
            inputs = PyModelInputs(
                input_hiddens=all_hidden[:4], attention_inputs=attention
            )
            inputs.lm_output_indexes = torch.tensor([5, 6])
            inputs.need_all_logits = need_all_logits
            inputs.need_all_hidden_states = True
            with patch(
                "rtp_llm.models_py.distributed.collective_torch.all_gather",
                return_value=all_hidden,
            ), torch.no_grad():
                output = model.forward(inputs)
                expected = model.compute_quantized_logits(
                    restored if need_all_logits else restored[[5, 6]]
                )
            self.assertTrue(torch.equal(output.hidden_states, restored))
            self.assertTrue(torch.equal(output.local_logits.float(), expected))

    def test_native_output_preserves_fp32_interface(self):
        # A BF16 linear isolates the output contract from the packed-weight
        # kernel. Production NVFP4 head/GEMM coverage is an on-device test.
        model = GptModelBase.__new__(GptModelBase)
        nn.Module.__init__(model)
        model.nvfp4_lm_head = nn.Linear(8, 16, bias=False, dtype=torch.bfloat16)
        with torch.no_grad():
            model.nvfp4_lm_head.weight.copy_(
                torch.arange(128).reshape(16, 8).to(torch.bfloat16) / 128
            )
        for dtype in (torch.bfloat16, torch.float16, torch.float32):
            for offset in (0.0, 0.25):
                hidden = (torch.arange(24).reshape(3, 8) / 16 + offset).to(dtype)
                expected = nn.functional.linear(
                    hidden.to(torch.bfloat16), model.nvfp4_lm_head.weight
                ).float()
                native = model._compute_quantized_logits_native(hidden)
                public = model.compute_quantized_logits(hidden)
                self.assertEqual(native.dtype, torch.bfloat16)
                self.assertEqual(public.dtype, torch.float32)
                self.assertTrue(torch.equal(native.float(), expected))
                self.assertTrue(torch.equal(public, expected))

    def test_uninitialized_head_fails_explicitly(self):
        model = GptModelBase.__new__(GptModelBase)
        nn.Module.__init__(model)
        model.nvfp4_lm_head = None
        for compute in (
            model.compute_quantized_logits,
            model._compute_quantized_logits_native,
        ):
            with self.assertRaisesRegex(ValueError, "No quantized LM head"):
                compute(torch.zeros(1, 8))


if __name__ == "__main__":
    unittest.main()
