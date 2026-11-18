"""CPU metadata/mocked-dispatch regression cases; no inference or AITER calls."""

import sys
import types
import unittest
from contextlib import ExitStack
from unittest.mock import patch

import torch

from rtp_llm.models_py.model_desc import rocm_eager_qwen3 as eager


class RocmEagerPrefillPlanTest(unittest.TestCase):
    def test_projection_accepts_column_major_swizzle_view_layout(self):
        # Physical swizzling changes values, not this shape/stride contract.
        weight = torch.empty(32, 64, dtype=torch.bfloat16).t()
        self.assertFalse(weight.is_contiguous())
        self.assertTrue(eager._projection_layout_supported(weight))
        self.assertTrue(eager._projection_layout_supported(weight.contiguous()))
        self.assertFalse(eager._projection_layout_supported(weight[::2, ::2]))
        self.assertFalse(eager._projection_layout_supported(torch.empty(0, 64)))

    def test_positive_plan_preserves_and_invalidates_strides(self):
        module = torch.nn.Module()
        module.weight = torch.empty(32, 32, dtype=torch.bfloat16).t()
        modules = (module,)
        plan = object.__new__(eager._Plan)
        plan.modules = modules
        plan.config = eager._configuration_signature(modules)
        plan.weights = [(module, module.weight, eager._weight_metadata(module.weight))]
        self.assertTrue(plan.matches(modules))
        # Same object, shape and pointer, but a different storage interpretation.
        module.weight.transpose_(0, 1)
        self.assertFalse(plan.matches(modules))

    def test_failed_plan_invalidates_weights_and_configuration(self):
        module = torch.nn.Module()
        module.weight = torch.empty(32, 32)
        module.tp_size = 2
        modules = (module,)
        failure = eager._FailedPlan(modules, "plan:unsupported decoder")
        self.assertTrue(failure.matches(modules))
        module.tp_size = 1
        self.assertFalse(failure.matches(modules))
        module.tp_size = 2
        module.weight = module.weight.clone()
        self.assertFalse(failure.matches(modules))

    def test_failed_constructor_is_not_repeated_until_weight_change(self):
        class Model(torch.nn.Module):
            pass

        model = Model()
        model.weight = torch.empty(32, 32)
        model._eager_prefill_plan = None
        model._eager_prefill_failure = None
        model._eager_prefill_fallback_reasons = set()
        qwen3 = types.ModuleType("rtp_llm.models_py.model_desc.qwen3")
        qwen3.Qwen3Model = Model
        inputs = types.SimpleNamespace(
            attention_inputs=types.SimpleNamespace(
                is_prefill=True, is_cuda_graph=False
            ),
            capture_hidden_states=False,
        )
        with ExitStack() as stack:
            stack.enter_context(patch.dict(sys.modules, {qwen3.__name__: qwen3}))
            stack.enter_context(
                patch.dict("os.environ", {"RTP_LLM_ROCM_EAGER_PREFILL_SCRATCH": "1"})
            )
            stack.enter_context(patch.object(torch.version, "hip", "mock"))
            for owner, name in (
                (torch, "is_grad_enabled"),
                (torch, "is_autocast_enabled"),
                (torch.jit, "is_tracing"),
                (torch.cuda, "is_current_stream_capturing"),
            ):
                stack.enter_context(patch.object(owner, name, return_value=False))
            stack.enter_context(
                patch.object(eager, "_native_ops", return_value=(object(), None))
            )
            constructor = stack.enter_context(
                patch.object(
                    eager, "_Plan", side_effect=ValueError("projection_weight_layout")
                )
            )
            log = stack.enter_context(patch.object(eager.logging, "info"))
            for _ in range(3):
                self.assertIsNone(
                    eager.prepare_workspace(model, inputs, model.weight, False)
                )
            self.assertEqual(constructor.call_count, 1)
            self.assertEqual(log.call_count, 1)
            model.weight = model.weight.clone()
            self.assertIsNone(
                eager.prepare_workspace(model, inputs, model.weight, False)
            )
            self.assertEqual(constructor.call_count, 2)
            self.assertEqual(log.call_count, 1)


if __name__ == "__main__":
    unittest.main()
