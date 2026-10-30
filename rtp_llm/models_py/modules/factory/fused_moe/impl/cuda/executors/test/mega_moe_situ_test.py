"""Activation selection and execution validation without loading GPU weights."""

import unittest
from types import SimpleNamespace
from unittest import mock

import torch
from rtp_llm.models_py.modules.factory.fused_moe.defs.fused_moe import (
    ExpertForwardPayload,
)
from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.fp8_fp4_base import (
    Fp8Fp4ExecutorBase,
)
from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.mega_moe import (
    MegaMoeExecutor,
)
from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.mega_moe_se import (
    MegaMoeSEExecutor,
)


class MegaMoeActivationTest(unittest.TestCase):
    def _make_executor(self, executor_type, activation):
        config = SimpleNamespace(activation_type=activation)
        with mock.patch.object(executor_type, "setup_weights") as setup:
            executor = executor_type(config, None, {})
        setup.assert_called_once_with({})
        return executor

    def test_silu_aliases_initialize_the_same_activation(self):
        for executor_type in (Fp8Fp4ExecutorBase, MegaMoeExecutor, MegaMoeSEExecutor):
            for activation in ("silu", "SiGLU", "SwiGLU"):
                with self.subTest(executor=executor_type, activation=activation):
                    executor = self._make_executor(executor_type, activation)
                    self.assertEqual(executor.activation, "swiglu")

    def test_unsupported_activation_is_rejected_before_loading_weights(self):
        for executor_type, activation in (
            (Fp8Fp4ExecutorBase, "situ"),
            (MegaMoeSEExecutor, "situ"),
            (MegaMoeExecutor, "gelu"),
        ):
            with self.subTest(executor=executor_type, activation=activation):
                with mock.patch.object(executor_type, "setup_weights") as setup:
                    with self.assertRaisesRegex(ValueError, "supports activations"):
                        executor_type(
                            SimpleNamespace(activation_type=activation), None, {}
                        )
                setup.assert_not_called()

    def test_call_must_match_the_initialized_kernel_activation(self):
        for configured, requested in (("situ", "silu"), ("swiglu", "situ")):
            with self.subTest(configured=configured, requested=requested):
                executor = self._make_executor(MegaMoeExecutor, configured)
                with mock.patch.object(executor, "forward") as forward:
                    with self.assertRaisesRegex(
                        ValueError, "was initialized with activation"
                    ):
                        executor.execute(None, requested, None, None, False, None)
                forward.assert_not_called()

    def test_matching_activation_preserves_executor_output(self):
        payload = ExpertForwardPayload(
            expert_x=torch.ones((2, 4), dtype=torch.bfloat16),
            expert_topk_weights=torch.ones((2, 1)),
            expert_topk_ids=torch.zeros((2, 1), dtype=torch.long),
        )
        output = torch.full_like(payload.expert_x, 3)
        for configured, requested in (
            ("situ", "SiTU"),
            ("swiglu", "silu"),
            ("swiglu", "SiGLU"),
            ("swiglu", "SwiGLU"),
        ):
            with self.subTest(configured=configured, requested=requested):
                executor = self._make_executor(MegaMoeExecutor, configured)
                with mock.patch.object(
                    executor, "forward", return_value=output
                ) as forward:
                    result = executor.execute(
                        payload, requested, None, None, False, None
                    )
                forward.assert_called_once_with(
                    payload.expert_x,
                    payload.expert_topk_weights,
                    payload.expert_topk_ids,
                )
                self.assertIs(result.fused_expert_output, output)


if __name__ == "__main__":
    unittest.main()
