"""Selection guard tests for the optional CuTeDSL FP4 backend."""

import unittest
from unittest.mock import MagicMock, patch

from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.cutedsl_fp4_executor import (
    CutedslFp4Executor,
)
from rtp_llm.models_py.modules.factory.fused_moe.utils.condition_checker import (
    ConditionChecker,
)


class CutedslAvailabilityTest(unittest.TestCase):
    def _conditions_pass(self, available: bool) -> bool:
        checker = ConditionChecker("CutedslFp4Executor.check_conditions")
        with patch(
            "rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors."
            "cutedsl_fp4_executor.is_flashinfer_cutedsl_fp4_available",
            return_value=available,
        ), patch(
            "rtp_llm.models_py.modules.factory.fused_moe.utils.config_resolver."
            "MoeConfigResolver.is_bf16",
            return_value=True,
        ), patch(
            "rtp_llm.models_py.modules.factory.fused_moe.utils.config_resolver."
            "MoeConfigResolver.has_quantization",
            return_value=True,
        ), patch(
            "rtp_llm.models_py.modules.factory.fused_moe.utils.config_resolver."
            "MoeConfigResolver.get_quant_method",
            return_value="modelopt_fp4",
        ):
            CutedslFp4Executor.check_conditions(checker, MagicMock())
        return checker.all_passed()

    def test_missing_backend_is_rejected_before_executor_construction(self) -> None:
        self.assertFalse(self._conditions_pass(available=False))

    def test_loaded_backend_remains_selectable(self) -> None:
        self.assertTrue(self._conditions_pass(available=True))


if __name__ == "__main__":
    unittest.main()
