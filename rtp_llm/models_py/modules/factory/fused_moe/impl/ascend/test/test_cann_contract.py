"""Static contract checks for the Ascend MoE implementation.

These checks deliberately do not import torch or torch_npu, so they can run in
ordinary CI.  They verify the wiring only; NPU numerical tests remain manual
targets because this repository's local CI image has no Ascend device.
"""

import ast
import os
import unittest
from pathlib import Path


_ROOT = "rtp_llm/models_py"


def _source(relative_path: str) -> str:
    runfiles = os.environ.get("TEST_SRCDIR")
    workspace = os.environ.get("TEST_WORKSPACE")
    if runfiles and workspace:
        return (
            Path(runfiles) / workspace / _ROOT / relative_path
        ).read_text(encoding="utf-8")
    return (Path(__file__).parents[8] / _ROOT / relative_path).read_text(
        encoding="utf-8"
    )


class AscendCannMoeContractTest(unittest.TestCase):
    def test_cann_executor_uses_all_required_ops(self) -> None:
        source = _source(
            "modules/factory/fused_moe/impl/ascend/executors/npu_fused_experts.py"
        )
        for op in (
            "npu_moe_init_routing_v2",
            "npu_grouped_matmul",
            "npu_swiglu",
            "npu_moe_token_unpermute",
        ):
            self.assertIn(op, source)
        self.assertNotIn("rtp_llm.models_py.triton_kernels", source)

    def test_w1_transform_swaps_qwen_up_gate_order(self) -> None:
        tree = ast.parse(
            _source(
                "modules/factory/fused_moe/impl/ascend/executors/npu_fused_experts.py"
            )
        )
        transform = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == "_prepare_w1"
        )
        text = ast.unparse(transform)
        # ast.unparse renders tuple unpacking with parentheses on Python 3.10
        # ("(up, gate) = ...") and without on newer versions; match the
        # paren-insensitive form so the contract holds on both.
        self.assertIn("up, gate) = w1.chunk(2, dim=1)", text)
        self.assertIn("torch.cat((gate, up), dim=1)", text)

    def test_select_topk_uses_cann_gating_and_optional_renorm(self) -> None:
        source = _source("modules/base/ascend/select_topk.py")
        self.assertIn("npu_moe_gating_top_k", source)
        self.assertIn("if self.config.has_moe_norm", source)
        self.assertIn("renorm=0", source)


if __name__ == "__main__":
    unittest.main()
