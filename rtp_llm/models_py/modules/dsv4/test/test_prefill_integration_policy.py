"""CPU source-policy contracts; CUDA kernels and service execution are tested separately."""

from __future__ import annotations

import ast
import os
import types
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = next(
    parent
    for parent in Path(__file__).resolve().parents
    if (parent / "rtp_llm/models_py/modules/dsv4/fp8/attention.py").is_file()
)
FP8 = "rtp_llm/models_py/modules/dsv4/fp8/"


def source_fn(path, name, ns, cls=None):
    """Compile the actual function, injecting only its dependencies."""
    nodes = ast.parse((ROOT / path).read_text()).body
    if cls:
        nodes = next(
            n for n in nodes if isinstance(n, ast.ClassDef) and n.name == cls
        ).body
    node = next(n for n in nodes if isinstance(n, ast.FunctionDef) and n.name == name)
    node.decorator_list = []
    mod = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            node,
        ],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(mod), str(ROOT / path), "exec"), ns)
    return ns[name]


class IntegrationPolicyTest(unittest.TestCase):
    def test_woa_and_warmup_choose_same_arm_at_invocation(self):
        def decision(path, method, needle, cls=None):
            ns = {"os": os, "is_sm120": lambda d: True}
            tree = ast.parse((ROOT / path).read_text())
            nodes = (
                tree.body
                if not cls
                else next(
                    n
                    for n in tree.body
                    if isinstance(n, ast.ClassDef) and n.name == cls
                ).body
            )
            # Locate by method anywhere in the file; class name is not part of the contract.
            f = next(
                n
                for n in ast.walk(tree)
                if isinstance(n, ast.FunctionDef) and n.name == method
            )
            test = next(
                n.test
                for n in ast.walk(f)
                if isinstance(n, ast.If) and needle in ast.unparse(n.test)
            )
            return eval(
                compile(ast.Expression(test), "<actual-guard>", "eval"),
                dict(ns, device=120, o_fp8=types.SimpleNamespace(device=120)),
            )

        for flag in ("0", "1"):
            with patch.dict(os.environ, {"DSV4_SM120_WOA_EINSUM": flag}, clear=True):
                self.assertEqual(
                    decision(
                        FP8 + "attention.py",
                        "_wo_a_einsum_from_fp8",
                        "DSV4_SM120_WOA_EINSUM",
                    ),
                    flag == "0",
                )
                self.assertEqual(
                    decision(
                        "rtp_llm/models_py/modules/dsv4/dsv4_kernel_jit_warmup.py",
                        "warmup_batched_fp8_einsum_jit",
                        "DSV4_SM120_WOA_EINSUM",
                    ),
                    flag == "0",
                )


if __name__ == "__main__":
    unittest.main(verbosity=2)
