"""CPU contracts for the actual K3 BF16 prefill adapter, without CUDA imports."""

import ast
import builtins
import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


class Tensor:
    def __init__(self, shape, *, dtype, device, strides=None):
        self.shape = (shape,) if isinstance(shape, int) else tuple(shape)
        self.dtype = dtype
        self.device = device
        if strides is None:
            strides = []
            stride = 1
            for size in reversed(self.shape):
                strides.append(stride)
                stride *= size
            strides.reverse()
        self.strides = tuple(strides)
        self.contiguous_result = None

    def transpose(self, dim0, dim1):
        shape, strides = list(self.shape), list(self.strides)
        shape[dim0], shape[dim1] = shape[dim1], shape[dim0]
        strides[dim0], strides[dim1] = strides[dim1], strides[dim0]
        return Tensor(shape, dtype=self.dtype, device=self.device, strides=strides)

    def contiguous(self):
        self.contiguous_result = Tensor(
            self.shape, dtype=self.dtype, device=self.device
        )
        return self.contiguous_result


def load_adapter(torch):
    source = Path(__file__).resolve().parent.parent / "mla.py"
    tree = ast.parse(source.read_text())
    workspace = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_bf16_flash_mla_workspace"
    )
    adapter = next(
        method
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        for method in node.body
        if isinstance(method, ast.FunctionDef) and method.name == "_run_bf16_partial"
    )
    module = ast.Module(body=[workspace, adapter], type_ignores=[])
    bindings = {"torch": torch, "_BF16_FLASH_MLA_WORKSPACES": {}}
    exec(compile(module, str(source), "exec"), bindings)
    return bindings["_run_bf16_partial"]


class FlashMlaPrefillApiTest(unittest.TestCase):
    def setUp(self):
        self.device = SimpleNamespace(index=2)
        self.torch = SimpleNamespace(
            bfloat16="bfloat16",
            float32="float32",
            uint8="uint8",
            empty=mock.Mock(side_effect=Tensor),
            ops=SimpleNamespace(flash_mla=SimpleNamespace()),
            cuda=SimpleNamespace(current_device=mock.Mock(return_value=2)),
        )
        self.run = load_adapter(self.torch)
        self.model = SimpleNamespace(num_heads=3, v_head_dim=128, scale=0.125)
        self.q = Tensor((7, 3, 192), dtype="bfloat16", device=self.device)
        self.k = Tensor((29, 1, 192), dtype="bfloat16", device=self.device)
        self.v = Tensor((29, 1, 128), dtype="bfloat16", device=self.device)
        self.qo_indptr = object()
        self.kv_indptr = object()
        self.legacy = mock.Mock()
        self.package = types.ModuleType("flash_mla")
        self.package.__path__ = []
        self.cuda = types.ModuleType("flash_mla.cuda")
        self.cuda.dense_prefill_fwd = self.legacy

    def invoke(self, *, causal=True):
        return self.run(
            self.model,
            self.q,
            self.k,
            self.v,
            self.qo_indptr,
            self.kv_indptr,
            max_q=13,
            max_k=29,
            causal=causal,
        )

    def patch_imports(self):
        return mock.patch.dict(
            sys.modules, {"flash_mla": self.package, "flash_mla.cuda": self.cuda}
        )

    def assert_contract(self, kernel, result, *, causal):
        output, returned_lse = result
        args = kernel.call_args.args
        self.assertEqual(len(args), 13)
        workspace, *_, lse, causal_arg, scale, max_q, max_k, varlen = args
        self.assertEqual(workspace.shape, (32 * 1024 * 1024,))
        self.assertEqual(workspace.dtype, self.torch.uint8)
        self.assertIs(workspace.device, self.device)
        self.assertEqual(
            args[1:8],
            (
                self.q,
                self.k,
                self.v,
                self.qo_indptr,
                self.kv_indptr,
                output,
                lse,
            ),
        )
        self.assertEqual(output.shape, (7, 3, 128))
        self.assertEqual(output.dtype, self.torch.bfloat16)
        self.assertIs(output.device, self.device)
        self.assertEqual(lse.shape, (7, 3))
        self.assertEqual(lse.strides, (1, 7))
        self.assertEqual(lse.dtype, self.torch.float32)
        self.assertIs(lse.device, self.device)
        self.assertIs(returned_lse, lse.contiguous_result)
        self.assertEqual(returned_lse.shape, lse.shape)
        self.assertEqual(returned_lse.strides, (3, 1))
        self.assertEqual(
            (causal_arg, scale, max_q, max_k, varlen),
            (int(causal), self.model.scale, 13, 29, True),
        )
        self.assertFalse(kernel.call_args.kwargs)

    def test_registered_operator_preferred_and_workspace_reused(self):
        kernel = mock.Mock()
        self.torch.ops.flash_mla.dense_prefill_fwd = kernel
        workspaces = []
        with self.patch_imports(), mock.patch(
            "builtins.__import__", wraps=builtins.__import__
        ) as imports:
            for causal in (False, True):
                with self.subTest(causal=causal):
                    result = self.invoke(causal=causal)
                    self.assert_contract(kernel, result, causal=causal)
                    workspaces.append(kernel.call_args.args[0])
        self.assertIs(workspaces[0], workspaces[1])
        self.assertEqual(kernel.call_count, 2)
        self.assertNotIn(
            "flash_mla.cuda", [call.args[0] for call in imports.call_args_list]
        )
        self.legacy.assert_not_called()
        self.torch.cuda.current_device.assert_not_called()

    def test_old_pybind_wheel_retains_identical_native_contract(self):
        with self.patch_imports():
            result = self.invoke()
        self.legacy.assert_called_once()
        self.assert_contract(self.legacy, result, causal=True)

    def test_registered_kernel_error_does_not_fall_back(self):
        failure = RuntimeError("registered kernel failed")
        kernel = mock.Mock(side_effect=failure)
        self.torch.ops.flash_mla.dense_prefill_fwd = kernel
        with self.patch_imports(), mock.patch(
            "builtins.__import__", wraps=builtins.__import__
        ) as imports:
            with self.assertRaises(RuntimeError) as raised:
                self.invoke()
        self.assertIs(raised.exception, failure)
        kernel.assert_called_once()
        self.legacy.assert_not_called()
        self.assertNotIn(
            "flash_mla.cuda", [call.args[0] for call in imports.call_args_list]
        )


if __name__ == "__main__":
    unittest.main()
