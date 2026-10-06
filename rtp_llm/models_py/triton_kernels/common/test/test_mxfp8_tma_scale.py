"""CPU production-API contracts by default; --gpu explicitly enables CUDA gate."""

import argparse
import ast
import copy
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import NamedTuple

COMMON = Path(__file__).resolve().parents[1]
GPU = False
FROZEN = None


class Tensor:
    def __init__(self, shape, dtype="bf16", strides=None):
        self.shape, self.dtype, self.device = tuple(shape), dtype, "cpu"
        self._strides = strides or ((max(1, shape[1]), 1) if len(shape) == 2 else (1,))
        self.is_cuda = False

    def dim(self):
        return len(self.shape)

    def stride(self, axis):
        return self._strides[axis]

    def transpose(self, a, b):
        return Tensor(self.shape[::-1], self.dtype, self._strides[::-1])

    def view(self, dtype):
        return Tensor((self.shape[0], self.shape[1] * 4), dtype)

    def __getitem__(self, s):
        return Tensor(
            (min(s.stop, self.shape[0]), self.shape[1]), self.dtype, self._strides
        )

    def is_contiguous(self):
        return True


class Launch:
    def __init__(self):
        self.calls = []

    def __getitem__(self, grid):
        def launch(*args, **kwargs):
            self.calls.append((grid, args, kwargs))

        return launch


def api_namespace():
    # Execute the *real* allocation/gating/wrapper AST, with no torch import,
    # CUDA inspection, JIT, or production extension load in the CPU gate.
    torch = SimpleNamespace(
        Tensor=Tensor,
        bfloat16="bf16",
        float32="fp32",
        int32="int32",
        uint8="u8",
        float8_e4m3fn="fp8",
        empty=lambda shape, dtype, device: Tensor(shape, dtype),
        finfo=lambda dtype: SimpleNamespace(max=448),
    )
    n = {
        "torch": torch,
        "triton": SimpleNamespace(
            next_power_of_2=lambda x: 1 << (x - 1).bit_length(),
            cdiv=lambda a, b: (a + b - 1) // b,
        ),
        "Tuple": tuple,
        "MAX_INREG_H": 8192,
        "_SILU_MUL_FP8_QUANT_M_THRESHOLD": 1024,
        "_select_num_warps": lambda h: 8,
        "fallbacks": [],
    }
    for name in (
        "_fused_add_rmsnorm_fp8_quant_singlepass_kernel",
        "_fused_add_rmsnorm_fp8_quant_dual_output_singlepass_kernel",
        "_silu_and_mul_post_quant_dense_packed_kernel",
    ):
        n[name] = Launch()

    def fallback(*args, **kw):
        n["fallbacks"].append((args, kw))
        return ("fallback",)

    for name in (
        "_baseline_add_rmsnorm_fp8_quant",
        "_baseline_add_rmsnorm_fp8_quant_with_bf16_output",
        "silu_and_mul_mxfp8_quant_tiled_fwd",
    ):
        n[name] = fallback
    n["create_per_token_group_quant_fp8_output_scale"] = lambda **kw: Tensor(
        (kw["x_shape"][0], kw["x_shape"][1] // kw["group_size"]),
        "int32" if kw["scale_ue8m0"] else "fp32",
    )
    wanted = {
        "allocate_mxfp8_tma_scale",
        "fused_add_rmsnorm_fp8_quant",
        "fused_add_rmsnorm_fp8_quant_with_bf16_output",
        "silu_and_mul_per_token_group_fp8_quant_dense_packed_fwd",
    }
    for filename in (
        "mxfp8_scale_layout.py",
        "fused_add_rmsnorm_fp8_quant.py",
        "activation.py",
    ):
        tree = ast.parse((COMMON / filename).read_text())
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name in wanted:
                node = copy.deepcopy(node)
                # Lazy imports are binding-only; substitute mocks already in n.
                node.body = [
                    s
                    for s in node.body
                    if not isinstance(s, (ast.Import, ast.ImportFrom))
                ]
                exec(
                    compile(
                        ast.fix_missing_locations(
                            ast.Module(body=[node], type_ignores=[])
                        ),
                        filename,
                        "exec",
                    ),
                    n,
                )
    return n


class CpuContract(unittest.TestCase):
    def test_release_scale_guard(self):
        node = next(
            n
            for n in ast.parse((COMMON / "mxfp8_scale_layout.py").read_text()).body
            if isinstance(n, ast.FunctionDef) and n.name == "checked_ue8m0_exponent"
        )
        asm = next(
            n
            for n in ast.walk(node)
            if isinstance(n, ast.Call)
            and ast.unparse(n.func) == "tl.inline_asm_elementwise"
        )
        self.assertIn("trap;", asm.args[0].value)
        self.assertFalse(
            next(kw.value.value for kw in asm.keywords if kw.arg == "is_pure")
        )
        mask = 0x807FFFFF
        for bits in (0, 0x3F800000, 0x7F800000, 0x7FC00000, 0x80000000, 1, 0xBF800000):
            # Same contract as old DeepGEMM pack: positive zero and Inf have
            # exponent-only bits, NaN/signed-zero/subnormal/negative do not.
            self.assertEqual(bool(bits & mask), bits not in (0, 0x3F800000, 0x7F800000))

    def test_default_and_small_shapes(self):
        for rows in (0, 1, 3, 16, 80, 128):
            for name in (
                "fused_add_rmsnorm_fp8_quant",
                "fused_add_rmsnorm_fp8_quant_with_bf16_output",
            ):
                n = api_namespace()
                x = Tensor((rows, 6144))
                w = Tensor((6144,))
                old = n[name](x, x, w, group_size=32, round_to_pow2=True)
                direct = n[name](
                    x, x, w, group_size=32, round_to_pow2=True, tma_packed_scales=True
                )
                self.assertEqual(old[-1].dtype, "fp32")
                self.assertEqual(direct[-1].dtype, "int32")
                self.assertEqual(direct[-1].shape, (rows, 48))
                if rows:
                    self.assertEqual(direct[-1].stride(1), (rows + 3) // 4 * 4)
                    launch = n[
                        "_fused_add_rmsnorm_fp8_quant_"
                        + ("dual_output_" if "bf16" in name else "")
                        + "singlepass_kernel"
                    ]
                    self.assertFalse(launch.calls[0][2]["TMA_PACKED_SCALES"])
                    self.assertTrue(launch.calls[1][2]["TMA_PACKED_SCALES"])
                    self.assertEqual(
                        launch.calls[1][1][5 if "bf16" in name else 4].dtype, "u8"
                    )
            n = api_namespace()
            x = Tensor((rows, 24576))
            f = n["silu_and_mul_per_token_group_fp8_quant_dense_packed_fwd"]
            old = f(x, quant_group_size=32, scale_ue8m0=False, round_to_pow2=True)
            direct = f(
                x,
                quant_group_size=32,
                scale_ue8m0=False,
                round_to_pow2=True,
                tma_packed_scales=True,
            )
            self.assertEqual(old[-1].dtype, "fp32")
            self.assertEqual(direct[-1].dtype, "int32")
            self.assertEqual(direct[-1].shape, (rows, 96))
            if rows:
                launch = n["_silu_and_mul_post_quant_dense_packed_kernel"]
                self.assertTrue(launch.calls[-1][2]["TMA_PACKED_SCALES"])
                self.assertEqual(launch.calls[-1][1][4].dtype, "u8")

    def test_only_mxfp8_consumers_request_packed(self):
        class OldFp8:
            scale_ue8m0 = True

        class Mxfp8:
            input_quant_tma_packed_scales = True

        for relative in ("modules/hybrid/dense_mlp.py", "model_desc/generic_moe.py"):
            source = COMMON.parents[1] / relative
            n = {
                "NamedTuple": NamedTuple,
                "Any": object,
                "Optional": __import__("typing").Optional,
                "CudaFp8GEMMLinear": OldFp8,
                "CudaMxfp8Linear": Mxfp8,
            }
            selected = [
                node
                for node in ast.parse(source.read_text()).body
                if isinstance(node, (ast.ClassDef, ast.FunctionDef))
                and node.name in ("_FusedFp8QuantParams", "_get_fused_fp8_quant_params")
            ]
            exec(
                compile(
                    ast.Module(body=selected, type_ignores=[]), str(source), "exec"
                ),
                n,
            )
            fn = n["_get_fused_fp8_quant_params"]
            self.assertFalse(fn(OldFp8()).tma_packed_scales)
            self.assertTrue(fn(Mxfp8()).tma_packed_scales)
            self.assertIsNone(fn(object()))

    def test_grouped_silu_dispatch_boundary(self):
        for rows in (1, 16, 63, 64, 65, 80, 127, 128, 129, 1023):
            for width in (3072, 12288, 6144):
                for direct in (False, True):
                    n = api_namespace()
                    n["silu_and_mul_per_token_group_fp8_quant_dense_packed_fwd"](
                        Tensor((rows, 2 * width)),
                        quant_group_size=32,
                        scale_ue8m0=False,
                        round_to_pow2=True,
                        gemm1_alpha=1.702,
                        gemm1_clamp_limit=7,
                        tma_packed_scales=direct,
                    )
                    grid, _, kw = n[
                        "_silu_and_mul_post_quant_dense_packed_kernel"
                    ].calls[-1]
                    groups = (
                        4
                        if direct and 64 <= rows <= 128 and width in (3072, 12288)
                        else 1
                    )
                    self.assertEqual(kw["GROUPS_PER_CTA"], groups)
                    self.assertEqual(grid, (width // 32 // groups, rows))

    def test_unsupported_and_legacy(self):
        for rows, width, group, ue, pow2 in (
            (16, 6144, 128, False, True),
            (16, 6176, 32, False, True),
            (1024, 6144, 32, False, True),
            (16, 6144, 32, False, False),
            (16, 6144, 128, True, False),
        ):
            n = api_namespace()
            x = Tensor((rows, width))
            w = Tensor((width,))
            f = n["fused_add_rmsnorm_fp8_quant"]
            a = f(x, x, w, group_size=group, scale_ue8m0=ue, round_to_pow2=pow2)
            b = f(
                x,
                x,
                w,
                group_size=group,
                scale_ue8m0=ue,
                round_to_pow2=pow2,
                tma_packed_scales=True,
            )
            self.assertEqual(a[-1].dtype, b[-1].dtype)
            self.assertFalse(
                n["_fused_add_rmsnorm_fp8_quant_singlepass_kernel"].calls[-1][2][
                    "TMA_PACKED_SCALES"
                ]
            )
        n = api_namespace()
        x = Tensor((3, 12288))
        w = Tensor((12288,))
        self.assertEqual(
            n["fused_add_rmsnorm_fp8_quant"](
                x, x, w, group_size=32, round_to_pow2=True, tma_packed_scales=True
            ),
            ("fallback",),
        )
        self.assertNotIn("tma_packed_scales", n["fallbacks"][0][1])
        n = api_namespace()
        x = Tensor((1024, 24576))
        self.assertEqual(
            n["silu_and_mul_per_token_group_fp8_quant_dense_packed_fwd"](
                x,
                quant_group_size=32,
                scale_ue8m0=False,
                round_to_pow2=True,
                gemm1_alpha=1.702,
                gemm1_clamp_limit=7,
                tma_packed_scales=True,
            ),
            ("fallback",),
        )

    def test_math_and_legacy_branch_ast(self):
        if FROZEN is None:
            self.skipTest("--frozen-probe supplies immutable baseline bodies")
        pairs = (
            (
                "fused_add_rmsnorm_fp8_quant.py",
                "norm_baseline.py",
                [
                    "_fused_add_rmsnorm_fp8_quant_singlepass_kernel",
                    "_fused_add_rmsnorm_fp8_quant_dual_output_singlepass_kernel",
                ],
            ),
            (
                "activation.py",
                "silu_baseline.py",
                ["_silu_and_mul_post_quant_dense_packed_kernel"],
            ),
        )

        class OldBranch(ast.NodeTransformer):
            def visit_If(self, node):
                if (
                    isinstance(node.test, ast.Name)
                    and node.test.id == "TMA_PACKED_SCALES"
                ):
                    return self.visit(
                        ast.Module(body=node.orelse, type_ignores=[])
                    ).body
                if (
                    isinstance(node.test, ast.Compare)
                    and isinstance(node.test.left, ast.Name)
                    and node.test.left.id == "GROUPS_PER_CTA"
                ):
                    return self.visit(
                        ast.Module(body=node.orelse, type_ignores=[])
                    ).body
                return self.generic_visit(node)

        for source, frozen, names in pairs:
            baseline = {
                n.name: n
                for n in ast.parse((FROZEN / frozen).read_text()).body
                if isinstance(n, ast.FunctionDef)
            }
            current = {
                n.name: n
                for n in ast.parse((COMMON / source).read_text()).body
                if isinstance(n, ast.FunctionDef)
            }
            for name in names:
                node = OldBranch().visit(copy.deepcopy(current[name]))
                # New constexpr is representation-only; remove it to compare.
                new_args = {"TMA_PACKED_SCALES", "GROUPS_PER_CTA"}
                node.args.kw_defaults = [
                    d
                    for a, d in zip(node.args.kwonlyargs, node.args.kw_defaults)
                    if a.arg not in new_args
                ]
                node.args.kwonlyargs = [
                    a for a in node.args.kwonlyargs if a.arg not in new_args
                ]
                while node.args.args[-1].arg in new_args:
                    node.args.args.pop()
                    node.args.defaults.pop()
                self.assertEqual(
                    ast.dump(node, include_attributes=False),
                    ast.dump(baseline[name], include_attributes=False),
                )


@unittest.skipUnless(GPU, "parent-owned --gpu opt-in required")
class GpuContract(unittest.TestCase):
    # Enabled explicitly in main after CLI parsing, not by test discovery.
    pass


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--gpu", action="store_true")
    p.add_argument("--frozen-probe", type=Path)
    args, rest = p.parse_known_args()
    GPU, FROZEN = args.gpu, args.frozen_probe
    if GPU:
        from test_mxfp8_tma_scale_gpu import GpuContract

        GpuContract.frozen = FROZEN
    unittest.main(argv=[__file__] + rest)
