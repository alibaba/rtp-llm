"""Parent-owned opt-in production API CUDA gate, imported only with --gpu."""

import importlib.util
import subprocess
import sys
import unittest
from pathlib import Path

import deep_gemm
import torch
import triton

from rtp_llm.models_py.triton_kernels.common.activation import (
    silu_and_mul_per_token_group_fp8_quant_dense_packed_fwd,
)
from rtp_llm.models_py.triton_kernels.common.fused_add_rmsnorm_fp8_quant import (
    fused_add_rmsnorm_fp8_quant,
    fused_add_rmsnorm_fp8_quant_with_bf16_output,
)


class GpuContract(unittest.TestCase):
    frozen = None

    @classmethod
    def setUpClass(cls):
        if cls.frozen is None:
            raise RuntimeError("--gpu requires --frozen-probe for immutable baseline")
        cls.reference = {}
        for name in ("norm_baseline", "silu_baseline"):
            spec = importlib.util.spec_from_file_location(
                name, cls.frozen / (name + ".py")
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            cls.reference[name] = module

    def exact(self, a, b, label):
        self.assertEqual(a.shape, b.shape, label)
        self.assertEqual(a.dtype, b.dtype, label)
        self.assertTrue(
            torch.equal(
                a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8)
            ),
            label,
        )

    def test_api_frozen_and_dynamic_graph(self):
        torch.manual_seed(61004)
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            for kind, k in [
                ("norm", 3072),
                ("norm", 6144),
                ("norm", 8192),
                ("single", 6144),
                ("silu", 3072),
                ("silu", 12288),
            ]:
                shapes = (
                    (0, 1, 3, 16, 63, 64, 65, 79, 80, 81, 127, 128, 129)
                    if kind == "silu"
                    else (0, 1, 3, 16, 80, 128)
                )
                for m in shapes:
                    with self.subTest(kind=kind, k=k, m=m):
                        x = torch.randn(
                            (m, 2 * k if kind == "silu" else k),
                            device="cuda",
                            dtype=torch.bfloat16,
                        )
                        seed = x.clone()
                        groups = k // 32
                        residual = torch.randn(
                            (m, k), device="cuda", dtype=torch.bfloat16
                        )
                        rb, rd, rr = (
                            residual.clone(),
                            residual.clone(),
                            residual.clone(),
                        )
                        weight = torch.randn((k,), device="cuda", dtype=torch.bfloat16)
                        qref = torch.empty(
                            (m, k), device="cuda", dtype=torch.float8_e4m3fn
                        )
                        yref = torch.empty((m, k), device="cuda", dtype=torch.bfloat16)
                        sfref = torch.empty(
                            (m, groups), device="cuda", dtype=torch.float32
                        )
                        state = {}

                        def produce(direct):
                            if kind == "silu":
                                return silu_and_mul_per_token_group_fp8_quant_dense_packed_fwd(
                                    x,
                                    quant_group_size=32,
                                    scale_ue8m0=False,
                                    round_to_pow2=True,
                                    gemm1_alpha=1.702,
                                    gemm1_clamp_limit=7,
                                    tma_packed_scales=direct,
                                )
                            r = rd if direct else rb
                            r.copy_(residual)
                            fn = (
                                fused_add_rmsnorm_fp8_quant
                                if kind == "single"
                                else fused_add_rmsnorm_fp8_quant_with_bf16_output
                            )
                            return fn(
                                x,
                                r,
                                weight,
                                group_size=32,
                                round_to_pow2=True,
                                tma_packed_scales=direct,
                            )

                        def frozen():
                            if not m:
                                return
                            if kind == "silu":
                                self.reference[
                                    "silu_baseline"
                                ]._silu_and_mul_post_quant_dense_packed_kernel[
                                    (groups, m)
                                ](
                                    x,
                                    x.stride(0),
                                    qref,
                                    k,
                                    sfref,
                                    groups,
                                    1,
                                    k,
                                    448.0,
                                    -448.0,
                                    BLOCK_N=32,
                                    NUM_STAGE=2,
                                    SCALE_UE8M0=False,
                                    ROUND_POW2=True,
                                    GEMM1_ALPHA=1.702,
                                    GEMM1_CLAMP_LIMIT=7.0,
                                    num_warps=1,
                                )
                            else:
                                rr.copy_(residual)
                                mod = self.reference["norm_baseline"]
                                fn = (
                                    mod._fused_add_rmsnorm_fp8_quant_singlepass_kernel
                                    if kind == "single"
                                    else mod._fused_add_rmsnorm_fp8_quant_dual_output_singlepass_kernel
                                )
                                ptrs = (
                                    [x, rr, weight]
                                    + ([] if kind == "single" else [yref])
                                    + [qref, sfref]
                                )
                                strides = (
                                    [k, k] + ([] if kind == "single" else [k]) + [k]
                                )
                                fn[(m,)](
                                    *ptrs,
                                    k,
                                    1e-6,
                                    448.0,
                                    -448.0,
                                    *strides,
                                    groups,
                                    1,
                                    BLOCK_N=triton.next_power_of_2(k),
                                    GROUP_SIZE=32,
                                    SCALE_UE8M0=False,
                                    ROUND_POW2=True,
                                    num_warps=8
                                )

                        for pattern in ("random", "zero", "tiny", "large"):
                            x.copy_(seed)
                            if pattern == "zero":
                                x.zero_()
                            if pattern == "tiny":
                                x.mul_(1e-12)
                            if pattern == "large":
                                x.mul_(32)
                            a, b = produce(False), produce(True)
                            frozen()
                            stream.synchronize()
                            self.assertEqual(a[-1].dtype, torch.float32)
                            self.assertEqual(b[-1].dtype, torch.int32)
                            self.assertEqual(b[-1].shape, (m, k // 128))
                            self.exact(a[-2], b[-2], "FP8 default/direct")
                            if m:
                                self.exact(a[-2], qref, "FP8 frozen")
                                self.exact(a[-1], sfref, "FP32 scale frozen")
                                packed = deep_gemm.transform_sf_into_required_layout(
                                    a[-1], mn=m, k=k, recipe=(1, 32)
                                )
                                self.exact(packed, b[-1], "packed words")
                                self.assertEqual(b[-1].stride(), packed.stride())
                            if kind != "silu":
                                self.exact(rb, rd, "residual")
                                self.exact(rb, rr, "frozen residual")
                            if kind == "norm":
                                self.exact(a[0], b[0], "BF16 norm")
                                self.exact(a[0], yref, "BF16 frozen")
                        if not m:
                            continue
                        n = (
                            9856
                            if kind == "norm" and k == 6144
                            else (24576 if kind == "single" else 6144)
                        )
                        w = torch.randn((n, k), device="cuda").to(torch.float8_e4m3fn)
                        sw = torch.ones((n, groups), device="cuda", dtype=torch.float32)
                        wsp = deep_gemm.transform_sf_into_required_layout(
                            sw, mn=n, k=k, recipe=(1, 32)
                        )
                        ob, od = torch.empty(
                            (m, n), device="cuda", dtype=torch.bfloat16
                        ), torch.empty((m, n), device="cuda", dtype=torch.bfloat16)

                        def chain(direct):
                            out = produce(direct)
                            state[direct] = out
                            scale = (
                                out[-1]
                                if direct
                                else deep_gemm.transform_sf_into_required_layout(
                                    out[-1], mn=m, k=k, recipe=(1, 32)
                                )
                            )
                            deep_gemm.fp8_fp4_gemm_nt(
                                (out[-2], scale),
                                (w, wsp),
                                od if direct else ob,
                                recipe_a=(1, 32),
                                recipe_b=(1, 32),
                                disable_ue8m0_cast=True,
                            )

                        graphs = []
                        for direct in (False, True):
                            for _ in range(3):
                                chain(direct)
                            graph = torch.cuda.CUDAGraph()
                            with torch.cuda.graph(graph, stream=stream):
                                chain(direct)
                            graphs.append(graph)
                        for turn in range(3):
                            x.copy_(seed * (turn + 1))
                            for graph in graphs:
                                graph.replay()
                            frozen()
                            stream.synchronize()
                            self.exact(state[False][-2], state[True][-2], "graph FP8")
                            self.exact(state[True][-2], qref, "graph FP8 frozen")
                            self.exact(ob, od, "graph GEMM BF16")
                            if kind == "norm":
                                self.exact(
                                    state[False][0], state[True][0], "graph norm"
                                )
                            if kind != "silu":
                                self.exact(rb, rd, "graph residual")
                            packed = deep_gemm.transform_sf_into_required_layout(
                                sfref, mn=m, k=k, recipe=(1, 32)
                            )
                            self.exact(packed, state[True][-1], "graph packed frozen")
        stream.synchronize()

    def test_defaults_and_fallbacks(self):
        # Existing UE8M0 branch and unsupported/new-request fallbacks stay
        # byte-identical; do not substitute a different math reference.
        for rows, k, group, ue, pow2 in (
            (3, 6144, 128, True, False),
            (3, 6144, 128, False, False),
            (3, 6176, 32, False, True),
            (1024, 6144, 32, False, True),
            (3, 12288, 32, False, True),
        ):
            x = torch.randn((rows, k), device="cuda", dtype=torch.bfloat16)
            r = torch.randn_like(x)
            w = torch.ones(k, device="cuda", dtype=torch.bfloat16)
            ra, rb = r.clone(), r.clone()
            a = fused_add_rmsnorm_fp8_quant(
                x, ra, w, group_size=group, scale_ue8m0=ue, round_to_pow2=pow2
            )
            b = fused_add_rmsnorm_fp8_quant(
                x,
                rb,
                w,
                group_size=group,
                scale_ue8m0=ue,
                round_to_pow2=pow2,
                tma_packed_scales=True,
            )
            self.exact(a[0], b[0], "legacy FP8")
            self.exact(a[1], b[1], "legacy scales")
            self.exact(ra, rb, "legacy residual")
        for rows, k, group, ue, pow2 in (
            (3, 3072, 128, True, False),
            (3, 3104, 32, False, True),
            (1024, 12288, 32, False, True),
        ):
            x = torch.randn((rows, 2 * k), device="cuda", dtype=torch.bfloat16)
            kw = dict(
                quant_group_size=group,
                scale_ue8m0=ue,
                round_to_pow2=pow2,
                gemm1_alpha=1.702,
                gemm1_clamp_limit=7.0,
            )
            a = silu_and_mul_per_token_group_fp8_quant_dense_packed_fwd(x, **kw)
            b = silu_and_mul_per_token_group_fp8_quant_dense_packed_fwd(
                x, **kw, tma_packed_scales=True
            )
            self.exact(a[0], b[0], "legacy SiLU FP8")
            self.exact(a[1], b[1], "legacy SiLU scale")

    def test_z_invalid_scales_fail_in_isolated_contexts(self):
        # Invalid-scale traps poison the child contexts only. Never catch and
        # continue CUDA operations in the normal comparison/profiler context.
        child = Path(__file__).with_name("mxfp8_invalid_scale_child.py")
        for bits in (0x7FC00000, 0x80000000, 0x3F800001):
            for legacy in (False, True):
                command = [sys.executable, "-B", str(child), "--bits", hex(bits)] + (
                    ["--legacy"] if legacy else []
                )
                result = subprocess.run(
                    command, text=True, capture_output=True, timeout=120
                )
                self.assertNotEqual(result.returncode, 0, (bits, legacy))
                self.assertNotIn("unexpectedly accepted", result.stderr, (bits, legacy))
                self.assertRegex(
                    result.stderr, r"CUDA|cuda|illegal instruction|device-side assert"
                )
