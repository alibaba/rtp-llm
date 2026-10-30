"""Inference CPU-path contracts: allocation, launch, hooks and stream ordering."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from rtp_llm.models_py.kernels.cuda import prefill_cpu_ops as ops
from rtp_llm.models_py.kernels.cuda.fp8_kernel import (
    create_per_token_group_quant_fp8_output_scale,
)
from rtp_llm.models_py.triton_kernels.common import fast_launch, mla_norm_pair
from rtp_llm.models_py.triton_kernels.common.fused_add_rmsnorm_fp8_quant import (
    fused_add_rmsnorm_fp8_quant_with_bf16_output,
)


class PrefillOptimizationSwitchTest(unittest.TestCase):
    def test_startup_defaults_and_bundle(self):
        import os
        import subprocess
        import sys

        code = """
from rtp_llm.models_py import prefill_optimization as opt
from rtp_llm.models_py.kernels.cuda import prefill_cpu_ops as ops
from rtp_llm.models_py.triton_kernels.common import fast_launch
from rtp_llm.models_py.modules.base.cuda.indexer_op import _fp8_prefill_topk_canonicalize
import os
expected = os.environ.get('RTP_GLM5_PREFILL_OPTIMIZE', '0') == '1'
assert opt.ENABLED == expected
assert all(getattr(ops, n) == expected for n in ('NATIVE_PLAN', 'NATIVE_ALLOC', 'DIRECT_ROUTER', 'NATIVE_LAUNCH', 'NATIVE_ROUTER'))
assert fast_launch.ENABLED == expected
assert _fp8_prefill_topk_canonicalize() == expected
"""
        for value in (None, "0", "1"):
            env = os.environ.copy()
            env.pop("RTP_GLM5_PREFILL_OPTIMIZE", None)
            env["DSV4_INDEXER_TOPK_CANONICALIZE"] = "0"
            if value is not None:
                env["RTP_GLM5_PREFILL_OPTIMIZE"] = value
            subprocess.run([sys.executable, "-c", code], env=env, check=True)


class PrefillHookContractTest(unittest.TestCase):
    def test_empty_triton_hook_chain_is_not_an_active_profiler(self):
        if fast_launch._hook_chain_type is None:
            self.skipTest("Triton HookChain API unavailable")
        chain = fast_launch._hook_chain_type()
        with patch.object(
            fast_launch.knobs.runtime, "launch_enter_hook", chain
        ), patch.object(fast_launch.knobs.runtime, "launch_exit_hook", None):
            self.assertTrue(fast_launch.hooks_idle())
            callback = lambda *args: None
            chain.add(callback)
            self.assertFalse(fast_launch.hooks_idle())
            chain.remove(callback)
            self.assertTrue(fast_launch.hooks_idle())

    def test_module_hook_and_compiled_call_fallback(self):
        module = torch.nn.Identity()
        self.assertTrue(ops.no_module_hooks(module))
        handle = module.register_forward_hook(lambda *args: None)
        self.assertFalse(ops.no_module_hooks(module))
        handle.remove()
        self.assertTrue(ops.no_module_hooks(module))
        module._compiled_call_impl = lambda x: x
        self.assertFalse(ops.no_module_hooks(module))

    def test_global_hook_fallback(self):
        module = torch.nn.Identity()
        handle = torch.nn.modules.module.register_module_forward_pre_hook(
            lambda *args: None
        )
        try:
            self.assertFalse(ops.no_module_hooks(module))
        finally:
            handle.remove()


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class PrefillCpuGpuTest(unittest.TestCase):
    def test_native_executor_retains_compiled_module_owner(self):
        import gc
        import weakref

        class Owner:
            pass

        library = ops.native()
        owner = Owner()
        reference = weakref.ref(owner)
        route = library.NativeLauncher(0, 0, 1, [0, 1, 2, 3], [0, 0, 0, 0], owner)
        pad = library.NativeLauncher(0, 0, 4, [0, 1, 2, 3], [0, 0, 1, 1], None)
        executor = library.NativeRouter(route, pad)
        del route, owner
        gc.collect()
        self.assertIsNotNone(reference())
        del executor
        gc.collect()
        self.assertIsNone(reference())

    def test_native_prefix_plan_against_scalar_oracle(self):
        from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.test.cp_kv_prefetch_test import (
            PrefixPlanTest,
        )

        oracle = PrefixPlanTest()
        with patch.object(ops, "NATIVE_PLAN", True):
            oracle.test_partial_owners_empty_requests_and_zero_owned_ranks()
            oracle.test_random_layouts_against_scalar_reference()
            oracle.test_invalid_plan_rejected()

    def test_native_key_distinguishes_scalar_types(self):
        library = ops.native()
        self.assertNotEqual(
            library.specialization_key((1,)), library.specialization_key((1.0,))
        )
        self.assertNotEqual(
            library.specialization_key((True,)), library.specialization_key((1,))
        )

    def test_native_router_fixed_m_and_multiple_chunks(self):
        from rtp_llm.models_py.model_desc.generic_moe import GenericMoeLayer
        from rtp_llm.models_py.modules.base.cuda import native_router
        from rtp_llm.models_py.modules.base.cuda.select_topk import GroupTopK

        class Experts(torch.nn.Module):
            topk_ids_dtype = torch.int64

            def forward(self, hidden_states, topk_weights, topk_ids, activation):
                return topk_weights, topk_ids

        layer = GenericMoeLayer.__new__(GenericMoeLayer)
        torch.nn.Module.__init__(layer)
        layer.gate_chunk_rows = 8192
        layer.gate = torch.nn.Linear(
            512, 256, bias=False, device="cuda", dtype=torch.bfloat16
        ).requires_grad_(False)
        layer.correction_bias = torch.randn(256, device="cuda", dtype=torch.float32)
        layer.group_topk = GroupTopK()
        layer.top_k = 8
        layer.config = SimpleNamespace(
            moe_n_group=1,
            moe_topk_group=1,
            has_moe_norm=True,
            routed_scaling_factor=2.5,
        )
        layer.fused_moe = Experts()
        layer.fake_balance_expert = None
        layer.ep_size = 8
        layer.shared_expert = None
        layer.shared_expert_gate = None
        from rtp_llm.models_py.modules.factory.linear.impl.cuda.f16_linear import (
            CudaF16Linear,
        )

        for transposed, ids_dtype in ((False, torch.int64), (True, torch.int32)):
            if transposed:
                layer.gate = CudaF16Linear(
                    torch.randn(512, 256, device="cuda", dtype=torch.bfloat16)
                )
                self.assertFalse(layer.gate.weight.is_contiguous())
            layer.fused_moe.topk_ids_dtype = ids_dtype
            for rows in (1, 17, 2024, 8193):
                x = torch.randn(rows, 512, device="cuda", dtype=torch.bfloat16)
                with patch.object(ops, "NATIVE_ROUTER", False):
                    expected = layer(x)
                with patch.object(native_router, "ENABLED", True):
                    result = native_router.try_route(layer, x)
                    self.assertIsNotNone(result)
                    self.assertTrue(
                        all(torch.equal(a, b) for a, b in zip(result, expected))
                    )
                    with patch.object(ops, "NATIVE_ROUTER", True):
                        got = layer(x)
                        self.assertTrue(
                            all(torch.equal(a, b) for a, b in zip(got, expected))
                        )
                    handle = layer.gate.register_forward_hook(lambda *args: None)
                    try:
                        self.assertIsNone(native_router.try_route(layer, x))
                    finally:
                        handle.remove()

    def test_native_allocation_layout_and_lifetime(self):
        native = ops.native()
        self.assertIsNotNone(native)
        for rows in (0, 1, 17, 2024, 4093):
            for width in (512, 2048, 6144):
                for ue8 in (False, True):
                    bf16, fp8, scale, kv = native.norm_outputs(
                        rows, width, 512, 0, ue8, True
                    )
                    ref = create_per_token_group_quant_fp8_output_scale(
                        (rows, width), "cuda", 128, True, True, ue8
                    )
                    self.assertEqual(scale.shape, ref.shape)
                    self.assertEqual(scale.stride(), ref.stride())
                    self.assertEqual(scale.storage_offset(), ref.storage_offset())
                    self.assertEqual(
                        scale.untyped_storage().nbytes(), ref.untyped_storage().nbytes()
                    )
                    self.assertEqual(bf16.dtype, torch.bfloat16)
                    self.assertEqual(fp8.dtype, torch.float8_e4m3fn)
                    self.assertEqual(kv.shape, (rows, 512))
                    if rows:
                        self.assertEqual(
                            len({x.data_ptr() for x in (bf16, fp8, scale, kv)}), 4
                        )
                    unused, q, s, unused_kv = native.norm_outputs(
                        rows, width, 0, 0, ue8, False
                    )
                    self.assertIsNone(unused)
                    self.assertIsNone(unused_kv)
                    self.assertEqual(q.shape, fp8.shape)
                    self.assertEqual(s.stride(), ref.stride())

    def test_norm_matrix_and_nondefault_stream(self):
        torch.manual_seed(1927)
        for rows in (1, 17, 2024):
            source = torch.randn(rows, 2624, device="cuda", dtype=torch.bfloat16)
            qw = torch.randn(2048, device="cuda", dtype=torch.bfloat16)
            kw = torch.randn(512, device="cuda", dtype=torch.bfloat16)
            args = (source[:, :2048], source[:, 2048:2560], qw, kw, 1e-5, 1e-6)
            with patch.object(ops, "NATIVE_ALLOC", False), patch.object(
                fast_launch, "ENABLED", False
            ):
                expected = mla_norm_pair.try_mla_norm_pair(*args)
            for native, fast in ((True, False), (False, True), (True, True)):
                with patch.object(ops, "NATIVE_ALLOC", native), patch.object(
                    fast_launch, "ENABLED", fast
                ):
                    stream = torch.cuda.Stream()
                    stream.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(stream):
                        for _ in range(3):
                            got = mla_norm_pair.try_mla_norm_pair(*args)
                    torch.cuda.current_stream().wait_stream(stream)
                    for a, b in zip(got, expected):
                        self.assertTrue(
                            torch.equal(
                                a.contiguous().view(torch.uint8),
                                b.contiguous().view(torch.uint8),
                            )
                        )

    def test_addnorm_exact_after_cache_warmup(self):
        for rows in (1, 17, 2024):
            h = torch.randn(rows, 6144, device="cuda", dtype=torch.bfloat16)
            residual = torch.randn_like(h)
            weight = torch.randn(6144, device="cuda", dtype=torch.bfloat16)
            for ue8 in (True, False):
                with patch.object(ops, "NATIVE_ALLOC", False), patch.object(
                    fast_launch, "ENABLED", False
                ):
                    reference_residual = residual.clone()
                    expected = fused_add_rmsnorm_fp8_quant_with_bf16_output(
                        h, reference_residual, weight, scale_ue8m0=ue8
                    )
                for _ in range(3):
                    with patch.object(ops, "NATIVE_ALLOC", True), patch.object(
                        fast_launch, "ENABLED", True
                    ):
                        r = residual.clone()
                        got = fused_add_rmsnorm_fp8_quant_with_bf16_output(
                            h, r, weight, scale_ue8m0=ue8
                        )
                    self.assertTrue(
                        torch.equal(
                            r.view(torch.uint8), reference_residual.view(torch.uint8)
                        )
                    )
                    for a, b in zip(got, expected):
                        self.assertTrue(
                            torch.equal(
                                a.contiguous().view(torch.uint8),
                                b.contiguous().view(torch.uint8),
                            )
                        )

    def test_direct_router_respects_module_hook(self):
        from rtp_llm.models_py.model_desc.generic_moe import GenericMoeLayer
        from rtp_llm.models_py.modules.base.cuda.select_topk import GroupTopK

        class Experts(torch.nn.Module):
            topk_ids_dtype = torch.int64

            def forward(self, hidden_states, topk_weights, topk_ids, activation):
                return topk_weights, topk_ids

        layer = GenericMoeLayer.__new__(GenericMoeLayer)
        torch.nn.Module.__init__(layer)
        layer.gate_chunk_rows = 0
        layer.gate = torch.nn.Identity()
        layer.correction_bias = torch.randn(256, device="cuda", dtype=torch.float32)
        layer.group_topk = GroupTopK()
        layer.top_k = 8
        layer.config = SimpleNamespace(
            moe_n_group=1,
            moe_topk_group=1,
            has_moe_norm=True,
            routed_scaling_factor=2.5,
        )
        layer.fused_moe = Experts()
        layer.fake_balance_expert = None
        layer.ep_size = 8
        layer.shared_expert = None
        layer.shared_expert_gate = None
        x = torch.randn(17, 256, device="cuda", dtype=torch.bfloat16)
        with patch.object(ops, "DIRECT_ROUTER", False), patch.object(
            ops, "NATIVE_ALLOC", False
        ):
            expected = layer(x)
        with patch.object(ops, "DIRECT_ROUTER", True), patch.object(
            ops, "NATIVE_ALLOC", True
        ):
            got = layer(x)
            self.assertTrue(all(torch.equal(a, b) for a, b in zip(got, expected)))
            calls = []
            handle = layer.group_topk.register_forward_hook(
                lambda *args: calls.append(True)
            )
            try:
                got = layer(x)
            finally:
                handle.remove()
            self.assertEqual(calls, [True])
            self.assertTrue(all(torch.equal(a, b) for a, b in zip(got, expected)))


if __name__ == "__main__":
    unittest.main()
