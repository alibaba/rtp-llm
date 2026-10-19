import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import torch


class MHCLazyBackwardTest(unittest.TestCase):
    def test_delayed_entry_uses_standalone_and_predecessor_uses_fused(self):
        from rtp_llm.models_py.modules.dsv41.hc import delayed, v41_prenorm

        class CudaMetadataTensor(torch.Tensor):
            @property
            def is_cuda(self):
                return True

        fn = torch.zeros(24, 20480)
        kwargs = dict(
            dim=5120, hc_mult=4, hc_sinkhorn_iters=20, norm_eps=1e-6, hc_eps=1e-6
        )
        first = delayed.DelayedHCUnit(fn, torch.zeros(24), torch.ones(3), **kwargs)
        following = delayed.DelayedHCUnit(fn, torch.zeros(24), torch.ones(3), **kwargs)
        following.set_previous(first)
        for tokens in (160, 192):
            with self.subTest(tokens=tokens):
                x = torch.zeros(1, tokens, 4, 5120, dtype=torch.bfloat16).as_subclass(
                    CudaMetadataTensor
                )
                pre = torch.ones(1, tokens, 4, 1)
                post, comb = pre.clone(), torch.ones(1, tokens, 4, 4)
                ops = Mock()
                ops.mhc_pre_norm_fn.return_value = torch.zeros(1, tokens, 24)
                ops.mhc_pre_split_mixes.return_value = pre, post, comb
                ops.sinkhorn_normalize.return_value = comb
                ops.mhc_pre_big_fuse.return_value = (
                    pre.squeeze(-1),
                    post,
                    comb,
                    x[:, :, 0, :],
                )
                with torch.inference_mode(), patch.object(
                    delayed, "_tile_ops", return_value=ops
                ), patch.object(
                    torch.cuda, "get_device_capability", return_value=(10, 0)
                ), patch.object(
                    v41_prenorm, "prenorm", return_value=None
                ), patch.dict(
                    "os.environ", {"DSV4_MHC_PRE_GEMM_BACKEND": "tilelang_single"}
                ):
                    first.pre(x)
                    ops.mhc_pre_norm_fn.assert_called_once()
                    ops.mhc_pre_big_fuse.assert_not_called()
                    following.pre(x)
                    ops.mhc_pre_norm_fn.assert_called_once()
                    ops.mhc_pre_big_fuse.assert_called_once()
                    self.assertIsNone(
                        ops.mhc_pre_big_fuse.call_args.kwargs["gemm_backend"]
                    )
                    self.assertIs(
                        ops.mhc_pre_big_fuse.call_args.kwargs["previous_pre_mix"],
                        first.pre_mix_out,
                    )

    def test_delayed_big_fuse_optional_deepgemm_gate(self):
        from rtp_llm.models_py.modules.dsv41.hc import v41_prenorm
        from rtp_llm.models_py.modules.dsv41.hc.delayed import can_fuse_big_pre

        with patch.dict("os.environ", {"DSV4_MHC_PRE_GEMM_BACKEND": "deepgemm"}):
            with patch.object(v41_prenorm, "_has_prenorm_gemm", return_value=False):
                self.assertFalse(can_fuse_big_pre())
        with patch.dict("os.environ", {"DSV4_MHC_PRE_GEMM_BACKEND": "tilelang_single"}):
            self.assertTrue(can_fuse_big_pre())

    def test_delayed_prefill_big_fuse_matches_separate_chain(self):
        if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
            self.skipTest("requires Blackwell CUDA")
        from rtp_llm.models_py.modules.dsv41.hc.delayed import _tile_ops

        ops = _tile_ops()
        torch.manual_seed(17)
        fn = torch.randn(24, 20480, device="cuda") / 20480**0.5
        scale = torch.randn(3, device="cuda") * 0.1
        base = torch.randn(24, device="cuda") * 0.1
        with torch.inference_mode():
            for rows in (1, 32, 64, 65, 2048):
                residual = torch.randn(
                    1, rows, 4, 5120, device="cuda", dtype=torch.bfloat16
                )
                previous = torch.randn(1, rows, 4, device="cuda")
                mixes = ops.mhc_pre_norm_fn(residual, fn, None, 1e-6, n_splits=1)
                pre, post, comb = ops.mhc_pre_split_mixes(
                    mixes, scale, base, 4, 2.0, 1e-6
                )
                comb = ops.sinkhorn_normalize(comb, repeat=20, eps=1e-6)
                readout = ops.mhc_pre_apply_mix(
                    residual, previous.unsqueeze(-1).contiguous()
                )
                fused = ops.mhc_pre_big_fuse(
                    residual,
                    fn,
                    scale,
                    base,
                    1e-6,
                    1e-6,
                    1e-6,
                    2.0,
                    20,
                    previous_pre_mix=previous,
                )
                for expected, actual in zip((pre, post, comb), fused[:3]):
                    torch.testing.assert_close(
                        actual.reshape(-1), expected.reshape(-1), rtol=1e-5, atol=1e-5
                    )
                torch.testing.assert_close(fused[3], readout, rtol=0, atol=0)

    def test_delayed_small_pre_uses_fused_path(self):
        if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
            self.skipTest("requires Blackwell CUDA")
        from rtp_llm.models_py.modules.dsv41.hc.delayed import DelayedHCUnit, _tile_ops
        from rtp_llm.models_py.modules.dsv41.hc.v41_prenorm import (
            _has_prenorm_gemm,
            prepare_tf32_weight,
        )

        torch.manual_seed(29)
        fn = torch.randn(24, 20480, device="cuda") / 20480**0.5
        base = torch.randn(24, device="cuda") * 0.1
        scale = torch.randn(3, device="cuda") * 0.1
        kwargs = dict(
            dim=5120, hc_mult=4, hc_sinkhorn_iters=20, norm_eps=1e-6, hc_eps=1e-6
        )
        previous = DelayedHCUnit(fn, base, scale, **kwargs)
        unit = DelayedHCUnit(fn, base, scale, **kwargs)
        unit.set_previous(previous)
        ops = _tile_ops()
        with torch.inference_mode():
            for rows in (1, 64):
                x = torch.randn(1, rows, 4, 5120, device="cuda", dtype=torch.bfloat16)
                previous.pre_mix_out = torch.randn(1, rows, 4, device="cuda")
                with patch(
                    "rtp_llm.models_py.modules.dsv41.hc.delayed.can_fuse_big_pre",
                    return_value=False,
                ):
                    expected = unit.pre(x)
                with patch.dict(
                    "os.environ", {"DSV4_MHC_PRE_GEMM_BACKEND": "tilelang_single"}
                ):
                    with patch.object(
                        ops, "mhc_pre_big_fuse", wraps=ops.mhc_pre_big_fuse
                    ) as fused:
                        actual = unit.pre(x)
                fused.assert_called_once()
                if _has_prenorm_gemm():
                    self.assertEqual(fused.call_args.kwargs["gemm_backend"], "deepgemm")
                    self.assertEqual(
                        fused.call_args.args[1].data_ptr(),
                        prepare_tf32_weight(unit.fn).data_ptr(),
                    )
                torch.testing.assert_close(actual[0], expected[0], rtol=0, atol=0)
                for got, ref in zip(actual[1:], expected[1:]):
                    torch.testing.assert_close(got, ref, rtol=1e-5, atol=1e-5)

    def test_delayed_nonstandard_shape_keeps_separate_path(self):
        if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
            self.skipTest("requires Blackwell CUDA")
        from rtp_llm.models_py.modules.dsv41.hc.delayed import DelayedHCUnit, _tile_ops

        fn = torch.randn(24, 512, device="cuda") / 512**0.5
        base = torch.randn(24, device="cuda") * 0.1
        scale = torch.randn(3, device="cuda") * 0.1
        kwargs = dict(
            dim=128, hc_mult=4, hc_sinkhorn_iters=20, norm_eps=1e-6, hc_eps=1e-6
        )
        previous = DelayedHCUnit(fn, base, scale, **kwargs)
        unit = DelayedHCUnit(fn, base, scale, **kwargs)
        unit.set_previous(previous)
        previous.pre_mix_out = torch.randn(1, 2, 4, device="cuda")
        x = torch.randn(1, 2, 4, 128, device="cuda", dtype=torch.bfloat16)
        with torch.inference_mode(), patch.dict(
            "os.environ", {"DSV4_MHC_PRE_GEMM_BACKEND": "tilelang_single"}
        ), patch.object(_tile_ops(), "mhc_pre_big_fuse") as fused:
            unit.pre(x)
        fused.assert_not_called()

    def load_kernels(self, name, kernels):
        root = Path(__file__).resolve().parents[3] / "3rdparty/tile_kernels"
        kernel = types.ModuleType(f"_test_tile.mhc.{name}_kernel")
        for key, value in kernels.items():
            setattr(kernel, key, value)
        config = types.ModuleType("_test_tile.config")
        config.get_num_sms = Mock(return_value=2)
        spec = importlib.util.spec_from_file_location(
            f"_test_tile.modeling.mhc.ops.{name}",
            root / f"modeling/mhc/ops/{name}.py",
        )
        module = importlib.util.module_from_spec(spec)
        with patch.dict(
            sys.modules, {kernel.__name__: kernel, config.__name__: config}
        ):
            spec.loader.exec_module(module)
        return module

    def load_op(self, name, forward, backward):
        compiler = Mock(return_value=backward)
        module = self.load_kernels(
            name,
            {
                f"_mhc_{name}_fwd": Mock(return_value=forward),
                f"_mhc_{name}_bwd": compiler,
            },
        )
        return module, compiler

    def load_norm(self, **overrides):
        kernels = {
            name: Mock()
            for name in (
                "_mhc_fn_normw_merge_bwd",
                "_mhc_fn_normw_merge_fwd",
                "_mhc_pre_norm_fn_bwd_mul",
                "_mhc_pre_norm_fn_bwd_norm",
                "_mhc_pre_norm_fn_fwd_mul",
                "_mhc_pre_norm_fn_fwd_norm",
            )
        }
        kernels["round_to_tf32"] = lambda x: x
        kernels.update(overrides)
        return self.load_kernels("norm_fn", kernels)

    def test_tilelang_split_planner_and_device_cache(self):
        op = self.load_norm()
        for tokens, expected in (
            (0, 1),
            (65, 1),
            (95, 1),
            (96, 20),
            (160, 20),
            (192, 20),
            (256, 16),
            (320, 10),
            (480, 8),
            (608, 5),
            (960, 4),
            (1216, 2),
            (2400, 1),
            (131072, 1),
        ):
            with self.subTest(tokens=tokens):
                self.assertEqual(op._tilelang_split_plan(tokens, 20480, 148), expected)
        for hidden in (0, 255, 256, 257):
            self.assertEqual(op._tilelang_split_plan(96, hidden, 148), 1)
        for hidden in (512, 768, 20480, 24576):
            for tokens in range(32, 4097, 32):
                splits = op._tilelang_split_plan(tokens, hidden, 132)
                self.assertEqual(hidden % (splits * 256), 0)
                self.assertLessEqual(splits, 20)
                if splits > 1:
                    self.assertLessEqual(tokens // 32 * splits, 132)
        with patch.object(torch.cuda, "get_device_properties") as properties:
            properties.side_effect = [
                types.SimpleNamespace(multi_processor_count=148),
                types.SimpleNamespace(multi_processor_count=16),
            ]
            for index, expected in ((0, 20), (0, 20), (1, 2)):
                x = types.SimpleNamespace(
                    is_cuda=True,
                    device=torch.device("cuda", index),
                    numel=lambda: 192 * 20480,
                )
                self.assertEqual(op._tilelang_num_splits(x, 20480), expected)
            self.assertEqual(properties.call_count, 2)
            self.assertEqual(op._tilelang_num_splits(torch.empty(1), 20480), 1)
            self.assertEqual(properties.call_count, 2)

    def test_tilelang_representatives_cover_only_reachable_split_keys(self):
        op = self.load_norm()
        for sms in (1, 16, 132, 148):
            for hidden in (256, 768, 20480, 24576):
                for minimum in (1, 65, 96):
                    for maximum in (0, 64, 65, 95, 96, 192, 256, 1024, 4096):
                        with self.subTest(
                            sms=sms, hidden=hidden, lo=minimum, hi=maximum
                        ):
                            reps = op._tilelang_representative_ms(
                                maximum, hidden, sms, min_m=minimum
                            )

                            def reference(m):
                                if m % 32:
                                    return 1
                                return max(
                                    s
                                    for s in range(1, 21)
                                    if (hidden // 256) % s == 0
                                    and (s == 1 or s * (m // 32) <= sms)
                                )

                            expected = {
                                reference(m) for m in range(minimum, maximum + 1)
                            }
                            self.assertEqual({reference(m) for m in reps}, expected)
                            self.assertEqual(len(reps), len(expected))
                            self.assertTrue(all(minimum <= m <= maximum for m in reps))
        self.assertEqual(
            op._tilelang_representative_ms(192, 20480, 148, min_m=65), (65, 96)
        )
        self.assertEqual(
            op._tilelang_representative_ms(131072, 20480, 148, min_m=65),
            (65, 96, 256, 320, 480, 608, 960, 1216),
        )

    def test_split_major_scratch_is_exact_and_reduced_with_full_hidden_size(self):
        for splits in (1, 2, 4, 5, 8, 10, 16, 20):
            with self.subTest(splits=splits):
                hidden, tokens = 20480, 32
                x = torch.randn(2, tokens // 2, 4, hidden // 4, dtype=torch.bfloat16)
                fn = torch.randn(24, hidden) / hidden**0.5

                def multiply(x, fn, output, sqrsum):
                    self.assertEqual(output.shape, (splits, tokens, 24))
                    self.assertEqual(sqrsum.shape, (splits, tokens))
                    self.assertEqual(
                        output.untyped_storage().nbytes(), splits * tokens * 24 * 4
                    )
                    self.assertEqual(
                        sqrsum.untyped_storage().nbytes(), splits * tokens * 4
                    )
                    for i in range(splits):
                        part = slice(i * hidden // splits, (i + 1) * hidden // splits)
                        output[i].copy_(x[:, part].float() @ fn[:, part].T)
                        sqrsum[i].copy_(x[:, part].float().square().sum(-1))

                def normalize(mul_splits, sqr_splits, mul, sqr, output):
                    mul.copy_(mul_splits.sum(0))
                    sqr.copy_(sqr_splits.sum(0))
                    output.copy_(mul[:, 0] * torch.rsqrt(sqr / hidden + 1e-6))

                op = self.load_norm(
                    _mhc_pre_norm_fn_fwd_mul=Mock(return_value=multiply),
                    _mhc_pre_norm_fn_fwd_norm=Mock(return_value=normalize),
                )
                with patch.object(op, "_tilelang_num_splits", return_value=splits):
                    actual = op.MHCPreNormFn.forward(None, x, fn, 1e-6, False, 128)
                flat = x.flatten(-2).float()
                expected = (flat @ fn.T) * torch.rsqrt(
                    flat.square().mean(-1, keepdim=True) + 1e-6
                )
                torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)
                op._mhc_pre_norm_fn_fwd_mul.assert_called_once_with(
                    24, splits, hidden // splits
                )
                op._mhc_pre_norm_fn_fwd_norm.assert_called_once_with(
                    24, 1, hidden, 1e-6, splits
                )

    def test_fused_tilelang_backend_stays_single_split(self):
        mul = Mock()
        norm = types.ModuleType("_test_tile.mhc.norm_fn_kernel")
        norm.round_to_tf32 = lambda x: x
        norm._mhc_pre_norm_fn_fwd_mul = Mock(return_value=mul)
        with patch.dict(sys.modules, {norm.__name__: norm}):
            op = self.load_kernels("pre_big_fuse", {"_mhc_pre_big_fuse": Mock()})
        partial = torch.empty(16, 192, 24)
        square = torch.empty(16, 192)
        self.assertEqual(
            op._run_tilelang_single_gemm(
                torch.empty(192, 4, 5120, dtype=torch.bfloat16),
                torch.empty(24, 20480),
                partial,
                square,
                24,
                20480,
            ),
            1,
        )
        norm._mhc_pre_norm_fn_fwd_mul.assert_called_once_with(24, 1, 20480)
        self.assertEqual(mul.call_args.args[2].shape, (1, 192, 24))
        self.assertEqual(mul.call_args.args[3].shape, (1, 192))
        self.assertEqual(mul.call_args.args[2].data_ptr(), partial.data_ptr())
        self.assertEqual(mul.call_args.args[3].data_ptr(), square.data_ptr())

    def test_sinkhorn_compiles_backward_only_when_differentiated(self):
        def forward(x, output):
            output.copy_(x)

        def backward(grad, x, output):
            output.copy_(grad)

        op, compiler = self.load_op("sinkhorn", forward, backward)
        x = torch.randn(2, 4, 4, requires_grad=True)
        with torch.inference_mode(), patch.object(
            op._SinkhornNormalize,
            "apply",
            side_effect=AssertionError("autograd dispatch"),
        ):
            torch.testing.assert_close(op.sinkhorn_normalize(x, 20, 1e-5), x)
        compiler.assert_not_called()
        output = op.sinkhorn_normalize(x, 20, 1e-5)
        compiler.assert_not_called()
        output.sum().backward()
        compiler.assert_called_once_with(4, 32, 20, 1e-5)
        torch.testing.assert_close(x.grad, torch.ones_like(x))

    def test_pre_apply_preserves_forward_and_backward_arguments(self):
        def forward(x, mix, output):
            output.copy_((x * mix.unsqueeze(-1)).sum(-2))

        def backward(grad, x, mix, x_grad):
            x_grad.copy_(grad.unsqueeze(-2) * mix.unsqueeze(-1))
            return (grad.unsqueeze(-2) * x).sum(-1)

        op, compiler = self.load_op("pre_apply_mix", forward, backward)
        x = torch.arange(24, dtype=torch.float32).reshape(1, 2, 4, 3).requires_grad_()
        mix = torch.ones(1, 2, 4, 1, requires_grad=True)
        with torch.inference_mode(), patch.object(
            op.MHCPreApplyMix, "apply", side_effect=AssertionError("autograd dispatch")
        ):
            out = torch.empty(1, 2, 3, dtype=torch.bfloat16)
            result = op.mhc_pre_apply_mix(x, mix, out)
            self.assertEqual(result.data_ptr(), out.data_ptr())
            torch.testing.assert_close(result, x.sum(-2).to(torch.bfloat16))
        compiler.assert_not_called()
        result = op.mhc_pre_apply_mix(x, mix)
        compiler.assert_not_called()
        result.sum().backward()
        compiler.assert_called_once_with(4, 3)
        torch.testing.assert_close(x.grad, torch.ones_like(x))
        torch.testing.assert_close(mix.grad, x.detach().sum(-1, keepdim=True))

    def test_split_preserves_backward_specialization_and_gradients(self):
        def forward(x, scale, base, pre, post, comb):
            pre.copy_(x[:, :4])
            post.copy_(x[:, 4:8])
            comb.copy_(x[:, 8:])

        def backward(
            pre_grad,
            post_grad,
            comb_grad,
            x,
            post,
            scale,
            base,
            x_grad,
            scale_grad,
            base_grad,
        ):
            x_grad.copy_(torch.cat((pre_grad, post_grad, comb_grad), dim=-1))
            scale_grad.zero_()
            base_grad.zero_()

        op, compiler = self.load_op("pre_split_mixes", forward, backward)
        x = torch.randn(1, 2, 24, requires_grad=True)
        scale = torch.ones(3, requires_grad=True)
        base = torch.zeros(24, requires_grad=True)
        with torch.inference_mode(), patch.object(
            op.MHCPreSplitMixes,
            "apply",
            side_effect=AssertionError("autograd dispatch"),
        ):
            pre, post, comb = op.mhc_pre_split_mixes(x, scale, base, 4, 2.5, 1e-5)
            torch.testing.assert_close(pre.flatten(2), x[:, :, :4])
            torch.testing.assert_close(post.flatten(2), x[:, :, 4:8])
            torch.testing.assert_close(comb.flatten(2), x[:, :, 8:])
        op.get_num_sms.assert_not_called()
        compiler.assert_not_called()
        outputs = op.mhc_pre_split_mixes(x, scale, base, 4, 2.5, 1e-5)
        compiler.assert_not_called()
        sum(output.sum() for output in outputs).backward()
        compiler.assert_called_once_with(4, 2.5, token_block_size=32, num_sms=2)
        torch.testing.assert_close(x.grad, torch.ones_like(x))
        torch.testing.assert_close(scale.grad, torch.zeros_like(scale))
        torch.testing.assert_close(base.grad, torch.zeros_like(base))

    def test_sinkhorn_cache_uses_static_parameters_and_device(self):
        op, _ = self.load_op("sinkhorn", lambda x, output: output.copy_(x), Mock())
        with torch.no_grad():
            for tokens in (1, 7, 32):
                x = torch.randn(tokens, 4, 4).transpose(-1, -2)
                torch.testing.assert_close(op.sinkhorn_normalize(x, 20, 1e-5), x)
            op._mhc_sinkhorn_fwd.assert_called_once_with(4, 1, 20, 1e-5)
            op.sinkhorn_normalize(torch.randn(2, 4, 4), 10, 1e-5)
            op.sinkhorn_normalize(torch.randn(2, 4, 4), 20, 1e-6)
            op.sinkhorn_normalize(torch.randn(2, 2, 2), 20, 1e-5)
            op._inference_kernel(torch.device("cpu", 0), 4, 20, 1e-5)
            self.assertEqual(op._mhc_sinkhorn_fwd.call_count, 5)

    def test_compile_retains_autograd_dispatch(self):
        op, _ = self.load_op("sinkhorn", Mock(), Mock())
        x = torch.randn(1, 4, 4)
        with torch.no_grad(), patch.object(
            torch.compiler, "is_compiling", return_value=True
        ), patch.object(op._SinkhornNormalize, "apply", return_value=x) as apply:
            torch.testing.assert_close(op.sinkhorn_normalize(x), x)
        apply.assert_called_once()
        op._mhc_sinkhorn_fwd.assert_not_called()

    def test_norm_inference_matches_autograd_forward_with_optional_weight(self):
        def merge(fn, weight, output):
            output.copy_(fn * weight)

        def multiply(x, fn, output, sqrsum):
            output.copy_((x.float() @ fn.T).unsqueeze(0))
            sqrsum.copy_(x.float().square().sum(-1).unsqueeze(0))

        def normalize(mul_splits, sqr_splits, mul, sqr, output):
            mul.copy_(mul_splits.sum(0))
            sqr.copy_(sqr_splits.sum(0))
            output.copy_(mul[:, 0] * torch.rsqrt(sqr / 12 + 1e-6))

        names = (
            "_mhc_fn_normw_merge_bwd",
            "_mhc_pre_norm_fn_bwd_mul",
            "_mhc_pre_norm_fn_bwd_norm",
        )
        kernels = {name: Mock() for name in names}
        kernels.update(
            {
                "_mhc_fn_normw_merge_fwd": Mock(return_value=merge),
                "_mhc_pre_norm_fn_fwd_mul": Mock(return_value=multiply),
                "_mhc_pre_norm_fn_fwd_norm": Mock(return_value=normalize),
                "round_to_tf32": lambda x: x,
            }
        )
        op = self.load_kernels("norm_fn", kernels)
        x = torch.randn(1, 2, 4, 3, dtype=torch.bfloat16)
        fn = torch.randn(24, 12)
        for weight in (None, torch.randn(12)):
            with self.subTest(weight=weight is not None), torch.no_grad():
                merged = fn if weight is None else op._MHCFnNormwMerge.apply(fn, weight)
                expected = op.MHCPreNormFn.apply(x, merged, 1e-6, False, 1)
                with patch.object(
                    op.MHCPreNormFn,
                    "apply",
                    side_effect=AssertionError("autograd dispatch"),
                ), patch.object(
                    op._MHCFnNormwMerge,
                    "apply",
                    side_effect=AssertionError("autograd dispatch"),
                ):
                    actual = op.mhc_pre_norm_fn(x, fn, weight, 1e-6, False, 1)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        for name in names:
            kernels[name].assert_not_called()

    def test_post_inference_alias_and_training_gradients(self):
        def forward(x, residual, post, comb, out):
            result = residual + x.unsqueeze(-2) * post + comb.sum(-1).unsqueeze(-1)
            if out is not None:
                out.copy_(result)
                return out
            return result

        def backward(x, residual, post, comb, grad):
            return (
                (grad * post).sum(-2),
                grad,
                (grad * x.unsqueeze(-2)).sum(-1, keepdim=True),
                grad.sum(-1, keepdim=True).expand_as(comb),
            )

        op = self.load_kernels(
            "post", {"mhc_post_fwd": forward, "mhc_post_bwd": backward}
        )
        x = torch.randn(2, 3, requires_grad=True)
        residual = torch.randn(2, 4, 3, requires_grad=True)
        post = torch.randn(2, 4, 1, requires_grad=True)
        comb = torch.randn(2, 4, 4, requires_grad=True)
        with torch.no_grad():
            expected = op.MHCPost.apply(x, residual, post, comb, None)
            for mode in (torch.no_grad, torch.inference_mode):
                out = residual.clone()
                with mode(), patch.object(
                    op.MHCPost, "apply", side_effect=AssertionError("autograd dispatch")
                ):
                    actual = op.mhc_post(x, out, post, comb, out=out)
                self.assertIs(actual, out)
                torch.testing.assert_close(actual, expected)
        op.mhc_post(x, residual, post, comb).sum().backward()
        torch.testing.assert_close(x.grad, post.detach().sum(-2).expand_as(x))
        torch.testing.assert_close(residual.grad, torch.ones_like(residual))
        torch.testing.assert_close(
            post.grad, x.detach().sum(-1).view(2, 1, 1).expand_as(post)
        )
        torch.testing.assert_close(comb.grad, torch.full_like(comb, 3))


if __name__ == "__main__":
    unittest.main()
