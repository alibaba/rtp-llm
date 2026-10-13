import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import torch


class MHCLazyBackwardTest(unittest.TestCase):
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
            output.copy_((x.float() @ fn.T).unsqueeze(1))
            sqrsum.copy_(x.float().square().sum(-1, keepdim=True))

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
