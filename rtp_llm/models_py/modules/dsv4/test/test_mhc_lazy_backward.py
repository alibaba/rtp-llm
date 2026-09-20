import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import torch


class MHCLazyBackwardTest(unittest.TestCase):
    def load_op(self, name, forward, backward):
        root = Path(__file__).resolve().parents[3] / "3rdparty/tile_kernels"
        kernel = types.ModuleType(f"_test_tile.mhc.{name}_kernel")
        setattr(kernel, f"_mhc_{name}_fwd", Mock(return_value=forward))
        setattr(kernel, f"_mhc_{name}_bwd", Mock(return_value=backward))
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
        return module, getattr(kernel, f"_mhc_{name}_bwd")

    def test_sinkhorn_compiles_backward_only_when_differentiated(self):
        def forward(x, output):
            output.copy_(x)

        def backward(grad, x, output):
            output.copy_(grad)

        op, compiler = self.load_op("sinkhorn", forward, backward)
        x = torch.randn(2, 4, 4, requires_grad=True)
        with torch.inference_mode():
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
        with torch.inference_mode():
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
        with torch.inference_mode():
            pre, post, comb = op.mhc_pre_split_mixes(x, scale, base, 4, 2.5, 1e-5)
            torch.testing.assert_close(pre.flatten(2), x[:, :, :4])
            torch.testing.assert_close(post.flatten(2), x[:, :, 4:8])
            torch.testing.assert_close(comb.flatten(2), x[:, :, 8:])
        compiler.assert_not_called()
        outputs = op.mhc_pre_split_mixes(x, scale, base, 4, 2.5, 1e-5)
        compiler.assert_not_called()
        sum(output.sum() for output in outputs).backward()
        compiler.assert_called_once_with(4, 2.5, token_block_size=32, num_sms=2)
        torch.testing.assert_close(x.grad, torch.ones_like(x))
        torch.testing.assert_close(scale.grad, torch.zeros_like(scale))
        torch.testing.assert_close(base.grad, torch.zeros_like(base))


if __name__ == "__main__":
    unittest.main()
