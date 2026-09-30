"""Checkpoint equations, lane ownership, mask and output-alias contracts."""

import os
import unittest

import torch

from rtp_llm.models.test.dsv41_cache_codec_test import load_codec_source

inject = load_codec_source("_dsv41_inject_test", "v41_engram_inject.py")


def checkpoint_reference(hidden, projected, q, k, valid, eps):
    # Preserve the checkpoint's original [batch, sequence, lane, dim] equations.
    x = hidden.unsqueeze(0)
    dim, lanes = x.shape[-1], x.shape[-2]
    kv = projected.unsqueeze(0)
    key, value = kv.split([lanes * dim, dim], dim=-1)
    key = key.float().unflatten(-1, (lanes, dim))
    weight = q.float() * k.float()
    h = x.float()
    rstd = torch.rsqrt(h.square().mean(-1) + eps) * torch.rsqrt(
        key.square().mean(-1) + eps
    )
    dot = (h * weight * key).sum(-1) * rstd * dim**-0.5
    gate = torch.sigmoid(torch.copysign(dot.abs().clamp_min(1e-6).sqrt(), dot))
    result = (h + gate.unsqueeze(-1) * value.float().unsqueeze(-2)).to(x.dtype)
    return torch.where(valid[None, :, None, None], result, x).squeeze(0)


def inputs(tokens=7, dim=64, dtype=torch.float32):
    generator = torch.Generator().manual_seed(4101)
    values = [
        torch.randn(shape, generator=generator).to(dtype)
        for shape in ((tokens, 4, dim), (tokens, 5 * dim), (4, dim), (4, dim))
    ]
    return (*values, torch.arange(tokens) % 3 != 0)


class V41EngramInjectTest(unittest.TestCase):
    def test_matches_checkpoint_equations_for_all_residual_dtypes(self):
        for dtype in (torch.float32, torch.bfloat16, torch.float16):
            args = inputs(dtype=dtype)
            before = args[0].clone()
            actual = inject.gated_engram_residual(*args, eps=1e-6, backend="eager")
            expected = checkpoint_reference(*args, 1e-6)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            torch.testing.assert_close(args[0], before, rtol=0, atol=0)

    def test_each_lane_has_its_own_gate_and_one_shared_value(self):
        hidden = (
            torch.tensor([1.0, -2.0, 0.0, 4.0])
            .reshape(1, 4, 1)
            .expand(1, 4, 32)
            .contiguous()
        )
        projection = torch.ones(1, 5 * 32)
        q = torch.ones(4, 32)
        k = torch.tensor([1.0, 1.0, 0.0, -1.0]).reshape(4, 1).expand(4, 32).contiguous()
        actual = inject.gated_engram_residual(
            hidden, projection, q, k, torch.tensor([True]), eps=1e-6, backend="eager"
        )
        update = actual - hidden
        self.assertGreater(float(update[0, 0, 0]), 0.9)
        self.assertLess(float(update[0, 1, 0]), 0.1)
        self.assertAlmostEqual(
            float(update[0, 2, 0]), float(torch.sigmoid(torch.tensor(0.001))), places=6
        )
        self.assertLess(float(update[0, 3, 0]), 0.1)

    def test_dead_rows_are_unchanged_even_with_nonfinite_projection(self):
        args = inputs()
        hidden, projected, _, _, valid = args
        hidden[0] = -0.0
        projected[~valid] = float("nan")
        actual = inject.gated_engram_residual(*args, eps=1e-6, backend="eager")
        torch.testing.assert_close(actual[~valid], hidden[~valid], rtol=0, atol=0)
        self.assertTrue(bool(torch.signbit(actual[0]).all()))

    def test_exact_hidden_alias_and_independent_output_match(self):
        args = inputs(dtype=torch.bfloat16)
        expected = inject.gated_engram_residual(*args, eps=1e-6, backend="eager")
        out = torch.empty_like(args[0])
        self.assertIs(
            inject.gated_engram_residual(*args, eps=1e-6, backend="eager", out=out), out
        )
        torch.testing.assert_close(out, expected, rtol=0, atol=0)
        with torch.inference_mode():
            actual = inject.gated_engram_residual(
                *args, eps=1e-6, backend="eager", out=args[0]
            )
        self.assertIs(actual, args[0])
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_partial_hidden_and_other_input_aliases_are_rejected(self):
        _, projected, q, k, valid = inputs()
        storage = torch.zeros(7 * 4 * 64 + 1)
        hidden = storage[:-1].view(7, 4, 64)
        partial = storage[1:].view_as(hidden)
        with self.assertRaisesRegex(ValueError, "exact"):
            inject.gated_engram_residual(
                hidden, projected, q, k, valid, eps=1e-6, backend="eager", out=partial
            )
        shared = projected.flatten()[: hidden.numel()].view_as(hidden)
        with self.assertRaisesRegex(ValueError, "share storage"):
            inject.gated_engram_residual(
                hidden, projected, q, k, valid, eps=1e-6, backend="eager", out=shared
            )
        torch.testing.assert_close(storage, torch.zeros_like(storage))

    def test_invalid_contracts_fail_before_writing(self):
        args = inputs()
        out = torch.full_like(args[0], 42)
        changes = [
            (0, args[0].transpose(1, 2)),
            (1, args[1][:, :-1]),
            (2, args[2].double()),
            (4, args[4].long()),
        ]
        for position, replacement in changes:
            bad = list(args)
            bad[position] = replacement
            with self.assertRaises(ValueError):
                inject.gated_engram_residual(*bad, eps=1e-6, backend="eager", out=out)
        for epsilon in (0, -1, float("nan"), float("inf"), True):
            with self.assertRaises(ValueError):
                inject.gated_engram_residual(
                    *args, eps=epsilon, backend="eager", out=out
                )
        with self.assertRaisesRegex(ValueError, "CUDA"):
            inject.gated_engram_residual(*args, eps=1e-6, out=out)
        with self.assertRaisesRegex(ValueError, "inference-only"):
            inject.gated_engram_residual(
                args[0].requires_grad_(), *args[1:], eps=1e-6, backend="eager", out=out
            )
        torch.testing.assert_close(out, torch.full_like(out, 42))

    def test_empty_and_permuted_chunked_tokens(self):
        args = inputs()
        expected = inject.gated_engram_residual(*args, eps=1e-6, backend="eager")
        order = torch.tensor([6, 2, 1, 3, 0, 5, 4])
        results = []
        for chunk in order.split(3):
            rows = (args[0][chunk], args[1][chunk], args[2], args[3], args[4][chunk])
            results.append(
                inject.gated_engram_residual(*rows, eps=1e-6, backend="eager")
            )
        torch.testing.assert_close(torch.cat(results), expected[order], rtol=0, atol=0)
        empty = inject.gated_engram_residual(
            *inputs(tokens=0), eps=1e-6, backend="eager"
        )
        self.assertEqual(empty.shape, (0, 4, 64))


@unittest.skipUnless(
    os.environ.get("TRITON_INTERPRET") == "1", "Set TRITON_INTERPRET=1"
)
class V41EngramInjectInterpreterTest(unittest.TestCase):
    def test_fused_reductions_tail_mask_and_exact_alias(self):
        kernels = load_codec_source(
            "_dsv41_inject_interpreter", "_v41_engram_inject_triton.py"
        )
        args = inputs(dim=37)
        args[1][~args[4]] = float("nan")
        args[0][0] = -0.0
        expected = checkpoint_reference(*args, 1e-6)
        for alias in (False, True):
            hidden = args[0].clone()
            output = hidden if alias else torch.empty_like(hidden)
            kernels.engram_inject_kernel[(7, 4)](
                hidden,
                *args[1:],
                output,
                DIM=37,
                HC=4,
                EPS=1e-6,
                BLOCK=64,
            )
            torch.testing.assert_close(output, expected, rtol=1e-6, atol=1e-6)
            self.assertTrue(bool(torch.signbit(output[0]).all()))


if __name__ == "__main__":
    unittest.main()
