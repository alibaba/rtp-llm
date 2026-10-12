"""Qualify selected FP32 mHC prenorm with held operands and changing graphs."""

import importlib
import json
import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from safetensors import safe_open

from rtp_llm.models_py.modules.dsv4.fp8.test.v41_mhc_precision import (
    fp32_mhc_prenorm,
    mhc_fp32,
)
from rtp_llm.models_py.modules.dsv4.hc import v41_prenorm
from rtp_llm.models_py.modules.dsv4.hc.delayed import DelayedHCUnit


def exact(actual, expected):
    actual, expected = actual.detach().cpu(), expected.detach().cpu()
    assert actual.dtype == expected.dtype and actual.shape == expected.shape
    if actual.numel():
        assert torch.equal(
            actual.contiguous().view(torch.uint8),
            expected.contiguous().view(torch.uint8),
        )


def unit(dim=64, *, prenorm_mode="fp32", weights=None, norm_eps=1e-6):
    if weights is None:
        weights = (
            torch.randn(24, 4 * dim, device="cuda", dtype=torch.float32) * 0.01,
            torch.randn(24, device="cuda", dtype=torch.float32) * 0.1,
            torch.ones(3, device="cuda", dtype=torch.float32),
        )
    return DelayedHCUnit(
        *weights,
        dim=dim,
        hc_mult=4,
        hc_sinkhorn_iters=20,
        norm_eps=norm_eps,
        hc_eps=1e-6,
        prenorm_mode=prenorm_mode,
    )


def coefficient_outputs(owner, residual):
    y, post, comb = owner.pre(residual)
    return y.clone(), owner.pre_mix_out.clone(), post.clone(), comb.clone()


class V41PrenormModeSM120Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (
            12,
            0,
        ):
            raise unittest.SkipTest("Requires a leased SM120 GPU")
        torch.set_num_threads(1)

    def setUp(self):
        precision = torch.get_float32_matmul_precision()
        self.addCleanup(torch.set_float32_matmul_precision, precision)
        torch.set_float32_matmul_precision("highest")

    @unittest.skipUnless(
        os.environ.get("DSV41_TEST_PRENORM_ARTIFACT_ROOT")
        and os.environ.get("DSV41_TEST_PRENORM_OUTPUT_ROOT")
        and os.environ.get("DSV41_TEST_CHECKPOINT"),
        "set admitted operand, checkpoint and fresh output directories",
    )
    @torch.inference_mode()
    def test_held_nas_weights_and_released_raw_mix_controls(self):
        root = Path(os.environ["DSV41_TEST_PRENORM_ARTIFACT_ROOT"])
        output = Path(os.environ["DSV41_TEST_PRENORM_OUTPUT_ROOT"])
        output.mkdir()
        checkpoint = Path(os.environ["DSV41_TEST_CHECKPOINT"])
        sys.path.insert(0, str(checkpoint / "inference"))
        try:
            released = importlib.import_module("model")
        finally:
            sys.path.pop(0)
        self.assertEqual(
            Path(released.__file__).resolve(), checkpoint / "inference/model.py"
        )
        frozen = torch.load(
            root / "attention-history-down-sparse-input-1.pt",
            map_location="cpu",
            weights_only=True,
        )
        index = json.loads((checkpoint / "model.safetensors.index.json").read_text())[
            "weight_map"
        ]
        admitted_weights = {}
        for layer in range(3):
            for kind in ("attn", "ffn"):
                values = []
                for suffix in ("fn", "base", "scale"):
                    name = f"layers.{layer}.hc_{kind}_{suffix}"
                    with safe_open(checkpoint / index[name], framework="pt") as reader:
                        value = reader.get_tensor(name)
                    admitted_weights[name] = value.clone()
                    values.append(value.cuda())
                admitted_weights[f"owner.{layer}.{kind}"] = values
        records = []
        text = json.loads((checkpoint / "config.json").read_text())["text_config"]
        norm_eps = text["rms_norm_eps"]
        block = SimpleNamespace(
            norm_eps=norm_eps,
            hc_mult=text["hc_mult"],
            hc_sinkhorn_iters=text["hc_sinkhorn_iters"],
            hc_eps=text["hc_eps"],
        )
        self.assertEqual(
            (block.hc_mult, block.hc_sinkhorn_iters, block.hc_eps), (4, 20, 1e-6)
        )
        for stream in ("native", "official"):
            steps = frozen[f"{stream}_mhc"]
            self.assertEqual(len(steps), 3)
            for step, layers in enumerate(steps):
                self.assertEqual(len(layers), 3)
                for layer, saved in enumerate(layers):
                    for kind in ("attn", "ffn"):
                        weights = admitted_weights[f"owner.{layer}.{kind}"]
                        selected = unit(5120, weights=weights, norm_eps=norm_eps)
                        default = unit(
                            5120, weights=weights, prenorm_mode=None, norm_eps=norm_eps
                        )
                        raw = saved[
                            "attention_residual" if kind == "attn" else "attn_post"
                        ]
                        residual = raw.cuda().reshape(1, -1, 4, 5120)
                        before = [v.clone() for v in (residual, *weights)]
                        expected_mix = fp32_mhc_prenorm(residual, weights[0], norm_eps)
                        mixes = v41_prenorm.prenorm_fp32(residual, weights[0], norm_eps)
                        exact(mixes, expected_mix)
                        exact(mixes, saved[f"{kind}_raw_mixes"].reshape_as(mixes))
                        with mhc_fp32(True):
                            diagnostic = coefficient_outputs(default, residual)
                        with patch.object(
                            v41_prenorm,
                            "prenorm",
                            side_effect=AssertionError(
                                "Selected prenorm used default dispatch"
                            ),
                        ):
                            actual = coefficient_outputs(selected, residual)
                        for value, control in zip(actual, diagnostic):
                            exact(value, control)
                        released_coefficients = released.Block.hc_mixes(
                            block, residual, weights[0], weights[2], weights[1]
                        )
                        saved_coefficients = [
                            saved[f"{kind}_{suffix}_mix"]
                            for suffix in ("pre", "post", "comb")
                        ]
                        controls = (
                            diagnostic[1:]
                            if stream == "native"
                            else released_coefficients
                        )
                        for value, control in zip(controls, saved_coefficients):
                            exact(value.reshape_as(control), control)
                        for value, control in zip((residual, *weights), before):
                            exact(value, control)
                        records.append(
                            dict(
                                stream=stream,
                                step=step,
                                layer=layer,
                                kind=kind,
                                mixes=mixes.cpu(),
                                diagnostic_mix=expected_mix.cpu(),
                                selected=[v.cpu() for v in actual],
                                diagnostic=[v.cpu() for v in diagnostic],
                                released=[v.cpu() for v in released_coefficients],
                                residual_after=residual.cpu(),
                            )
                        )
        self.assertEqual(len(records), 36)
        weights = {
            key: value
            for key, value in admitted_weights.items()
            if key.startswith("layers.")
        }
        torch.save(dict(cases=records, weights=weights), output / "held.pt")
        report = dict(
            cases=36,
            norm_eps=norm_eps,
            raw_mix_controls_exact=36,
            diagnostic_coefficient_outputs_exact=144,
            native_saved_coefficients_exact=54,
            released_saved_coefficients_exact=54,
            unchanged_operands=144,
            default_dispatch_bypassed=True,
            precision_unchanged=True,
        )
        self.assertEqual(torch.get_float32_matmul_precision(), "highest")
        (output / "held.json").write_text(json.dumps(report, sort_keys=True) + "\n")
        print(
            "DSV41_FP32_PRENORM_HELD " + json.dumps(report, sort_keys=True), flush=True
        )

    @torch.inference_mode()
    def test_analytic_layouts_empty_and_strided_padding(self):
        fn = torch.ones(24, 128, device="cuda", dtype=torch.float32)
        fn[1::2].neg_()
        for leading in ((1,), (5,), (2, 3), (0,), (2, 0)):
            residual = torch.ones(*leading, 4, 32, device="cuda", dtype=torch.bfloat16)
            actual = v41_prenorm.prenorm_fp32(residual, fn, 3.0)
            expected = torch.full((*leading, 24), 64.0, device="cuda")
            expected[..., 1::2].neg_()
            exact(actual, expected)
            backing = residual.repeat_interleave(2, dim=-1)
            padding = backing[..., ::2].clone()
            before = backing.clone()
            exact(v41_prenorm.prenorm_fp32(backing[..., 1::2], fn, 3.0), expected)
            exact(backing, before)
            exact(backing[..., ::2], padding)

    @torch.inference_mode()
    def test_invalid_operands_and_precision_reject_before_owner_mutation(self):
        owner = unit()
        residual = torch.ones(2, 4, 64, device="cuda", dtype=torch.bfloat16)
        fn = owner.fn
        owner.pre(residual)
        saved_mix = owner.pre_mix_out
        before = residual.clone()
        for bad, weight, eps, error in (
            (residual.float(), fn, 1e-6, TypeError),
            (residual, fn.bfloat16(), 1e-6, TypeError),
            (residual, fn[:23], 1e-6, ValueError),
            (residual[:, 0], fn, 1e-6, ValueError),
            (residual, fn.cpu(), 1e-6, ValueError),
            (residual[:, :0], fn, 1e-6, ValueError),
            *(
                (residual, fn, eps, ValueError)
                for eps in (0, -1, float("nan"), float("inf"), True)
            ),
        ):
            with self.assertRaises(error):
                v41_prenorm.prenorm_fp32(bad, weight, eps)
        for precision in ("high", "medium"):
            torch.set_float32_matmul_precision(precision)
            with self.assertRaisesRegex(ValueError, "highest FP32"):
                owner.pre(residual)
            with self.assertRaisesRegex(ValueError, "highest FP32"):
                unit()
            self.assertEqual(torch.get_float32_matmul_precision(), precision)
            self.assertIs(owner.pre_mix_out, saved_mix)
            exact(residual, before)
        torch.set_float32_matmul_precision("highest")
        owner.pre(residual)

    @torch.inference_mode()
    def test_omitted_dispatch_and_read_only_selection(self):
        residual = torch.ones(2, 4, 64, device="cuda", dtype=torch.bfloat16)
        selected, default = unit(), unit(prenorm_mode=None)
        self.assertIsNone(default.prenorm_mode)
        self.assertEqual(selected.prenorm_mode, "fp32")
        with self.assertRaises(AttributeError):
            selected.prenorm_mode = None
        for mode in (False, True, 0, [], "", "torch", "FP32"):
            with self.assertRaisesRegex(ValueError, "prenorm_mode"):
                unit(prenorm_mode=mode)
        with patch.object(
            v41_prenorm, "prenorm", wraps=v41_prenorm.prenorm
        ) as fallback:
            default.pre(residual)
            self.assertEqual(fallback.call_count, 1)
            selected.pre(residual)
            self.assertEqual(fallback.call_count, 1)

    @torch.inference_mode()
    def test_changing_input_and_weight_graph_replays(self):
        torch.manual_seed(41025)
        for leading in ((1,), (5,), (2, 3)):
            with self.subTest(leading=leading):
                selected = unit(5120)
                default = unit(
                    5120,
                    weights=(selected.fn, selected.base, selected.scale),
                    prenorm_mode=None,
                )
                residual = torch.randn(
                    *leading, 4, 5120, device="cuda", dtype=torch.bfloat16
                )
                for _ in range(3):
                    selected.pre(residual)
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with patch.object(
                    v41_prenorm,
                    "prenorm",
                    side_effect=AssertionError("Selected graph used default prenorm"),
                ), torch.cuda.graph(graph):
                    y, post, comb = selected.pre(residual)
                    pre = selected.pre_mix_out
                for _ in range(3):
                    residual.copy_(torch.randn_like(residual))
                    selected.fn.copy_(torch.randn_like(selected.fn) * 0.01)
                    with mhc_fp32(True):
                        expected = coefficient_outputs(default, residual)
                    graph.replay()
                    for actual, control in zip((y, pre, post, comb), expected):
                        exact(actual, control)
        print("DSV41_FP32_PRENORM_GRAPHS captures=3 changing_replays=9", flush=True)


if __name__ == "__main__":
    unittest.main()
