"""Maintained ordered delayed-mHC readout against frozen released operands."""

import json
import os
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from rtp_llm.models_py.modules.dsv4.hc import delayed
from rtp_llm.models_py.modules.dsv4.hc.delayed import (
    DelayedHCHead,
    DelayedHCUnit,
    collapse_ordered_fp32,
)


class Predecessor:
    def __init__(self, mix):
        self.pre_mix_out = mix


def unit():
    return DelayedHCUnit(
        torch.zeros(24, 20480, dtype=torch.float32, device="cuda"),
        torch.zeros(24, dtype=torch.float32, device="cuda"),
        torch.ones(3, dtype=torch.float32, device="cuda"),
        dim=5120,
        hc_mult=4,
        hc_sinkhorn_iters=20,
        norm_eps=1e-6,
        hc_eps=1e-6,
        collapse_mode="ordered_fp32",
    )


def exact(actual, expected):
    actual, expected = actual.detach().cpu(), expected.detach().cpu()
    if actual.dtype != expected.dtype or actual.shape != expected.shape:
        raise AssertionError("Collapse output shape/dtype differs")
    if actual.numel() == 0:
        return
    if not torch.equal(
        actual.contiguous().view(torch.uint8), expected.contiguous().view(torch.uint8)
    ):
        raise AssertionError("Collapse output bytes differ")


class V41CollapseModeSM120Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (
            12,
            0,
        ):
            raise unittest.SkipTest("Requires a leased SM120 GPU")
        torch.set_num_threads(1)

    @unittest.skipUnless(
        os.environ.get("DSV41_TEST_COLLAPSE_ARTIFACT_ROOT")
        and os.environ.get("DSV41_TEST_COLLAPSE_OUTPUT_ROOT"),
        "set admitted frozen operand and fresh output directories",
    )
    @torch.inference_mode()
    def test_frozen_native_and_released_controls(self):
        root = Path(os.environ["DSV41_TEST_COLLAPSE_ARTIFACT_ROOT"])
        output = Path(os.environ["DSV41_TEST_COLLAPSE_OUTPUT_ROOT"])
        output.mkdir()
        selected = unit()
        head = DelayedHCHead(selected)
        records = []
        counts = {}
        for prompt in ("france", "arithmetic"):
            frozen = torch.load(
                root / f"mhc-collapse-{prompt}-1.pt",
                map_location="cpu",
                weights_only=True,
            )
            cases = frozen["cases"]
            self.assertEqual(len(cases), 84 if prompt == "france" else 112)
            for case in cases:
                before = case["operands_before"]
                residual = before["residual"].cuda().reshape(1, -1, 4, 5120)
                mix = before["pre_mix"].cuda().reshape(1, -1, 4)
                previous = Predecessor(mix)
                selected.set_previous(None if case["entry"] else previous)
                expected = case["outputs"]["released"]
                baseline = (
                    residual[..., 0, :].clone()
                    if case["entry"]
                    else delayed.collapse_delayed(residual, mix)
                )
                exact(baseline.reshape_as(expected), case["outputs"]["baseline"])
                with patch.object(
                    delayed,
                    "collapse_delayed",
                    side_effect=AssertionError(
                        "Selected readout used default dispatch"
                    ),
                ):
                    actual, _, _ = selected.pre(residual)
                    selected.pre_mix_out = mix
                    readout = head.head(residual)
                actual, readout = actual.reshape_as(expected), readout.reshape_as(
                    expected
                )
                exact(actual, expected)
                exact(readout, expected)
                after = {
                    "residual": residual.cpu().reshape_as(before["residual"]),
                    "pre_mix": mix.cpu().reshape_as(before["pre_mix"]),
                }
                for name, tensor in before.items():
                    exact(after[name], tensor)
                records.append(
                    {
                        "prompt": prompt,
                        "stream": case["stream"],
                        "step": case["step"],
                        "layer": case["layer"],
                        "kind": case["kind"],
                        "entry": case["entry"],
                        "operands_after": after,
                        "baseline": baseline.cpu().reshape_as(expected),
                        "selected_unit": actual.cpu(),
                        "selected_head": readout.cpu(),
                    }
                )
            counts[prompt] = len(cases)
        torch.save({"cases": records}, output / "held.pt")
        report = {
            "cases": len(records),
            "prompt_cases": counts,
            "baseline_controls_exact": len(records),
            "selected_unit_controls_exact": len(records),
            "selected_head_controls_exact": len(records),
            "unchanged_operands": 2 * len(records),
            "default_dispatch_bypassed": True,
        }
        (output / "held.json").write_text(json.dumps(report, indent=2) + "\n")
        print("DSV41_ORDERED_COLLAPSE_HELD " + json.dumps(report), flush=True)

    @torch.inference_mode()
    def test_strided_flat_batched_and_empty_layouts(self):
        torch.manual_seed(41012)
        selected = unit()
        for shape in ((1, 4, 5120), (7, 4, 5120), (2, 3, 4, 5120), (0, 4, 5120)):
            with self.subTest(shape=shape):
                residual = torch.randn(
                    *shape[:-1], shape[-1] * 2, device="cuda", dtype=torch.bfloat16
                )[..., ::2]
                mix = torch.randn(
                    *shape[:-2], shape[-2] * 2, device="cuda", dtype=torch.float32
                )[..., ::2]
                before = residual.clone(), mix.clone()
                expected = (mix.unsqueeze(-1) * residual.float()).sum(-2).bfloat16()
                exact(collapse_ordered_fp32(residual, mix), expected)
                previous = Predecessor(mix)
                selected.set_previous(previous)
                actual, _, _ = selected.pre(residual)
                exact(actual, expected)
                exact(residual, before[0])
                exact(mix, before[1])

    @torch.inference_mode()
    def test_invalid_operands_reject_before_coefficient_writes(self):
        selected = unit()
        residual = torch.ones(2, 4, 5120, device="cuda", dtype=torch.bfloat16)
        sentinel = torch.full((2, 4), -11.0, device="cuda", dtype=torch.float32)
        selected.pre_mix_out = sentinel
        previous = Predecessor(None)
        selected.set_previous(previous)
        for mix, error in (
            (None, RuntimeError),
            (torch.ones(2, 3, device="cuda"), ValueError),
            (torch.ones(2, 4, device="cuda", dtype=torch.bfloat16), TypeError),
            (torch.ones(2, 4, dtype=torch.float32), ValueError),
        ):
            with self.subTest(error=error):
                previous.pre_mix_out = mix
                with self.assertRaises(error):
                    selected.pre(residual)
                self.assertIs(selected.pre_mix_out, sentinel)
                exact(sentinel, torch.full_like(sentinel, -11.0))
        with self.assertRaises(ValueError):
            collapse_ordered_fp32(residual[:, 0], torch.ones(2, device="cuda"))
        with self.assertRaises(ValueError):
            collapse_ordered_fp32(
                torch.empty(2, 0, 5120, device="cuda", dtype=torch.bfloat16),
                torch.empty(2, 0, device="cuda", dtype=torch.float32),
            )

    @torch.inference_mode()
    def test_unit_and_head_changing_input_graph_replays(self):
        torch.manual_seed(41013)
        for shape in ((3, 4, 5120), (2, 2, 4, 5120)):
            with self.subTest(shape=shape):
                selected = unit()
                residual = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
                mix = torch.randn(shape[:-1], device="cuda", dtype=torch.float32)
                previous = Predecessor(mix)
                selected.set_previous(previous)
                head = DelayedHCHead(selected)

                def run():
                    collapsed, _, _ = selected.pre(residual)
                    return collapsed, head.head(residual)

                for _ in range(3):
                    run()
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    actual = run()
                for iteration in range(3):
                    residual.copy_(torch.randn_like(residual) * (iteration + 1))
                    mix.copy_(torch.randn_like(mix) * (iteration - 1.5))
                    expected = tuple(value.clone() for value in run())
                    graph.replay()
                    torch.cuda.synchronize()
                    for value, reference in zip(actual, expected):
                        exact(value, reference)
        print("DSV41_ORDERED_COLLAPSE_GRAPHS captures=2 changing_replays=6", flush=True)


if __name__ == "__main__":
    unittest.main()
