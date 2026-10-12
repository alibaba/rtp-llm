"""Qualify maintained delayed-mHC post selection against released operands."""

import importlib
import json
import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from rtp_llm.models_py.modules.dsv4.hc import delayed
from rtp_llm.models_py.modules.dsv4.hc.delayed import DelayedHCUnit, post_ordered_fp32


def unit(dim=5120, *, post_mode="ordered_fp32"):
    return DelayedHCUnit(
        torch.zeros(24, 4 * dim, device="cuda", dtype=torch.float32),
        torch.zeros(24, device="cuda", dtype=torch.float32),
        torch.ones(3, device="cuda", dtype=torch.float32),
        dim=dim,
        hc_mult=4,
        hc_sinkhorn_iters=20,
        hc_eps=1e-6,
        norm_eps=1e-6,
        post_mode=post_mode,
    )


def exact(actual, expected):
    actual, expected = actual.detach().cpu(), expected.detach().cpu()
    if actual.dtype != expected.dtype or actual.shape != expected.shape:
        raise AssertionError("Post output shape/dtype differs")
    if actual.numel() and not torch.equal(
        actual.contiguous().view(torch.uint8), expected.contiguous().view(torch.uint8)
    ):
        raise AssertionError("Post output bytes differ")


def values(leading, dim=5120):
    return (
        torch.randn(*leading, dim, device="cuda", dtype=torch.bfloat16),
        torch.randn(*leading, 4, dim, device="cuda", dtype=torch.bfloat16),
        torch.randn(*leading, 4, 1, device="cuda", dtype=torch.float32),
        torch.randn(*leading, 4, 4, device="cuda", dtype=torch.float32),
    )


class V41PostModeSM120Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (
            12,
            0,
        ):
            raise unittest.SkipTest("Requires a leased SM120 GPU")
        torch.set_num_threads(1)

    @unittest.skipUnless(
        os.environ.get("DSV41_TEST_POST_ARTIFACT_ROOT")
        and os.environ.get("DSV41_TEST_POST_OUTPUT_ROOT")
        and os.environ.get("DSV41_TEST_CHECKPOINT"),
        "set admitted operand, checkpoint and fresh output directories",
    )
    @torch.inference_mode()
    def test_frozen_native_and_released_controls(self):
        root = Path(os.environ["DSV41_TEST_POST_ARTIFACT_ROOT"])
        output = Path(os.environ["DSV41_TEST_POST_OUTPUT_ROOT"])
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
        selected, default = unit(), unit(post_mode=None)
        block = SimpleNamespace(hc_mult=4)
        records = []
        for stream in ("official", "native"):
            steps = frozen[f"{stream}_mhc"]
            self.assertEqual(len(steps), 3)
            for step, layers in enumerate(steps):
                self.assertEqual(len(layers), 3)
                for layer, saved in enumerate(layers):
                    for kind in ("attn", "ffn"):
                        if kind == "ffn" and layer == 2:
                            continue
                        x = saved[
                            "attention_output" if kind == "attn" else "ffn_output"
                        ]
                        residual = saved[
                            "attention_residual" if kind == "attn" else "attn_post"
                        ]
                        operands = (
                            x.cuda().reshape(1, -1, 5120),
                            residual.cuda().reshape(1, -1, 4, 5120),
                            saved[f"{kind}_post_mix"].cuda().reshape(1, -1, 4, 1),
                            saved[f"{kind}_comb_mix"].cuda().reshape(1, -1, 4, 4),
                        )
                        x, residual, post, comb = operands
                        before = [value.clone() for value in operands]
                        baseline = default.post(x, residual.clone(), post, comb).clone()
                        expected = released.Block.hc_post(
                            block, x, residual, post.squeeze(-1), comb
                        )
                        control = baseline if stream == "native" else expected
                        exact(
                            control.reshape_as(saved[f"{kind}_post"]),
                            saved[f"{kind}_post"],
                        )
                        working = residual.clone()
                        with patch.object(
                            delayed,
                            "post_delayed",
                            side_effect=AssertionError(
                                "Selected post used default dispatch"
                            ),
                        ):
                            actual = selected.post(x, working, post, comb)
                        self.assertIs(actual, working)
                        exact(actual, expected)
                        for old, current in zip(before, operands):
                            exact(current, old)
                        records.append(
                            {
                                "stream": stream,
                                "step": step,
                                "layer": layer,
                                "kind": kind,
                                "baseline": baseline.cpu(),
                                "released": expected.cpu(),
                                "selected": actual.cpu(),
                                "operands_after": [value.cpu() for value in operands],
                            }
                        )
        self.assertEqual(len(records), 30)
        torch.save({"cases": records}, output / "held.pt")
        report = {
            "cases": 30,
            "native_controls_exact": 15,
            "released_controls_exact": 15,
            "selected_controls_exact": 30,
            "unchanged_operands": 120,
            "inplace_writebacks": 30,
            "default_dispatch_bypassed": True,
        }
        (output / "held.json").write_text(json.dumps(report, sort_keys=True) + "\n")
        print(
            "DSV41_ORDERED_POST_HELD " + json.dumps(report, sort_keys=True), flush=True
        )

    @torch.inference_mode()
    def test_scalar_oracle_orientation_layouts_and_empty_writeback(self):
        owner = unit(32)
        residual_values = [-8.0, 4.0, 2.0, -1.0]
        post_values = [0.5, 0.25, 1.0, 2.0]
        comb_values = [
            [1, 0.25, 0.5, 2],
            [0.5, 1, 2, 0.25],
            [2, 0.5, 0.25, 1],
            [0.25, 2, 1, 0.5],
        ]
        expected = torch.tensor(
            [
                4 * post_values[d]
                + sum(comb_values[s][d] * residual_values[s] for s in range(4))
                for d in range(4)
            ],
            dtype=torch.bfloat16,
        )
        for leading in ((1,), (5,), (1, 1), (2, 3), (0,), (2, 0)):
            with self.subTest(leading=leading):
                x = torch.full((*leading, 32), 4.0, device="cuda", dtype=torch.bfloat16)
                residual = torch.tensor(
                    residual_values, device="cuda", dtype=torch.bfloat16
                )
                residual = residual.reshape(4, 1).expand(*leading, 4, 32).clone()
                post = torch.tensor(post_values, device="cuda").reshape(4, 1)
                post = post.expand(*leading, 4, 1).clone()
                comb = (
                    torch.tensor(comb_values, device="cuda")
                    .expand(*leading, 4, 4)
                    .clone()
                )
                before = [v.clone() for v in (x, post, comb)]
                actual = owner.post(x, residual, post, comb)
                self.assertIs(actual, residual)
                exact(actual, expected.reshape(4, 1).expand(*leading, 4, 32))
                for value, saved in zip((x, post, comb), before):
                    exact(value, saved)

    @torch.inference_mode()
    def test_strided_operands_preserve_padding_and_alias_reads(self):
        torch.manual_seed(41021)
        owner = unit()
        for leading in ((3,), (2, 3)):
            with self.subTest(leading=leading):
                dense = values(leading)
                expected = owner.post(
                    dense[0], dense[1].clone(), dense[2], dense[3]
                ).clone()
                backings = [v.repeat_interleave(2, dim=-1) for v in dense]
                strided = [v[..., 1::2] for v in backings]
                padding = [v[..., 0::2].clone() for v in backings]
                before = [v.clone() for v in strided]
                actual = owner.post(*strided)
                self.assertIs(actual, strided[1])
                exact(actual, expected)
                for index in (0, 2, 3):
                    exact(strided[index], before[index])
                for backing, saved in zip(backings, padding):
                    exact(backing[..., 0::2], saved)
                x = dense[1][..., 0, :]
                expected = owner.post(
                    x.clone(), dense[1].clone(), dense[2], dense[3]
                ).clone()
                actual = owner.post(x, dense[1], dense[2], dense[3])
                exact(actual, expected)

    @torch.inference_mode()
    def test_invalid_operands_rejected_before_writeback(self):
        owner, good = unit(), values((2,))
        invalid_cases = []
        for index in range(4):
            changed = list(good)
            changed[index] = changed[index][:1]
            invalid_cases.append((changed, ValueError))
            changed = list(good)
            changed[index] = changed[index].cpu()
            invalid_cases.append((changed, ValueError))
            for dtype in (torch.float16, torch.float64):
                changed = list(good)
                changed[index] = changed[index].to(dtype)
                invalid_cases.append((changed, TypeError))
        for index, (invalid, error) in enumerate(invalid_cases):
            with self.subTest(index=index):
                before = [v.clone() for v in invalid]
                with self.assertRaises(error):
                    owner.post(*invalid)
                with self.assertRaises(error):
                    post_ordered_fp32(*invalid)
                for actual, saved in zip(invalid, before):
                    exact(actual, saved)
        for residual in (
            torch.empty(2, 5120, device="cuda", dtype=torch.bfloat16),
            torch.empty(2, 0, 5120, device="cuda", dtype=torch.bfloat16),
        ):
            with self.assertRaises(ValueError):
                post_ordered_fp32(good[0], residual, good[2], good[3])

    @torch.inference_mode()
    def test_changing_input_graph_replays_preserve_writeback(self):
        torch.manual_seed(41022)
        owner = unit()
        for leading in ((1,), (5,), (2, 3)):
            with self.subTest(leading=leading), patch.object(
                delayed,
                "post_delayed",
                side_effect=AssertionError("Selected graph used default dispatch"),
            ):
                static = values(leading)
                for _ in range(3):
                    owner.post(*static)
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    captured = owner.post(*static)
                self.assertIs(captured, static[1])
                for _ in range(3):
                    changed = values(leading)
                    expected = owner.post(
                        changed[0], changed[1].clone(), changed[2], changed[3]
                    ).clone()
                    for destination, source in zip(static, changed):
                        destination.copy_(source)
                    graph.replay()
                    exact(captured, expected)
        print("DSV41_ORDERED_POST_GRAPHS captures=3 changing_replays=9", flush=True)


if __name__ == "__main__":
    unittest.main()
