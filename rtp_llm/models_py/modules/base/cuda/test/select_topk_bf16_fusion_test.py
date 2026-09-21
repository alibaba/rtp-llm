"""BF16 cast+topk production dispatch, exact arithmetic and graph regression."""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.models_py.modules.base.cuda.select_topk import SelectTopk
from rtp_llm.ops.compute_ops import SelectTopkOp


def config(k=10, norm=True, experts=512):
    c = ModelConfig()
    c.attn_config.head_num = 1
    c.attn_config.size_per_head = 128
    c.num_layers = c.max_seq_len = 1
    c.vocab_size = 5120
    c.expert_num, c.moe_k, c.has_moe_norm = experts, k, norm
    return c


def outputs(n, k=10, dtype=torch.int64):
    return (
        torch.empty((n, k), device="cuda", dtype=dtype),
        torch.empty((n, k), device="cuda", dtype=torch.float32),
    )


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class SelectTopkBf16FusionTest(unittest.TestCase):
    def assert_bits_equal(self, left, right):
        for a, b in zip(left, right):
            self.assertTrue(torch.equal(a.view(torch.uint8), b.view(torch.uint8)))

    def test_bitwise_and_graph(self):
        torch.manual_seed(20260921)
        c = config()
        old = SelectTopk(c, use_fused_512=True, fuse_bf16_cast=False)
        new = SelectTopk(c, use_fused_512=True, fuse_bf16_cast=True)
        for n in (0, 1, 7, 257, 4095, 4096, 4097, 8192, 16384, 32768, 65536):
            for dtype in (torch.int32, torch.int64):
                with self.subTest(n=n, dtype=dtype):
                    x = torch.randn(n, 512, device="cuda", dtype=torch.bfloat16)
                    ref, got = outputs(n, dtype=dtype), outputs(n, dtype=dtype)
                    old(x, *ref)
                    new(x, *got)
                    self.assert_bits_equal(ref, got)
                    if n in (7, 4097):
                        for _ in range(3):
                            new(x, *got)
                        torch.cuda.synchronize()
                        graph = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(graph):
                            new(x, *got)
                        x.neg_()
                        graph.replay()
                        old(x, *ref)
                        self.assert_bits_equal(ref, got)

    def test_ties_extremes_and_nonfinite(self):
        c = config()
        old = SelectTopk(c, use_fused_512=True, fuse_bf16_cast=False)
        new = SelectTopk(c, use_fused_512=True, fuse_bf16_cast=True)
        for kind in ("zeros", "ties", "extreme", "nonfinite"):
            x = torch.randn(4097, 512, device="cuda", dtype=torch.bfloat16)
            if kind == "zeros":
                x.zero_()
            elif kind == "ties":
                x.copy_(torch.randint(-3, 4, x.shape, device="cuda"))
            elif kind == "extreme":
                x.mul_(1000)
            else:
                x[0, 0], x[1, 1], x[2, 2] = float("nan"), float("inf"), -float("inf")
            ref, got = outputs(4097), outputs(4097)
            old(x, *ref)
            new(x, *got)
            self.assert_bits_equal(ref, got)

    def test_fallbacks(self):
        for c in (config(), config(k=1), config(norm=False), config(experts=256)):
            for kind in ("fp32", "fp16", "strided_bf16"):
                dtype = {
                    "fp32": torch.float32,
                    "fp16": torch.float16,
                    "strided_bf16": torch.bfloat16,
                }[kind]
                x = torch.randn(7, c.expert_num * 2, device="cuda", dtype=dtype)
                x = (
                    x[:, ::2]
                    if kind == "strided_bf16"
                    else x[:, : c.expert_num].contiguous()
                )
                old = SelectTopk(c, use_fused_512=True, fuse_bf16_cast=False)
                new = SelectTopk(c, use_fused_512=True, fuse_bf16_cast=True)
                ref, got = outputs(7, c.moe_k), outputs(7, c.moe_k)
                old(x, *ref)
                new(x, *got)
                self.assert_bits_equal(ref, got)
                if c.moe_k != 10 or c.expert_num != 512 or not c.has_moe_norm:
                    self.assertFalse(new.fuse_bf16_cast)

    def test_environment_dispatch(self):
        x = torch.randn(7, 512, device="cuda", dtype=torch.bfloat16)
        # Explicit enable/disable takes precedence over the environment.
        cases = [
            (None, {}, True, True),
            ("0", {}, False, False),
            ("1", {}, True, True),
            (None, {"use_fused_512": True}, True, True),
            ("0", {"use_fused_512": True}, True, True),
            ("1", {"use_fused_512": False}, False, False),
            ("1", {"fuse_bf16_cast": False}, True, False),
        ]
        for env, kwargs, fused, cast in cases:
            with self.subTest(env=env, kwargs=kwargs), patch.dict(os.environ):
                os.environ.pop("RTP_FUSED_TOPK_512", None)
                if env is not None:
                    os.environ["RTP_FUSED_TOPK_512"] = env
                module = SelectTopk(config(), **kwargs)
                self.assertEqual(module.fuse_bf16_cast, cast)
                reference = SelectTopk(
                    config(), use_fused_512=fused, fuse_bf16_cast=False
                )
                ref, got = outputs(7), outputs(7)
                reference(x, *ref)
                module(x, *got)
                self.assert_bits_equal(ref, got)
                # Inspect the native boundary to detect a standalone cast.
                with patch(
                    "rtp_llm.models_py.modules.base.cuda.select_topk.compute_ops.SelectTopkOp"
                ) as native:
                    probe = SelectTopk(config(), **kwargs)
                    probe(x, *got)
                    native.assert_called_once_with(probe.config, use_fused_512=fused)
                    forwarded = native.return_value.forward.call_args.args[0]
                    self.assertEqual(
                        forwarded.dtype, torch.bfloat16 if cast else torch.float32
                    )
                    if cast:
                        self.assertIs(forwarded, x)

    def test_native_validation(self):
        op = SelectTopkOp(config(), True)
        x = torch.zeros(7, 512, device="cuda", dtype=torch.bfloat16)
        ids, weights = outputs(7)
        invalid = [
            (x[:, :511], ids, weights),
            (x.cpu(), ids, weights),
            (x, ids[:, :9], weights),
            (x, ids.float(), weights),
            (x, ids, weights.bfloat16()),
        ]
        for args in invalid:
            with self.assertRaises(RuntimeError):
                op.forward(*args)
        for c in (config(k=1), config(norm=False), config(experts=256)):
            with self.assertRaises(RuntimeError):
                SelectTopkOp(c, True).forward(x, ids, weights)

    def test_single_routing_launch(self):
        with patch.dict(os.environ, {"RTP_FUSED_TOPK_512": "1"}):
            new = SelectTopk(config())
        x = torch.randn(4097, 512, device="cuda", dtype=torch.bfloat16)
        out = outputs(4097)
        for _ in range(5):
            new(x, *out)
        torch.cuda.synchronize()
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ]
        ) as prof:
            for _ in range(3):
                new(x, *out)
            torch.cuda.synchronize()
        with tempfile.TemporaryDirectory() as tmp:
            trace = Path(tmp) / "trace.json"
            prof.export_chrome_trace(str(trace))
            kernels = [
                e
                for e in json.loads(trace.read_text())["traceEvents"]
                if e.get("cat") == "kernel"
            ]
        self.assertEqual(len(kernels), 3, [e["name"] for e in kernels])
        self.assertTrue(
            all(
                "topkGatingSoftmax" in e["name"] and "bfloat16" in e["name"]
                for e in kernels
            )
        )


if __name__ == "__main__":
    unittest.main()
