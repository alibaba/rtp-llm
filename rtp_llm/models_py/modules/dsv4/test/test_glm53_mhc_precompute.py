"""Exact mHC precomputed weight and CPU submission regression on CUDA."""

import importlib
import json
import os
import statistics
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from rtp_llm.models_py.modules.dsv4 import tilelang_kernels  # noqa: F401
from rtp_llm.models_py.modules.dsv4.hc.tilelang_impl import TileLangHCUnit

_ops = importlib.import_module(
    "rtp_llm.models_py.3rdparty.tile_kernels.modeling.mhc.ops.pre_big_fuse"
)


class TestMhcPrecompute(unittest.TestCase):
    @torch.inference_mode()
    def test_exact_and_submission(self):
        self.assertTrue(torch.cuda.is_available())
        torch.manual_seed(731)
        records = []
        for dim in (256, 4096, 6144):
            fn = torch.randn(24, 4 * dim, device="cuda") * 0.02
            bits = torch.tensor(
                [0, -2147483648, 0x3F800FFF, 0x3F801000, -1082126337, 1, -2147483647],
                dtype=torch.int32,
                device="cuda",
            )
            fn.view(torch.int32).flatten()[: bits.numel()].copy_(bits)
            original = fn.clone()
            scale = torch.ones(3, device="cuda")
            base = torch.randn(24, device="cuda") * 0.02
            layers = []
            for enabled in ("0", "1"):
                with patch.dict(
                    os.environ,
                    {
                        "GLM53_MHC_PRECOMPUTE_FN": enabled,
                        "DSV4_MHC_PRE_GEMM_BACKEND": "tilelang_single",
                    },
                ):
                    layers.append(
                        TileLangHCUnit(
                            fn,
                            base,
                            scale,
                            dim=dim,
                            hc_mult=4,
                            hc_sinkhorn_iters=20,
                            norm_eps=1e-6,
                            hc_eps=1e-6,
                        )
                    )
            self.assertIsNone(layers[0]._fn_rounded)
            self.assertTrue(torch.equal(layers[1]._fn_rounded, _ops.round_to_tf32(fn)))
            self.assertNotIn("_fn_rounded", layers[1].state_dict())
            with patch.dict(
                os.environ, {"DSV4_MHC_PRE_GEMM_BACKEND": "tilelang_single"}
            ):
                for tokens in (1, 31, 80, 81, 257, 1021, 4093):
                    x = torch.randn(tokens, 4, dim, device="cuda", dtype=torch.bfloat16)
                    expected = layers[0].pre(x)
                    with patch.object(
                        _ops,
                        "round_to_tf32",
                        side_effect=AssertionError("weight rounded during forward"),
                    ):
                        actual = layers[1].pre(x)
                    for a, b in zip(expected, actual):
                        dtype = (
                            torch.int16 if a.dtype == torch.bfloat16 else torch.int32
                        )
                        self.assertTrue(
                            torch.equal(a.view(dtype), b.view(dtype)), (dim, tokens)
                        )
                    if dim == 256:
                        continue
                    row = {
                        "dim": dim,
                        "tokens": tokens,
                        "weight_bytes": fn.numel() * fn.element_size(),
                    }
                    for name, layer in zip(("baseline", "candidate"), layers):
                        for _ in range(10):
                            layer.pre(x)
                        gpu, cpu = [], []
                        for _ in range(3):
                            torch.cuda.synchronize()
                            start, end = torch.cuda.Event(
                                enable_timing=True
                            ), torch.cuda.Event(enable_timing=True)
                            start.record()
                            before = time.perf_counter()
                            for _ in range(50):
                                layer.pre(x)
                            cpu.append((time.perf_counter() - before) * 1e6 / 50)
                            end.record()
                            end.synchronize()
                            gpu.append(start.elapsed_time(end) * 1000 / 50)
                        row[name] = {
                            "cpu_submit_us": statistics.median(cpu),
                            "event_stream_us": statistics.median(gpu),
                        }
                    records.append(row)
            self.assertTrue(
                torch.equal(fn.view(torch.int32), original.view(torch.int32))
            )
        output = Path(os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR", "/tmp"))
        (output / "mhc_precompute.json").write_text(
            json.dumps(
                {
                    "metric": "CUDA event stream latency including CPU launch gaps, not kernel-only",
                    "warmup": 10,
                    "iterations": 50,
                    "blocks": 3,
                    "records": records,
                },
                indent=2,
            )
        )
        print(json.dumps(records), flush=True)


if __name__ == "__main__":
    unittest.main()
