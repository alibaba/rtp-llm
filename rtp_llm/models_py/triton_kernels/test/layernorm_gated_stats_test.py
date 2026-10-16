"""Omitting unused normalization statistics must preserve every output bit."""

import json
import os
import statistics
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
import triton
from rtp_llm.models_py.triton_kernels.common import layernorm_gated
from rtp_llm.models_py.triton_kernels.common.layernorm_gated import (
    RmsNormGated,
    layer_norm_fwd,
)


class LayerNormStatsTest(unittest.TestCase):
    def test_exact_output_and_stats_contract(self):
        torch.manual_seed(5354)
        for dtype in (torch.bfloat16, torch.float16, torch.float32):
            for rows, width, group in (
                (1, 128, 128),
                (13, 512, 128),
                (81, 6144, 256),
                (419, 512, 512),
            ):
                # Noncontiguous row stride exercises grouped address calculations.
                x = torch.randn(rows, width * 2, device="cuda", dtype=dtype)[
                    :, ::2
                ].contiguous()
                x = torch.cat((x, x), dim=1)[:, :width]
                z, weight = torch.randn_like(x), torch.randn(
                    width, device="cuda", dtype=dtype
                )
                for rms in (False, True):
                    for activation in ("silu", "sigmoid"):
                        for bias in (None, torch.randn_like(weight)):
                            for before in (False, True):
                                kwargs = dict(
                                    z=z,
                                    group_size=group,
                                    is_rms_norm=rms,
                                    activation=activation,
                                    norm_before_gate=before,
                                )
                                expected, mean, rstd = layer_norm_fwd(
                                    x, weight, bias, 1e-6, **kwargs
                                )
                                output, no_mean, no_rstd = layer_norm_fwd(
                                    x, weight, bias, 1e-6, return_stats=False, **kwargs
                                )
                                bits = (
                                    torch.int32
                                    if dtype == torch.float32
                                    else torch.int16
                                )
                                self.assertTrue(
                                    torch.equal(output.view(bits), expected.view(bits))
                                )
                                self.assertEqual(rstd.shape, (rows * width // group,))
                                self.assertEqual(mean is None, rms)
                                self.assertIsNone(no_mean)
                                self.assertIsNone(no_rstd)
                                if rms and before:
                                    actual = RmsNormGated(
                                        weight, bias, group, activation=activation
                                    )(x, z)
                                    self.assertTrue(
                                        torch.equal(
                                            actual.view(bits), expected.view(bits)
                                        )
                                    )

    def test_large_input_stride(self):
        storage = torch.empty(2 * (1 << 30) + 128, device="cuda", dtype=torch.bfloat16)
        x = storage.as_strided((3, 128), (1 << 30, 1))
        x.copy_(torch.randn(3, 128, device="cuda", dtype=x.dtype))
        z = torch.randn(3, 128, device="cuda", dtype=x.dtype)
        w = torch.ones(128, device="cuda", dtype=x.dtype)
        expected = layer_norm_fwd(x, w, None, 1e-6, z=z, is_rms_norm=True)[0]
        actual = layer_norm_fwd(
            x, w, None, 1e-6, z=z, is_rms_norm=True, return_stats=False
        )[0]
        self.assertTrue(
            torch.equal(actual.view(torch.int16), expected.view(torch.int16))
        )

    def test_heuristics_submission(self):
        direct = layernorm_gated._layer_norm_fwd_1pass_kernel
        legacy = triton.heuristics({"HAS_BIAS": lambda args: args["B"] is not None})(
            triton.heuristics({"HAS_Z": lambda args: args["Z"] is not None})(direct)
        )
        x = torch.randn(81, 6144, device="cuda", dtype=torch.bfloat16)
        z = torch.randn_like(x)
        w = torch.ones(6144, device="cuda", dtype=x.dtype)
        records = {}
        outputs = []
        for name, kernel in (("heuristics", legacy), ("explicit", direct)):
            with patch.object(layernorm_gated, "_layer_norm_fwd_1pass_kernel", kernel):

                def run():
                    return layer_norm_fwd(
                        x,
                        w,
                        None,
                        1e-6,
                        z=z,
                        group_size=128,
                        is_rms_norm=True,
                        return_stats=False,
                    )[0]

                outputs.append(run())
                for _ in range(20):
                    run()
                cpu = []
                for _ in range(5):
                    torch.cuda.synchronize()
                    before = time.perf_counter()
                    for _ in range(100):
                        run()
                    cpu.append((time.perf_counter() - before) * 1e4)
                torch.cuda.synchronize()
                records[name + "_cpu_us"] = statistics.median(cpu)
        self.assertTrue(
            torch.equal(outputs[0].view(torch.int16), outputs[1].view(torch.int16))
        )
        Path(
            os.environ["TEST_UNDECLARED_OUTPUTS_DIR"], "layernorm_heuristics.json"
        ).write_text(json.dumps(records, indent=2))

    def test_submission(self):
        records = []
        for rows in (1, 81, 419, 9889):
            x = torch.randn(rows, 6144, device="cuda", dtype=torch.bfloat16)
            z = torch.randn_like(x)
            weight = torch.randn(6144, device="cuda", dtype=x.dtype)
            record = {"tokens": rows}
            for stats in (True, False):

                def run():
                    return layer_norm_fwd(
                        x,
                        weight,
                        None,
                        1e-6,
                        z=z,
                        group_size=128,
                        is_rms_norm=True,
                        activation="sigmoid",
                        return_stats=stats,
                    )

                for _ in range(20):
                    run()
                cpu = []
                for _ in range(5):
                    torch.cuda.synchronize()
                    before = time.perf_counter()
                    for _ in range(100):
                        run()
                    cpu.append((time.perf_counter() - before) * 1e4)
                torch.cuda.synchronize()
                record["stats_cpu_us" if stats else "no_stats_cpu_us"] = (
                    statistics.median(cpu)
                )
            records.append(record)
        Path(
            os.environ["TEST_UNDECLARED_OUTPUTS_DIR"], "layernorm_stats.json"
        ).write_text(json.dumps(records, indent=2))


if __name__ == "__main__":
    unittest.main()
