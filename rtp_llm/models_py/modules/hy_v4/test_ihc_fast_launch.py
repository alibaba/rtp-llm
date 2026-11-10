import json
import os
import statistics
import time
import unittest
from pathlib import Path
from unittest import mock

import torch

os.environ.setdefault(
    "DG_JIT_CACHE_DIR",
    os.path.join(os.environ.get("TEST_TMPDIR", "/tmp"), "deep_gemm_cache"),
)

from rtp_llm.models_py.modules.hy_v4 import ihc_triton as ihc


class IHCFastLaunchTest(unittest.TestCase):
    def setUp(self):
        self.assertTrue(torch.cuda.is_available())
        self.assertEqual(torch.cuda.get_device_capability()[0], 10)
        self.assertTrue(ihc._has_deepgemm_prenorm())
        torch.manual_seed(71)
        self.env = mock.patch.dict(
            os.environ, {"RTP_LLM_HY4_IHC_PRE_BACKEND": "deepgemm"}
        )
        self.env.start()
        self.addCleanup(self.env.stop)
        self.inputs = [
            torch.randn(82, 4, 6144, device="cuda", dtype=torch.bfloat16),
            torch.randn(8, 4 * 6144, device="cuda") * 0.01,
            torch.tensor([0.1, 0.2], device="cuda"),
            torch.randn(8, device="cuda") * 0.1,
            torch.randn(6144, device="cuda", dtype=torch.bfloat16),
        ]

    def run_pair(self, emit_mxfp8=False):
        result = ihc.maybe_fused_ihc_pre_normed_grouped(
            *self.inputs,
            magnitude=2.0,
            hc_eps=1e-6,
            ihc_norm_eps=1e-5,
            read_norm_eps=1e-5,
            chunk_size=1024,
            emit_mxfp8=emit_mxfp8,
        )
        self.assertIsNotNone(result)
        read, gate = result[:2]
        post = ihc.maybe_fused_ihc_post(read, self.inputs[0], gate)
        self.assertIsNotNone(post)
        return (read, gate, post, *result[2:])

    def oracle(self):
        channels, weight, scale, base, norm = self.inputs
        flat = channels.flatten(1).float()
        mixes = torch.nn.functional.linear(flat, weight) * torch.rsqrt(
            flat.square().mean(-1, keepdim=True) + 1e-5
        )
        pre = torch.sigmoid(mixes[:, :4] * scale[0] + base[:4]) + 1e-6
        gate = 2 * torch.sigmoid(mixes[:, 4:] * scale[1] + base[4:]) + 1e-6
        read = (channels.float() * pre.unsqueeze(-1)).sum(1).bfloat16().float()
        read = (
            read
            * torch.rsqrt(read.square().mean(-1, keepdim=True) + 1e-5)
            * norm.float()
        ).bfloat16()
        post = (
            channels.float() + gate.unsqueeze(-1) * read.float().unsqueeze(1)
        ).bfloat16()
        return read, gate, post

    def assert_oracle(self, actual):
        read, gate, _ = self.oracle()
        post = (
            self.inputs[0].float()
            + actual[1].unsqueeze(-1) * actual[0].float().unsqueeze(1)
        ).bfloat16()
        for index, (value, reference) in enumerate(zip(actual, (read, gate, post))):
            tolerance = 5e-4 if index == 1 else 2e-2
            torch.testing.assert_close(value, reference, rtol=tolerance, atol=tolerance)

    def record_oracle_diagnostics(self, baseline, fast):
        expected = self.oracle()
        error = (baseline[2].float() - expected[2].float()).abs()
        failures = error > (0.02 + 0.02 * expected[2].float().abs())
        witnesses = []
        for position in failures.nonzero()[:8].tolist():
            row, channel, column = position
            witnesses.append(
                {
                    "index": position,
                    "standard_post": baseline[2][row, channel, column].item(),
                    "fast_post": fast[2][row, channel, column].item(),
                    "whole_chain_oracle_post": expected[2][row, channel, column].item(),
                    "residual": self.inputs[0][row, channel, column].item(),
                    "standard_read": baseline[0][row, column].item(),
                    "oracle_read": expected[0][row, column].item(),
                    "standard_gate": baseline[1][row, channel].item(),
                    "oracle_gate": expected[1][row, channel].item(),
                }
            )
        report = {
            "tokens": self.inputs[0].shape[0],
            "whole_chain_standard_post_mismatch_count": int(failures.sum()),
            "whole_chain_standard_post_max_abs": error.max().item(),
            "standard_fast_bitexact": all(
                self.tensors_equal(a, b) for a, b in zip(baseline, fast)
            ),
            "witnesses": witnesses,
        }
        root = Path(os.environ["TEST_UNDECLARED_OUTPUTS_DIR"])
        root.mkdir(parents=True, exist_ok=True)
        with (root / "ihc_oracle_diagnostics.jsonl").open("a") as output:
            output.write(json.dumps(report) + "\n")

    @staticmethod
    def tensors_equal(value, reference):
        if value.dtype == torch.float8_e4m3fn:
            return torch.equal(value.view(torch.uint8), reference.view(torch.uint8))
        return torch.equal(value, reference)

    def assert_exact(self, actual, expected):
        self.assertEqual(len(actual), len(expected))
        for value, reference in zip(actual, expected):
            self.assertTrue(self.tensors_equal(value, reference))

    def verify_stream_graph(self, emit_mxfp8=False):
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for tensor in self.inputs:
                tensor.add_(0.015625)
            streamed = self.run_pair(emit_mxfp8)
            self.assert_oracle(streamed)
            if emit_mxfp8:
                from rtp_llm.models_py.kernels.cuda.mxfp8_ops import (
                    mxfp8_quant_act_packed,
                )

                with mock.patch.dict(os.environ, {"RTP_LLM_MXFP8_FUSED_QUANT": "0"}):
                    reference = mxfp8_quant_act_packed(streamed[0])
                self.assert_exact(streamed[3:], reference)
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = self.run_pair(emit_mxfp8)
        for tensor in self.inputs:
            tensor.add_(0.015625)
        graph.replay()
        eager = self.run_pair(emit_mxfp8)
        self.assert_exact(captured, eager)

    @torch.no_grad()
    def test_single_token_draft(self):
        self.inputs[0] = self.inputs[0][:1].clone()
        with mock.patch.object(ihc._cpu_launch, "ENABLED", False):
            baseline = self.run_pair()
        with mock.patch.object(ihc._cpu_launch, "ENABLED", True):
            fast = self.run_pair()
            self.assert_exact(fast, baseline)
            self.record_oracle_diagnostics(baseline, fast)
            self.assert_oracle(fast)
            self.verify_stream_graph()

    @torch.no_grad()
    def test_mxfp8_single_token_and_prefill(self):
        from rtp_llm.models_py.kernels.cuda.mxfp8_ops import mxfp8_quant_act_packed

        all_channels = self.inputs[0]
        for tokens in (1, 82):
            with self.subTest(tokens=tokens):
                self.inputs[0] = all_channels[:tokens].clone()
                with mock.patch.object(ihc._cpu_launch, "ENABLED", False):
                    baseline = self.run_pair(emit_mxfp8=True)
                with mock.patch.object(ihc._cpu_launch, "ENABLED", True):
                    fast = self.run_pair(emit_mxfp8=True)
                    self.assert_exact(fast, baseline)
                    self.record_oracle_diagnostics(baseline, fast)
                    self.assert_oracle(fast)
                    with mock.patch.dict(
                        os.environ, {"RTP_LLM_MXFP8_FUSED_QUANT": "0"}
                    ):
                        reference = mxfp8_quant_act_packed(fast[0])
                    self.assert_exact(fast[3:], reference)
                    self.verify_stream_graph(emit_mxfp8=True)

    @torch.no_grad()
    def test_precision_stream_graph_and_host_submit(self):
        with mock.patch.object(ihc._cpu_launch, "ENABLED", False):
            baseline = self.run_pair()
        with mock.patch.object(ihc._cpu_launch, "ENABLED", True):
            fast = self.run_pair()
            self.assert_exact(fast, baseline)
            self.record_oracle_diagnostics(baseline, fast)
            counts_before = dict(ihc._cpu_launch.counts)
            with mock.patch.object(ihc._cpu_launch, "TRACE", True):
                self.run_pair()
            launch_counts = {
                key: value - counts_before[key]
                for key, value in ihc._cpu_launch.counts.items()
            }
            self.assertEqual(launch_counts["jit_miss"], 0)
            self.assertEqual(
                launch_counts["raw"] + launch_counts["compiled_fallback"], 2
            )
            self.assert_oracle(fast)
            self.verify_stream_graph()

        samples = {"baseline": [], "fast": []}
        for enabled in (False, True):
            with mock.patch.object(ihc._cpu_launch, "ENABLED", enabled):
                for _ in range(5):
                    self.run_pair()
        for round_index in range(20):
            for enabled in (False, True) if round_index % 2 == 0 else (True, False):
                with mock.patch.object(ihc._cpu_launch, "ENABLED", enabled):
                    torch.cuda.synchronize()
                    start = time.perf_counter_ns()
                    for _ in range(10):
                        self.run_pair()
                    elapsed = (time.perf_counter_ns() - start) / 10 / 1000
                    samples["fast" if enabled else "baseline"].append(elapsed)
        torch.cuda.synchronize()
        report = {
            "shape": [82, 4, 6144],
            "host_submit_us": samples,
            "median_host_submit_us": {
                key: statistics.median(value) for key, value in samples.items()
            },
            "scope": "DeepGEMM pre plus Triton pre RMSNorm and post; unchanged GPU kernels",
            "warm_pair_launch_counts": launch_counts,
            "precision_bitexact_ab": True,
            "nondefault_stream_oracle": True,
            "graph_changed_all_live_inputs": True,
        }
        root = Path(os.environ["TEST_UNDECLARED_OUTPUTS_DIR"])
        root.mkdir(parents=True, exist_ok=True)
        (root / "ihc_fast_launch_benchmark.json").write_text(
            json.dumps(report, indent=2)
        )
        print(json.dumps(report))


if __name__ == "__main__":
    unittest.main()
