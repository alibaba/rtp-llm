"""512-expert fused routing: actual binding A/B correctness and opt-in benchmarks."""

import itertools
import json
import os
import statistics
import unittest
from pathlib import Path

import torch

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.ops.compute_ops import SelectTopkOp


def config(e=512, k=10, norm=True):
    c = ModelConfig()
    c.attn_config.head_num = 1
    c.attn_config.size_per_head = 128
    c.num_layers = 1
    c.max_seq_len = 1
    c.vocab_size = 5120
    c.expert_num = e
    c.moe_k = k
    c.has_moe_norm = norm
    return c


def outputs(x, k, dtype):
    return (
        torch.empty((x.shape[0], k), device=x.device, dtype=dtype),
        torch.empty((x.shape[0], k), device=x.device, dtype=torch.float32),
    )


class SelectTopk512FusionTest(unittest.TestCase):
    def setUp(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA required")
        torch.manual_seed(20260917)
        self.out = Path(os.environ.get("TOPK_BENCH_OUTPUT", "/tmp/topk512"))
        self.out.mkdir(parents=True, exist_ok=True)

    def test_01_correctness(self):
        results = []
        cases = [
            (n, 512, k, norm, dtype, kind)
            for n, k, norm, dtype, kind in itertools.product(
                [0, 1, 7, 37, 257],
                [1, 10, 16],
                [False, True],
                [torch.int32, torch.int64],
                ["random", "bf16"],
            )
        ]
        cases += [
            (83, 512, 10, norm, dtype, kind)
            for norm, dtype, kind in itertools.product(
                [False, True],
                [torch.int32, torch.int64],
                ["zeros", "ties", "extreme", "noncontiguous"],
            )
        ]
        cases += [
            (n, 512, 10, True, torch.int64, "bf16") for n in [2048, 10007, 24601, 49202]
        ]
        cases += [(37, e, 10, True, torch.int32, "random") for e in [256, 513]]
        for n, e, k, norm, dtype, kind in cases:
            x = torch.randn((n, e), device="cuda", dtype=torch.float32)
            if kind == "bf16":
                x = x.bfloat16().float()
            elif kind == "zeros":
                x.zero_()
            elif kind == "ties":
                x = torch.randint(-3, 4, (n, e), device="cuda").float()
            elif kind == "extreme":
                x.mul_(1000)
            elif kind == "noncontiguous":
                x = torch.randn((e, n), device="cuda").t()
            c = config(e, k, norm)
            baseline, fused = SelectTopkOp(c), SelectTopkOp(c, True)
            bi, bw = outputs(x, k, dtype)
            fi, fw = outputs(x, k, dtype)
            baseline.forward(x, bi, bw)
            fused.forward(x, fi, fw)
            torch.cuda.synchronize()
            label = dict(n=n, e=e, k=k, norm=norm, dtype=str(dtype), kind=kind)
            self.assertTrue(torch.equal(bi, fi), label)
            if not torch.equal(fw, bw):
                diagnostic = dict(
                    case=label,
                    max_abs=float((fw - bw).abs().max()),
                    unequal=int((fw != bw).sum()),
                )
                (self.out / "bitwise-failure.json").write_text(
                    json.dumps(diagnostic, indent=2)
                )
                torch.save(
                    dict(
                        x=x,
                        baseline_ids=bi,
                        fused_ids=fi,
                        baseline_weights=bw,
                        fused_weights=fw,
                    ),
                    self.out / "bitwise-failure.pt",
                )
                print("BITWISE_FAILURE", json.dumps(diagnostic), flush=True)
            self.assertTrue(torch.equal(fw, bw), label)
            if n:
                probs = torch.softmax(x, dim=-1)
                # torch.topk has unspecified tie order; compare values at the exact selected IDs.
                ref = probs.gather(1, fi.long())
                if norm:
                    ref = ref / ref.sum(-1, keepdim=True)
                torch.testing.assert_close(
                    fw, ref, atol=1e-6, rtol=1e-5, msg=str(label)
                )
                self.assertTrue(bool((fi[:, 1:] != fi[:, :-1]).all()), label)
                chosen = probs.gather(1, fi.long())
                inversions = chosen[:, :-1] < chosen[:, 1:]
                if bool(inversions.any()):
                    rr, cc = inversions.nonzero(as_tuple=True)
                    diagnostic = dict(
                        case=label,
                        inversions=int(rr.numel()),
                        left_prob=chosen[rr, cc].tolist(),
                        right_prob=chosen[rr, cc + 1].tolist(),
                        left_weight=fw[rr, cc].tolist(),
                        right_weight=fw[rr, cc + 1].tolist(),
                        baseline_ids_equal=bool(torch.equal(bi, fi)),
                        max_weight_error=float((fw - bw).abs().max()),
                        left_logit=x[rr, fi[rr, cc].long()].tolist(),
                        right_logit=x[rr, fi[rr, cc + 1].long()].tolist(),
                    )
                    (self.out / "sort-diagnostic.json").write_text(
                        json.dumps(diagnostic, indent=2)
                    )
                    torch.save(
                        dict(
                            x=x,
                            baseline_ids=bi,
                            fused_ids=fi,
                            baseline_weights=bw,
                            fused_weights=fw,
                            torch_probs=probs,
                        ),
                        self.out / "sort-diagnostic.pt",
                    )
                    print("SORT_DIAGNOSTIC", json.dumps(diagnostic), flush=True)
                    # RTP CUDA fast math flushes subnormal probabilities to zero.
                    # The baseline and candidate must still agree exactly on IDs.
                    self.assertTrue(
                        bool(
                            (chosen[rr, cc + 1] < torch.finfo(torch.float32).tiny).all()
                        ),
                        label,
                    )
                    self.assertTrue(
                        bool((fw[rr, cc] == 0).all() and (fw[rr, cc + 1] == 0).all()),
                        label,
                    )
                comparable = chosen.masked_fill(
                    chosen < torch.finfo(torch.float32).tiny, 0
                )
                self.assertTrue(
                    bool((comparable[:, :-1] >= comparable[:, 1:]).all()), label
                )
                self.assertTrue(bool((fw[:, :-1] >= fw[:, 1:]).all()), label)
                expected_values = torch.topk(probs, k, dim=-1).values
                if norm:
                    expected_values /= expected_values.sum(-1, keepdim=True)
                torch.testing.assert_close(
                    fw, expected_values, atol=1e-6, rtol=1e-5, msg=str(label)
                )
                sorted_ids = fi.sort(dim=-1).values
                self.assertTrue(
                    bool((sorted_ids[:, 1:] != sorted_ids[:, :-1]).all()), label
                )
                self.assertTrue(bool(((fi >= 0) & (fi < e)).all()), label)
            label["max_weight_abs_error"] = (fw - bw).abs().max().item() if n else 0.0
            results.append(label)
        # Exercise the production Python boundary, including BF16 -> FP32 conversion.
        from unittest.mock import patch

        from rtp_llm.models_py.modules.base.cuda.select_topk import SelectTopk

        x = torch.randn((37, 512), device="cuda", dtype=torch.bfloat16)
        saved = []
        for flag in ["0", "1"]:
            with patch.dict(os.environ, {"RTP_FUSED_TOPK_512": flag}):
                module = SelectTopk(config())
                ids, weights = outputs(x, 10, torch.int64)
                module(x, ids, weights)
                saved.append((ids, weights))
        self.assertTrue(torch.equal(saved[0][0], saved[1][0]))
        self.assertTrue(torch.equal(saved[0][1], saved[1][1]))
        (self.out / "wrapper.json").write_text(
            json.dumps({"flag_0_vs_1": "PASS", "input_dtype": "bfloat16"})
        )
        (self.out / "correctness.json").write_text(json.dumps(results, indent=2))
        print("TOPK_CORRECTNESS_PASS", len(results), flush=True)

    def test_02_benchmark(self):
        if os.environ.get("TOPK_BENCH", "0") != "1":
            self.skipTest("benchmark disabled")
        self.assertTrue(
            (self.out / "correctness.json").exists(), "correctness must pass first"
        )
        rows = []
        for n in [1, 37, 2048, 10007, 24601, 40009]:
            x = torch.randn((n, 512), device="cuda").bfloat16().float()
            c = config()
            ops = {"baseline": SelectTopkOp(c), "fused": SelectTopkOp(c, True)}
            buffers = {name: outputs(x, 10, torch.int64) for name in ops}

            def run(name):
                ops[name].forward(x, *buffers[name])

            for _ in range(30):
                run("baseline")
                run("fused")
            torch.cuda.synchronize()
            samples = {name: [] for name in ops}
            for round_id in range(6):
                for name in (
                    ["baseline", "fused"]
                    if round_id % 2 == 0
                    else ["fused", "baseline"]
                ):
                    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
                        enable_timing=True
                    )
                    torch.cuda.synchronize()
                    start.record()
                    for _ in range(100):
                        run(name)
                    end.record()
                    end.synchronize()
                    samples[name].append(start.elapsed_time(end) * 1000 / 100)
            for name in ops:
                trace = self.out / f"trace-{n}-{name}.json"
                with torch.profiler.profile(
                    activities=[
                        torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA,
                    ]
                ) as p:
                    for _ in range(10):
                        run(name)
                    torch.cuda.synchronize()
                p.export_chrome_trace(str(trace))
                events = [
                    ev
                    for ev in json.loads(trace.read_text())["traceEvents"]
                    if ev.get("cat") == "kernel"
                ]
                expected = (
                    ["moeSoftmax", "moeTopK"]
                    if name == "baseline"
                    else ["topkGatingSoftmax"]
                )
                counts = {
                    needle: sum(needle in ev["name"] for ev in events)
                    for needle in expected
                }
                self.assertEqual(counts, {needle: 10 for needle in expected})
                self.assertEqual(len(events), 10 * len(expected))
                row = dict(
                    n=n,
                    experts=512,
                    k=10,
                    norm=True,
                    name=name,
                    input_dtype="float32 (BF16-origin)",
                    index_dtype="int64",
                    event_stream_us=samples[name],
                    event_stream_median_us=statistics.median(samples[name]),
                    kernel_sum_us_per_iter=sum(ev["dur"] for ev in events) / 10,
                    kernel_names=sorted(set(ev["name"] for ev in events)),
                    launches_per_iter=len(events) / 10,
                    dispatch_verified=True,
                )
                rows.append(row)
                print("TOPK_PERF", json.dumps(row), flush=True)
            (self.out / "performance.json").write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    unittest.main()
