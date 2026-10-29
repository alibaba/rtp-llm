"""SM90 prefill-score dispatch regression against a frozen prior baseline.

Check FP64 score accuracy, causal top-512 membership, shape rejection,
CUDA Graph input refresh, and whole-call timings on synthetic inputs.
"""

import ast
import contextlib
import hashlib
import json
import os
import unittest
from unittest.mock import patch

import indexer_prefill_score_preintegration as frozen
import torch
import triton
from indexer_prefill_score_test_utils import (
    _profile_arm,
    _summarize,
    _timed_step,
    make_inputs,
    reference_fp64_chunked,
    run_fp32_fallback,
    tail256_gate,
    tf32_disabled,
)

from rtp_llm.models_py.modules.qwen4_exp.qsa import indexer_prefill_score as live

FROZEN_SHA = "3a9718ea817ad772d0795ee5c43cb3047d622b68ede4ddeb8f2addfa909cea82"


def sha(path):
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def save(name, record):
    with open(
        os.path.join(os.environ["TEST_UNDECLARED_OUTPUTS_DIR"], name), "w"
    ) as handle:
        json.dump(record, handle, indent=2)


def dispatch(module, q, k):
    out = module.try_prefill_score(q, k)
    return run_fp32_fallback(q, k) if out is None else out


@contextlib.contextmanager
def capture(module):
    original, calls = module._prefill_score, []

    class Capture:
        def __getitem__(self, grid):
            launcher = original[grid]

            def wrapped(*args, **kwargs):
                compiled = launcher(*args, **kwargs)
                meta = getattr(compiled, "metadata", None)
                calls.append(
                    {
                        "grid": list(grid),
                        "BM": kwargs.get("BM"),
                        "BN": kwargs.get("BN"),
                        "warps": kwargs.get("num_warps"),
                        "stages": kwargs.get("num_stages"),
                        "n_regs": getattr(compiled, "n_regs", None),
                        "shared_bytes": getattr(meta, "shared", None),
                        "compiled_hash": str(getattr(compiled, "hash", "unavailable")),
                    }
                )
                return compiled

            return wrapped

    with patch.object(module, "_prefill_score", Capture()):
        yield calls


def captured_dispatch(module, q, k):
    with capture(module) as calls:
        out = module.try_prefill_score(q, k)
    fused = out is not None
    return (out if fused else run_fp32_fallback(q, k)), fused, calls


def full_gate(test, a, b, q, k):
    reference = reference_fp64_chunked(q, k)
    errors = {}
    for name, scores in (("prechange", a), ("new_dispatch", b)):
        test.assertTrue(bool(torch.isfinite(scores).all()), name)
        torch.testing.assert_close(scores.double(), reference, atol=5e-6, rtol=5e-6)
        errors[name] = float((scores.double() - reference).abs().max())
    length, n = q.shape[1], k.shape[1]
    # Rectangular scorer inputs can have N < S/4. Metadata is clipped to N.
    visible_count = ((torch.arange(length, device=q.device) + 1) // 4).clamp(max=n)
    visible = torch.arange(n, device=q.device)[None] < visible_count[:, None]
    width = min(512, n)
    at = a.masked_fill(~visible[None], float("-inf")).topk(width, dim=-1, sorted=True)
    bt = b.masked_fill(~visible[None], float("-inf")).topk(width, dim=-1, sorted=True)
    gate = tail256_gate(
        at.indices.reshape(-1, width),
        bt.indices.reshape(-1, width),
        at.values.reshape(-1, width),
        bt.values.reshape(-1, width),
        visible_count.repeat(q.shape[0]),
        n,
    )
    test.assertTrue(gate["all_rows_pass"], gate)
    return {
        "FP64_atol": 5e-6,
        "FP64_rtol": 5e-6,
        "max_abs_errors": errors,
        "tail256": gate,
    }


def fixtures():
    rows = []
    for s in (5120, 10240, 15360, 20480):
        for kind, seed in (
            ("old_correctness", 31000 + s),
            ("exact_benchmark", 20261029 + s),
            ("independent_1", 741 + s),
            ("independent_2", 17011 + s),
            ("independent_3", 93209 + s),
        ):
            rows.append(dict(kind=kind, S=s, N=s // 4, seed=seed, B=1, strided=False))
    for s in (8191, 8192, 8193):
        rows.append(
            dict(kind="boundary", S=s, N=s // 4, seed=83000 + s, B=1, strided=False)
        )
    for kind, s, seed, batch, stride in (
        ("batch2", 10240, 123009, 2, False),
        ("batch2_strided", 10240, 123010, 2, True),
        ("small127", 127, 60201, 1, False),
        ("small128", 128, 60202, 1, False),
        ("zeros", 4096, 60203, 1, False),
        ("identical_keys_ties", 4096, 60204, 1, False),
        ("negative_relu_zero", 4096, 60205, 1, False),
    ):
        rows.append(dict(kind=kind, S=s, N=s // 4, seed=seed, B=batch, strided=stride))
    for i, (kind, s, n) in enumerate(
        (
            ("corner_lowS_highN", 128, 5120),
            ("corner_highS_lowN", 20480, 32),
            ("ragged_highS", 20479, 5119),
            ("ragged_highN", 20480, 5119),
            ("upper_corner", 20480, 5120),
            ("reject_S", 20481, 32),
            ("reject_N", 128, 5121),
        )
    ):
        rows.append(dict(kind=kind, S=s, N=n, seed=82001 + i, B=1, strided=False))
    return rows


class IntegrationTest(unittest.TestCase):
    def check_frozen_and_kernel(self):
        self.assertEqual(sha(frozen.__file__), FROZEN_SHA)

        def kernel_ast(module):
            with open(module.__file__) as handle:
                return ast.dump(
                    next(
                        n
                        for n in ast.parse(handle.read()).body
                        if isinstance(n, ast.FunctionDef) and n.name == "_prefill_score"
                    )
                )

        self.assertEqual(kernel_ast(frozen), kernel_ast(live))

    def base_record(self):
        self.check_frozen_and_kernel()
        return {
            "contract": "integration-tail256-v1",
            "status": "running",
            "frozen_module_sha256": FROZEN_SHA,
            "new_module_sha256": sha(live.__file__),
            "harness_sha256": sha(__file__),
            "scope": "synthetic prefill-score regression",
            "baseline": "frozen prechange try_prefill_score, then original FP32 fallback on None",
            "candidate": "real new try_prefill_score, then original FP32 fallback on None",
        }

    def gpu_metadata(self):
        props = torch.cuda.get_device_properties(0)
        return {
            "name": props.name,
            "uuid": str(getattr(props, "uuid", "unavailable")),
            "capability": list(torch.cuda.get_device_capability(0)),
            "total_memory_bytes": props.total_memory,
            "sm_count": props.multi_processor_count,
            "torch": str(torch.__version__),
            "triton": str(triton.__version__),
            "cuda": torch.version.cuda,
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        }

    def test_cpu_contract_positive_and_adversarial_rows(self):
        records = []

        def check(
            name,
            a,
            b,
            visible=1024,
            n=1024,
            expected=True,
            scores_a=None,
            scores_b=None,
        ):
            width = min(512, n)
            sa = (
                torch.arange(width, 0, -1, dtype=torch.float32)
                if scores_a is None
                else scores_a
            )
            sb = (
                torch.arange(width, 0, -1, dtype=torch.float32)
                if scores_b is None
                else scores_b
            )
            result = tail256_gate(
                torch.tensor([a], dtype=torch.int64),
                torch.tensor([b], dtype=torch.int64),
                sa[None],
                sb[None],
                torch.tensor([visible]),
                n,
            )
            records.append({"name": name, "expected_pass": expected, "result": result})
            self.assertEqual(result["all_rows_pass"], expected, name)
            return result

        base = list(range(512))
        check("exact_zero", base, base)
        check("one_tail", base, base[:-1] + [600])
        r = check("three_tail", base, base[:-3] + [600, 601, 602])
        self.assertEqual((r["max_replacements"], r["max_symmetric_difference"]), (3, 6))
        check("four_tail", base, base[:-4] + [600, 601, 602, 603], expected=False)
        rank256 = base.copy()
        rank256[255] = 600
        check("rank256_both_fail", base, rank256, expected=False)
        rank257 = base.copy()
        rank257[256] = 600
        check("rank257_pass", base, rank257)
        check("A_head_only_fail", base, base[:255] + base[256:] + [600], expected=False)
        check("B_head_only_fail", base, [600] + base[:-1], expected=False)
        check("common_internal_reorder", base, list(reversed(base)))
        duplicate = base.copy()
        duplicate[-1] = 510
        check("duplicate_finite", base, duplicate, expected=False)
        check("negative_finite", base, base[:-1] + [-1], expected=False)
        check("outside_visible_finite", base, base[:-1] + [1024], expected=False)
        for value, name in ((float("nan"), "nan"), (float("inf"), "positive_inf")):
            sb = torch.arange(512, 0, -1, dtype=torch.float32)
            sb[-1] = value
            check(name, base, base, scores_b=sb, expected=False)
        short_scores = torch.full((512,), float("-inf"))
        short_scores[:300] = torch.arange(300, 0, -1)
        check(
            "short_exact",
            base,
            base,
            visible=300,
            scores_a=short_scores,
            scores_b=short_scores,
        )
        placeholders = list(range(300)) + [-1] * 212
        check(
            "short_sentinel_excluded",
            base,
            placeholders,
            visible=300,
            scores_a=short_scores,
            scores_b=short_scores,
        )
        missing = short_scores.clone()
        missing[299] = float("-inf")
        check(
            "short_missing_legal",
            base,
            base,
            visible=300,
            scores_a=short_scores,
            scores_b=missing,
            expected=False,
        )
        check(
            "N_under512",
            list(range(128)),
            list(reversed(range(128))),
            visible=128,
            n=128,
        )
        empty = torch.full((512,), float("-inf"))
        check(
            "empty_visible", base, [-1] * 512, visible=0, scores_a=empty, scores_b=empty
        )
        check("N_zero", [], [], visible=0, n=0)
        out = os.path.join(
            os.environ["TEST_UNDECLARED_OUTPUTS_DIR"], "tail256_cpu_contract.json"
        )
        with open(out, "w") as handle:
            json.dump({"contract": "tail256-v1", "cases": records}, handle, indent=2)
        print(f"[tail256-cpu] {len(records)} independent contract cases passed")

    def test_cpu_mock_guards(self):
        record = self.base_record()
        record["scope"] = (
            "CPU guard mocks only; not execution on other GPU architectures"
        )
        record["cases"] = []

        class Fake:
            def __init__(
                self, shape, dtype=torch.bfloat16, cuda=True, device="cuda:0", stride=1
            ):
                self.shape, self.dtype, self.is_cuda, self.device, self.last_stride = (
                    shape,
                    dtype,
                    cuda,
                    torch.device(device),
                    stride,
                )

            def dim(self):
                return len(self.shape)

            def stride(self, index):
                return self.last_stride

        class Dummy:
            def __getitem__(self, grid):
                def call(*args, **kw):
                    self.kw = kw

                return call

        for major in (7, 8, 9, 10, 12):
            for s, n in (
                (128, 32),
                (5120, 1280),
                (8192, 2048),
                (8193, 2048),
                (128, 2049),
                (20480, 5120),
                (20481, 32),
                (128, 5121),
                (127, 32),
                (128, 31),
            ):
                q, k = Fake((1, s, 4, 128)), Fake((1, n, 128))
                dummy = Dummy()
                sentinel = object()
                expected = (
                    major >= 8
                    and 128 <= s <= (20480 if major == 9 else 8192)
                    and 32 <= n <= (5120 if major == 9 else 2048)
                )
                with patch.object(
                    torch.cuda, "get_device_capability", return_value=(major, 0)
                ), patch.object(torch, "empty", return_value=sentinel), patch.object(
                    live, "_prefill_score", dummy
                ):
                    out = live.try_prefill_score(q, k)
                self.assertEqual(out is sentinel, expected, (major, s, n))
                self.assertTrue(out is sentinel or out is None)
                if expected:
                    self.assertEqual(
                        (
                            dummy.kw["BM"],
                            dummy.kw["BN"],
                            dummy.kw["num_warps"],
                            dummy.kw["num_stages"],
                        ),
                        (64, 128, 4, 1),
                    )
                record["cases"].append(
                    {
                        "major_mock": major,
                        "S": s,
                        "N": n,
                        "expected_fused": expected,
                        "pass": True,
                    }
                )
        goodq, goodk = Fake((1, 128, 4, 128)), Fake((1, 32, 128))
        bad = (
            ("cpu", Fake((1, 128, 4, 128), cuda=False, device="cpu"), goodk),
            ("dtype", Fake((1, 128, 4, 128), dtype=torch.float32), goodk),
            ("heads", Fake((1, 128, 8, 128)), goodk),
            ("head_dim", Fake((1, 128, 4, 64)), Fake((1, 32, 64))),
            ("q_rank", Fake((128, 4, 128)), goodk),
            ("k_rank", goodq, Fake((1, 1, 32, 128))),
            ("batch", goodq, Fake((2, 32, 128))),
            ("stride", Fake((1, 128, 4, 128), stride=2), goodk),
            ("device", Fake((1, 128, 4, 128), device="cuda:1"), goodk),
        )
        for name, q, k in bad:
            with patch.object(
                torch.cuda, "get_device_capability", return_value=(9, 0)
            ) as capability, patch.object(live, "_prefill_score", Dummy()):
                self.assertIsNone(live.try_prefill_score(q, k))
                capability.assert_not_called()
            record["cases"].append({"hard_reject": name, "pass": True})
        with patch.object(torch.version, "hip", "mock-hip"), patch.object(
            torch.cuda, "get_device_capability", return_value=(9, 0)
        ) as capability:
            self.assertIsNone(live.try_prefill_score(goodq, goodk))
            capability.assert_not_called()
        record["cases"].append({"hard_reject": "HIP_early_return", "pass": True})
        record["status"] = "PASS"
        save("integration_cpu_guards.json", record)

    def test_real_dispatch_matrix(self):
        self.assertEqual(torch.cuda.get_device_capability()[0], 9)
        record = self.base_record()
        record["gpu"] = self.gpu_metadata()
        record["fixtures"] = []
        try:
            for spec in fixtures():
                q, k = make_inputs(
                    spec["S"],
                    blocks=spec["N"],
                    seed=spec["seed"],
                    batch=spec["B"],
                    strided=spec["strided"],
                )
                if spec["kind"] == "zeros":
                    q.zero_()
                    k.zero_()
                elif spec["kind"] == "identical_keys_ties":
                    k.copy_(k[:, :1].expand_as(k).clone())
                elif spec["kind"] == "negative_relu_zero":
                    q.fill_(1)
                    k.fill_(-1)
                entry = {**spec, "status": "running"}
                record["fixtures"].append(entry)
                with tf32_disabled():
                    a, old_fused, old_calls = captured_dispatch(frozen, q, k)
                    b, new_fused, new_calls = captured_dispatch(live, q, k)
                    self.assertEqual(
                        new_fused, 128 <= spec["S"] <= 20480 and 32 <= spec["N"] <= 5120
                    )
                    self.assertEqual(
                        old_fused, 128 <= spec["S"] <= 8192 and 32 <= spec["N"] <= 2048
                    )
                    self.assertEqual(len(old_calls), int(old_fused))
                    self.assertEqual(len(new_calls), int(new_fused))
                    entry.update(
                        prechange_fused=old_fused,
                        new_fused=new_fused,
                        old_kernel_calls=old_calls,
                        new_kernel_calls=new_calls,
                    )
                    entry["gate"] = full_gate(self, a, b, q, k)
                entry["status"] = "PASS"
                save("integration_matrix.json", record)
                print(
                    f"[integration] {spec['kind']} S={spec['S']} N={spec['N']} old_fused={old_fused} new_fused={new_fused} PASS"
                )
            record["status"] = "PASS"
        except Exception as exc:
            record["status"] = "FAIL"
            record["failure"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            save("integration_matrix.json", record)

    def test_real_dispatch_graph_input_refresh(self):
        record = self.base_record()
        record["gpu"] = self.gpu_metadata()
        record["replays"] = []
        try:
            for length in (10240, 20480):
                q, k = make_inputs(length, seed=95000 + length, batch=1)
                for _ in range(3):
                    self.assertIsNotNone(live.try_prefill_score(q, k))
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with capture(live) as calls:
                    with torch.cuda.graph(graph):
                        out = live.try_prefill_score(q, k)
                self.assertEqual(len(calls), 1)
                self.assertIsNotNone(out)
                previous = None
                for step in range(2):
                    nq, nk = make_inputs(length, seed=95100 + length + step, batch=1)
                    q.copy_(nq)
                    k.copy_(nk)
                    graph.replay()
                    torch.cuda.synchronize()
                    with tf32_disabled():
                        a = dispatch(frozen, q, k)
                        gate = full_gate(self, a, out, q, k)
                    if previous is not None:
                        self.assertFalse(torch.equal(previous, out))
                    previous = out.clone()
                    record["replays"].append(
                        {
                            "S": length,
                            "step": step,
                            "capture_calls": calls,
                            "gate": gate,
                        }
                    )
            record["status"] = "PASS"
        except Exception as exc:
            record["status"] = "FAIL"
            record["failure"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            save("integration_graph.json", record)

    def test_prechange_vs_real_dispatch_abba(self):
        record = self.base_record()
        record["gpu"] = self.gpu_metadata()
        record["shapes"] = []
        record["protocol"] = {
            "warmup": 3,
            "rounds": 3,
            "order": ["prechange", "new_dispatch", "new_dispatch", "prechange"],
            "samples_per_step": 5,
            "timing": "CUDA event whole-call; gates/profile excluded",
        }
        arms = {
            "prechange": lambda q, k: dispatch(frozen, q, k),
            "new_dispatch": lambda q, k: dispatch(live, q, k),
        }
        try:
            with tf32_disabled():
                for length in (5120, 10240, 15360, 20480):
                    q, k = make_inputs(length, seed=20261029 + length, batch=1)
                    shape = {
                        "S": length,
                        "N": length // 4,
                        "B": 1,
                        "seed": 20261029 + length,
                        "status": "running",
                    }
                    record["shapes"].append(shape)
                    a, af, ac = captured_dispatch(frozen, q, k)
                    b, bf, bc = captured_dispatch(live, q, k)
                    self.assertTrue(bf)
                    self.assertEqual(len(bc), 1)
                    shape.update(
                        prechange_fused=af,
                        new_fused=bf,
                        prechange_calls=ac,
                        new_calls=bc,
                        gate=full_gate(self, a, b, q, k),
                    )
                    del a, b
                    torch.cuda.empty_cache()
                    shape["peak_alloc_bytes"] = {}
                    for name, fn in arms.items():
                        base = torch.cuda.memory_allocated()
                        torch.cuda.reset_peak_memory_stats()
                        fn(q, k)
                        torch.cuda.synchronize()
                        peak = torch.cuda.max_memory_allocated()
                        shape["peak_alloc_bytes"][name] = {
                            "base": base,
                            "absolute_peak": peak,
                            "extra": peak - base,
                        }
                        for _ in range(3):
                            fn(q, k)
                    torch.cuda.synchronize()
                    samples = []
                    for rnd in range(3):
                        for step, name in enumerate(record["protocol"]["order"]):
                            for i, ms in enumerate(_timed_step(arms[name], q, k, 5)):
                                samples.append(
                                    dict(
                                        round=rnd,
                                        step=step,
                                        arm=name,
                                        sample=i,
                                        elapsed_ms=ms,
                                    )
                                )
                    shape["samples"] = samples
                    shape["summary_ms"] = {
                        name: _summarize(samples, name) for name in arms
                    }
                    for summary in shape["summary_ms"].values():
                        self.assertEqual(summary["count"], 30)
                    shape["profile"] = {
                        name: _profile_arm(fn, q, k) for name, fn in arms.items()
                    }
                    shape["status"] = "PASS"
                    shape["median_speedup"] = (
                        shape["summary_ms"]["prechange"]["median_ms"]
                        / shape["summary_ms"]["new_dispatch"]["median_ms"]
                    )
                    save("integration_abba.json", record)
                    print(
                        f"[integration-abba] S={length} {shape['summary_ms']} speedup={shape['median_speedup']:.4f}"
                    )
            record["status"] = "PASS"
        except Exception as exc:
            record["status"] = "FAIL"
            record["failure"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            save("integration_abba.json", record)


if __name__ == "__main__":
    unittest.main()
