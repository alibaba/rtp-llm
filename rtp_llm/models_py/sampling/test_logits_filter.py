"""Independent threshold reference for RTP logits filtering."""

import hashlib
import json
import math
import os
import statistics
import time
import unittest
from collections import Counter
from collections.abc import Callable
from pathlib import Path

import torch

from rtp_llm.models_py.sampling.logits_filter import filter_logits


def reference_filter(logits: torch.Tensor, k: int, p: float) -> torch.Tensor:
    """Scalar FP64 reference: accumulate whole groups of equal logit values."""
    values = logits.tolist()
    ordered = sorted(values, reverse=True)
    k_cutoff = ordered[min(k, len(values)) - 1] if k else -math.inf
    candidates = [value for value in values if value >= k_cutoff]
    maximum = max(candidates)
    total = math.fsum(math.exp(value - maximum) for value in candidates)
    cutoff = k_cutoff
    if p < 1:
        mass = 0.0
        counts = Counter(candidates)
        for value in sorted(counts, reverse=True):
            mass += counts[value] * math.exp(value - maximum) / total
            cutoff = value
            if mass >= p:
                break
    return logits.masked_fill(logits < cutoff, -math.inf)


class LogitsFilterTest(unittest.TestCase):
    def check_case(self, values: torch.Tensor, k: list[int], p: list[float]) -> None:
        top_k = torch.tensor(k, dtype=torch.int32)
        top_p = torch.tensor(p, dtype=torch.float32)
        source = values.to(device="cuda", dtype=torch.float32)
        before = source.clone()
        actual = filter_logits(source, top_k, top_p).cpu()
        expected = torch.stack(
            [
                reference_filter(row, int(ki), float(pi))
                for row, ki, pi in zip(values, top_k, top_p)
            ]
        )
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(source, before, rtol=0, atol=0)

    def test_mixed_rows_and_ties(self) -> None:
        row = torch.tensor([4.0, 3.0, 3.0, 2.0, 1.0, -math.inf])
        self.check_case(
            row.repeat(8, 1),
            [0, 2, 2, 0, 1, 99, 0, 3],
            [1.0, 1.0, 0.8, 0.8, 0.0, 0.8, 0.0, 0.5],
        )
        self.check_case(torch.zeros(4, 9), [0, 2, 9, 1], [0.5, 1.0, 0.0, 0.7])

    def test_topk_is_normalized_before_topp(self) -> None:
        values = torch.tensor([[0.4, 0.3, 0.2, 0.1]]).log()
        self.check_case(values, [2], [0.55])

    def test_extreme_logits_and_strided_input(self) -> None:
        values = torch.tensor([[10000.0, 9999.0, 9999.0, 9998.0, -10000.0, -math.inf]])
        self.check_case(values.repeat(4, 1), [0, 2, 0, 0], [0.8, 0.8, 0.0, 1.0])
        self.check_case(values.repeat(2, 2)[:, ::2], [2, 0], [0.8, 0.8])

    def test_random_and_large_topk(self) -> None:
        generator = torch.Generator().manual_seed(917)
        values = torch.randn(6, 8193, generator=generator)
        self.check_case(
            values, [0, 1, 40, 4097, 8193, 9000], [0.9, 0.9, 0.9, 0.8, 1.0, 0.95]
        )

    def test_empty_batch(self) -> None:
        values = torch.empty(0, 7, device="cuda", dtype=torch.float32)
        result = filter_logits(
            values, torch.empty(0, dtype=torch.int32), torch.empty(0)
        )
        self.assertIs(result, values)

    @unittest.skipUnless(
        os.environ.get("RTP_LLM_BENCH_LOGITS_FILTER") == "1",
        "opt-in FlashInfer correctness and timing experiment",
    )
    def test_flashinfer_comparison_and_timing(self) -> None:
        import flashinfer
        from flashinfer.sampling import top_k_renorm_probs, top_p_renorm_probs

        artifact_root = Path(
            os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR", ".omo/evidence")
        )
        artifact_root.mkdir(parents=True, exist_ok=True)
        artifact = artifact_root / "logits_filter_benchmark.jsonl"
        vocab = 154880
        iterations = 50
        blocks = 5

        def baseline(
            logits: torch.Tensor, k: torch.Tensor, p: torch.Tensor
        ) -> torch.Tensor:
            probs = logits.softmax(dim=-1)
            probs = top_k_renorm_probs(probs, k.to(device=logits.device))
            probs = top_p_renorm_probs(probs, p.to(device=logits.device))
            return probs.log()

        def measure(fn: Callable[[], torch.Tensor]) -> tuple[float, float]:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            wall_start = time.perf_counter()
            start.record()
            for _ in range(iterations):
                fn()
            end.record()
            end.synchronize()
            wall_ms = (time.perf_counter() - wall_start) * 1000 / iterations
            return start.elapsed_time(end) / iterations, wall_ms

        with artifact.open("w") as output:
            metadata = {
                "event": "metadata",
                "gpu": torch.cuda.get_device_name(),
                "torch": torch.__version__,
                "flashinfer": flashinfer.__version__,
                "cuda": torch.version.cuda,
                "vocab": vocab,
                "top_k": 50,
                "top_p": 0.9,
                "warmup_calls": 20,
                "iterations_per_block": iterations,
                "blocks": blocks,
                "includes_parameter_h2d": True,
            }
            print(json.dumps(metadata), file=output, flush=True)
            inputs = []
            # Gate all shapes and distributions before collecting any timings.
            for batch in (1, 32, 128):
                seed = 917 + batch
                generator = torch.Generator().manual_seed(seed)
                random_cpu = torch.randn(batch, vocab, generator=generator)
                source = random_cpu.cuda()
                k = torch.full((batch,), 50, dtype=torch.int32)
                p = torch.full((batch,), 0.9, dtype=torch.float32)
                digest = hashlib.sha256(random_cpu.numpy().tobytes()).hexdigest()
                inputs.append((source, k, p))
                for distribution in ("random", "ties", "extreme"):
                    values = source
                    if distribution == "ties":
                        values = torch.full_like(source, -10.0)
                        values[:, :20] = 4.0
                        values[:, 20:80] = 3.0
                    if distribution == "extreme":
                        values = torch.full_like(source, -10000.0)
                        values[:, :64] = 10000.0
                        values[:, 64:128] = 9999.0
                        values[:, -1] = -math.inf
                    expected = baseline(values, k, p)
                    actual = filter_logits(values, k, p)
                    mismatches = int(
                        (torch.isfinite(actual) != torch.isfinite(expected)).sum()
                    )
                    record = {
                        "event": "correctness",
                        "batch": batch,
                        "distribution": distribution,
                        "seed": seed,
                        "random_input_sha256": digest,
                        "mask_mismatches": mismatches,
                        "probability_check": "pending",
                    }
                    print(json.dumps(record), file=output, flush=True)
                    self.assertEqual(mismatches, 0, msg=str(record))
                    torch.testing.assert_close(
                        actual.softmax(-1),
                        expected.softmax(-1),
                        rtol=1e-5,
                        atol=1e-7,
                    )
                    record["probability_check"] = "passed"
                    print(json.dumps(record), file=output, flush=True)
                    print("LOGITS_FILTER_CORRECTNESS " + json.dumps(record), flush=True)

            for source, k, p in inputs:

                def legacy() -> torch.Tensor:
                    return baseline(source, k, p)

                def candidate() -> torch.Tensor:
                    return filter_logits(source, k, p)

                for _ in range(20):
                    legacy()
                    candidate()
                timings = {"legacy": [], "candidate": []}
                for block in range(blocks):
                    order = (("legacy", legacy), ("candidate", candidate))
                    if block % 2:
                        order = tuple(reversed(order))
                    for name, fn in order:
                        event_ms, wall_ms = measure(fn)
                        timings[name].append((event_ms, wall_ms))
                        record = {
                            "event": "timing_block",
                            "batch": source.shape[0],
                            "implementation": name,
                            "block": block,
                            "cuda_ms_per_call": event_ms,
                            "wall_ms_per_call": wall_ms,
                        }
                        print(json.dumps(record), file=output, flush=True)
                legacy_ms = statistics.median(t[0] for t in timings["legacy"])
                candidate_ms = statistics.median(t[0] for t in timings["candidate"])
                summary = {
                    "event": "timing_summary",
                    "batch": source.shape[0],
                    "legacy_cuda_ms": legacy_ms,
                    "candidate_cuda_ms": candidate_ms,
                    "legacy_wall_ms": statistics.median(
                        t[1] for t in timings["legacy"]
                    ),
                    "candidate_wall_ms": statistics.median(
                        t[1] for t in timings["candidate"]
                    ),
                    "candidate_over_legacy": candidate_ms / legacy_ms,
                    "performance_acceptance": "not_asserted",
                }
                print(json.dumps(summary), file=output, flush=True)
                print("LOGITS_FILTER_TIMING " + json.dumps(summary), flush=True)
        print(f"LOGITS_FILTER_ARTIFACT {artifact}", flush=True)


if __name__ == "__main__":
    unittest.main()
