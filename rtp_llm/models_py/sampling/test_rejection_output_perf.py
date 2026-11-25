"""Run :test_rejection_output_perf with --test_env=REJECTION_OUTPUT_BENCH=1.

GPU event timings use warm inputs and exclude profiling and model forward.
"""

import json
import os
import statistics
import unittest
from collections.abc import Callable
from pathlib import Path

import torch

from rtp_llm.models_py.sampling import adapter
from rtp_llm.models_py.sampling.logits_filter import filter_logits


Run = Callable[[], torch.Tensor | tuple[torch.Tensor, torch.Tensor]]


def measure_pair(name: str, baseline: Run, candidate: Run) -> None:
    runs = (baseline, candidate)
    for run in runs:
        for _ in range(5):
            run()
    torch.cuda.synchronize()
    timings = [[], []]
    for block in range(5):
        for index in (block % 2, 1 - block % 2):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(20):
                runs[index]()
            end.record()
            end.synchronize()
            timings[index].append(start.elapsed_time(end) * 1000 / 20)
    result = {
        "case": name,
        "baseline_us": statistics.median(timings[0]),
        "candidate_us": statistics.median(timings[1]),
    }
    print(json.dumps(result), flush=True)
    directory = os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR")
    if directory:
        path = Path(directory)
        (path / f"{name}.json").write_text(json.dumps(result))
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ]
        ) as profile:
            for label, run in zip(("baseline", "candidate"), runs):
                with torch.profiler.record_function(label):
                    run()
            torch.cuda.synchronize()
        profile.export_chrome_trace(str(path / f"{name}.trace.json"))


@unittest.skipUnless(
    os.environ.get("REJECTION_OUTPUT_BENCH") == "1", "opt-in GPU benchmark"
)
class SamplingPerfTest(unittest.TestCase):
    def test_prepared_target(self) -> None:
        for batch, steps in ((16, 3), (41, 5)):
            vocab = 154880
            gen = torch.Generator(device="cuda").manual_seed(20261110)
            logits = torch.randn(batch, steps + 1, vocab, device="cuda", generator=gen)
            ids = logits[:, :steps].argmax(-1).int()
            q = logits[:, :steps].softmax(-1)
            seed = torch.arange(batch, device="cuda", dtype=torch.int64)
            pos = seed + 4096
            sample = torch.ones(batch, device="cuda", dtype=torch.bool)
            args = (ids, seed, pos, sample)

            def full_draft(fused: bool) -> tuple[torch.Tensor, torch.Tensor]:
                return adapter._rejection_sample_logits_eager(
                    logits, q, *args, fuse_draft_probs_log=fused
                )

            reference = full_draft(False)
            torch.testing.assert_close(full_draft(True), reference, rtol=0, atol=0)
            case = f"b{batch}_s{steps}"
            measure_pair(
                f"{case}_draft_log", lambda: full_draft(False), lambda: full_draft(True)
            )

            tokens = torch.empty((batch, steps + 1), device="cuda", dtype=torch.int32)
            lengths = torch.empty(batch, device="cuda", dtype=torch.int32)

            def output_path(cached: bool) -> tuple[torch.Tensor, torch.Tensor]:
                compute = (
                    adapter.rejection_sample_logits
                    if cached
                    else adapter._rejection_sample_logits_eager
                )
                return compute(
                    logits,
                    None,
                    *args,
                    output_ids=tokens,
                    output_lengths=lengths,
                    clean_output=True,
                )

            expected = tuple(t.clone() for t in output_path(False))
            torch.testing.assert_close(output_path(True), expected, rtol=0, atol=0)
            measure_pair(
                f"{case}_cached_output",
                lambda: output_path(False),
                lambda: output_path(True),
            )

    def test_gumbel_against_torch(self) -> None:
        for batch in (1, 32, 64, 128):
            logits = torch.randn(batch, 154880, device="cuda")
            seeds = torch.arange(batch, device="cuda", dtype=torch.int64)
            positions = seeds + 4096
            measure_pair(
                f"b{batch}_torch_multinomial_vs_gumbel",
                lambda: torch.multinomial(logits.softmax(-1), 1),
                lambda: adapter.sample(logits, seeds, positions),
            )

    def test_reference_filter(self) -> None:
        from flashinfer.sampling import top_k_renorm_probs, top_p_renorm_probs

        generator = torch.Generator(device="cuda").manual_seed(20261110)
        logits = torch.randn((16, 154880), device="cuda", generator=generator)
        k = torch.full((16,), 50, dtype=torch.int32)
        p = torch.full((16,), 0.9)
        gpu_k, gpu_p = k.cuda(), p.cuda()

        def flashinfer() -> torch.Tensor:
            return top_p_renorm_probs(
                top_k_renorm_probs(logits.softmax(-1), gpu_k), gpu_p
            )

        def reference() -> torch.Tensor:
            return filter_logits(logits, k, p).softmax(-1)

        torch.testing.assert_close(reference(), flashinfer(), rtol=1e-5, atol=1e-7)
        measure_pair("b16_reference_filter", flashinfer, reference)


if __name__ == "__main__":
    unittest.main()
