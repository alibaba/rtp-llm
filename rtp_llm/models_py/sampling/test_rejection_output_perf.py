import json
import os
import statistics
import unittest
from pathlib import Path

import torch
from rtp_llm.models_py.sampling import adapter


@unittest.skipUnless(
    os.environ.get("REJECTION_OUTPUT_BENCH") == "1", "opt-in GPU benchmark"
)
class RejectionOutputPerfTest(unittest.TestCase):
    def test_output_fusion(self) -> None:
        torch.manual_seed(20261110)
        directory = Path(os.environ["TEST_UNDECLARED_OUTPUTS_DIR"])
        results = []
        graph_bench = os.environ.get("REJECTION_GRAPH_BENCH") == "1"
        for batch, steps in ((16, 3), (41, 5)):
            for random in (False, True):
                for forced in (False, True):
                    vocab = 154880
                    logits = torch.randn(batch, steps + 1, vocab, device="cuda")
                    ids = torch.randint(
                        vocab, (batch, steps), device="cuda", dtype=torch.int32
                    )
                    seeds = torch.arange(batch, device="cuda", dtype=torch.int64)
                    positions = seeds + 4096
                    sample = torch.full(
                        (batch,), random, device="cuda", dtype=torch.bool
                    )
                    outputs = [
                        torch.empty(batch, steps + 1, device="cuda", dtype=torch.int32)
                        for _ in range(2)
                    ]
                    lengths = [
                        torch.empty(batch, device="cuda", dtype=torch.int32)
                        for _ in range(2)
                    ]

                    def run(direct: bool) -> None:
                        index = int(direct)
                        compute = (
                            adapter.rejection_sample_logits
                            if graph_bench and direct
                            else adapter._rejection_sample_logits_eager
                        )
                        compute(
                            logits,
                            None,
                            ids,
                            seeds,
                            positions,
                            sample,
                            [forced] * batch,
                            output_ids=outputs[index],
                            output_lengths=lengths[index],
                            clean_output=direct or graph_bench,
                        )
                        if not direct and not graph_bench:
                            outputs[index].masked_fill_(outputs[index] == -1, 0)

                    for _ in range(10):
                        run(False)
                        run(True)
                    torch.testing.assert_close(outputs[0], outputs[1], rtol=0, atol=0)
                    torch.testing.assert_close(lengths[0], lengths[1], rtol=0, atol=0)
                    graphs = []
                    for direct in () if graph_bench else (False, True):
                        graph = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(graph):
                            run(direct)
                        graphs.append(graph)
                    times = {"eager": [[], []]}
                    if not graph_bench:
                        times["graph"] = [[], []]
                    for mode in times:
                        for repeat in range(5):
                            for index in (0, 1) if repeat % 2 == 0 else (1, 0):
                                start, stop = torch.cuda.Event(
                                    enable_timing=True
                                ), torch.cuda.Event(enable_timing=True)
                                start.record()
                                for _ in range(50):
                                    if mode == "graph":
                                        graphs[index].replay()
                                    else:
                                        run(bool(index))
                                stop.record()
                                stop.synchronize()
                                times[mode][index].append(
                                    start.elapsed_time(stop) * 1000 / 50
                                )
                    row = {
                        "batch": batch,
                        "steps": steps,
                        "vocab": vocab,
                        "target_random": random,
                        "forced_accept": forced,
                        "draft": "point_mass",
                        "accepted_lengths": lengths[1].cpu().tolist(),
                        "times_us": times,
                        "medians_us": {
                            key: [statistics.median(v) for v in values]
                            for key, values in times.items()
                        },
                    }
                    results.append(row)
                    print(json.dumps(row), flush=True)
                    label = f"b{batch}_s{steps}_random{int(random)}_forced{int(forced)}"
                    with torch.profiler.profile(
                        activities=[
                            torch.profiler.ProfilerActivity.CPU,
                            torch.profiler.ProfilerActivity.CUDA,
                        ]
                    ) as profiler:
                        for index, name in enumerate(
                            (
                                ("gumbel_eager", "gumbel_cached_graph")
                                if graph_bench
                                else (
                                    "baseline_finalize_then_clean",
                                    "direct_output_and_clean",
                                )
                            )
                        ):
                            for _ in range(5):
                                with torch.profiler.record_function(
                                    name
                                ), torch.cuda.nvtx.range(name):
                                    run(bool(index))
                                torch.cuda.synchronize()
                    profiler.export_chrome_trace(str(directory / f"{label}.json"))
        (directory / "results.json").write_text(
            json.dumps(
                {
                    "device": torch.cuda.get_device_name(),
                    "torch": torch.__version__,
                    "results": results,
                    "scope": "prepared target logits through rejection output and tail cleanup; no model forward or target filtering",
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
