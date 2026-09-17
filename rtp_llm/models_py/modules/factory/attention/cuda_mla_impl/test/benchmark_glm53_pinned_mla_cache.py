"""Diagnostic SparseMLA A/B benchmark, excluding indexer and Q expansion.

Reuses the correctness fixture and validates outputs before timing. Each CUDA
Graph contains 20 complete SparseMLA calls. Results are per-call GPU times,
not model TPOT. With three rows, both selections fit the larger working set;
that alternating case measures hits after warmup, not sustained cold misses.
"""

import argparse
import importlib.util
import json
import os
from pathlib import Path
from unittest.mock import patch

import torch

from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.flashmla_sparse_impl import (
    SparseMlaImpl,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    options = parser.parse_args()
    path = Path(__file__).with_name("glm53_pinned_mla_cache_test.py")
    spec = importlib.util.spec_from_file_location("host_test", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    cases = {}
    original = SparseMlaImpl.forward

    def record(self, *args, **kwargs):
        key = (args[0].shape[0], bool(getattr(self, "pinned_mla_groups", {})))
        cases[key] = (self, args)
        return original(self, *args, **kwargs)

    test = m.Glm53PinnedMlaCacheTest()
    test.setUp()
    with patch.object(SparseMlaImpl, "forward", record):
        test.test_full_sparse_mla_decode_and_chunked_prefill()
    # Production sizes resident capacity for all simultaneous verify rows.
    for rows in (3,):
        op, args = cases[(rows, True)]
        old = op.pinned_mla_groups[0][0]
        capacity = ((rows * 2051 + 127) // 128) * 128
        hbm = torch.empty(
            ((256 + capacity) // 128, 128, 528), dtype=torch.uint8, device="cuda"
        )
        hbm[:2].copy_(old.resident[0][:2])
        op.pinned_mla_groups = {
            0: (
                m.PinnedMlaWorkingSet(
                    old.backing,
                    capacity,
                    128,
                    torch.device("cuda", 0),
                    allocator_block_size=256,
                    hbm_tokens=256,
                    hbm_cache=[hbm],
                ),
                0,
            )
        }
    results = []
    for rows in (1, 3):
        for cold in (False, True):
            for host in (False, True):
                op, args = cases[(rows, host)]
                topk = args[5]
                low = torch.arange(512, device="cuda", dtype=torch.int32).repeat(
                    rows, 1
                )
                high = low + 700
                topk.copy_(low)
                for _ in range(3):
                    original(op, *args)
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                steps = 20
                with torch.cuda.graph(graph):
                    for step in range(steps):
                        topk.copy_(high if cold and step % 2 else low)
                        original(op, *args)
                for _ in range(5):
                    graph.replay()
                times = []
                for _ in range(25):
                    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
                        enable_timing=True
                    )
                    start.record()
                    graph.replay()
                    end.record()
                    end.synchronize()
                    times.append(start.elapsed_time(end) * 1000 / steps)
                times.sort()
                results.append(
                    dict(
                        rows=rows,
                        cache="host" if host else "hbm",
                        selection="alternating_disjoint" if cold else "repeated",
                        median_us=times[len(times) // 2],
                        p95_us=times[int(len(times) * 0.95)],
                    )
                )
                print(json.dumps(results[-1]), flush=True)
    with open(options.output, "w") as f:
        json.dump(
            dict(
                gpu=torch.cuda.get_device_name(),
                torch=torch.__version__,
                fetch_rows=os.environ.get("RTP_LLM_DSA_MLA_FETCH_ROWS", "0"),
                results=results,
            ),
            f,
            indent=2,
        )


if __name__ == "__main__":
    main()
