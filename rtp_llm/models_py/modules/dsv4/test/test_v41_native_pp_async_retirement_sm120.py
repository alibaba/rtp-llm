"""T7 of the C1b matrix: retirement with a rejected chunk outstanding.

The window under test
---------------------
`test_v41_native_pp_async_rejection_sm120` proves the T2 rejection round itself: stage 1
rejects, stage 0 admits and writes, stage 1 drains the frame and sends a terminal error,
and the error reaches stage 0's root via the ordered result channel at slot rotation --
six ticks later. During those six ticks stage 0 is carrying an admitted, written chunk
whose terminal state it cannot yet see. That window is exactly "a rejected chunk
outstanding", and the round after the error lands is where `retireV41Requests` reclaims
the request's identity and pages.

The hazard retirement poses in that window is a participation mismatch: the retirement
fan-out is a stage-local broadcast of the plan's terminal ids (`PPExecutor.cc:565-591`),
and a root-only branch there is what stranded the non-root ranks in run 119 -- roots
entered a broadcast non-roots never joined. Every per-rank tick in this test is
wall-clock bounded, so that failure mode fails the test instead of passing silently.

Assertions
----------
  R1. no tick ever blocks: every scheduler_tick on every rank finishes well inside the
      gloo watchdog, through rejection, result rotation AND the retirement rounds;
  R2. the terminal error still surfaces exactly once with `outstanding == 0` (this is
      async_rejection's R3, kept, so the extension cannot weaken the rejection path);
  R3. after the retirement rounds the admission session holds nothing: `active_global`
      reaches 0, i.e. the rejected request's identity was retired exactly once;
  R4. its pages are released exactly once: `free_blocks` returns to the guard-only level
      and stays there across the retirement rounds -- never below (no double release)
      and never stuck above (no leak);
  R5. an unrelated live request's pages are untouched throughout.

Run with `RTP_LLM_V41_PP_ASYNC=1` (fixed on by this target's BUILD env).
"""

import gc
import json
import os
import socket
import sys
import tempfile
import time
import traceback
import unittest
from datetime import timedelta
from pathlib import Path
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from rtp_llm.cpp.normal_engine.test import libth_v41_native_pp_transport_test as ops
from rtp_llm.models_py.distributed import collective_torch as collective
from rtp_llm.models_py.modules.dsv4.fp8.test.test_v41_worker_sm120 import (
    checkpoint,
    resources,
)
from rtp_llm.models_py.modules.dsv4.test.test_native_l20_transfer import CONTRACT
from rtp_llm.models_py.modules.dsv4.test.test_v41_cp_stage_worker_sm120 import (
    make_worker,
    parallelism_for,
)
from rtp_llm.models_py.modules.dsv4.test.test_v41_native_pp_execution_sm120 import (
    raw_snapshot,
    require,
    world_check,
)
from rtp_llm.models_py.modules.dsv4.test.test_v41_native_pp_transport_sm120 import (
    cache_config,
)
from rtp_llm.ops import NcclCommConfig


def changed_pages(before, after):
    """(layer, tag, page) triples whose bytes differ between two raw snapshots."""
    changed = set()
    for key, value in after.items():
        expected = before[key]
        if torch.equal(value, expected):
            continue
        layer, tag = key
        for page in range(value.size(0)):
            if not torch.equal(value[page], expected[page]):
                changed.add((layer, tag, page))
    return changed


# An unrelated live request whose physical pages must survive the whole sequence (R5).
GUARD_REQUEST = 9701
# The rejected request: enqueue_scheduled takes TOKEN IDS, bounded by vocab_size (128 in
# this synthetic checkpoint) -- see the T4 note on the same trap.
TOKENS = [7, 8, 9]
# A terminal result sent in round N is read by stage 0 when that slot rotates back
# (pp_size + 1 slots), so the error needs this many ticks to surface (async_rejection R3).
RESULT_LATENCY_TICKS = 6
# Rounds after the error surfaces, during which the plan carries the finished id into
# retireV41Requests. Generous: the property under test is bounded progress, not a precise
# tick count.
RETIREMENT_TICKS = 5
WATCHDOG_SECONDS = 180
BOUNDED_TICK_SECONDS = 60


def run_rank(rank, port, config):
    torch.cuda.set_device(rank)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    os.environ.update(
        {
            "NCCL_P2P_DISABLE": "1",
            "DSV41_ENGRAM_HOST_BYTES": str(2**20),
            "DSV41_ENGRAM_CACHE_DIR": "/dev/shm",
            "DSV4_PP_EP_ENABLE": "1",
            "DSV4_PP_EP_BACKEND": "fork_nccl_mxfp8",
            "DSV4_MOE_STRATEGY": "auto",
            "RTP_V41_BATCHED_PRODUCER": "0",
            "DSV4_PREFILL_FAST_PATH": "1",
            "MOEDBG": "0",
        }
    )
    dist.init_process_group(
        "nccl",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=8,
        timeout=timedelta(seconds=WATCHDOG_SECONDS),
    )
    control = dist.new_group(
        backend="gloo", timeout=timedelta(seconds=WATCHDOG_SECONDS)
    )
    native = worker = guard = None
    new_group = dist.new_group

    def bounded_group(*args, **kwargs):
        kwargs["timeout"] = timedelta(seconds=WATCHDOG_SECONDS)
        return new_group(*args, **kwargs)

    with patch.dict(sys.modules, {"librtp_compute_ops": ops}), patch.object(
        dist, "new_group", side_effect=bounded_group
    ):
        try:
            collective.init_distributed_environment(
                parallelism_for(rank), NcclCommConfig(nccl_ip="127.0.0.1"), port
            )
            with torch.inference_mode():
                native = ops.NativePPTransport(cache_config(), parallelism_for(rank))
                cache = native.cache
                worker = make_worker(rank, config)
                assert worker.initialize(resources(cache))
                native.attach_model(
                    worker, worker.v4.head_weight if rank >= 4 else torch.empty(0)
                )
                native.start_scheduler(43)

                guard = native.allocate(GUARD_REQUEST, 256)
                guard_pages = {
                    tag: set(pages) for tag, pages in guard.block_tables.items()
                }
                free_with_guard = native.free_blocks
                for layer in range(cache.layer_count):
                    for group in cache.get_layer_cache_groups(layer):
                        group.kv_cache_base.view(torch.uint8).fill_(37)
                before = raw_snapshot(cache)

                # Stage-1-only rejection: rank 4 alone lacks the engine contract.
                if rank != 4:
                    worker._configure_native_pp_engine(CONTRACT)
                rejected_ids = None
                if rank == 0:
                    rejected_ids = native.enqueue_scheduled([TOKENS], False)

                tick_times = []

                def tick(label):
                    started = time.monotonic()
                    native.scheduler_tick()
                    tick_times.append((label, time.monotonic() - started))

                tick("rejection")
                for index in range(RESULT_LATENCY_TICKS):
                    tick("rotation-%d" % index)
                rows = native.scheduled_outputs() if rank == 0 else []
                for index in range(RETIREMENT_TICKS):
                    tick("retirement-%d" % index)

                changed = changed_pages(before, raw_snapshot(cache))
                foreign = sorted(
                    (layer, tag, page)
                    for (layer, tag, page) in changed
                    if page in guard_pages.get(tag, ())
                )

                slowest = max(elapsed for _, elapsed in tick_times)
                pages_back = native.free_blocks == free_with_guard
                retired = native.active_global == 0

                # All checks are collective on `control`; every rank runs the same
                # sequence, with per-rank conditions inside require().
                world_check(
                    lambda: require(
                        slowest < BOUNDED_TICK_SECONDS,
                        f"rank {rank} had a {slowest:.1f}s tick; a rank stranded by a "
                        f"root-only retirement branch can only be released by the "
                        f"{WATCHDOG_SECONDS}s watchdog",
                    ),
                    "no tick blocks through rejection, rotation or retirement (R1)",
                    control,
                )
                world_check(
                    lambda: require(
                        rank != 0
                        or (
                            [row["request_id"] for row in rows] == rejected_ids
                            and all(row["invalid_params"] for row in rows)
                            and all(row["outstanding"] == 0 for row in rows)
                            and all(row["error"] for row in rows)
                        ),
                        f"terminal error did not surface exactly once: {rows}",
                    ),
                    "terminal error still surfaces exactly once (R2)",
                    control,
                )
                world_check(
                    lambda: require(
                        retired,
                        f"admission session still holds {native.active_global} requests "
                        "after the retirement rounds; the rejected request's identity "
                        "survived past its lifetime boundary or was retired twice",
                    ),
                    "rejected request's identity is retired exactly once (R3)",
                    control,
                )
                world_check(
                    lambda: require(
                        pages_back,
                        f"free_blocks={native.free_blocks} expected={free_with_guard}: "
                        "pages leaked or were released twice around retirement",
                    ),
                    "rejected request's pages are released exactly once (R4)",
                    control,
                )
                world_check(
                    lambda: require(
                        not foreign,
                        f"retirement round wrote an unrelated request's pages: {foreign[:8]}",
                    ),
                    "retirement confined to the retired request's pages (R5)",
                    control,
                )

                print(
                    "DSV41_NATIVE_PP_ASYNC_RETIREMENT "
                    + json.dumps(
                        {
                            "rank": rank,
                            "stage": rank // 4,
                            "ticks": [[label, round(t, 3)] for label, t in tick_times],
                            "slowest_tick": round(slowest, 3),
                            "active_global": native.active_global,
                            "free_blocks": native.free_blocks,
                            "free_blocks_expected": free_with_guard,
                            "foreign_pages_touched": len(foreign),
                        }
                    ),
                    flush=True,
                )

                guard.release_owner()
                guard.release_lease()
                guard = None
                del worker
                gc.collect()
                native.close()
                native = None
            dist.barrier(group=control)
        except BaseException:
            traceback.print_exc()
            raise
        finally:
            if guard is not None:
                guard.release_owner()
                guard.release_lease()
            for fixture in (native,):
                if fixture is not None:
                    fixture.wait()
                    fixture.close()
            collective.destroy_distributed_environment()


class V41NativePPAsyncRetirementSM120Test(unittest.TestCase):
    def test_retirement_with_a_rejected_chunk_outstanding(self):
        self.assertEqual(
            os.environ.get("RTP_LLM_V41_PP_ASYNC"),
            "1",
            "this target asserts the asynchronous contract; run it with RTP_LLM_V41_PP_ASYNC=1",
        )
        self.assertGreaterEqual(torch.cuda.device_count(), 8)
        for rank in range(8):
            self.assertEqual(torch.cuda.get_device_capability(rank), (12, 0))
        with tempfile.TemporaryDirectory() as directory:
            torch.manual_seed(41702)
            config = checkpoint(Path(directory), full_schedule=True, hidden_size=5120)
            with socket.socket() as listener:
                listener.bind(("127.0.0.1", 0))
                port = listener.getsockname()[1]
            mp.spawn(run_rank, args=(port, config), nprocs=8, join=True)


if __name__ == "__main__":
    unittest.main()
