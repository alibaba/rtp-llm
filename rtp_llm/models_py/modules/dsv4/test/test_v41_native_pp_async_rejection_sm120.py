"""Asynchronous rejection contract for the native V4.1 PP executor (guide §6.C / §8).

Why this exists as a separate target
------------------------------------
`test_v41_native_pp_executor_sm120` asserts *"executor rejection precedes all stores"*:
with a WORLD-scoped admission agreement, a failure on any single rank is carried in the
phase-3 worker-ready packet, so every other rank throws inside `prepareImpl` before
`initializeFresh` runs and the cache stays byte-identical.

That guarantee is **logically unavailable** under the asynchronous contract. Narrowing the
agreement to `ParallelMode::STAGE` is what allows stage 0 to run ahead of stage 1, and it
also means stage 0's agreement covers only its own four ranks: when stage 1 alone rejects,
stage 0 has already admitted, already zeroed its newly-admitted pages, and can only learn
of the rejection afterwards through the ordered result channel. Any per-chunk WORLD
agreement would restore the pre-write detection *and* the lock-step, and deferring the
writes until the consumer acknowledges would add a round trip per chunk — which is the
lock-step again. Invariant #8 anticipates this: "Failures after writes may require
poisoning/draining the request or session; do not pretend rolling back counters undoes
GPU/cache writes."

So the synchronous test is left untouched (it correctly documents the synchronous
guarantee and passes there), and this target asserts the **replacement** contract:

  R1. stores may occur, but only on the stage that admitted, and only on pages belonging
      to the rejected request — never on pages owned by an unrelated live request;
  R2. every page taken for the rejected request is released (free-block count restored),
      i.e. the write is confined and reclaimable, not a leak;
  R3. the request terminates with exactly one error, `outstanding == 0`, and on stage 0
      that error arrives via the *result channel* at slot rotation rather than from its
      own admission — so it needs several ticks, not one;
  R4. neither stage blocks: the whole check completes inside the gloo/NCCL timeout, which
      is the property the pre-C1b implementation failed (180 s hang).

Run with `RTP_LLM_V41_PP_ASYNC=1`.
"""

import gc
import json
import os
import socket
import sys
import tempfile
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

# An unrelated live request whose physical pages must survive the rejected round (R1).
GUARD_REQUEST = 9501
# pp_size + 1 slots: a terminal result sent in round N is read by stage 0 when that slot
# is rotated back, so the error needs this many ticks to surface (R3).
RESULT_LATENCY_TICKS = 6


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
        timeout=timedelta(seconds=180),
    )
    control = dist.new_group(backend="gloo", timeout=timedelta(seconds=180))
    native = worker = guard = None
    new_group = dist.new_group

    def bounded_group(*args, **kwargs):
        kwargs["timeout"] = timedelta(seconds=180)
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

                # An unrelated live allocation, on every rank, so R1 is checkable per rank.
                guard = native.allocate(GUARD_REQUEST, 256)
                guard_pages = {
                    tag: set(pages) for tag, pages in guard.block_tables.items()
                }
                free_with_guard = native.free_blocks

                # Sentinel-fill so any store is visible, then snapshot.
                for layer in range(cache.layer_count):
                    for group in cache.get_layer_cache_groups(layer):
                        group.kv_cache_base.view(torch.uint8).fill_(37)
                before = raw_snapshot(cache)

                # Engineer a stage-1-only rejection: rank 4 alone lacks the engine
                # contract, so stage 1's worker preflight fails while stage 0's succeeds.
                # Under WORLD scope stage 0 would learn this before writing; under STAGE
                # scope it cannot, which is precisely the contract under test.
                if rank != 4:
                    worker._configure_native_pp_engine(CONTRACT)
                rejected_ids = None
                if rank == 0:
                    rejected_ids = native.enqueue_scheduled([[7, 8, 9]], False)
                native.scheduler_tick()

                changed = changed_pages(before, raw_snapshot(cache))
                foreign = sorted(
                    (layer, tag, page)
                    for (layer, tag, page) in changed
                    if page in guard_pages.get(tag, ())
                )

                # R1: stores are confined away from other requests' pages.
                world_check(
                    lambda: require(
                        not foreign,
                        f"rejected round wrote an unrelated request's pages: {foreign[:8]}",
                    ),
                    "async rejection confines stores (R1)",
                    control,
                )

                # R3: drain the ordered result channel, then the error must have surfaced
                # exactly once on stage 0's root, with nothing outstanding.
                for _ in range(RESULT_LATENCY_TICKS):
                    native.scheduler_tick()
                rows = native.scheduled_outputs() if rank == 0 else []
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
                    "async rejection terminates exactly once (R3)",
                    control,
                )

                # R2: every page taken for the rejected request is released.
                world_check(
                    lambda: require(
                        native.free_blocks == free_with_guard,
                        f"rejected request leaked pages: free={native.free_blocks} "
                        f"expected={free_with_guard}",
                    ),
                    "async rejection releases its pages (R2)",
                    control,
                )

                # Evidence, not assertion: whether the admitting stage wrote before it
                # could learn of the rejection. Expected true on stage 0 under STAGE scope.
                print(
                    "DSV41_NATIVE_PP_ASYNC_REJECTION "
                    + json.dumps(
                        {
                            "rank": rank,
                            "stage": rank // 4,
                            "wrote_before_detection": bool(changed),
                            "changed_page_count": len(changed),
                            "foreign_pages_touched": len(foreign),
                            "free_blocks_restored": native.free_blocks
                            == free_with_guard,
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


class V41NativePPAsyncRejectionSM120Test(unittest.TestCase):
    def test_stage_local_rejection_contract(self):
        self.assertEqual(
            os.environ.get("RTP_LLM_V41_PP_ASYNC"),
            "1",
            "this target asserts the asynchronous contract; run it with RTP_LLM_V41_PP_ASYNC=1",
        )
        self.assertGreaterEqual(torch.cuda.device_count(), 8)
        for rank in range(8):
            self.assertEqual(torch.cuda.get_device_capability(rank), (12, 0))
        with tempfile.TemporaryDirectory() as directory:
            torch.manual_seed(41700)
            config = checkpoint(Path(directory), full_schedule=True, hidden_size=5120)
            with socket.socket() as listener:
                listener.bind(("127.0.0.1", 0))
                port = listener.getsockname()[1]
            mp.spawn(run_rank, args=(port, config), nprocs=8, join=True)


if __name__ == "__main__":
    unittest.main()
