"""T4 of the C1b matrix: failure AFTER tagged writes must reach the peer stage.

Why this case is different from the admission-rejection contract
---------------------------------------------------------------
`test_v41_native_pp_async_rejection_sm120` covers a rejection decided during
`prepareWithWorker`. That decision is caught at PPExecutor.cc:1297 and turned into a
`rejected`-status frame, so the peer stage learns about it on the ordered activation
channel and takes the path at :1323-1335 (drain, abort, advance nothing).

A failure *after* admission had no such channel. `PPExecutor::process` had exactly one
try/catch, around `prepareWithWorker`; everything after it -- `initializeFresh` (which
performs the tagged `index_fill_` writes), `beginExecution`, the forward, the frame send,
`completeExecution`, `commit` -- was unprotected. Under the synchronous contract that was
safe, because the two agreements inside `initializeFresh` were WORLD-scoped and therefore
*were* the cross-stage failure channel: one stage's `status[0] = 2` made `!same` true on all
eight ranks, so all eight threw together. C2 narrowed those agreements to
`ParallelMode::STAGE`, which is what allows stage 0 to run ahead, and it also removed that
channel. A post-admission failure on stage 0 then left stage 1 blocked in `receiveObject`
with `waitTicket` unbounded while running, until the ~180 s gloo watchdog fired -- and stage 1
never reached the code that aborts its own batch, so its leases and replica frontier were
released by process death rather than by protocol.

The fix (C8) records which matched traffic the round has already emitted and, on unwind,
supplies only what is still owed: a producer that has not sent its frame sends a
`rejected`-status empty one; a consumer whose producer admitted sends the terminal result.
This target asserts the consequence.

Assertions
----------
  F1. stage 0's tick fails with the injected error and its session is poisoned -- a later
      `prepare_global` must refuse to enter it, so no further chunk of that request is
      admitted, and the batch is never committed (no false rollback claimed);
  F2. stage 1 does **not** block: its tick returns inside a wall-clock bound far below the
      180 s gloo timeout, which is the property the pre-C8 code failed;
  F3. stage 1 aborts its own batch, so the pages it took are released;
  F4. neither stage writes an unrelated live request's pages;
  F5. the injected failure really did occur after the tagged writes -- stage 0 must have
      written, which is what distinguishes T4 from an admission-time rejection;
  F6. and stage 0 still holds those pages afterwards. A poisoned session retains the batch
      and its leases until the caller drains and aborts; returning the blocks here would
      claim a rollback that never happened (invariant #8).

Injection is `RTP_LLM_V41_TEST_FAIL_AFTER_TAGGED_WRITES_STAGE=0`, fixed on by this target's
BUILD env together with `RTP_LLM_V41_PP_ASYNC=1`. It throws inside `initializeFresh` after
the write loop and after the second agreement, landing in the catch that sets `failed_` and
retains `writing_batch_`.
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

# An unrelated live allocation whose physical pages must survive the failed round (F4).
# `allocate` takes a request id, which may be large; `enqueue_scheduled` takes TOKEN IDS,
# which must be < vocab_size (128 in this synthetic checkpoint). Conflating the two makes
# buildPlan fail on the stage root only, which returns an error status there while the other
# ranks block on a plan broadcast that never comes.
GUARD_REQUEST = 9601
FIRST_TOKENS = [7, 8, 9]
SECOND_TOKENS = [11, 12, 13]
# The gloo/NCCL timeout every group in this file is created with. F2's bound is a fraction
# of it: a stage that blocks in receiveObject can only be released by the watchdog, so
# finishing well inside it is what proves the peer was told rather than abandoned.
WATCHDOG_SECONDS = 180
BOUNDED_SECONDS = 60


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
                worker._configure_native_pp_engine(CONTRACT)
                native.attach_model(
                    worker, worker.v4.head_weight if rank >= 4 else torch.empty(0)
                )
                native.start_scheduler(43)

                # An unrelated live allocation on every rank, so F4 is checkable per rank.
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

                # Drive one real chunk through the executor round loop. Stage 0 admits,
                # performs the tagged writes, and then hits the injected failure; stage 1
                # is healthy and is waiting for stage 0's frame.
                if rank == 0:
                    native.enqueue_scheduled([FIRST_TOKENS], False)
                started = time.monotonic()
                tick_error = None
                try:
                    native.scheduler_tick()
                except (
                    BaseException
                ) as exception:  # noqa: BLE001 - recorded, then compared
                    tick_error = f"{type(exception).__name__}: {exception}"
                elapsed = time.monotonic() - started

                # A second request, so "no later chunk admitted" is observed on the real
                # scheduler path instead of inferred from a flag. Stage 0's session is
                # poisoned and must refuse it; stage 1 is healthy and must be told about
                # that refusal by C8's rejected frame rather than left blocking.
                if rank == 0:
                    native.enqueue_scheduled([SECOND_TOKENS], False)
                later_started = time.monotonic()
                later_error = None
                try:
                    native.scheduler_tick()
                except BaseException as exception:  # noqa: BLE001
                    later_error = f"{type(exception).__name__}: {exception}"
                later_elapsed = time.monotonic() - later_started

                changed = changed_pages(before, raw_snapshot(cache))
                foreign = sorted(
                    (layer, tag, page)
                    for (layer, tag, page) in changed
                    if page in guard_pages.get(tag, ())
                )
                stage = rank // 4
                poisoned = stage == 0 and "injected" in (tick_error or "")
                refused = stage == 0 and "failed" in (later_error or "")
                pages_back = native.free_blocks == free_with_guard

                # Every world_check below is a collective on `control`, so all eight ranks
                # must run the same sequence in the same order; the per-stage conditions
                # live inside require() and are vacuously true on the other stage.

                # F2: a stage that was never told about the failure can only be released
                # by the watchdog, so finishing well inside it is the property under test.
                world_check(
                    lambda: require(
                        elapsed < BOUNDED_SECONDS,
                        f"rank {rank} took {elapsed:.1f}s in one tick; a stage blocked in "
                        f"receiveObject can only be released by the {WATCHDOG_SECONDS}s "
                        "watchdog, so the peer was not told about the failure",
                    ),
                    "post-admission failure is delivered, not waited out (F2)",
                    control,
                )
                world_check(
                    lambda: require(
                        later_elapsed < BOUNDED_SECONDS,
                        f"rank {rank} took {later_elapsed:.1f}s on the later tick; the "
                        "poisoned producer's refusal was not delivered to its peer",
                    ),
                    "the later refusal is delivered too (F2)",
                    control,
                )

                # F4: stores stay away from other requests' pages on both stages.
                world_check(
                    lambda: require(
                        not foreign,
                        f"failed round wrote an unrelated request's pages: {foreign[:8]}",
                    ),
                    "failed round confines stores (F4)",
                    control,
                )

                # F1: the producer really did fail, and its session refuses later chunks.
                world_check(
                    lambda: require(
                        stage != 0 or poisoned,
                        f"stage 0 did not report the injected failure: {tick_error}",
                    ),
                    "injected failure reached the producer (F1)",
                    control,
                )
                world_check(
                    lambda: require(
                        stage != 0 or refused,
                        f"a poisoned session admitted a later chunk: {later_error}",
                    ),
                    "poisoned session admits no later chunk (F1)",
                    control,
                )

                # F5: the failure came after the tagged writes, which is what separates
                # T4 from an admission-time rejection.
                world_check(
                    lambda: require(
                        stage != 0 or bool(changed),
                        "stage 0 wrote nothing, so the failure was not after the tagged "
                        "writes and this is an admission rejection, not T4",
                    ),
                    "failure occurred after tagged writes (F5)",
                    control,
                )

                # F6: and the pages it wrote are STILL held. A poisoned session retains
                # writing_batch_ and its leases until the caller drains and aborts, because
                # returning the blocks here would claim a rollback that never happened --
                # invariant #8: do not pretend rolling back counters undoes GPU/cache writes.
                world_check(
                    lambda: require(
                        stage != 0 or native.free_blocks < free_with_guard,
                        f"stage 0 released the failed round's pages "
                        f"(free={native.free_blocks}, before={free_with_guard}), claiming a "
                        "rollback of writes that are still on the device",
                    ),
                    "producer retains the written pages, no false rollback (F6)",
                    control,
                )

                # F3: the consumer was told, abandoned the round and gave its pages back.
                world_check(
                    lambda: require(
                        stage != 1 or tick_error is None,
                        f"stage 1 should abandon a rejected frame cleanly, got "
                        f"{tick_error}",
                    ),
                    "consumer abandons the round cleanly (F3)",
                    control,
                )
                world_check(
                    lambda: require(
                        stage != 1 or pages_back,
                        f"stage 1 leaked the failed round's pages: "
                        f"free={native.free_blocks} expected={free_with_guard}",
                    ),
                    "consumer releases the failed round's pages (F3)",
                    control,
                )

                print(
                    "DSV41_NATIVE_PP_ASYNC_FAILURE_AFTER_WRITES "
                    + json.dumps(
                        {
                            "rank": rank,
                            "stage": stage,
                            "tick_seconds": round(elapsed, 3),
                            "tick_error": tick_error,
                            "later_tick_seconds": round(later_elapsed, 3),
                            "later_tick_error": later_error,
                            "wrote_before_failure": bool(changed),
                            "changed_page_count": len(changed),
                            "foreign_pages_touched": len(foreign),
                            "free_blocks": native.free_blocks,
                            "free_blocks_expected": free_with_guard,
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


class V41NativePPAsyncFailureAfterWritesSM120Test(unittest.TestCase):
    def test_failure_after_tagged_writes_reaches_the_peer_stage(self):
        self.assertEqual(
            os.environ.get("RTP_LLM_V41_PP_ASYNC"),
            "1",
            "this target asserts the asynchronous contract; run it with RTP_LLM_V41_PP_ASYNC=1",
        )
        self.assertEqual(
            os.environ.get("RTP_LLM_V41_TEST_FAIL_AFTER_TAGGED_WRITES_STAGE"),
            "0",
            "this target needs the post-write injection on stage 0; it is set by BUILD env",
        )
        self.assertGreaterEqual(torch.cuda.device_count(), 8)
        for rank in range(8):
            self.assertEqual(torch.cuda.get_device_capability(rank), (12, 0))
        with tempfile.TemporaryDirectory() as directory:
            torch.manual_seed(41701)
            config = checkpoint(Path(directory), full_schedule=True, hidden_size=5120)
            with socket.socket() as listener:
                listener.bind(("127.0.0.1", 0))
                port = listener.getsockname()[1]
            mp.spawn(run_rank, args=(port, config), nprocs=8, join=True)


if __name__ == "__main__":
    unittest.main()
