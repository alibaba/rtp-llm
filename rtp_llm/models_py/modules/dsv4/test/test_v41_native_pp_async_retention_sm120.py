"""Asynchronous retained-dispatch contract for the V4.1 native PP local cache.

The synchronous contract allows exactly one prepared-but-uncommitted dispatch, which
chains `prepare -> ... -> WORLD completeExecution -> commit` and therefore forbids
preparing chunk N+1 until chunk N has completed across both PP stages. The
asynchronous contract keeps stage-local execution serial but lets a dispatch that
already completed locally stay *retained* (pinned by its pipeline slot) while the
next chunk prepares, so the two stages can overlap.

This target covers the allocator-level invariants that chaining introduces:
  * preparation chains from the retained dispatch's prospective request map, so a
    retained batch's cursor progress is neither lost nor resurrected;
  * commits publish oldest-first (a newer snapshot is a superset, so the reverse
    order would resurrect pre-chunk request state);
  * aborts unwind newest-first (a chained successor is unusable once its prefix is
    aborted, so none may survive a poisoned prefix);
  * retention is bounded by the pp_size+1 slot ring, not an unlimited queue;
  * retirement is refused while a retained dispatch still binds that request;
  * leases pin the retained dispatch's pages and are fully released afterwards;
  * an abandoned dispatch (never committed nor aborted) still expires, preserving
    the synchronous escape hatch.

It deliberately does not exercise the worker/forward path; the execution-lifecycle
precondition ("the retained dispatch completed locally") is enforced by
V41PPAdmission::prepareImpl and is covered by the admission/execution targets.
"""

import gc
import json
import os
import socket
import sys
import traceback
import unittest
from datetime import timedelta
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from rtp_llm.cpp.normal_engine.test import libth_v41_native_pp_transport_test as ops
from rtp_llm.models_py.distributed import collective_torch as collective
from rtp_llm.models_py.modules.dsv4.test.test_v41_cp_stage_worker_sm120 import (
    parallelism_for,
)
from rtp_llm.models_py.modules.dsv4.test.test_v41_native_pp_local_cache_sm120 import (
    rejected,
    release_allocations,
    wire,
)
from rtp_llm.models_py.modules.dsv4.test.test_v41_native_pp_transport_sm120 import (
    REQUESTS,
    cache_config,
)
from rtp_llm.models_py.modules.dsv4.v41_cache_transfer import (
    V41CacheIdentity,
    V41L20Span,
)
from rtp_llm.ops import NcclCommConfig

# pp_size + 1 for the qualified CEP4PP2 topology.
RING_BOUND = 3


def async_retention_rank(rank, port):
    torch.cuda.set_device(rank)
    torch.set_num_threads(1)
    os.environ["NCCL_P2P_DISABLE"] = "1"
    dist.init_process_group(
        "nccl",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=8,
        timeout=timedelta(seconds=180),
    )
    new_group = dist.new_group

    def bounded_group(*args, **kwargs):
        kwargs["timeout"] = timedelta(seconds=180)
        return new_group(*args, **kwargs)

    native = None
    with patch.dict(sys.modules, {"librtp_compute_ops": ops}), patch.object(
        dist, "new_group", side_effect=bounded_group
    ):
        try:
            pcfg = parallelism_for(rank)
            collective.init_distributed_environment(
                pcfg, NcclCommConfig(nccl_ip="127.0.0.1"), port
            )
            with torch.inference_mode():
                native = ops.NativePPTransport(cache_config(), pcfg)
                native.enable_async_local(True)
                request = REQUESTS[0]
                first = V41CacheIdentity(request, 1, 1)
                # Baseline BEFORE any root allocation. Rank 0 is the only rank that
                # owns root leases (borrow_root_ = pp_rank==0 && tp_rank==0), so
                # capturing after `allocate` would compare the final release against
                # a baseline that still includes those pages.
                free_before = native.free_blocks
                roots = [native.allocate(request, 256)] if rank == 0 else []
                pinned_baseline = native.free_blocks
                assert native.retained_local == 0

                def spans_for(identity, chunk):
                    return wire([V41L20Span(identity, chunk, chunk, 1)])

                # --- retention is bounded by the slot ring -----------------------
                retained = []
                for chunk in range(RING_BOUND):
                    retained.append(
                        native.prepare_local(spans_for(first, chunk), roots)
                    )
                    assert native.retained_local == chunk + 1
                rejected(
                    lambda: native.prepare_local(spans_for(first, RING_BOUND), roots)
                )
                assert native.retained_local == RING_BOUND
                # Chained preparation must not have released the retained pages.
                # (<= because rank 0 borrows its root lease and mallocs nothing.)
                assert native.free_blocks <= pinned_baseline

                # --- commits publish oldest-first --------------------------------
                rejected(lambda: native.commit_local(retained[1]))
                rejected(lambda: native.commit_local(retained[2]))
                assert native.retained_local == RING_BOUND
                native.commit_local(retained[0])
                assert native.retained_local == 2 and native.active_local == 1

                # --- retirement waits for retained dispatches --------------------
                rejected(lambda: native.retire_local(list(first.__dict__.values())))

                native.commit_local(retained[1])
                native.commit_local(retained[2])
                assert native.retained_local == 0 and native.active_local == 1
                # The chained snapshots really advanced the cursor: a replayed
                # chunk-1 span is stale rather than silently accepted.
                rejected(lambda: native.prepare_local(spans_for(first, 1), roots))

                native.retire_local(list(first.__dict__.values()))
                assert native.active_local == 0
                del retained
                gc.collect()
                release_allocations(roots)
                assert native.free_blocks == free_before

                # --- aborts unwind newest-first ----------------------------------
                second = V41CacheIdentity(request, 2, 1)
                roots2 = [native.allocate(request, 256)] if rank == 0 else []
                kept = [
                    native.prepare_local(spans_for(second, 0), roots2),
                    native.prepare_local(spans_for(second, 1), roots2),
                ]
                rejected(lambda: native.abort_local(kept[0]))
                assert native.retained_local == 2
                native.abort_local(kept[1])
                assert native.retained_local == 1
                native.abort_local(kept[0])
                assert native.retained_local == 0 and native.active_local == 0
                del kept
                gc.collect()
                release_allocations(roots2)
                assert native.free_blocks == free_before

                # --- an abandoned dispatch still expires -------------------------
                third = V41CacheIdentity(request, 3, 1)
                roots3 = [native.allocate(request, 256)] if rank == 0 else []
                dropped = native.prepare_local(spans_for(third, 0), roots3)
                del dropped
                gc.collect()
                assert native.retained_local == 0
                revived = native.prepare_local(spans_for(third, 0), roots3)
                native.commit_local(revived)
                native.retire_local(list(third.__dict__.values()))
                del revived
                gc.collect()
                release_allocations(roots3)
                assert native.free_blocks == free_before

                print(
                    "DSV41_NATIVE_PP_ASYNC_RETENTION "
                    + json.dumps(
                        {
                            "rank": rank,
                            "ring_bound": RING_BOUND,
                            "chained_preparation": True,
                            "ordered_commit": True,
                            "newest_first_abort": True,
                            "retirement_blocked_while_retained": True,
                            "stale_chunk_rejected_after_chained_commits": True,
                            "leases_pinned_then_released": True,
                        }
                    ),
                    flush=True,
                )
                native.close()
                native = None
            dist.barrier()
        except BaseException:
            traceback.print_exc()
            raise
        finally:
            if native is not None:
                native.wait()
                native.close()
            collective.destroy_distributed_environment()


class V41NativePPAsyncRetentionSM120Test(unittest.TestCase):
    def test_retained_dispatch_chaining(self):
        self.assertGreaterEqual(torch.cuda.device_count(), 8)
        for rank in range(8):
            self.assertEqual(torch.cuda.get_device_capability(rank), (12, 0))
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        mp.spawn(async_retention_rank, args=(port,), nprocs=8, join=True)


if __name__ == "__main__":
    unittest.main()
