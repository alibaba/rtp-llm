"""Production local cache admission over the actual four native PP lane pairs."""

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
from rtp_llm.models_py.modules.dsv4.fp8.test.test_v41_cp_owner_sm120 import zigzag
from rtp_llm.models_py.modules.dsv4.native_l20_transfer import (
    NativeL20ReplicaReceiver,
    export_native_l20_delta,
    native_l20_cache_views,
)
from rtp_llm.models_py.modules.dsv4.test.test_v41_cp_stage_worker_sm120 import (
    parallelism_for,
)
from rtp_llm.models_py.modules.dsv4.test.test_v41_native_pp_transport_sm120 import (
    CONTRACT,
    FRAME_KEYS,
    REQUESTS,
    TAGS,
    TEXT,
    WIRE_ROWS,
    cache_config,
    expected_delta,
    frame_for_round,
    pattern,
    require_snapshot,
    snapshot,
    update_expected,
)
from rtp_llm.models_py.modules.dsv4.v41_cache_transfer import (
    V41CacheIdentity,
    V41L20Span,
)
from rtp_llm.models_py.modules.dsv4.v41_pp import import_v41_pp_frame, pack_v41_pp_frame
from rtp_llm.ops import NcclCommConfig

ROUNDS = ([1, 1], [6, 4], [122, 124], [126, 126], [1, 1])


def rejected(fn, kind=ValueError):
    try:
        fn()
    except kind:
        return
    raise AssertionError("invalid native local admission succeeded")


def wire(spans):
    return torch.tensor([list(s.wire_values()) for s in spans], dtype=torch.int64)


def root_allocations(native, rank):
    return [native.allocate(request, 256) for request in REQUESTS] if rank == 0 else []


def release_allocations(allocations):
    for allocation in allocations:
        allocation.release_owner()
        allocation.release_lease()


def controls(native, other, rank, pcfg, initial, roots):
    before = native.free_blocks
    raw = snapshot(native.cache)
    valid = wire(initial)
    for row, column, value in (
        (1, 5, 0),
        (1, 0, REQUESTS[0]),
        (1, 1, 1),
        (1, 4, 1),
        (1, 5, 257),
    ):
        bad = valid.clone()
        bad[row, column] = value
        rejected(lambda: native.prepare_local(bad, roots))
        assert native.active_local == 0 and native.free_blocks == before
    rejected(lambda: native.prepare_local(valid.int(), roots))
    rejected(lambda: native.prepare_local(valid[:, :5], roots))
    if rank == 0:
        rejected(lambda: native.prepare_local(valid))
        rejected(lambda: native.prepare_local(valid, [roots[0], roots[0]]))
        oversized = native.allocate(9001, 257)
        rejected(lambda: native.prepare_local(valid, [roots[0], oversized]))
        release_allocations([oversized])
    else:
        borrowed = native.allocate(9001, 256)
        rejected(lambda: native.prepare_local(valid, [borrowed, borrowed]))
        release_allocations([borrowed])
    config = cache_config()
    config.gen_num_per_cycle = 1
    rejected(lambda: native.check_local_config(config, pcfg))
    wrong = parallelism_for(rank)
    wrong.prefill_cp_config.kv_cache_sharded = True
    rejected(lambda: native.check_local_config(cache_config(), wrong))
    wrong = parallelism_for(rank)
    wrong.pp_ep_enabled = False
    rejected(lambda: native.check_local_config(cache_config(), wrong))
    batch = native.prepare_local(valid, roots)
    assert native.active_local == 0 and batch.new_requests == [True, True]
    rejected(lambda: native.prepare_local(valid, roots))
    rejected(lambda: native.retire_local(list(initial[0].identity.__dict__.values())))
    del batch
    gc.collect()
    assert native.free_blocks == before
    batch = native.prepare_local(valid, roots)
    native.abort_local(batch)
    rejected(lambda: native.commit_local(batch))
    del batch
    assert native.free_blocks == before
    alien_roots = root_allocations(other, rank)
    alien = other.prepare_local(valid, alien_roots)
    rejected(lambda: native.commit_local(alien))
    rejected(lambda: native.abort_local(alien))
    other.abort_local(alien)
    del alien
    release_allocations(alien_roots)
    rolled_back = None
    if rank != 0:
        groups = len(native.cache.get_layer_cache_groups(0)) if rank >= 4 else 5
        # Stage-one slot zero owns all three tags; the first stage has five pools.
        pressure = []
        while native.free_blocks > groups:
            pressure.append(native.allocate(10000 + len(pressure), 256))
        remaining = native.free_blocks
        assert remaining == groups
        rejected(lambda: native.prepare_local(valid), RuntimeError)
        assert native.free_blocks == remaining and native.active_local == 0
        release_allocations(pressure)
        assert native.free_blocks == before
        rolled_back = True
    require_snapshot(native.cache, raw)
    return rolled_back


def run_rank(rank, port):
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

    native = other = None
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
                other = ops.NativePPTransport(cache_config(), pcfg)
                cache = native.cache
                free_before = native.free_blocks
                for layer in range(cache.layer_count):
                    for group in cache.get_layer_cache_groups(layer):
                        group.kv_cache_base.zero_()
                identities = [
                    V41CacheIdentity(request, i + 1, 1)
                    for i, request in enumerate(REQUESTS)
                ]
                initial = [V41L20Span(identity, 0, 0, 1) for identity in identities]
                roots = root_allocations(native, rank)
                rolled_back = controls(native, other, rank, pcfg, initial, roots)
                other.close()
                other = None
                blocker = [native.allocate(9501, 256)] if rank >= 4 else []
                views = native_l20_cache_views(TEXT, cache, rank // 4)
                receiver = None
                prefixes = [0, 0]
                first_tables = first_generations = None
                expected_cache = None
                for step, lengths in enumerate(ROUNDS):
                    spans = [
                        V41L20Span(identity, step, start, count)
                        for identity, start, count in zip(identities, prefixes, lengths)
                    ]
                    batch = native.prepare_local(wire(spans), roots)
                    tables = {
                        identity: {tag: tuple(table[tag]) for tag in TAGS}
                        for identity, table in zip(identities, batch.block_tables)
                    }
                    full_tables = batch.block_tables
                    kernel = batch.kernel_block_table
                    assert kernel.dtype == torch.int32 and kernel.shape == (
                        2,
                        max(batch.canonical_columns.values()) + 1,
                        1,
                    )
                    for row, pages in enumerate(full_tables):
                        for tag, column in batch.canonical_columns.items():
                            assert kernel[row, column, 0].item() == pages[tag][0]
                    if step == 0:
                        first_tables, first_generations = (
                            full_tables,
                            batch.local_generations,
                        )
                        all_tables = [None] * 8
                        dist.all_gather_object(all_tables, tables)
                        source_tables = all_tables[rank % 4]
                        for identity in identities:
                            assert (
                                all_tables[rank % 4][identity]
                                != all_tables[rank % 4 + 4][identity]
                            )
                        if rank < 4:
                            for tag, view in views.items():
                                offsets = torch.arange(view.raw.shape[1], device="cuda")
                                for pages in tables.values():
                                    for page in pages[tag]:
                                        view.raw[page].copy_(
                                            pattern(page, offsets, tag, rank % 4)
                                        )
                        else:
                            receiver = NativeL20ReplicaReceiver(
                                TEXT, cache, CONTRACT, rank % 4
                            )
                            for identity in identities:
                                receiver.begin_request(identity, tables[identity])
                        expected_cache = snapshot(cache)
                    else:
                        assert (
                            full_tables == first_tables
                            and batch.local_generations == first_generations
                        )
                        assert batch.new_requests == [False, False]
                    cp, _, _ = zigzag(
                        lengths, prefixes, rank % 4, torch.device("cuda", rank)
                    )
                    rows = torch.stack(
                        (
                            cp.req_id_per_token.long(),
                            cp.global_positions,
                            cp.local_is_real.long(),
                        ),
                        -1,
                    ).cpu()
                    rows[~rows[:, 2].bool()] = torch.tensor([-1, -1, 0])
                    expected_frame = frame_for_round(
                        rows, step, rank % 4, torch.device("cuda", rank)
                    )
                    deltas = [
                        expected_delta(spans, source_tables, tag, rank % 4)
                        for tag in TAGS
                    ]
                    if rank < 4:
                        outgoing = pack_v41_pp_frame(
                            export_native_l20_delta(
                                TEXT, cache, CONTRACT, rank % 4, step, spans, tables
                            ),
                            expected_frame,
                            spans,
                            rows,
                        )
                        native.send_local(outgoing, batch)
                        del outgoing
                    else:
                        incoming = native.receive_local(batch)
                    assert native.retained_leases == 2 and native.pending_tickets > 0
                    native.wait()
                    if rank >= 4:
                        for key, expected in zip(
                            FRAME_KEYS, expected_frame.__dict__.values()
                        ):
                            torch.testing.assert_close(
                                incoming[key], expected, rtol=0, atol=0
                            )
                        for key, expected in zip(WIRE_ROWS, deltas):
                            torch.testing.assert_close(
                                incoming[key].cpu(), expected, rtol=0, atol=0
                            )
                        import_v41_pp_frame(incoming, receiver, spans, rows)
                        update_expected(expected_cache, spans, tables, deltas)
                    require_snapshot(cache, expected_cache)
                    native.finish()
                    native.commit_local(batch)
                    assert native.active_local == 2
                    rejected(lambda: native.commit_local(batch))
                    rejected(lambda: native.prepare_local(wire(spans), roots))
                    if step == 0:
                        gap = wire(spans)
                        gap[:, 3] = 1
                        gap[:, 4] = 2
                        rejected(lambda: native.prepare_local(gap, roots))
                        if rank == 0:
                            alias = wire(
                                [V41L20Span(V41CacheIdentity(9100, 3, 1), 0, 0, 1)]
                            )
                            rejected(lambda: native.prepare_local(alias, [roots[0]]))
                    prefixes = [p + n for p, n in zip(prefixes, lengths)]
                    print(
                        "DSV41_NATIVE_PP_LOCAL_HANDOFF "
                        + json.dumps(
                            {
                                "rank": rank,
                                "step": step,
                                "ends": prefixes,
                                "native_bytes_exact": True,
                                "canonical_tables_exact": True,
                                "independent_local_allocations": True,
                            }
                        ),
                        flush=True,
                    )
                    if step != len(ROUNDS) - 1:
                        del batch
                occupied = native.free_blocks
                for identity in identities:
                    rejected(
                        lambda: native.retire_local(
                            [
                                identity.request_id,
                                identity.generation + 1,
                                identity.cache_incarnation,
                            ]
                        )
                    )
                    native.retire_local(list(identity.__dict__.values()))
                    if receiver is not None:
                        receiver.cancel(identity)
                        receiver.release(identity)
                release_allocations(roots)
                assert native.active_local == 0 and native.free_blocks == occupied
                probe = native.allocate(9901, 256)
                for tag in TAGS:
                    assert not set(probe.block_tables[tag]).intersection(
                        page for pages in tables.values() for page in pages[tag]
                    )
                release_allocations([probe])
                del batch
                gc.collect()
                release_allocations(blocker)
                assert native.free_blocks == free_before
                rejected(lambda: native.prepare_local(wire(initial), roots))
                fresh_identity = V41CacheIdentity(REQUESTS[0], 3, 1)
                fresh_span = wire([V41L20Span(fresh_identity, 0, 0, 1)])
                fresh_root = [native.allocate(REQUESTS[0], 256)] if rank == 0 else []
                fresh = native.prepare_local(fresh_span, fresh_root)
                assert fresh.local_generations[0] > max(first_generations)
                native.commit_local(fresh)
                native.retire_local(list(fresh_identity.__dict__.values()))
                release_allocations(fresh_root)
                del fresh
                assert native.free_blocks == free_before
                print(
                    "DSV41_NATIVE_PP_LOCAL_LIFETIME "
                    + json.dumps(
                        {
                            "rank": rank,
                            "free_before": free_before,
                            "free_after": native.free_blocks,
                            "whole_batch_allocation_rollback": rolled_back,
                            "abandoned_preparation_rollback": True,
                            "retired_pages_pinned_by_batch": True,
                            "fresh_local_generation": True,
                            "stale_wire_rejected": True,
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
            for fixture in (other, native):
                if fixture is not None:
                    fixture.wait()
                    fixture.close()
            collective.destroy_distributed_environment()


class V41NativePPLocalCacheSM120Test(unittest.TestCase):
    def test_local_admission_over_actual_pp(self):
        self.assertGreaterEqual(torch.cuda.device_count(), 8)
        for rank in range(8):
            self.assertEqual(torch.cuda.get_device_capability(rank), (12, 0))
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        mp.spawn(run_rank, args=(port,), nprocs=8, join=True)


if __name__ == "__main__":
    unittest.main()
