"""Coordinated fresh-page initialization in actual independent native pools."""

import gc
import json
import os
import socket
import sys
import tempfile
import traceback
import unittest
from contextlib import ExitStack
from datetime import timedelta
from pathlib import Path
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from rtp_llm.cpp.normal_engine.test import libth_v41_native_pp_transport_test as ops
from rtp_llm.models_py.distributed import collective_torch as collective
from rtp_llm.models_py.model_desc.deepseek_v4_model import Dsv4SharedRuntimeBufferStore
from rtp_llm.models_py.modules.dsv4.fp8.test.test_v41_worker_sm120 import (
    checkpoint,
    resources,
)
from rtp_llm.models_py.modules.dsv4.test.test_v41_cp_stage_worker_sm120 import (
    make_worker,
    parallelism_for,
)
from rtp_llm.models_py.modules.dsv4.test.test_v41_native_pp_admission_sm120 import (
    tensors_for,
)
from rtp_llm.models_py.modules.dsv4.test.test_v41_native_pp_local_cache_sm120 import (
    release_allocations,
    root_allocations,
    wire,
)
from rtp_llm.models_py.modules.dsv4.test.test_v41_native_pp_preflight_sm120 import (
    clean_worker,
    global_tokens,
)
from rtp_llm.models_py.modules.dsv4.test.test_v41_native_pp_transport_sm120 import (
    REQUESTS,
    cache_config,
    require_snapshot,
    snapshot,
)
from rtp_llm.models_py.modules.dsv4.v41_cache_transfer import (
    V41CacheIdentity,
    V41L20Span,
)
from rtp_llm.ops import NcclCommConfig


def fill_pages(cache):
    for layer in range(cache.layer_count):
        for group in cache.get_layer_cache_groups(layer):
            raw = group.kv_cache_base.view(torch.uint8)
            for page in range(raw.shape[0]):
                raw[page].fill_(1 + (layer * 11 + page * 7) % 250)


def zero_oracle(expected, batch):
    for row, fresh in zip(batch.block_tables, batch.new_requests):
        if fresh:
            for (layer, tag), raw in expected.items():
                for page in row[tag]:
                    raw[page].zero_()


def rejection(native, candidate, raw, rank, variant, actor):
    try:
        native.initialize_global(candidate)
    except ops.NativeInputRejected:
        pass
    else:
        raise AssertionError("invalid tagged initialization reached cache writes")
    assert not native.global_failed
    require_snapshot(native.cache, raw)
    print(
        "DSV41_NATIVE_PP_INITIALIZATION_REJECTION "
        + json.dumps({"rank": rank, "variant": variant, "actor": actor}),
        flush=True,
    )


def run_rank(rank, port, config):
    torch.cuda.set_device(rank)
    torch.set_num_threads(1)
    os.environ.update(
        {
            "NCCL_P2P_DISABLE": "1",
            "DSV41_ENGRAM_HOST_BYTES": str(2**20),
            "DSV41_ENGRAM_CACHE_DIR": "/dev/shm",
            "DSV4_PP_EP_ENABLE": "1",
            "DSV4_PP_EP_BACKEND": "fork_nccl_mxfp8",
            "DSV4_MOE_STRATEGY": "auto",
            "MOEDBG": "0",
        }
    )
    dist.init_process_group(
        "nccl",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=8,
        timeout=timedelta(seconds=120),
    )
    new_group = dist.new_group

    def bounded_group(*args, **kwargs):
        kwargs["timeout"] = timedelta(seconds=120)
        return new_group(*args, **kwargs)

    native = model = None
    with patch.dict(sys.modules, {"librtp_compute_ops": ops}), patch.object(
        dist, "new_group", side_effect=bounded_group
    ):
        try:
            collective.init_distributed_environment(
                parallelism_for(rank), NcclCommConfig(nccl_ip="127.0.0.1"), port
            )
            with torch.inference_mode(), ExitStack() as stack:
                native = ops.NativePPTransport(cache_config(), parallelism_for(rank))
                cache = native.cache
                free_before = native.free_blocks
                # A longer unrelated request separates page numbers in the
                # independent full/SWA and single-state pools on stage zero.
                blocker = native.allocate(88000 + rank, 257)
                roots = root_allocations(native, rank)
                model = make_worker(rank, config)
                assert model.initialize(resources(cache))
                for layer in model.v4.layers:
                    stack.enter_context(
                        patch.object(
                            layer,
                            "forward",
                            side_effect=AssertionError("initialization ran a block"),
                        )
                    )
                fill_pages(cache)
                expected = snapshot(cache)
                identities = [
                    V41CacheIdentity(request, index + 1, 1)
                    for index, request in enumerate(REQUESTS)
                ]
                spans = [V41L20Span(identity, 0, 0, 1) for identity in identities]
                ids, windows = global_tokens([1, 1], [[], []], 0)

                source = native.global_inputs(tensors_for(spans, ids, windows), roots)
                source_only = native.prepare_global(source)
                rejection(native, source_only, expected, rank, "source-only", -1)
                assert not source_only.initialized
                native.abort_global(source_only)
                rejection(native, source_only, expected, rank, "aborted", -1)
                del source, source_only

                source = native.global_inputs(tensors_for(spans, ids, windows), roots)
                first = native.prepare_global_worker(
                    source, model.prepare_native_pp_prefill
                )
                assert first.new_requests == [True, True]
                if rank < 4:
                    assert len({v[0] for v in first.block_tables[0].values()}) > 1
                alien = native.prepare_local(wire(spans), roots)
                rejection(
                    native,
                    alien if rank == 4 else first,
                    expected,
                    rank,
                    "foreign-owner",
                    4,
                )
                rejection(
                    native,
                    None if rank == 0 else first,
                    expected,
                    rank,
                    "missing-batch",
                    0,
                )
                assert not first.initialized and native.active_global == 0
                native.abort_local(alien)
                del alien
                native.initialize_global(first)
                assert first.initialized
                zero_oracle(expected, first)
                require_snapshot(cache, expected)
                rejection(native, first, expected, rank, "repeated", -1)
                native.commit_global(first)
                rejection(native, first, expected, rank, "committed", -1)
                print(
                    "DSV41_NATIVE_PP_INITIALIZATION "
                    + json.dumps({"rank": rank, "step": 0, "fresh": [True, True]}),
                    flush=True,
                )
                del source, first

                # Model writes are represented by distinguishable bytes here;
                # no attention/expert arithmetic is part of this qualification.
                fill_pages(cache)
                expected = snapshot(cache)
                ids, windows = global_tokens([6, 4], [[3], [4]], 1)
                spans = [
                    V41L20Span(identity, 1, 1, count)
                    for identity, count in zip(identities, [6, 4])
                ]
                source = native.global_inputs(tensors_for(spans, ids, windows), roots)
                continued = native.prepare_global_worker(
                    source, model.prepare_native_pp_prefill
                )
                assert continued.new_requests == [False, False]
                native.initialize_global(continued)
                assert continued.initialized
                require_snapshot(cache, expected)
                native.commit_global(continued)
                print(
                    "DSV41_NATIVE_PP_INITIALIZATION "
                    + json.dumps({"rank": rank, "step": 1, "fresh": [False, False]}),
                    flush=True,
                )
                del source, continued

                # Recycle the retired second request's dirty pages into a new
                # source identity. Only that row may clear; row zero continues.
                native.retire_global(list(identities[1].__dict__.values()))
                if rank == 0:
                    release_allocations([roots.pop()])
                    roots.append(native.allocate(99002, 256))
                fresh = V41CacheIdentity(99002, 3, 1)
                spans = [
                    V41L20Span(identities[0], 2, 7, 1),
                    V41L20Span(fresh, 0, 0, 1),
                ]
                ids, windows = global_tokens([1, 1], [[23, 24, 25], []], 2)
                source = native.global_inputs(tensors_for(spans, ids, windows), roots)
                mixed = native.prepare_global_worker(
                    source, model.prepare_native_pp_prefill
                )
                assert mixed.new_requests == [False, True]
                native.initialize_global(mixed)
                zero_oracle(expected, mixed)
                require_snapshot(cache, expected)
                native.commit_global(mixed)
                clean_worker(model)
                print(
                    "DSV41_NATIVE_PP_INITIALIZATION "
                    + json.dumps({"rank": rank, "step": 2, "fresh": [False, True]}),
                    flush=True,
                )
                occupied = native.free_blocks
                held_dispatch = source.dispatch
                for identity in (identities[0], fresh):
                    native.retire_global(list(identity.__dict__.values()))
                release_allocations(roots)
                del source, mixed
                assert native.active_global == 0 and native.free_blocks == occupied
                del held_dispatch
                release_allocations([blocker])
                gc.collect()
                assert native.free_blocks == free_before
                print(
                    "DSV41_NATIVE_PP_INITIALIZATION_LIFETIME "
                    + json.dumps(
                        {
                            "rank": rank,
                            "free_before": free_before,
                            "free_after": native.free_blocks,
                        }
                    ),
                    flush=True,
                )

                # Fail the completion agreement after actual stores. The
                # coordinator must own the allocation after callers drop it.
                fatal_roots = (
                    [native.allocate(99101, 256), native.allocate(99102, 256)]
                    if rank == 0
                    else []
                )
                fatal_spans = [
                    V41L20Span(V41CacheIdentity(99101 + row, 4 + row, 1), 0, 0, count)
                    for row, count in enumerate([7, 3])
                ]
                ids, windows = global_tokens([7, 3], [[], []], 3)
                fill_pages(cache)
                expected = snapshot(cache)
                source = native.global_inputs(
                    tensors_for(fatal_spans, ids, windows), fatal_roots
                )
                fatal = native.prepare_global_worker(
                    source, model.prepare_native_pp_prefill
                )
                occupied = native.free_blocks
                zero_oracle(expected, fatal)
                original_gather = dist.all_gather_into_tensor

                def failed_completion(output, packet, *args, **kwargs):
                    if packet.dtype == torch.int64 and tuple(packet.shape) == (3,):
                        if int(packet[1]) == 5:
                            raise RuntimeError(
                                "injected tagged initialization completion"
                            )
                    return original_gather(output, packet, *args, **kwargs)

                with patch.object(
                    dist, "all_gather_into_tensor", new=failed_completion
                ):
                    try:
                        native.initialize_global(fatal)
                    except RuntimeError:
                        pass
                    else:
                        raise AssertionError(
                            "fatal completion published initialization"
                        )
                assert native.global_failed and not fatal.initialized
                require_snapshot(cache, expected)
                try:
                    native.commit_global(fatal)
                except RuntimeError:
                    pass
                else:
                    raise AssertionError("failed session committed its cursors")
                release_allocations(fatal_roots)
                del source, fatal
                gc.collect()
                assert native.active_global == 0 and native.free_blocks == occupied
                torch.cuda.synchronize()
                native.wait()
                native.abort_global(None)
                assert native.free_blocks == free_before
                clean_worker(model)
                print(
                    "DSV41_NATIVE_PP_INITIALIZATION_FATAL "
                    + json.dumps(
                        {
                            "rank": rank,
                            "writes_exact": True,
                            "retained_after_caller_drop": True,
                            "free_before": free_before,
                            "free_after_drain_abort": native.free_blocks,
                        }
                    ),
                    flush=True,
                )
                model.close()
                model = None
                native.close()
                native = None
            dist.barrier()
        except BaseException:
            print(f"DSV41_INITIALIZATION_FAILURE rank={rank}", flush=True)
            traceback.print_exc()
            raise
        finally:
            if model is not None:
                model.close()
            if native is not None:
                native.wait()
                native.close()
            Dsv4SharedRuntimeBufferStore._reset_for_test()
            collective.destroy_distributed_environment()


class V41NativePPInitializationSM120Test(unittest.TestCase):
    def test_fresh_tagged_pages_after_world_worker_readiness(self):
        self.assertGreaterEqual(torch.cuda.device_count(), 8)
        with tempfile.TemporaryDirectory() as root, socket.socket() as sock:
            torch.manual_seed(41700)
            config = checkpoint(Path(root), full_schedule=True, hidden_size=5120)
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
            sock.close()
            mp.spawn(run_rank, args=(port, config), nprocs=8, join=True)


if __name__ == "__main__":
    unittest.main()
