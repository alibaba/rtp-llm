"""Whole-WORLD native allocation admission before maintained CP worker entry."""

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
from rtp_llm.models_py.modules.dsv4.fp8.test.test_v41_cp_owner_sm120 import zigzag
from rtp_llm.models_py.modules.dsv4.fp8.test.test_v41_worker_sm120 import (
    checkpoint,
    resources,
)
from rtp_llm.models_py.modules.dsv4.test.test_v41_cp_stage_worker_sm120 import (
    make_worker,
    parallelism_for,
)
from rtp_llm.models_py.modules.dsv4.test.test_v41_native_pp_local_cache_sm120 import (
    release_allocations,
    root_allocations,
    wire,
)
from rtp_llm.models_py.modules.dsv4.test.test_v41_native_pp_preflight_sm120 import (
    ROUNDS,
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


def tensors_for(spans, ids, windows):
    return {
        "spans": wire(spans),
        "ids": ids.clone(),
        "windows": windows.clone(),
        "lengths": torch.tensor([s.count for s in spans], dtype=torch.int32),
        "prefixes": torch.tensor([s.start for s in spans], dtype=torch.int32),
        "decode_lengths": torch.empty(0, dtype=torch.int32),
        "positions": torch.cat(
            [torch.arange(s.start, s.start + s.count, dtype=torch.int32) for s in spans]
        ),
        # Source addresses deliberately cannot describe the independent pools.
        "table": torch.full((1, len(spans), 1), -99, dtype=torch.int32),
        "physical_table": torch.full((1, len(spans), 1), -98, dtype=torch.int32),
        "zero_blocks": torch.tensor([999999], dtype=torch.int64),
        "copy_mapping": torch.empty((0, 3), dtype=torch.int32),
    }


def caller_snapshot(source):
    return {
        name: (value.data_ptr(), value.clone())
        for name, value in source.tensors.items()
    }, source.lease_count


def require_caller(source, before):
    tensors, leases = before
    assert source.lease_count == leases and source.tensors.keys() == tensors.keys()
    for name, (pointer, value) in tensors.items():
        current = source.tensors[name]
        assert current.data_ptr() == pointer
        torch.testing.assert_close(current, value, rtol=0, atol=0)


def require_rejection(native, source, raw, free, active):
    before = caller_snapshot(source)
    try:
        native.prepare_global(source)
    except ops.NativeInputRejected:
        pass
    else:
        raise AssertionError("native WORLD admitted a rejected peer/dispatch")
    assert not native.global_failed and source.batch is None
    assert native.free_blocks == free and native.active_global == active
    require_caller(source, before)
    require_snapshot(native.cache, raw)


def corrupt(source, variant):
    tensors = source.tensors
    if variant == "different-spans":
        tensors["spans"][0, 1] += 10
    elif variant == "different-tokens":
        tensors["ids"][0] += 1
        tensors["windows"][0, 0] = tensors["ids"][0]
    elif variant == "different-history":
        tensors["windows"][0, 1] = 7
    elif variant in ("lengths", "prefixes", "positions"):
        tensors[variant][0] += 1
    elif variant == "spans-dtype":
        source.set_tensor("spans", tensors["spans"].int())
    elif variant == "windows-shape":
        source.set_tensor("windows", tensors["windows"][:, :3].contiguous())
    elif variant == "negative-token":
        tensors["ids"][0] = -1
    elif variant == "copy-mapping":
        source.set_tensor("copy_mapping", torch.zeros((1, 3), dtype=torch.int32))
    elif variant == "mask":
        source.set_tensor("mask", torch.ones(1))
    else:
        source.control(variant)


def controls(native, other, rank, spans, roots, model):
    ids, windows = global_tokens([1, 1], [[], []], 0)
    free = native.free_blocks
    raw = snapshot(native.cache)
    negatives = [
        ("different-spans", 4),
        ("different-tokens", 2),
        ("different-history", 6),
        ("lengths", 0),
        ("prefixes", 5),
        ("positions", 4),
        ("spans-dtype", 0),
        ("windows-shape", 6),
        ("negative-token", 2),
        ("copy-mapping", 4),
        ("mask", 0),
        ("shutdown", 7),
        ("warmup", 0),
        ("verify", 4),
        ("multimodal", 6),
        ("pp_frame", 0),
        ("missing-root", 0),
        ("foreign-root", 0),
        ("nonroot-lease", 4),
    ]
    for variant, actor in negatives:
        borrowed = []
        selected = roots
        if rank == actor:
            if variant == "missing-root":
                selected = []
            elif variant == "foreign-root":
                borrowed = root_allocations(other, rank)
                selected = borrowed
            elif variant == "nonroot-lease":
                borrowed = [native.allocate(9100, 256)]
                selected = borrowed
        source = native.global_inputs(tensors_for(spans, ids, windows), selected)
        if rank == actor and variant not in (
            "missing-root",
            "foreign-root",
            "nonroot-lease",
        ):
            corrupt(source, variant)
        require_rejection(native, source, raw, native.free_blocks, 0)
        del source
        release_allocations(borrowed)
        assert native.free_blocks == free
        # Retry the exact rejected round without advancing committed cursors.
        retry = native.global_inputs(tensors_for(spans, ids, windows), roots)
        batch = native.prepare_global(retry)
        request = native.split_global_inputs(retry)
        dispatch = model.prepare_native_pp_prefill(request)
        assert dispatch.schedule == tuple(s.wire_values() for s in spans)
        clean_worker(model)
        native.abort_global(batch)
        del dispatch, request, retry, batch
        gc.collect()
        assert native.free_blocks == free and native.active_global == 0
        require_snapshot(native.cache, raw)
        print(
            "DSV41_NATIVE_PP_ADMISSION_REJECTION "
            + json.dumps(
                {
                    "rank": rank,
                    "variant": variant,
                    "actor": actor,
                    "world_rejected": True,
                    "caller_unchanged": True,
                    "read_only": True,
                    "retry": True,
                }
            ),
            flush=True,
        )

    # Real independent-pool exhaustion halfway through a two-request batch.
    for actor in (2, 6):
        pressure = []
        if rank == actor:
            groups = 5 if rank < 4 else 3
            while native.free_blocks > groups:
                pressure.append(native.allocate(10000 + len(pressure), 256))
            assert native.free_blocks == groups
        source = native.global_inputs(tensors_for(spans, ids, windows), roots)
        require_rejection(native, source, raw, native.free_blocks, 0)
        del source
        release_allocations(pressure)
        assert native.free_blocks == free
        retry = native.global_inputs(tensors_for(spans, ids, windows), roots)
        batch = native.prepare_global(retry)
        native.abort_global(batch)
        del retry, batch
        assert native.free_blocks == free
        print(
            "DSV41_NATIVE_PP_ADMISSION_REJECTION "
            + json.dumps(
                {
                    "rank": rank,
                    "variant": "allocation-exhaustion",
                    "actor": actor,
                    "world_rejected": True,
                    "caller_unchanged": True,
                    "read_only": True,
                    "retry": True,
                }
            ),
            flush=True,
        )

    # Only one rank retains a pending preparation. A second WORLD transaction
    # must roll back peers while preserving the unrelated pending batch.
    pending = native.global_inputs(tensors_for(spans, ids, windows), roots)
    held = native.prepare_global(pending)
    tables = held.block_tables
    if rank != 2:
        native.abort_global(held)
        del held, pending
    overlap = native.global_inputs(tensors_for(spans, ids, windows), roots)
    require_rejection(native, overlap, raw, native.free_blocks, 0)
    del overlap
    if rank == 2:
        assert held.block_tables == tables
        native.abort_global(held)
        del held, pending
    assert native.free_blocks == free
    print(
        "DSV41_NATIVE_PP_ADMISSION_REJECTION "
        + json.dumps(
            {
                "rank": rank,
                "variant": "overlap",
                "actor": 2,
                "world_rejected": True,
                "caller_unchanged": True,
                "read_only": True,
                "pending_preserved": True,
            }
        ),
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
        timeout=timedelta(seconds=180),
    )
    new_group = dist.new_group

    def bounded_group(*args, **kwargs):
        kwargs["timeout"] = timedelta(seconds=180)
        return new_group(*args, **kwargs)

    native = other = model = None
    calls = {"world": 0, "stage": 0}
    cpp_modes = []
    register = ops.register_comm_ops

    def checked_registration(broadcast, reduce, gather):
        def checked_gather(receive, mode, send, inplace):
            cpp_modes.append(mode)
            return gather(receive, mode, send, inplace)

        return register(broadcast, reduce, checked_gather)

    with patch.dict(sys.modules, {"librtp_compute_ops": ops}), patch.object(
        dist, "new_group", side_effect=bounded_group
    ), patch.object(ops, "register_comm_ops", side_effect=checked_registration):
        try:
            collective.init_distributed_environment(
                parallelism_for(rank), NcclCommConfig(nccl_ip="127.0.0.1"), port
            )
            assert collective._get_group(collective.Group.WORLD) is dist.group.WORLD
            with torch.inference_mode(), ExitStack() as stack:
                native = ops.NativePPTransport(cache_config(), parallelism_for(rank))
                other = ops.NativePPTransport(cache_config(), parallelism_for(rank))
                cache = native.cache
                free_before = native.free_blocks
                roots = root_allocations(native, rank)
                model = make_worker(rank, config)
                assert model.initialize(resources(cache))
                for layer in range(cache.layer_count):
                    for group in cache.get_layer_cache_groups(layer):
                        group.kv_cache_base.zero_()
                gather = dist.all_gather_into_tensor

                def admission_only(output, value, **kwargs):
                    if kwargs["group"] is dist.group.WORLD:
                        assert value.shape == (2576,) and output.shape == (20608,)
                        calls["world"] += 1
                    else:
                        assert kwargs["group"] is model._dp_group
                        assert value.shape == (1, 33) and output.shape == (4, 33)
                        calls["stage"] += 1
                    return gather(output, value, **kwargs)

                stack.enter_context(
                    patch.object(
                        dist, "all_gather_into_tensor", side_effect=admission_only
                    )
                )
                for layer in model.v4.layers:
                    stack.enter_context(
                        patch.object(
                            layer,
                            "forward",
                            side_effect=AssertionError(
                                "admission executed a maintained block"
                            ),
                        )
                    )
                identities = [
                    V41CacheIdentity(request, i + 1, 1)
                    for i, request in enumerate(REQUESTS)
                ]
                initial = [V41L20Span(identity, 0, 0, 1) for identity in identities]
                controls(native, other, rank, initial, roots, model)
                prefixes, histories = [0, 0], [[], []]
                for step, lengths in enumerate(ROUNDS):
                    spans = [
                        V41L20Span(identity, step, start, count)
                        for identity, start, count in zip(identities, prefixes, lengths)
                    ]
                    ids, windows = global_tokens(lengths, histories, step)
                    raw = snapshot(cache)
                    source_tensors = tensors_for(spans, ids, windows)
                    source = native.global_inputs(source_tensors, roots)
                    batch = native.prepare_global(source)
                    assert source.lease_count == 2 and source.batch is not None
                    assert native.active_global == (0 if step == 0 else 2)
                    for name in (
                        "spans",
                        "ids",
                        "windows",
                        "lengths",
                        "prefixes",
                        "positions",
                    ):
                        torch.testing.assert_close(
                            source.tensors[name], source_tensors[name], rtol=0, atol=0
                        )
                        assert (
                            source.tensors[name].data_ptr()
                            != source_tensors[name].data_ptr()
                        )
                    assert (
                        "zero_blocks" not in source.tensors
                        and "copy_mapping" not in source.tensors
                    )
                    expected = batch.kernel_block_table.permute(1, 0, 2).contiguous()
                    torch.testing.assert_close(
                        source.tensors["table"].cpu(), expected, rtol=0, atol=0
                    )
                    torch.testing.assert_close(
                        source.tensors["physical_table"].cpu(), expected, rtol=0, atol=0
                    )
                    for name in (
                        "spans",
                        "ids",
                        "windows",
                        "lengths",
                        "prefixes",
                        "positions",
                    ):
                        source_tensors[name].zero_()
                    request = native.split_global_inputs(source)
                    dispatch = model.prepare_native_pp_prefill(request)
                    cp, indices, _ = zigzag(
                        lengths, prefixes, rank % 4, torch.device("cuda", rank)
                    )
                    local_ids = ids.index_select(0, indices.cpu()).cuda()
                    local_ids[~cp.local_is_real] = 0
                    local_windows = windows.index_select(0, indices.cpu()).cuda()
                    local_windows[~cp.local_is_real] = -1
                    torch.testing.assert_close(
                        dispatch.input_ids, local_ids, rtol=0, atol=0
                    )
                    torch.testing.assert_close(
                        dispatch.token_windows, local_windows, rtol=0, atol=0
                    )
                    assert cache.native_allocation_tables(request) == batch.block_tables
                    clean_worker(model)
                    require_snapshot(cache, raw)
                    if rank < 4:
                        native.send_local(
                            {
                                "admitted_ids": dispatch.input_ids,
                                "admitted_rows": dispatch.rows.cuda(),
                            },
                            batch,
                        )
                    else:
                        received = native.receive_local(batch)
                    native.wait()
                    if rank >= 4:
                        torch.testing.assert_close(
                            received["admitted_ids"], dispatch.input_ids, rtol=0, atol=0
                        )
                        torch.testing.assert_close(
                            received["admitted_rows"].cpu(),
                            dispatch.rows,
                            rtol=0,
                            atol=0,
                        )
                        del received
                    native.commit_global(batch)
                    native.finish()
                    assert native.active_global == 2
                    values = ids.tolist()
                    histories[0].extend(values[: lengths[0]])
                    histories[1].extend(values[lengths[0] :])
                    prefixes = [p + n for p, n in zip(prefixes, lengths)]
                    print(
                        "DSV41_NATIVE_PP_ADMISSION "
                        + json.dumps(
                            {
                                "rank": rank,
                                "step": step,
                                "ends": prefixes,
                                "canonical_local_tables": True,
                                "cpp_zigzag_exact": True,
                                "caller_snapshot_independent": True,
                                "read_only": True,
                                "native_named_tensor_handoff": True,
                            }
                        ),
                        flush=True,
                    )
                    del source, request, batch
                    if step != len(ROUNDS) - 1:
                        del dispatch
                occupied = native.free_blocks
                for identity in identities:
                    native.retire_global(list(identity.__dict__.values()))
                release_allocations(roots)
                assert native.free_blocks == occupied and native.active_global == 0
                del dispatch
                gc.collect()
                assert native.free_blocks == free_before
                assert not native.global_failed
                # An unexpected collective implementation failure poisons the
                # session instead of reporting a retryable malformed dispatch.
                fresh_identity = V41CacheIdentity(REQUESTS[0], 3, 1)
                fresh = [V41L20Span(fresh_identity, 0, 0, 1)]
                fresh_roots = [native.allocate(REQUESTS[0], 256)] if rank == 0 else []
                source = native.global_inputs(
                    tensors_for(fresh, ids[:1], windows[:1]), fresh_roots
                )
                before = caller_snapshot(source)
                available = native.free_blocks
                with patch.object(
                    dist,
                    "all_gather_into_tensor",
                    side_effect=RuntimeError("injected WORLD callback failure"),
                ):
                    try:
                        native.prepare_global(source)
                    except RuntimeError:
                        assert native.global_failed
                    else:
                        raise AssertionError("unexpected callback failure succeeded")
                require_caller(source, before)
                assert native.free_blocks == available and source.batch is None
                try:
                    native.prepare_global(source)
                except RuntimeError:
                    assert native.global_failed
                else:
                    raise AssertionError("failed WORLD session retried")
                del source
                release_allocations(fresh_roots)
                assert native.free_blocks == free_before
                assert calls == {"world": 47, "stage": 22}
                assert cpp_modes.count(2) == 48
                model.close()
                model = None
                native.close()
                other.close()
                native = other = None
                print(
                    "DSV41_NATIVE_PP_ADMISSION_LIFETIME "
                    + json.dumps(
                        {
                            "rank": rank,
                            "calls": calls,
                            "cpp_world_calls": cpp_modes.count(2),
                            "free_before": free_before,
                            "free_after": free_before,
                            "descriptor_holds_native_leases": True,
                            "unexpected_failure_poisoned": True,
                        }
                    ),
                    flush=True,
                )
            dist.barrier()
        except BaseException:
            print(
                "DSV41_ADMISSION_FAILURE "
                + json.dumps(
                    {
                        "rank": rank,
                        "calls": calls,
                        "cpp_modes": cpp_modes,
                        "groups": {
                            str(k): v.size() for k, v in collective._group_map.items()
                        },
                    }
                ),
                flush=True,
            )
            traceback.print_exc()
            raise
        finally:
            if model is not None:
                model.close()
            for fixture in (other, native):
                if fixture is not None:
                    fixture.wait()
                    fixture.close()
            Dsv4SharedRuntimeBufferStore._reset_for_test()
            collective.destroy_distributed_environment()


class V41NativePPAdmissionSM120Test(unittest.TestCase):
    def test_world_admission_then_native_cp_worker_preflight(self):
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
