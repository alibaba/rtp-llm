"""Native lease provenance and read-only stage-agreed maintained CP preflight."""

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
from rtp_llm.models_py.model_desc.v41_errors import V41InputRejected
from rtp_llm.models_py.modules.dsv4.engram import make_token_windows
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
    rejected,
    release_allocations,
    root_allocations,
    wire,
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
from rtp_llm.ops.compute_ops import PyModelInputs

ROUNDS = ([1, 1], [6, 4], [122, 124])


def clean_worker(model):
    for layer in model.v4.layers:
        assert layer.engram_hashes is None and layer.engram_token_mask is None
        assert layer.attn_hc.pre_mix_out is None and layer.ffn_hc.pre_mix_out is None
        assert layer.attn._prefill_meta_shared is None and layer.attn._cp_ctx is None
    shared = model.v4.layers[0].attn._shared_attention
    assert (
        shared.get("global", {}) == {}
        and shared.get("topk", {}) == {}
        and shared.get("candidates") is None
    )
    assert model._worker_state == "ready"


def global_tokens(lengths, histories, step):
    ids = (torch.arange(sum(lengths), dtype=torch.int32) + 3 + step * 17).remainder(128)
    lookback = [list(reversed(h[-3:])) for h in histories]
    lookback = [h + [-1] * (3 - len(h)) for h in lookback]
    cu = torch.tensor([0, lengths[0], sum(lengths)], dtype=torch.int32)
    return ids, make_token_windows(ids, cu, torch.tensor(lookback)).int().contiguous()


def corrupt(request, variant, cp):
    tagged = request.attention_inputs
    first = next(iter(tagged))
    if variant == "missing-lease":
        return PyModelInputs(
            input_ids=request.input_ids,
            attention_inputs=tagged,
            engram_token_windows=request.engram_token_windows,
            v41_cache_spans=request.v41_cache_spans,
        )
    if variant == "table":
        tagged[first].kv_cache_kernel_block_id_device = torch.zeros_like(
            tagged[first].kv_cache_kernel_block_id_device
        )
    elif variant == "common":
        tagged[first].prefix_lengths = tagged[first].prefix_lengths + 1
    elif variant == "restore":
        info = tagged[first].context_parallel_info
        info.prefill_qkv_restore_indice = torch.full_like(
            info.prefill_qkv_restore_indice, -1
        )
        tagged[first].context_parallel_info = info
    elif variant == "mask":
        info = tagged[first].context_parallel_info
        info.prefill_qkv_padding_mask = torch.zeros_like(info.prefill_qkv_padding_mask)
        tagged[first].context_parallel_info = info
    elif variant == "cp-missing":
        for value in tagged.values():
            value.context_parallel_info = None
    elif variant == "windows":
        real = cp.local_is_real.to(request.engram_token_windows.device)
        request.engram_token_windows[real, 0] = -1
    elif variant == "tokens":
        request.input_ids[0] = -1
    elif variant == "position":
        for value in tagged.values():
            value.combo_position_ids = cp.global_positions.clone()
            value.combo_position_ids[cp.local_is_real] += 1
    elif variant == "spans":
        request.v41_cache_spans[0, 5] += 1
    else:
        raise AssertionError(variant)
    request.attention_inputs = tagged
    return request


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
                roots = root_allocations(native, rank)
                model = make_worker(rank, config)
                assert model.initialize(resources(cache))
                for layer in range(cache.layer_count):
                    for group in cache.get_layer_cache_groups(layer):
                        group.kv_cache_base.zero_()
                histories = [[], []]
                identities = [
                    V41CacheIdentity(request, i + 1, 1)
                    for i, request in enumerate(REQUESTS)
                ]
                calls = [0]
                gather = dist.all_gather_into_tensor

                def agreement_only(output, value, **kwargs):
                    assert kwargs["group"] is model._dp_group
                    assert value.shape == (1, 33) and output.shape == (4, 33)
                    calls[0] += 1
                    return gather(output, value, **kwargs)

                with ExitStack() as stack:
                    stack.enter_context(
                        patch.object(
                            dist, "all_gather_into_tensor", side_effect=agreement_only
                        )
                    )
                    for name in ("all_reduce", "broadcast", "all_gather"):
                        stack.enter_context(
                            patch.object(
                                dist,
                                name,
                                side_effect=AssertionError(
                                    "preflight entered a model collective"
                                ),
                            )
                        )
                    for layer in model.v4.layers:
                        stack.enter_context(
                            patch.object(
                                layer,
                                "forward",
                                side_effect=AssertionError(
                                    "preflight executed a maintained block"
                                ),
                            )
                        )
                    prefixes = [0, 0]
                    for step, lengths in enumerate(ROUNDS):
                        spans = [
                            V41L20Span(identity, step, start, count)
                            for identity, start, count in zip(
                                identities, prefixes, lengths
                            )
                        ]
                        batch = native.prepare_local(wire(spans), roots)
                        ids, windows = global_tokens(lengths, histories, step)
                        cp, indices, _ = zigzag(
                            lengths, prefixes, rank % 4, torch.device("cuda", rank)
                        )
                        request = native.split_local_inputs(batch, ids, windows)
                        assert (
                            cache.native_allocation_tables(request)
                            == batch.block_tables
                        )
                        copied = cache.native_allocation_tables(request)
                        copied[0][next(iter(copied[0]))][0] = 0
                        assert (
                            cache.native_allocation_tables(request)
                            == batch.block_tables
                        )
                        rejected(lambda: other.cache.native_allocation_tables(request))
                        raw = snapshot(cache)
                        dispatch = model.prepare_native_pp_prefill(request)
                        expected_ids = ids.index_select(0, indices.cpu()).to(
                            request.input_ids.device
                        )
                        expected_ids[~cp.local_is_real] = 0
                        expected_windows = windows.index_select(0, indices.cpu()).to(
                            request.input_ids.device
                        )
                        expected_windows[~cp.local_is_real] = -1
                        torch.testing.assert_close(
                            dispatch.input_ids, expected_ids, rtol=0, atol=0
                        )
                        torch.testing.assert_close(
                            dispatch.token_windows, expected_windows, rtol=0, atol=0
                        )
                        expected_rows = torch.stack(
                            (
                                cp.req_id_per_token.long(),
                                cp.global_positions,
                                cp.local_is_real.long(),
                            ),
                            -1,
                        ).cpu()
                        expected_rows[~expected_rows[:, 2].bool()] = torch.tensor(
                            [-1, -1, 0]
                        )
                        torch.testing.assert_close(
                            dispatch.rows, expected_rows, rtol=0, atol=0
                        )
                        assert dispatch.schedule == tuple(
                            s.wire_values() for s in spans
                        )
                        if step == 0:
                            for variant, actor in (
                                ("missing-lease", 0),
                                ("table", 4),
                                ("common", 0),
                                ("restore", 4),
                                ("mask", 0),
                                ("cp-missing", 4),
                                ("windows", 0),
                                ("tokens", 2),
                                ("position", 4),
                                ("spans", 0),
                            ):
                                negative = native.split_local_inputs(
                                    batch, ids, windows
                                )
                                if rank == actor:
                                    negative = corrupt(negative, variant, cp)
                                try:
                                    admitted = model.prepare_native_pp_prefill(negative)
                                except V41InputRejected:
                                    assert rank // 4 == actor // 4
                                else:
                                    assert rank // 4 != actor // 4
                                    del admitted
                                del negative
                                clean_worker(model)
                                require_snapshot(cache, raw)
                                retry = model.prepare_native_pp_prefill(request)
                                assert retry.schedule == dispatch.schedule
                                del retry
                            alien_roots = root_allocations(other, rank)
                            alien = other.prepare_local(wire(spans), alien_roots)
                            wrong = (
                                other.split_local_inputs(alien, ids, windows)
                                if rank == 4
                                else request
                            )
                            try:
                                admitted = model.prepare_native_pp_prefill(wrong)
                            except V41InputRejected:
                                assert rank >= 4
                            else:
                                assert rank < 4
                                del admitted
                            del wrong
                            other.abort_local(alien)
                            del alien
                            release_allocations(alien_roots)
                            model.prepare_native_pp_prefill(request)
                        # The returned descriptor owns copies of all forward inputs.
                        saved = {
                            tag: value.clone() for tag, value in dispatch.tables.items()
                        }
                        request.input_ids.zero_()
                        request.engram_token_windows.zero_()
                        for value in request.attention_inputs.values():
                            value.kv_cache_kernel_block_id_device.zero_()
                            value.context_parallel_info.prefill_qkv_restore_indice.zero_()
                        torch.testing.assert_close(
                            dispatch.input_ids, expected_ids, rtol=0, atol=0
                        )
                        torch.testing.assert_close(
                            dispatch.token_windows, expected_windows, rtol=0, atol=0
                        )
                        torch.testing.assert_close(
                            dispatch.rows, expected_rows, rtol=0, atol=0
                        )
                        torch.testing.assert_close(
                            dispatch.cp.unpad_restore, cp.unpad_restore, rtol=0, atol=0
                        )
                        for tag, value in saved.items():
                            torch.testing.assert_close(
                                dispatch.tables[tag], value, rtol=0, atol=0
                            )
                        clean_worker(model)
                        require_snapshot(cache, raw)
                        native.commit_local(batch)
                        values = ids.tolist()
                        histories[0].extend(values[: lengths[0]])
                        histories[1].extend(values[lengths[0] :])
                        prefixes = [p + n for p, n in zip(prefixes, lengths)]
                        print(
                            "DSV41_NATIVE_PP_PREFLIGHT "
                            + json.dumps(
                                {
                                    "rank": rank,
                                    "step": step,
                                    "ends": prefixes,
                                    "real_cp_rows": int(cp.local_is_real.sum()),
                                    "cpp_zigzag_exact": True,
                                    "native_tables_exact": True,
                                    "read_only": True,
                                    "snapshot_independent": True,
                                    "stage_rejection_retry": True,
                                }
                            ),
                            flush=True,
                        )
                        del request, batch
                        if step != len(ROUNDS) - 1:
                            del dispatch
                    occupied = native.free_blocks
                    torch.cuda.synchronize()
                    for identity in identities:
                        native.retire_local(list(identity.__dict__.values()))
                    release_allocations(roots)
                    assert native.free_blocks == occupied
                    del dispatch
                    gc.collect()
                    assert native.free_blocks == free_before
                model.close()
                model = None
                native.close()
                other.close()
                native = other = None
                print(
                    "DSV41_NATIVE_PP_PREFLIGHT_LIFETIME "
                    + json.dumps(
                        {
                            "rank": rank,
                            "agreement_calls": calls[0],
                            "descriptor_holds_native_leases": True,
                            "free_before": free_before,
                            "free_after": free_before,
                        }
                    ),
                    flush=True,
                )
            dist.barrier()
        except BaseException:
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


class V41NativePPPreflightSM120Test(unittest.TestCase):
    def test_native_cp_preflight_and_stage_retries(self):
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
