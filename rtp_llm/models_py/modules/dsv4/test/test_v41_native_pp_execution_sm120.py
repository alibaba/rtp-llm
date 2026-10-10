"""Actual maintained workers, native allocations and four PP frame lanes."""

import gc
import json
import os
import socket
import sys
import tempfile
import traceback
import unittest
from dataclasses import replace
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from rtp_llm.cpp.normal_engine.test import libth_v41_native_pp_transport_test as ops
from rtp_llm.models.dsv41_kv_cache import dsv41_cache_bindings
from rtp_llm.models_py.distributed import collective_torch as collective
from rtp_llm.models_py.model_desc.deepseek_v4_model import Dsv4SharedRuntimeBufferStore
from rtp_llm.models_py.model_desc.deepseek_v41_model import DeepSeekV41Model
from rtp_llm.models_py.model_desc.v41_errors import V41InputRejected
from rtp_llm.models_py.modules.dsv4.fp8.test.test_v41_attention_owner_sm120 import (
    NativeTaggedCache,
)
from rtp_llm.models_py.modules.dsv4.fp8.test.test_v41_cp_owner_sm120 import restore
from rtp_llm.models_py.modules.dsv4.fp8.test.test_v41_worker_sm120 import (
    checkpoint,
    model_config,
    resources,
)
from rtp_llm.models_py.modules.dsv4.native_l20_transfer import (
    NativeL20ReplicaReceiver,
    export_native_l20_delta,
)
from rtp_llm.models_py.modules.dsv4.native_pp_execution import l20_tables
from rtp_llm.models_py.modules.dsv4.test.test_native_l20_transfer import CONTRACT, TEXT
from rtp_llm.models_py.modules.dsv4.test.test_v41_cp_stage_slices_sm120 import call
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
)
from rtp_llm.models_py.modules.dsv4.test.test_v41_native_pp_preflight_sm120 import (
    ROUNDS,
    clean_worker,
    global_tokens,
)
from rtp_llm.models_py.modules.dsv4.test.test_v41_native_pp_transport_sm120 import (
    REQUESTS,
    cache_config,
)
from rtp_llm.models_py.modules.dsv4.test.test_v41_stage_slices_sm120 import (
    capture_frame,
    model_for,
    wide_fixture,
)
from rtp_llm.models_py.modules.dsv4.v41_cache_transfer import (
    V41CacheIdentity,
    V41L20Span,
)
from rtp_llm.models_py.modules.dsv4.v41_pp import pack_v41_pp_frame
from rtp_llm.ops import NcclCommConfig, ParallelismConfig, RoleType
from rtp_llm.utils.model_weight import W


def raw_snapshot(cache):
    return {
        (layer, group.tag): group.kv_cache_base.view(torch.uint8).cpu().clone()
        for layer in range(cache.layer_count)
        for group in cache.get_layer_cache_groups(layer)
    }


def require_raw(cache, expected):
    for key, value in raw_snapshot(cache).items():
        torch.testing.assert_close(value, expected[key], rtol=0, atol=0)


def world_check(check, label, group=None):
    error = None
    try:
        check()
    except Exception:
        error = traceback.format_exc()
    errors = [None] * dist.get_world_size(group)
    dist.all_gather_object(errors, error, group=group)
    if any(errors):
        raise AssertionError(f"{label}: " + str(errors))


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def full_owner(rank, config, worker):
    # A separate PP1 worker supplies the same checkpoint's maintained Engram
    # tables. The arithmetic oracle owns an independent complete layer stack.
    torch.manual_seed(41701)
    args, weights = wide_fixture()
    local = ParallelismConfig()
    local.role_type = RoleType.PREFILL
    reference = DeepSeekV41Model(model_config(args, config), local, weights, None, 2)
    with patch.dict(os.environ, {"DSV4_PP_EP_ENABLE": "0", "DSV4_PP_EP_BACKEND": ""}):
        assert reference.initialize(resources())
    torch.manual_seed(41701)
    args, weights = wide_fixture()
    for layer in weights.weights:
        for projection in ("w1", "w2", "w3"):
            for kind in ("w", "s"):
                key = getattr(W, f"v4_routed_{projection}_{kind}")
                layer[key] = layer[key][rank % 4 : rank % 4 + 1].contiguous()
    args = replace(
        args,
        ep_size=4,
        ep_rank=rank % 4,
        world_size=8,
        world_rank=rank,
        moe_tp_size=4,
        moe_tp_rank=rank % 4,
        moe_cp_size=4,
        moe_cp_enabled=True,
        moe_stage_context=worker._stage_agreement,
        v41_config={**args.v41_config, **config},
    )
    model = model_for(args, weights)
    for layer, _ in reference._engram_layers:
        model.layers[layer.layer_id].engram = layer.engram
    return model, reference


def oracle_forward(oracle, reference, cache, dispatch, lengths, prefixes):
    reference._prepare_engram(
        SimpleNamespace(
            input_ids=dispatch.input_ids,
            engram_token_windows=dispatch.token_windows,
        )
    )
    for layer, _ in reference._engram_layers:
        owner = oracle.layers[layer.layer_id]
        owner.engram_hashes = layer.engram_hashes
        owner.engram_token_mask = layer.engram_token_mask
    captures = []
    handle = oracle.layers[20].register_forward_hook(
        lambda block, inputs, hidden: captures.append(
            capture_frame(block, dispatch.rows, hidden)
        )
    )
    pages = {
        tag: dispatch.tables.get(
            tag,
            torch.tensor(
                [[1], [2]], device=dispatch.input_ids.device, dtype=torch.int32
            ),
        )
        for tag in cache.layouts
    }
    try:
        with patch.dict(os.environ, {"DSV4_PREFILL_FAST_PATH": "0"}):
            hidden = call(
                oracle,
                cache,
                dispatch.input_ids,
                dispatch.cp.global_positions,
                lengths,
                prefixes,
                pages,
                dispatch.cp,
            )
        frame = captures.pop()
        frame.topk[~dispatch.cp.local_is_real] = -1
        proxy = SimpleNamespace(
            layer_count=21,
            get_layer_cache=cache.get_layer_cache,
            get_seq_size_per_block=cache.get_seq_size_per_block,
            get_kernel_seq_size_per_block=cache.get_kernel_seq_size_per_block,
        )
        delta = export_native_l20_delta(
            TEXT,
            proxy,
            CONTRACT,
            dispatch.cp.cp_rank,
            dispatch.spans[0].chunk_id,
            dispatch.spans,
            l20_tables(dispatch),
        )
        return (
            hidden,
            pack_v41_pp_frame(delta, frame, dispatch.spans, dispatch.rows),
            pages,
        )
    finally:
        handle.remove()
        reference._clear_forward_state()
        for layer in oracle.layers:
            layer.engram_hashes = None
            layer.engram_token_mask = None


def expected_storage(cache, oracle_cache, dispatch, pp_rank, before):
    result = {key: value.clone() for key, value in before.items()}
    for binding in dsv41_cache_bindings(TEXT, 2, pp_rank):
        raw = oracle_cache.get_layer_cache(
            binding.source_layer_id, binding.tag
        ).kv_cache_base
        for page in dispatch.tables[binding.tag].flatten().tolist():
            result[binding.storage_layer_id, binding.tag][page].view(-1).copy_(
                raw[page].contiguous().view(torch.uint8).flatten().cpu()
            )
    return result


def receiver_controls(worker, dispatch, receiver, payload, before, rank, step):
    for variant in ("pre-mix", "row-map", "delta", "owner", "descriptor", "round"):
        candidate, incoming, round_id = dispatch, dict(payload), step
        original_owner = receiver._native_cache
        if rank == 4:
            if variant == "pre-mix":
                incoming["v41_prev_pre_mix"] = payload["v41_prev_pre_mix"].bfloat16()
            elif variant == "row-map":
                incoming["v41_row_map"] = payload["v41_row_map"].clone()
                incoming["v41_row_map"][0, 0] = 999
            elif variant == "delta":
                incoming["v41_cache_fingerprint"] = payload[
                    "v41_cache_fingerprint"
                ].clone()
                incoming["v41_cache_fingerprint"][0] ^= 1
            elif variant == "owner":
                receiver._native_cache = None
            elif variant == "descriptor":
                candidate = replace(dispatch, worker_owner=object())
            elif variant == "round":
                round_id += 1
        try:
            try:
                worker._execute_native_pp_prefill(
                    candidate, CONTRACT, round_id, receiver=receiver, tensors=incoming
                )
            except V41InputRejected:
                pass
            else:
                raise AssertionError(
                    "malformed received frame entered maintained layers"
                )
        finally:
            receiver._native_cache = original_owner
        require_raw(worker.kv_cache, before)
        clean_worker(worker)
        assert receiver._next_round == step
        print(
            "DSV41_NATIVE_PP_EXECUTION_REJECTION "
            + json.dumps({"rank": rank, "variant": variant, "actor": 4}),
            flush=True,
        )


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

    native = worker = reference = None
    with patch.dict(sys.modules, {"librtp_compute_ops": ops}), patch.object(
        dist, "new_group", new=bounded_group
    ):
        try:
            collective.init_distributed_environment(
                parallelism_for(rank), NcclCommConfig(nccl_ip="127.0.0.1"), port
            )
            with torch.inference_mode():
                native = ops.NativePPTransport(cache_config(), parallelism_for(rank))
                cache = native.cache
                free_before = native.free_blocks
                roots = root_allocations(native, rank)
                worker = make_worker(rank, config)
                assert worker.initialize(resources(cache))
                oracle, reference = full_owner(rank, config, worker)
                oracle_cache = NativeTaggedCache(
                    [layer.attn for layer in oracle.layers],
                    gen=0,
                    padding=0,
                    pages=native.capacity,
                )
                for layer in range(cache.layer_count):
                    for group in cache.get_layer_cache_groups(layer):
                        group.kv_cache_base.view(torch.uint8).fill_(37)
                receiver = (
                    NativeL20ReplicaReceiver(TEXT, cache, CONTRACT, rank % 4)
                    if rank >= 4
                    else None
                )
                identities = [
                    V41CacheIdentity(request, row + 1, 1)
                    for row, request in enumerate(REQUESTS)
                ]
                prefixes, histories = [0, 0], [[], []]
                for step, lengths in enumerate(ROUNDS):
                    os.environ["DSV4_PREFILL_FAST_PATH"] = "0" if step == 0 else "1"
                    ids, windows = global_tokens(lengths, histories, step)
                    spans = [
                        V41L20Span(identity, step, start, count)
                        for identity, start, count in zip(identities, prefixes, lengths)
                    ]
                    source = native.global_inputs(
                        tensors_for(spans, ids, windows), roots
                    )
                    batch = native.prepare_global_worker(
                        source, worker.prepare_native_pp_prefill
                    )
                    dispatch = source.dispatch
                    if step == 0:
                        before = raw_snapshot(cache)
                        try:
                            native.begin_global_execution(batch)
                        except ops.NativeInputRejected:
                            pass
                        else:
                            raise AssertionError(
                                "execution began before tagged initialization"
                            )
                        require_raw(cache, before)
                    native.initialize_global(batch)
                    if step == 0:
                        try:
                            native.begin_global_execution(None if rank == 4 else batch)
                        except ops.NativeInputRejected:
                            pass
                        else:
                            raise AssertionError(
                                "execution began with a missing WORLD batch"
                            )
                        if receiver is not None:
                            for identity, tables in l20_tables(dispatch).items():
                                receiver.begin_request(identity, tables)
                    round_id = native.begin_global_execution(batch)
                    world_check(
                        lambda: require(round_id == step, "transfer round drift"),
                        "execution begin",
                    )
                    if step == 0:
                        try:
                            native.begin_global_execution(batch)
                        except ops.NativeInputRejected:
                            pass
                        else:
                            raise AssertionError("execution began twice")
                        try:
                            native.commit_global(batch)
                        except ValueError:
                            pass
                        else:
                            raise AssertionError(
                                "cursor committed before WORLD completion"
                            )
                    before = raw_snapshot(cache)
                    expected, expected_payload, pages = oracle_forward(
                        oracle, reference, oracle_cache, dispatch, lengths, prefixes
                    )
                    if rank < 4:
                        outputs = worker._execute_native_pp_prefill(
                            dispatch, CONTRACT, round_id
                        )
                        payload = outputs.pp_intermediates
                        native.send_local(payload, batch)
                    else:
                        payload = native.receive_local(batch)
                    native.wait()
                    if rank >= 4:
                        if step == 0:
                            receiver_controls(
                                worker, dispatch, receiver, payload, before, rank, step
                            )
                        outputs = worker._execute_native_pp_prefill(
                            dispatch,
                            CONTRACT,
                            round_id,
                            receiver=receiver,
                            tensors=payload,
                        )
                    native.wait()
                    restored = restored_expected = None
                    if rank >= 4:
                        real = dispatch.cp.local_is_real
                        different = outputs.hidden_states != expected
                        print(
                            "DSV41_NATIVE_PP_EXECUTION_ROWS "
                            + json.dumps(
                                {
                                    "rank": rank,
                                    "step": step,
                                    "real_rows": int(real.sum()),
                                    "padding_rows": int((~real).sum()),
                                    "real_mismatches": int(different[real].sum()),
                                    "padding_mismatches": int(different[~real].sum()),
                                }
                            ),
                            flush=True,
                        )
                        group = collective._get_group(collective.Group.TP)
                        restored = restore(outputs.hidden_states, dispatch.cp, group)
                        restored_expected = restore(expected, dispatch.cp, group)

                    def verify():
                        clean_worker(worker)
                        assert set(payload) == set(expected_payload)
                        for key, value in expected_payload.items():
                            torch.testing.assert_close(
                                payload[key], value, rtol=0, atol=0
                            )
                        target = (
                            expected_payload["hidden_states"] if rank < 4 else expected
                        )
                        if rank < 4:
                            torch.testing.assert_close(
                                outputs.hidden_states, target, rtol=0, atol=0
                            )
                        else:
                            torch.testing.assert_close(
                                outputs.hidden_states[real],
                                target[real],
                                rtol=0,
                                atol=0,
                            )
                            torch.testing.assert_close(
                                restored, restored_expected, rtol=0, atol=0
                            )
                        require_raw(
                            cache,
                            expected_storage(
                                cache, oracle_cache, dispatch, rank // 4, before
                            ),
                        )
                        if rank >= 4:
                            actual_logits = (
                                restored.to(worker.v4.head_weight.dtype)
                                @ worker.v4.head_weight.t()
                            )
                            expected_logits = (
                                restored_expected.to(oracle.head_weight.dtype)
                                @ oracle.head_weight.t()
                            )
                            torch.testing.assert_close(
                                actual_logits, expected_logits, rtol=0, atol=0
                            )
                        assert native.pending_tickets == 0 and not native.global_failed

                    world_check(verify, f"native PP output/cache round {step}")
                    native.wait()
                    native.complete_global_execution(batch, True)
                    native.commit_global(batch)
                    print(
                        "DSV41_NATIVE_PP_EXECUTION "
                        + json.dumps(
                            {
                                "rank": rank,
                                "step": step,
                                "layers": len(worker.v4.layers),
                                "hidden_frame_cache_exact": True,
                                "engram_owned": rank < 4,
                                "native_transport": True,
                                "round": round_id,
                            }
                        ),
                        flush=True,
                    )
                    if step < 2:
                        native.finish()
                    for row, count in enumerate(lengths):
                        histories[row].extend(
                            ids[
                                sum(lengths[:row]) : sum(lengths[:row]) + count
                            ].tolist()
                        )
                        prefixes[row] += count
                    del source, dispatch, batch, outputs, payload
                occupied = native.free_blocks
                for identity in identities:
                    native.retire_global(list(identity.__dict__.values()))
                    if receiver is not None:
                        receiver.release(identity)
                release_allocations(roots)
                gc.collect()
                world_check(
                    lambda: require(
                        native.free_blocks == occupied, "consumer lease released early"
                    ),
                    "retained consumer lifetime",
                )
                native.finish()
                world_check(
                    lambda: require(
                        native.free_blocks == free_before,
                        "allocation capacity did not recover",
                    ),
                    "consumer drain",
                )
                print(
                    "DSV41_NATIVE_PP_EXECUTION_LIFETIME "
                    + json.dumps(
                        {
                            "rank": rank,
                            "free_before": free_before,
                            "free_after": native.free_blocks,
                        }
                    ),
                    flush=True,
                )
                # A single failed stage-one completion poisons WORLD. The
                # coordinator must retain pages after every caller drops them.
                fatal_roots = (
                    [native.allocate(99201 + row, 256) for row in range(2)]
                    if rank == 0
                    else []
                )
                fatal_spans = [
                    V41L20Span(V41CacheIdentity(99201 + row, 3 + row, 1), 0, 0, 1)
                    for row in range(2)
                ]
                ids, windows = global_tokens([1, 1], [[], []], 3)
                source = native.global_inputs(
                    tensors_for(fatal_spans, ids, windows), fatal_roots
                )
                fatal = native.prepare_global_worker(
                    source, worker.prepare_native_pp_prefill
                )
                native.initialize_global(fatal)
                assert native.begin_global_execution(fatal) == 3
                occupied = native.free_blocks
                try:
                    native.complete_global_execution(fatal, rank != 4)
                except RuntimeError:
                    pass
                else:
                    raise AssertionError("failed WORLD execution committed")
                assert native.global_failed and native.active_global == 0
                release_allocations(fatal_roots)
                del source, fatal
                gc.collect()
                world_check(
                    lambda: require(
                        native.free_blocks == occupied,
                        "failed execution lost its pages",
                    ),
                    "fatal retention",
                )
                native.wait()
                native.abort_global(None)
                world_check(
                    lambda: require(
                        native.free_blocks == free_before,
                        "failed execution did not release after drain",
                    ),
                    "fatal drain",
                )
                print(
                    "DSV41_NATIVE_PP_EXECUTION_FATAL "
                    + json.dumps(
                        {
                            "rank": rank,
                            "failed_actor": 4,
                            "world_poisoned": True,
                            "free_after_drain": native.free_blocks,
                        }
                    ),
                    flush=True,
                )
                worker.close()
                worker = None
                reference.close()
                reference = None
                native.close()
                native = None
            dist.barrier()
        except BaseException:
            print(f"DSV41_EXECUTION_FAILURE rank={rank}", flush=True)
            traceback.print_exc()
            raise
        finally:
            if worker is not None:
                worker.close()
            if reference is not None:
                reference.close()
            if native is not None:
                native.wait()
                native.close()
            Dsv4SharedRuntimeBufferStore._reset_for_test()
            collective.destroy_distributed_environment()


class V41NativePPExecutionSM120Test(unittest.TestCase):
    def test_actual_workers_with_native_frames_and_world_completion(self):
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
