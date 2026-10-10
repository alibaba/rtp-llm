"""Actual PPScheduler chunks, PPExecutor slots and native V4.1 model owners."""

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
from rtp_llm.models_py.model_desc.deepseek_v4_model import Dsv4SharedRuntimeBufferStore
from rtp_llm.models_py.modules.dsv4.fp8.test.test_v41_attention_owner_sm120 import (
    NativeTaggedCache,
)
from rtp_llm.models_py.modules.dsv4.fp8.test.test_v41_cp_owner_sm120 import restore
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
    expected_storage,
    full_owner,
    oracle_forward,
    raw_snapshot,
    require,
    require_raw,
    world_check,
)
from rtp_llm.models_py.modules.dsv4.test.test_v41_native_pp_preflight_sm120 import (
    clean_worker,
)
from rtp_llm.models_py.modules.dsv4.test.test_v41_native_pp_transport_sm120 import (
    cache_config,
)
from rtp_llm.ops import NcclCommConfig


def cache_diagnostics(cache, expected, before):
    differences = []
    for key, value in raw_snapshot(cache).items():
        if torch.equal(value, expected[key]):
            continue
        pages = []
        for page in range(value.size(0)):
            offsets = (value[page] != expected[key][page]).nonzero().flatten()
            if not offsets.numel():
                continue
            sample = offsets[:12]
            pages.append(
                {
                    "page": page,
                    "mismatches": offsets.numel(),
                    "first_offsets": sample.tolist(),
                    "actual": value[page][sample].tolist(),
                    "expected": expected[key][page][sample].tolist(),
                    "actual_changes": int((value[page] != before[key][page]).sum()),
                    "expected_changes": int(
                        (expected[key][page] != before[key][page]).sum()
                    ),
                }
            )
        differences.append({"storage_layer_tag": key, "pages": pages})
    return differences


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
    stage_control = None
    for ranks in (list(range(4)), list(range(4, 8))):
        group = dist.new_group(ranks, backend="gloo", timeout=timedelta(seconds=180))
        if rank in ranks:
            stage_control = group
    native = worker = reference = None
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
                free_before = native.free_blocks
                worker = make_worker(rank, config)
                assert worker.initialize(resources(cache))
                native.attach_model(
                    worker, worker.v4.head_weight if rank >= 4 else torch.empty(0)
                )
                native.start_scheduler(43)
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
                before_rejection = raw_snapshot(cache)
                if rank != 4:
                    worker._configure_native_pp_engine(CONTRACT)
                if rank == 0:
                    rejected_ids = native.enqueue_scheduled([[7, 8, 9]], False)
                native.scheduler_tick()
                world_check(
                    lambda: require_raw(cache, before_rejection),
                    "executor rejection precedes all stores",
                    control,
                )
                if rank == 0:
                    rejected = native.scheduled_outputs()
                    assert [row["request_id"] for row in rejected] == rejected_ids
                    assert all(row["invalid_params"] for row in rejected)
                    assert all(row["outstanding"] == 0 for row in rejected)
                    assert all("worker" in row["error"] for row in rejected), rejected
                worker._configure_native_pp_engine(CONTRACT)
                for _ in range(4):
                    stats = native.scheduler_tick()
                world_check(
                    lambda: require(
                        stats["free_blocks"] == free_before,
                        "rejected scheduler pages leaked",
                    ),
                    "executor rejection capacity",
                    control,
                )
                if rank == 0:
                    pd_ids = native.enqueue_scheduled([[7, 8, 9]], False, True)
                native.scheduler_tick()
                world_check(
                    lambda: require_raw(cache, before_rejection),
                    "actual PD rejection precedes all stores",
                    control,
                )
                if rank == 0:
                    rejected = native.scheduled_outputs()
                    assert [row["request_id"] for row in rejected] == pd_ids
                    assert all(row["invalid_params"] for row in rejected)
                    assert all(row["outstanding"] == 0 for row in rejected)
                    assert all("eager text-only" in row["error"] for row in rejected)
                for _ in range(4):
                    stats = native.scheduler_tick()
                world_check(
                    lambda: require(
                        stats["free_blocks"] == free_before,
                        "rejected PD scheduler pages leaked",
                    ),
                    "PD rejection capacity",
                    control,
                )
                expected_outputs = {}
                expected_hidden_chunks = {}
                execution_records = []
                original_execute = worker._execute_native_pp_engine
                cp_group = collective._get_group(collective.Group.TP)

                def execute(dispatch, transfer_round, tensors=None):
                    lengths = [span.count for span in dispatch.spans]
                    prefixes = [span.start for span in dispatch.spans]
                    # Reclaimed physical pages are independent of logical
                    # request history. Fresh oracle requests start at zero.
                    for (_, tag), group in oracle_cache.groups.items():
                        pages = dispatch.tables.get(tag, dispatch.tables["swa_kv"])
                        for row, span in enumerate(dispatch.spans):
                            if span.start == 0:
                                page = int(pages[row, 0])
                                group.kv_cache_base[page].zero_()
                    before = raw_snapshot(cache)
                    expected, expected_payload, _ = oracle_forward(
                        oracle,
                        reference,
                        oracle_cache,
                        dispatch,
                        lengths,
                        prefixes,
                        transfer_round=transfer_round,
                    )
                    restored = restore(expected, dispatch.cp, cp_group)
                    outputs = original_execute(dispatch, transfer_round, tensors)

                    def verify():
                        clean_worker(worker)
                        expected_cache = expected_storage(
                            cache, oracle_cache, dispatch, rank // 4, before
                        )
                        differences = cache_diagnostics(cache, expected_cache, before)
                        if differences:
                            print(
                                "DSV41_NATIVE_PP_CACHE_DIAGNOSTIC "
                                + json.dumps(
                                    {
                                        "rank": rank,
                                        "round": transfer_round,
                                        "spans": [
                                            span.wire_values()
                                            for span in dispatch.spans
                                        ],
                                        "tables": {
                                            tag: table.cpu().tolist()
                                            for tag, table in dispatch.tables.items()
                                        },
                                        "real_hidden_equal": (
                                            None
                                            if rank < 4
                                            else torch.equal(
                                                outputs.hidden_states[
                                                    dispatch.cp.local_is_real
                                                ],
                                                expected[dispatch.cp.local_is_real],
                                            )
                                        ),
                                        "incoming_frame_equal": (
                                            None
                                            if tensors is None
                                            else {
                                                key: torch.equal(
                                                    tensors[key], expected_payload[key]
                                                )
                                                for key in (
                                                    "hidden_states",
                                                    "v41_row_map",
                                                    "v41_prev_pre_mix",
                                                    "v41_l20_topk",
                                                    "v41_candidate_blocks",
                                                    "v41_candidate_counts",
                                                )
                                            }
                                        ),
                                        "differences": differences,
                                    }
                                ),
                                flush=True,
                            )
                        require_raw(cache, expected_cache)
                        if rank < 4:
                            assert set(outputs.pp_intermediates) == set(
                                expected_payload
                            )
                            for key, value in expected_payload.items():
                                torch.testing.assert_close(
                                    outputs.pp_intermediates[key], value, rtol=0, atol=0
                                )
                        else:
                            for key in (
                                "hidden_states",
                                "v41_row_map",
                                "v41_prev_pre_mix",
                                "v41_l20_topk",
                                "v41_candidate_blocks",
                                "v41_candidate_counts",
                            ):
                                torch.testing.assert_close(
                                    tensors[key], expected_payload[key], rtol=0, atol=0
                                )
                            real = dispatch.cp.local_is_real
                            torch.testing.assert_close(
                                outputs.hidden_states[real],
                                expected[real],
                                rtol=0,
                                atol=0,
                            )

                    world_check(
                        verify, "executor stage arithmetic/cache", stage_control
                    )
                    if rank >= 4:
                        indexes = torch.tensor(
                            [
                                sum(lengths[: row + 1]) - 1
                                for row in range(len(lengths))
                            ],
                            device="cuda",
                            dtype=torch.long,
                        )
                        # Returning all hidden states selects a full CP exit,
                        # but these requests ask only for last-token logits.
                        # Match that head shape independently of hidden mode.
                        hidden_for_head = restored[indexes]
                        width = oracle.head_weight.size(0) // 4
                        head = oracle.head_weight.narrow(
                            0, rank % 4 * width, width
                        ).contiguous()
                        local = ops.native_lm_logits(hidden_for_head, head)
                        shards = [torch.empty_like(local) for _ in range(4)]
                        dist.all_gather(shards, local, group=cp_group)
                        logits = torch.cat(shards, 1)
                        for row, span in enumerate(dispatch.spans):
                            begin, end = sum(lengths[:row]), sum(lengths[: row + 1])
                            selected = logits[row : row + 1]
                            all_hidden = None
                            if output_modes[span.identity.request_id]:
                                chunks = expected_hidden_chunks.setdefault(
                                    span.identity.request_id, []
                                )
                                chunks.append(restored[begin:end].cpu())
                                all_hidden = torch.cat(chunks, 0)
                            expected_outputs[span.identity.request_id] = {
                                "token": int(selected.argmax(-1).item()),
                                "hidden": restored[end - 1 : end].cpu(),
                                "logits": selected.cpu(),
                                "all_hidden": all_hidden,
                            }
                    execution_records.append(
                        {
                            "rank": rank,
                            "round": transfer_round,
                            "spans": [span.wire_values() for span in dispatch.spans],
                            "exact_frame_cache_real_hidden": True,
                        }
                    )
                    print(
                        "DSV41_NATIVE_PP_EXECUTOR " + json.dumps(execution_records[-1]),
                        flush=True,
                    )
                    return outputs

                worker._execute_native_pp_engine = execute
                output_modes = {}
                prompts = [
                    [1 + index % 124 for index in range(129)],
                    list(range(2, 88)),
                ]
                if rank == 0:
                    ids = native.enqueue_scheduled(prompts, True)
                else:
                    ids = None
                ids_box = [ids]
                dist.broadcast_object_list(ids_box, src=0, group=control)
                output_modes.update({request: True for request in ids_box[0]})
                returned = set()
                expected_count = 3
                for tick in range(18):
                    stats = native.scheduler_tick()
                    if tick == 1:
                        if rank == 0:
                            late_ids = native.enqueue_scheduled(
                                [[3, 4, 5, 6, 7]], False
                            )
                        else:
                            late_ids = None
                        box = [late_ids]
                        dist.broadcast_object_list(box, src=0, group=control)
                        output_modes.update({request: False for request in box[0]})
                    records = [None] * 8
                    dist.all_gather_object(
                        records, expected_outputs if rank == 4 else {}, group=control
                    )

                    def verify_outputs():
                        if rank != 0:
                            return
                        for output in native.scheduled_outputs():
                            identity = output["request_id"]
                            assert "error" not in output, output
                            expected = records[4][identity]
                            assert output["outstanding"] == 0
                            assert output["tokens"][-1] == expected["token"]
                            torch.testing.assert_close(
                                output["hidden"], expected["hidden"], rtol=0, atol=0
                            )
                            torch.testing.assert_close(
                                output["logits"], expected["logits"], rtol=0, atol=0
                            )
                            if expected["all_hidden"] is not None:
                                torch.testing.assert_close(
                                    output["all_hidden"],
                                    expected["all_hidden"],
                                    rtol=0,
                                    atol=0,
                                )
                            returned.add(identity)

                    world_check(
                        verify_outputs, "scheduler native returned outputs", control
                    )
                    print(
                        "DSV41_NATIVE_PP_EXECUTOR_TICK "
                        + json.dumps({"rank": rank, "tick": tick, **stats}),
                        flush=True,
                    )
                world_check(
                    lambda: (
                        require(
                            rank != 0 or len(returned) == expected_count,
                            "missing scheduled output",
                        ),
                        require(
                            stats["active_requests"] == 0
                            and stats["identity_count"] == 0,
                            "terminal identity leaked",
                        ),
                        require(
                            stats["retained_slots"] == 0,
                            "drained slots retain descriptors",
                        ),
                        require(
                            native.free_blocks == free_before,
                            "scheduler pool capacity not recovered",
                        ),
                        require(
                            worker._native_pp_receiver is None
                            or not worker._native_pp_receiver._requests,
                            "receiver identity leaked",
                        ),
                    ),
                    "scheduler retirement/drain",
                    control,
                )
                # A later read-only rejection must drain older chunk results
                # before terminal retirement can reclaim their pages.
                interrupted_prompts = [
                    [1 + index % 124 for index in range(172)],
                    prompts[1],
                ]
                if rank == 0:
                    interrupted_ids = native.enqueue_scheduled(
                        interrupted_prompts, False
                    )
                else:
                    interrupted_ids = None
                box = [interrupted_ids]
                dist.broadcast_object_list(box, src=0, group=control)
                interrupted_ids = box[0]
                output_modes.update({request: False for request in interrupted_ids})
                for _ in range(2):
                    stats = native.scheduler_tick()
                before_interruption = raw_snapshot(cache)
                with patch.object(
                    worker,
                    "_native_pp_contract",
                    None if rank == 4 else worker._native_pp_contract,
                ):
                    for _ in range(6):
                        stats = native.scheduler_tick()
                        world_check(
                            lambda: require_raw(cache, before_interruption),
                            "later chunk rejection/drain is read-only",
                            control,
                        )
                expected_box = [None] * 8
                dist.all_gather_object(
                    expected_box, expected_outputs if rank == 4 else {}, group=control
                )

                def verify_interruption():
                    clean_worker(worker)
                    require_raw(cache, before_interruption)
                    require(
                        stats["active_requests"] == 0
                        and stats["identity_count"] == 0
                        and stats["retained_slots"] == 0
                        and native.free_blocks == free_before,
                        "interrupted requests retained consumers or allocations",
                    )
                    require(
                        worker._native_pp_receiver is None
                        or not worker._native_pp_receiver._requests,
                        "interrupted receiver identity leaked",
                    )
                    if rank != 0:
                        return
                    outputs = native.scheduled_outputs()
                    assert {row["request_id"] for row in outputs} == set(
                        interrupted_ids
                    )
                    for output in outputs:
                        assert output["outstanding"] == 0, output
                        if output["request_id"] == interrupted_ids[0]:
                            assert output["invalid_params"], output
                            assert "worker" in output["error"], output
                        else:
                            assert "error" not in output, output
                            expected = expected_box[4][output["request_id"]]
                            assert output["tokens"][-1] == expected["token"]
                            torch.testing.assert_close(
                                output["hidden"], expected["hidden"], rtol=0, atol=0
                            )
                            torch.testing.assert_close(
                                output["logits"], expected["logits"], rtol=0, atol=0
                            )

                world_check(
                    verify_interruption, "older results and retirement", control
                )
                if rank == 0:
                    recovery_ids = native.enqueue_scheduled([[3, 4, 5, 6, 7]], False)
                else:
                    recovery_ids = None
                box = [recovery_ids]
                dist.broadcast_object_list(box, src=0, group=control)
                recovery_ids = box[0]
                output_modes.update({request: False for request in recovery_ids})
                for _ in range(6):
                    stats = native.scheduler_tick()
                dist.all_gather_object(
                    expected_box, expected_outputs if rank == 4 else {}, group=control
                )

                def verify_recovery():
                    clean_worker(worker)
                    require(
                        stats["active_requests"] == 0
                        and stats["identity_count"] == 0
                        and stats["retained_slots"] == 0
                        and native.free_blocks == free_before,
                        "post-interruption recovery leaked consumers or allocations",
                    )
                    require(
                        worker._native_pp_receiver is None
                        or not worker._native_pp_receiver._requests,
                        "recovered receiver identity leaked",
                    )
                    if rank != 0:
                        return
                    outputs = native.scheduled_outputs()
                    assert [row["request_id"] for row in outputs] == recovery_ids
                    for output in outputs:
                        assert "error" not in output, output
                        assert output["outstanding"] == 0, output
                        expected = expected_box[4][output["request_id"]]
                        assert output["tokens"][-1] == expected["token"]
                        torch.testing.assert_close(
                            output["hidden"], expected["hidden"], rtol=0, atol=0
                        )
                        torch.testing.assert_close(
                            output["logits"], expected["logits"], rtol=0, atol=0
                        )

                world_check(verify_recovery, "post-interruption recovery", control)
                native.shutdown_scheduler()
                for _ in range(6):
                    stats = native.scheduler_tick()
                    stopped = [None] * 8
                    dist.all_gather_object(stopped, stats["shutdown"], group=control)
                    if all(stopped):
                        break
                assert all(stopped)
                print(
                    "DSV41_NATIVE_PP_EXECUTOR_LIFETIME "
                    + json.dumps(
                        {"rank": rank, "capacity_recovered": True, "shutdown": True}
                    ),
                    flush=True,
                )
                native.close()
                native = None
                worker.close()
                worker = None
                reference.close()
                reference = None
        except BaseException:
            print(f"DSV41_EXECUTOR_FAILURE rank={rank}", flush=True)
            traceback.print_exc()
            raise
        finally:
            if native is not None:
                native.wait()
                native.close()
            if worker is not None:
                worker.close()
            if reference is not None:
                reference.close()
            gc.collect()
            Dsv4SharedRuntimeBufferStore._reset_for_test()
            collective.destroy_distributed_environment()


class V41NativePPExecutorSM120Test(unittest.TestCase):
    def test_actual_scheduler_chunks_slots_rejection_retirement_and_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            torch.manual_seed(41700)
            config = checkpoint(Path(directory), full_schedule=True, hidden_size=5120)
            with socket.socket() as listener:
                listener.bind(("127.0.0.1", 0))
                port = listener.getsockname()[1]
            mp.spawn(run_rank, args=(port, config), nprocs=8, join=True)


if __name__ == "__main__":
    unittest.main()
