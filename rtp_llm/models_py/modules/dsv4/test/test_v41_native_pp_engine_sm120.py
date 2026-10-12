"""Cold CEP4PP2 NormalEngine background execution, delivery, stop and reload."""

import gc
import hashlib
import json
import os
import socket
import sys
import tempfile
import threading
import traceback
import unittest
from datetime import timedelta
from pathlib import Path
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from rtp_llm.cpp.normal_engine.test import libth_v41_native_engine_test as ops
from rtp_llm.models_py.distributed import collective_torch as collective
from rtp_llm.models_py.model_desc.deepseek_v4_model import Dsv4SharedRuntimeBufferStore
from rtp_llm.models_py.modules.dsv4.fp8.test.test_v41_attention_owner_sm120 import (
    NativeTaggedCache,
)
from rtp_llm.models_py.modules.dsv4.fp8.test.test_v41_cp_owner_sm120 import restore
from rtp_llm.models_py.modules.dsv4.fp8.test.test_v41_worker_sm120 import (
    checkpoint,
    model_config,
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
from rtp_llm.ops import NcclCommConfig
from rtp_llm.ops.compute_ops import PyModelInitResources
from rtp_llm.utils.model_weight import W


def tensor_hash(value):
    return hashlib.sha256(
        value.contiguous().view(torch.uint8).numpy().tobytes()
    ).hexdigest()


def verify_outputs(actual, expected, request_ids, prompts, full_outputs):
    assert [row["request_id"] for row in actual] == request_ids
    for row, prompt in zip(actual, prompts):
        wanted = expected[row["request_id"]]
        assert row["outstanding"] == 0, row
        assert row["tokens"] == prompt + [wanted["token"]], row
        torch.testing.assert_close(row["hidden"], wanted["hidden"], rtol=0, atol=0)
        torch.testing.assert_close(row["logits"], wanted["logits"], rtol=0, atol=0)
        if full_outputs:
            torch.testing.assert_close(
                row["all_hidden"], wanted["all_hidden"], rtol=0, atol=0
            )
        else:
            assert row["all_hidden"] is None


def run_rank(rank, port, config, worker_factory=None, head_is_sharded=False):
    torch.cuda.set_device(rank)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    os.environ.update(
        {
            "NCCL_P2P_DISABLE": "1",
            "DSV41_ENGRAM_HOST_BYTES": str(2**20),
            "DSV41_ENGRAM_CACHE_DIR": str(Path(config["checkpoint_path"]) / "cache"),
            "DSV4_PP_EP_ENABLE": "1",
            "DSV4_PP_EP_BACKEND": "fork_nccl_mxfp8",
            "DSV4_MOE_STRATEGY": "auto",
            "RTP_V41_BATCHED_PRODUCER": "0",
            "DSV4_PREFILL_FAST_PATH": "1",
            "RTP_LLM_STREAM_ASYNC": "0",
            "RTP_LLM_DEVICE_INPUT": "0",
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
    # Main-thread controls must not share the background native WORLD group.
    control = dist.new_group(backend="gloo", timeout=timedelta(seconds=180))
    stage_control = None
    for ranks in (list(range(4)), list(range(4, 8))):
        group = dist.new_group(ranks, backend="gloo", timeout=timedelta(seconds=180))
        if rank in ranks:
            stage_control = group
    new_group = dist.new_group

    def bounded_group(*args, **kwargs):
        kwargs["timeout"] = timedelta(seconds=180)
        return new_group(*args, **kwargs)

    worker = engine = None
    state = {}
    with patch.dict(sys.modules, {"librtp_compute_ops": ops}), patch.object(
        dist, "new_group", side_effect=bounded_group
    ):
        try:
            collective.init_distributed_environment(
                parallelism_for(rank), NcclCommConfig(nccl_ip="127.0.0.1"), port
            )
            cp_group = collective._get_group(collective.Group.TP)
            with torch.inference_mode():
                for epoch in range(2):
                    full_outputs = epoch == 0
                    state = {}
                    expected_outputs = {}
                    hidden_chunks = {}
                    initialization_calls = []
                    records = []
                    gate_armed = threading.Event()
                    gate_entered = threading.Event()
                    gate_release = threading.Event()
                    worker = (worker_factory or make_worker)(rank, config)
                    assert not worker._materialized and worker.v4 is None
                    initialize = worker.initialize
                    original_execute = worker._execute_native_pp_engine
                    if rank >= 4:
                        # This bounded fixture supplies a full head. The real
                        # loader already shards it and must not be sliced again.
                        head = worker.weight.global_weights[W.lm_head]
                        if head_is_sharded:
                            assert head.size(0) * 4 == worker._v4_args.vocab_size
                        else:
                            width = head.size(0) // 4
                            head = head.narrow(0, rank % 4 * width, width).contiguous()
                    else:
                        head = torch.empty(0)

                    def checked_initialize(resource):
                        assert isinstance(resource, PyModelInitResources)
                        assert resource.kv_cache is not None
                        initialization_calls.append(worker._materialized)
                        assert initialize(resource)
                        worker._configure_native_pp_engine(CONTRACT)
                        oracle, reference = full_owner(rank, config, worker)
                        state.update(
                            cache=resource.kv_cache,
                            oracle=oracle,
                            reference=reference,
                            oracle_cache=NativeTaggedCache(
                                [layer.attn for layer in oracle.layers],
                                gen=0,
                                padding=0,
                                pages=max(
                                    group.kv_cache_base.size(0)
                                    for layer in range(resource.kv_cache.layer_count)
                                    for group in resource.kv_cache.get_layer_cache_groups(
                                        layer
                                    )
                                ),
                            ),
                        )
                        for layer in range(resource.kv_cache.layer_count):
                            for group in resource.kv_cache.get_layer_cache_groups(
                                layer
                            ):
                                group.kv_cache_base.view(torch.uint8).fill_(37)
                        return True

                    def execute(dispatch, transfer_round, tensors=None):
                        try:
                            cache, oracle_cache = state["cache"], state["oracle_cache"]
                            oracle, reference = state["oracle"], state["reference"]
                            lengths = [span.count for span in dispatch.spans]
                            prefixes = [span.start for span in dispatch.spans]
                            for (_, tag), group in oracle_cache.groups.items():
                                pages = dispatch.tables.get(
                                    tag, dispatch.tables["swa_kv"]
                                )
                                for row, span in enumerate(dispatch.spans):
                                    if span.start == 0:
                                        group.kv_cache_base[int(pages[row, 0])].zero_()
                            before = raw_snapshot(cache)
                            expected, payload, _ = oracle_forward(
                                oracle,
                                reference,
                                oracle_cache,
                                dispatch,
                                lengths,
                                prefixes,
                                transfer_round=transfer_round,
                            )
                            restored = restore(expected, dispatch.cp, cp_group)
                            output = original_execute(dispatch, transfer_round, tensors)

                            def verify():
                                clean_worker(worker)
                                require_raw(
                                    cache,
                                    expected_storage(
                                        cache, oracle_cache, dispatch, rank // 4, before
                                    ),
                                )
                                if rank < 4:
                                    assert set(output.pp_intermediates) == set(payload)
                                    actual = output.pp_intermediates
                                    keys = payload
                                else:
                                    actual = tensors
                                    keys = (
                                        "hidden_states",
                                        "v41_row_map",
                                        "v41_prev_pre_mix",
                                        "v41_l20_topk",
                                        "v41_candidate_blocks",
                                        "v41_candidate_counts",
                                    )
                                    real = dispatch.cp.local_is_real
                                    torch.testing.assert_close(
                                        output.hidden_states[real],
                                        expected[real],
                                        rtol=0,
                                        atol=0,
                                    )
                                for key in keys:
                                    torch.testing.assert_close(
                                        actual[key], payload[key], rtol=0, atol=0
                                    )

                            world_check(
                                verify,
                                "cold engine exact stage execution",
                                stage_control,
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
                                width = oracle.head_weight.size(0) // 4
                                oracle_head = oracle.head_weight.narrow(
                                    0, rank % 4 * width, width
                                ).contiguous()
                                local = ops.native_lm_logits(
                                    restored[indexes], oracle_head
                                )
                                shards = [torch.empty_like(local) for _ in range(4)]
                                dist.all_gather(shards, local, group=cp_group)
                                logits = torch.cat(shards, 1)
                                if rank == 4:
                                    for row, span in enumerate(dispatch.spans):
                                        begin, end = sum(lengths[:row]), sum(
                                            lengths[: row + 1]
                                        )
                                        selected = logits[row : row + 1].cpu()
                                        complete = None
                                        if full_outputs:
                                            chunks = hidden_chunks.setdefault(
                                                span.identity.request_id, []
                                            )
                                            chunks.append(restored[begin:end].cpu())
                                            complete = torch.cat(chunks, 0)
                                        expected_outputs[span.identity.request_id] = {
                                            "token": int(selected.argmax(-1).item()),
                                            "hidden": restored[end - 1 : end].cpu(),
                                            "logits": selected,
                                            "all_hidden": complete,
                                        }
                            record = {
                                "rank": rank,
                                "epoch": epoch,
                                "round": transfer_round,
                                "spans": [
                                    span.wire_values() for span in dispatch.spans
                                ],
                                "exact_frame_cache_real_hidden": True,
                            }
                            records.append(record)
                            print(
                                "DSV41_NATIVE_PP_ENGINE " + json.dumps(record),
                                flush=True,
                            )
                            # Stage one awaits stage zero's activation while
                            # this gate is held. Only stage-zero main threads
                            # wait for it; all main controls use their own group.
                            if (
                                rank < 4
                                and gate_armed.is_set()
                                and any(span.start == 43 for span in dispatch.spans)
                            ):
                                gate_armed.clear()
                                gate_entered.set()
                                assert gate_release.wait(
                                    60
                                ), "late-arrival gate was not released"
                            return output
                        except BaseException:
                            print(
                                f"DSV41_PP_ENGINE_EXECUTION_FAILURE rank={rank}",
                                flush=True,
                            )
                            traceback.print_exc()
                            raise

                    worker.initialize = checked_initialize
                    worker._execute_native_pp_engine = execute
                    engine = ops.NativeEngine(
                        worker,
                        model_config(worker._v4_args, config),
                        worker.parallelism_config,
                        head,
                        pp_chunk_size=43,
                    )
                    assert initialization_calls == [False]
                    assert worker._materialized and not hasattr(worker, "weight")
                    table_paths = [
                        table._path for table in worker._engram_tables.values()
                    ]
                    prompts = [
                        [1 + index % 124 for index in range(129)],
                        list(range(2, 88)),
                    ]
                    late = [[13, 14, 15, 16, 17]]
                    previous = None
                    for wave in range(2):
                        gate_entered.clear()
                        gate_release.clear()
                        gate_armed.set()
                        dist.barrier(group=control)
                        request_ids = (
                            engine.enqueue(prompts, full_outputs=full_outputs)
                            if rank == 0
                            else None
                        )
                        reached = rank >= 4 or gate_entered.wait(60)
                        world_check(
                            lambda: require(
                                reached,
                                "second chunk did not reach background engine gate",
                            ),
                            "background late-arrival boundary",
                            control,
                        )
                        late_ids = (
                            engine.enqueue(late, full_outputs=full_outputs)
                            if rank == 0
                            else None
                        )
                        box = [(request_ids, late_ids)]
                        dist.broadcast_object_list(box, src=0, group=control)
                        request_ids, late_ids = box[0]
                        gate_release.set()
                        actual = engine.outputs()["outputs"] if rank == 0 else None
                        dist.barrier(group=control)
                        expected_box = [expected_outputs if rank == 4 else None]
                        dist.broadcast_object_list(expected_box, src=4, group=control)

                        def verify_delivered():
                            if rank != 0:
                                return
                            verify_outputs(
                                actual,
                                expected_box[0],
                                request_ids + late_ids,
                                prompts + late,
                                full_outputs,
                            )
                            if previous is not None:
                                for older, newer in zip(previous, actual):
                                    assert older["tokens"] == newer["tokens"]
                                    for key in ("hidden", "logits") + (
                                        ("all_hidden",) if full_outputs else ()
                                    ):
                                        torch.testing.assert_close(
                                            older[key], newer[key], rtol=0, atol=0
                                        )

                        world_check(
                            verify_delivered,
                            "cold engine returned outputs and repetition",
                            control,
                        )
                        if rank == 0:
                            previous = actual
                            print(
                                "DSV41_NATIVE_PP_ENGINE_OUTPUT "
                                + json.dumps(
                                    {
                                        "epoch": epoch,
                                        "wave": wave,
                                        "request_ids": request_ids + late_ids,
                                        "tokens": [row["tokens"][-1] for row in actual],
                                        "hidden_sha256": [
                                            tensor_hash(row["hidden"]) for row in actual
                                        ],
                                        "full_prompt": full_outputs,
                                        "exact_outputs": True,
                                    }
                                ),
                                flush=True,
                            )
                    dist.barrier(group=control)
                    stopped = engine.stop()
                    engine = None

                    def verify_stopped():
                        assert stopped["shutdown"]
                        assert stopped["active_requests"] == 0, stopped
                        assert stopped["identity_count"] == 0, stopped
                        assert stopped["retained_slots"] == 0, stopped
                        assert stopped["free_before"] == stopped["free_after"], stopped
                        assert (
                            worker._native_pp_receiver is None
                            or not worker._native_pp_receiver._requests
                        )
                        clean_worker(worker)
                        assert records and len(initialization_calls) == 1

                    world_check(
                        verify_stopped,
                        "cold engine production shutdown and capacity",
                        control,
                    )
                    del worker.initialize
                    del worker._execute_native_pp_engine
                    worker.close()
                    state["reference"].close()
                    state.clear()
                    dist.barrier(group=control)

                    def verify_closed():
                        if rank == 0:
                            assert all(not path.exists() for path in table_paths)
                            assert not list(worker._engram_cache_dir.glob("*.lease.*"))

                    world_check(
                        verify_closed,
                        "cold engine Engram cleanup before reload",
                        control,
                    )
                    print(
                        "DSV41_NATIVE_PP_ENGINE_LIFETIME "
                        + json.dumps(
                            {
                                "rank": rank,
                                "epoch": epoch,
                                "cold_initializations": len(initialization_calls),
                                "executions": len(records),
                                "shutdown_capacity_engram_cleanup": True,
                                **stopped,
                            }
                        ),
                        flush=True,
                    )
                    worker = None
                    del initialize, original_execute, head
                    gc.collect()
                    Dsv4SharedRuntimeBufferStore._reset_for_test()
        except BaseException:
            print(f"DSV41_PP_ENGINE_FAILURE rank={rank}", flush=True)
            traceback.print_exc()
            raise
        finally:
            if engine is not None:
                gate_release.set()
                engine.stop()
            if worker is not None:
                worker.close()
            if state.get("reference") is not None:
                state["reference"].close()
            state.clear()
            gc.collect()
            Dsv4SharedRuntimeBufferStore._reset_for_test()
            collective.destroy_distributed_environment()


class V41NativePPEngineSM120Test(unittest.TestCase):
    def test_cold_background_chunks_late_arrivals_outputs_stop_and_reload(self):
        with tempfile.TemporaryDirectory() as directory:
            torch.manual_seed(41700)
            config = checkpoint(Path(directory), full_schedule=True, hidden_size=5120)
            (Path(directory) / "cache").mkdir()
            with socket.socket() as listener:
                listener.bind(("127.0.0.1", 0))
                port = listener.getsockname()[1]
            mp.spawn(run_rank, args=(port, config), nprocs=8, join=True)


if __name__ == "__main__":
    unittest.main()
