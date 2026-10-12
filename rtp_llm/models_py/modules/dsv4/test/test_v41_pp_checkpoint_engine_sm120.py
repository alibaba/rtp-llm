"""Maintained checkpoint-to-NormalEngine CEP4PP2 token and lifecycle smoke."""

import gc
import hashlib
import json
import os
import shutil
import socket
import sys
import tempfile
import traceback
import unittest
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from rtp_llm.cpp.normal_engine.test import libth_v41_native_engine_test as ops
from rtp_llm.models_py.distributed import collective_torch as collective
from rtp_llm.models_py.model_desc.deepseek_v4_model import Dsv4SharedRuntimeBufferStore
from rtp_llm.models_py.model_desc.deepseek_v41_model import DeepSeekV41Model
from rtp_llm.models_py.modules.dsv4.fp8.test.test_v41_cp_owner_sm120 import restore
from rtp_llm.models_py.modules.dsv4.fp8.test.test_v41_loader_worker_sm120 import (
    load_configuration,
    save_checkpoint,
)
from rtp_llm.models_py.modules.dsv4.test.test_v41_cp_stage_worker_sm120 import (
    parallelism_for,
)
from rtp_llm.models_py.modules.dsv4.test.test_v41_native_pp_engine_sm120 import (
    tensor_hash,
    verify_outputs,
)
from rtp_llm.models_py.modules.dsv4.test.test_v41_native_pp_execution_sm120 import (
    world_check,
)
from rtp_llm.models_py.modules.dsv4.test.test_v41_native_pp_preflight_sm120 import (
    clean_worker,
)
from rtp_llm.models_py.modules.dsv4.test.test_v41_pp_loader_sm120 import make_pp_loader
from rtp_llm.models_py.modules.dsv4.test.test_v41_stage_slices_sm120 import wide_fixture
from rtp_llm.models_py.modules.dsv4.v41_cache_transfer import V41L20Contract
from rtp_llm.models_py.modules.dsv4.v41_engram_host import plan_stage_engram_tables
from rtp_llm.ops import NcclCommConfig
from rtp_llm.ops.compute_ops import PyModelInitResources
from rtp_llm.tools.dsv41_pp_smoke import checkpoint_identities
from rtp_llm.utils.model_weight import W


def run_rank(rank, port, checkpoint, directory, prompts, full, first_ids):
    torch.cuda.set_device(rank)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    os.environ.update(
        {
            "NCCL_P2P_DISABLE": "1",
            "DSV41_ENGRAM_HOST_BYTES": str(192 * 2**30 if full else 2**20),
            "DSV41_ENGRAM_CACHE_DIR": directory,
            "DSV4_PP_EP_ENABLE": "1",
            "DSV4_PP_EP_BACKEND": "fork_nccl_mxfp8",
            "DSV4_MOE_STRATEGY": "auto",
            "RTP_V41_BATCHED_PRODUCER": "0",
            "DSV4_PREFILL_FAST_PATH": "1",
            "FAKE_BALANCE_EXPERT": "0",
            "RTP_LLM_STREAM_ASYNC": "0",
            "RTP_LLM_DEVICE_INPUT": "0",
            "MOEDBG": "0",
        }
    )
    timeout = timedelta(seconds=3600 if full else 180)
    dist.init_process_group(
        "nccl",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=8,
        timeout=timeout,
    )
    # Stage one must not enter native WORLD scheduling during stage-zero
    # Engram staging. Neither initialization nor main-thread controls may
    # compete with the native background scheduler's WORLD collectives.
    startup = dist.new_group(backend="gloo", timeout=timeout)
    control = dist.new_group(backend="gloo", timeout=timeout)
    new_group = dist.new_group

    def bounded_group(*args, **kwargs):
        kwargs["timeout"] = timeout
        return new_group(*args, **kwargs)

    worker = engine = None
    with patch.dict(sys.modules, {"librtp_compute_ops": ops}), patch.object(
        dist, "new_group", side_effect=bounded_group
    ):
        try:
            collective.init_distributed_environment(
                parallelism_for(rank), NcclCommConfig(nccl_ip="127.0.0.1"), port
            )
            cp_group = collective._get_group(collective.Group.TP)
            with torch.inference_mode():
                config, kv_config = load_configuration(Path(checkpoint))
                parallelism, hw, loader = make_pp_loader(config, kv_config, rank)
                loaded = loader.load_weights(f"cuda:{rank}")
                head = loaded.global_weights[W.lm_head] if rank >= 4 else torch.empty(0)
                if rank >= 4:
                    assert head.size(0) * 4 == config.vocab_size
                worker = DeepSeekV41Model(
                    config, parallelism, loaded, None, 2, py_hw_kernel_config=hw
                )
                del loaded, loader
                identities = checkpoint_identities(Path(checkpoint))
                contract = V41L20Contract(**identities, max_sequence_length=256)
                initialize = worker.initialize
                original_execute = worker._execute_native_pp_engine
                initialization_calls = []
                records = []
                expected_outputs = {}
                phase = [0]
                previous_stage_hashes = {}

                def checked_initialize(resource):
                    assert isinstance(resource, PyModelInitResources)
                    assert resource.kv_cache is not None and not worker._materialized
                    initialization_calls.append(worker._materialized)
                    assert initialize(resource)
                    worker._configure_native_pp_engine(contract)
                    dist.barrier(group=startup)
                    return True

                def execute(dispatch, transfer_round, tensors=None):
                    try:
                        output = original_execute(dispatch, transfer_round, tensors)
                        clean_worker(worker)
                        value = (
                            output.pp_intermediates["hidden_states"]
                            if rank < 4
                            else output.hidden_states
                        )
                        assert torch.isfinite(value).all()
                        hashes = {"hidden": tensor_hash(value.cpu())}
                        if rank < 4:
                            hashes["pre_mix"] = tensor_hash(
                                output.pp_intermediates["v41_prev_pre_mix"].cpu()
                            )
                        else:
                            # Compare native delivered results with the actual
                            # maintained stage output and native LM-head GEMM.
                            # This checks delivery/argmax, not reference math.
                            restored = restore(value, dispatch.cp, cp_group)
                            lengths = [span.count for span in dispatch.spans]
                            indexes = torch.tensor(
                                [
                                    sum(lengths[: row + 1]) - 1
                                    for row in range(len(lengths))
                                ],
                                device="cuda",
                                dtype=torch.long,
                            )
                            local = ops.native_lm_logits(restored[indexes], head)
                            shards = [torch.empty_like(local) for _ in range(4)]
                            dist.all_gather(shards, local, group=cp_group)
                            logits = torch.cat(shards, 1)
                            hashes["logits"] = tensor_hash(logits.cpu())
                            if rank == 4:
                                for row, span in enumerate(dispatch.spans):
                                    begin = sum(lengths[:row])
                                    end = begin + lengths[row]
                                    selected = logits[row : row + 1].cpu()
                                    expected_outputs[span.identity.request_id] = {
                                        "token": int(selected.argmax(-1).item()),
                                        "hidden": restored[end - 1 : end].cpu(),
                                        "logits": selected,
                                        "all_hidden": restored[begin:end].cpu(),
                                    }
                        if phase[0] == 0:
                            previous_stage_hashes.update(hashes)
                        if phase[0] == 1:
                            assert hashes == previous_stage_hashes
                        record = {
                            "rank": rank,
                            "stage": rank // 4,
                            "phase": phase[0],
                            "round": transfer_round,
                            "spans": [span.wire_values() for span in dispatch.spans],
                            "contract_fingerprint": contract.fingerprint.hex(),
                            "max_sequence_length": contract.max_sequence_length,
                            "identities": identities,
                            "hashes": hashes,
                            "full_prompt": phase[0] < 2,
                            "maintained_arithmetic": True,
                        }
                        records.append(record)
                        print(
                            "DSV41_PP_CHECKPOINT_EXECUTION " + json.dumps(record),
                            flush=True,
                        )
                        return output
                    except BaseException:
                        traceback.print_exc()
                        raise

                worker.initialize = checked_initialize
                worker._execute_native_pp_engine = execute
                engine = ops.NativeEngine(
                    worker, config, parallelism, head, pp_chunk_size=256
                )
                assert initialization_calls == [False]
                assert worker._materialized and not hasattr(worker, "weight")
                table_paths = [table._path for table in worker._engram_tables.values()]
                previous_outputs = None
                for step in range(4):
                    phase[0] = step
                    full_outputs = step < 2
                    selected_prompts = prompts if full_outputs else [prompts[step - 2]]
                    expected_outputs.clear()
                    dist.barrier(group=control)
                    ids = (
                        engine.enqueue(selected_prompts, full_outputs=full_outputs)
                        if rank == 0
                        else None
                    )
                    box = [ids]
                    dist.broadcast_object_list(box, src=0, group=control)
                    ids = box[0]
                    result = engine.outputs() if rank == 0 else None
                    dist.barrier(group=control)
                    expected_box = [expected_outputs if rank == 4 else None]
                    dist.broadcast_object_list(expected_box, src=4, group=control)

                    def verify_delivered():
                        assert len(records) == step + 1
                        assert records[-1]["phase"] == step
                        assert [span[0] for span in records[-1]["spans"]] == ids
                        if rank != 0:
                            return
                        assert result["steps"] == [1] * len(selected_prompts)
                        actual = result["outputs"]
                        verify_outputs(
                            actual, expected_box[0], ids, selected_prompts, full_outputs
                        )
                        if full:
                            wanted = (
                                first_ids if full_outputs else [first_ids[step - 2]]
                            )
                            assert [row["tokens"][-1] for row in actual] == wanted
                        if step == 1:
                            for older, newer in zip(previous_outputs, actual):
                                assert older["tokens"] == newer["tokens"]
                                for key in ("hidden", "logits", "all_hidden"):
                                    torch.testing.assert_close(
                                        older[key], newer[key], rtol=0, atol=0
                                    )

                    world_check(
                        verify_delivered, "checkpoint PP native results", control
                    )
                    if rank == 0:
                        actual = result["outputs"]
                        if step == 0:
                            previous_outputs = actual
                        print(
                            "DSV41_PP_CHECKPOINT_OUTPUT "
                            + json.dumps(
                                {
                                    "phase": step,
                                    "full_prompt": full_outputs,
                                    "full_nas": full,
                                    "request_ids": ids,
                                    "prompts": selected_prompts,
                                    "tokens": [row["tokens"][-1] for row in actual],
                                    "hidden_sha256": [
                                        tensor_hash(row["hidden"]) for row in actual
                                    ],
                                    "logits_sha256": [
                                        tensor_hash(row["logits"]) for row in actual
                                    ],
                                    "all_hidden_sha256": [
                                        (
                                            tensor_hash(row["all_hidden"])
                                            if full_outputs
                                            else None
                                        )
                                        for row in actual
                                    ],
                                    "native_argmax_exact": True,
                                    "native_delivery_exact": True,
                                }
                            ),
                            flush=True,
                        )
                dist.barrier(group=control)
                stopped = engine.stop()
                engine = None

                def verify_stopped():
                    assert stopped["shutdown"]
                    assert stopped["active_requests"] == stopped["identity_count"] == 0
                    assert stopped["retained_slots"] == 0
                    assert stopped["free_before"] == stopped["free_after"]
                    assert stopped["cache_blocks"] == 8
                    assert (
                        worker._native_pp_receiver is None
                        or not worker._native_pp_receiver._requests
                    )
                    clean_worker(worker)
                    assert len(records) == 4

                world_check(
                    verify_stopped, "checkpoint PP production shutdown", control
                )
                del worker.initialize, worker._execute_native_pp_engine
                worker.close()
                dist.barrier(group=control)

                def verify_closed():
                    assert all(not path.exists() for path in table_paths)
                    if rank == 0:
                        assert not list(Path(directory).glob("*.lease.*"))
                        assert not list(Path(directory).glob("*.json"))

                world_check(verify_closed, "checkpoint PP Engram cleanup", control)
                print(
                    "DSV41_PP_CHECKPOINT_LIFETIME "
                    + json.dumps(
                        {
                            "rank": rank,
                            "cold_initializations": len(initialization_calls),
                            "executions": len(records),
                            "receiver_empty": True,
                            "engram_cleanup": True,
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
            print(f"DSV41_PP_CHECKPOINT_FAILURE rank={rank}", flush=True)
            traceback.print_exc()
            sys.stdout.flush()
            sys.stderr.flush()
            # A unilateral stop could wait forever on peers still scheduling.
            # mp.spawn owns and terminates only this test's failed peers.
            os._exit(1)
        finally:
            if engine is not None:
                engine.stop()
            if worker is not None:
                worker.close()
            collective.destroy_distributed_environment()


class V41PPCheckpointEngineSM120Test(unittest.TestCase):
    def run_checkpoint(self, root, directory, prompts, *, full, first_ids=None):
        self.assertEqual(torch.cuda.device_count(), 8)
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            port = listener.getsockname()[1]
        mp.spawn(
            run_rank,
            args=(port, str(root), str(directory), prompts, full, first_ids),
            nprocs=8,
            join=True,
        )

    def test_bounded_named_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            torch.manual_seed(41701)
            args, weights = wide_fixture()
            save_checkpoint(root, args, weights)
            del weights
            gc.collect()
            torch.cuda.empty_cache()
            cache = root / "cache"
            cache.mkdir()
            self.run_checkpoint(
                root, cache, [[3, 4, 5, 6, 7], [8, 9, 10, 11, 12]], full=False
            )

    @unittest.skipUnless(
        os.environ.get("DSV41_RUN_FULL_PP_ENGINE") == "1",
        "full CEP4PP2 checkpoint requires explicit host/device admission",
    )
    def test_full_nas_cep4pp2_prefill(self):
        from transformers import AutoTokenizer

        root = Path(os.environ["DSV41_TEST_CHECKPOINT"]).resolve(strict=True)
        storage = Path(os.environ["DSV41_ENGRAM_CACHE_DIR"]).resolve(strict=True)
        config, _ = load_configuration(root)
        identity = hashlib.sha256(
            json.dumps(
                config.deepseek_v41_config, sort_keys=True, separators=(",", ":")
            ).encode()
        ).hexdigest()
        sources = plan_stage_engram_tables(
            SimpleNamespace(text=config.deepseek_v41_config), root, [1, 14], identity
        )
        total = sum(source.nbytes for source in sources.values())
        self.assertEqual(total, 202758032400)
        reserve_gib = int(os.environ.get("DSV41_TEST_STORAGE_RESERVE_GIB", "32"))
        self.assertGreaterEqual(reserve_gib, 32)
        free_storage = shutil.disk_usage(storage).free
        self.assertGreater(free_storage, total + reserve_gib * 2**30)
        memory = {
            key: int(value.split()[0]) * 1024
            for key, value in (
                line.split(":", 1)
                for line in Path("/proc/meminfo").read_text().splitlines()
            )
        }
        self.assertGreater(memory["MemAvailable"], 2 * total + 128 * 2**30)
        limits = dict(
            line.split()
            for line in Path("/sys/fs/cgroup/memory/memory.stat")
            .read_text()
            .splitlines()
        )
        self.assertGreaterEqual(
            int(limits["hierarchical_memory_limit"]), memory["MemTotal"]
        )
        tokenizer = AutoTokenizer.from_pretrained(root, trust_remote_code=False)
        prompts = [
            tokenizer.encode(text, add_special_tokens=False)
            for text in ("The capital of France is", "2 + 2 =")
        ]
        self.assertEqual(
            prompts, [[671, 6102, 294, 8760, 344], [20, 940, 223, 20, 438]]
        )
        print(
            "DSV41_PP_CHECKPOINT_ADMISSION "
            + json.dumps(
                {
                    "full_nas": True,
                    "engram_bytes": total,
                    "available_ram": memory["MemAvailable"],
                    "free_storage": free_storage,
                    "reserve_gib": reserve_gib,
                    "fresh_staging": True,
                    "prompts": prompts,
                }
            ),
            flush=True,
        )
        with tempfile.TemporaryDirectory(
            prefix="v41-full-pp-engine-", dir=storage
        ) as directory:
            self.run_checkpoint(
                root, directory, prompts, full=True, first_ids=[11111, 223]
            )


if __name__ == "__main__":
    unittest.main()
