"""Checkpoint-to-NormalEngine prefill; full NAS execution is explicitly opt-in."""

import gc
import hashlib
import json
import os
import shutil
import socket
import sys
import tempfile
import threading
import time
import traceback
import unittest
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from rtp_llm.cpp.normal_engine.test import libth_v41_native_engine_test as native_ops
from rtp_llm.models_py.distributed import collective_torch as collective
from rtp_llm.models_py.model_desc.deepseek_v41_model import DeepSeekV41Model
from rtp_llm.models_py.modules.dsv4.fp8.test.test_v41_loader_worker_sm120 import (
    load_configuration,
    make_loader,
    save_checkpoint,
)
from rtp_llm.models_py.modules.dsv4.fp8.test.test_v41_transformer_sm120 import fixture
from rtp_llm.models_py.modules.dsv4.v41_engram_host import plan_stage_engram_tables
from rtp_llm.ops.compute_ops import PyModelInitResources, PyModelOutputs
from rtp_llm.utils.model_weight import W


def run_rank(rank, port, checkpoint, directory, prompts, full):
    torch.set_num_threads(1)
    torch.cuda.set_device(rank)
    os.environ.update(
        {
            "NCCL_P2P_DISABLE": "1",
            "DSV41_ENGRAM_HOST_BYTES": str(192 * 2**30 if full else 2**20),
            "DSV41_ENGRAM_CACHE_DIR": directory,
            "DSV4_MOE_STRATEGY": "auto",
            "DSV4_MOE_CHUNK_PREFILL": "1",
            "DSV4_MOE_CHUNK_TOKENS": "32",
            "DSV4_MOE_EXTENT_COUNT_FUSE": "1",
            "FAKE_BALANCE_EXPERT": "0",
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
    control = dist.new_group(backend="gloo", timeout=timeout)
    model = engine = None
    completed = threading.Event()
    stop_requested = threading.Event()
    stopped = threading.Event()
    try:
        with torch.inference_mode():
            assert torch.cuda.get_device_capability() == (12, 0)
            if full:
                assert torch.cuda.mem_get_info()[0] > 60 * 2**30
            config, kv_config = load_configuration(Path(checkpoint))
            parallelism, loader = make_loader(config, kv_config, 8, rank)
            started = time.monotonic()
            weights = loader.load_weights(f"cuda:{rank}")
            head = weights.global_weights[W.lm_head]
            print(
                f"rank={rank} checkpoint loaded seconds={time.monotonic() - started:.2f} "
                f"allocated={torch.cuda.memory_allocated()}",
                flush=True,
            )
            model = DeepSeekV41Model(config, parallelism, weights, None, 2)
            del weights, loader
            assert not model._materialized
            with patch.dict(
                sys.modules, {"librtp_compute_ops": native_ops}
            ), patch.dict(
                collective._group_map,
                {
                    collective.Group.WORLD: dist.group.WORLD,
                    collective.Group.STAGE: dist.group.WORLD,
                },
                clear=True,
            ), patch.object(
                collective, "_parallelism_config", parallelism
            ):
                collective._register_process_groups_to_cpp()
            forward = model.forward
            initialize = model.initialize
            calls = {"initialize": 0, "active": 0, "idle": 0}
            observations = {}

            def checked_initialize(resource):
                assert isinstance(resource, PyModelInitResources)
                assert resource.kv_cache is not None and not model._materialized
                calls["initialize"] += 1
                began = time.monotonic()
                result = initialize(resource)
                print(
                    f"rank={rank} native initialization seconds={time.monotonic() - began:.2f} "
                    f"allocated={torch.cuda.memory_allocated()}",
                    flush=True,
                )
                return result

            def checked_forward(request, fmha_impl=None):
                stop = torch.tensor([int(stop_requested.is_set())], device="cuda")
                dist.all_reduce(stop, op=dist.ReduceOp.MIN)
                if stop.item():
                    engine.request_stop()
                    stopped.set()
                hidden = forward(request, fmha_impl).hidden_states
                assert hidden.shape == (request.input_ids.numel(), config.hidden_size)
                assert bool(hidden.isfinite().all())
                if request.input_ids.numel():
                    key = tuple(request.input_ids.tolist())
                    value = hidden.cpu()
                    if key in observations:
                        torch.testing.assert_close(
                            value, observations[key], rtol=0.03, atol=0.05
                        )
                    observations[key] = value
                    calls["active"] += 1
                    completed.set()
                else:
                    calls["idle"] += 1
                return PyModelOutputs(hidden)

            with patch.object(model, "initialize", checked_initialize), patch.object(
                model, "forward", checked_forward
            ):
                engine = native_ops.NativeEngine(model, config, parallelism, head)
                assert calls["initialize"] == 1 and model._materialized
                assert not hasattr(model, "weight")
                tables = list(model._engram_tables.values())
                assert len(tables) == 2
                identities = [
                    (t._path.stat().st_dev, t._path.stat().st_ino) for t in tables
                ]
                peers = [None] * 8
                dist.all_gather_object(peers, identities, group=control)
                assert all(peer == identities for peer in peers)
                for phase in range(4):
                    active = phase in (1, 2) or rank == (0 if phase == 0 else 7)
                    requests = [prompts[phase % 2]] if active else []
                    completed.clear()
                    engine.enqueue(requests)
                    if active:
                        assert completed.wait(300), "prefill forward did not complete"
                    result = engine.outputs()
                    assert result["steps"] == [1] * len(requests)
                    assert len(result["tokens"]) == len(requests)
                    for prompt, tokens in zip(requests, result["tokens"]):
                        assert tokens[:-1] == prompt and len(tokens) == len(prompt) + 1
                        assert 0 <= tokens[-1] < config.vocab_size
                        digest = hashlib.sha256(
                            observations[tuple(prompt)]
                            .view(torch.uint8)
                            .numpy()
                            .tobytes()
                        ).hexdigest()
                        print(
                            f"rank={rank} phase={phase} output_token={tokens[-1]} hidden_sha256={digest}",
                            flush=True,
                        )
                    dist.barrier(group=control)
                stop_requested.set()
                assert stopped.wait(60), "engine missed coordinated stop"
                result = engine.stop()
                engine = None
                assert result["free_before"] == result["free_after"], result
                assert result["cache_blocks"] == 8, result
            assert calls["active"] >= 2 and calls["idle"] >= 1, calls
            model.close()
            dist.barrier(group=control)
            if rank == 0:
                assert all(not table._path.exists() for table in tables)
                assert not list(Path(directory).glob("*.lease.*"))
                assert not list(Path(directory).glob("*.json"))
                print(
                    f"checkpoint-to-native-engine full_nas={full} prefill liveness PASS",
                    flush=True,
                )
    except BaseException:
        # Let mp.spawn terminate this test's peer processes. A unilateral engine
        # stop can otherwise block on peers still running NCCL scheduler rounds.
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)
    finally:
        if engine is not None:
            engine.stop()
        if model is not None:
            model.close()
        dist.destroy_process_group()


class V41CheckpointEngineSM120Test(unittest.TestCase):
    def run_checkpoint(self, checkpoint, directory, prompts, *, full):
        self.assertEqual(torch.cuda.device_count(), 8)
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        mp.spawn(
            run_rank,
            args=(port, str(checkpoint), str(directory), prompts, full),
            nprocs=8,
            join=True,
        )

    def test_bounded_checkpoint_cold_engine(self):
        torch.manual_seed(41990)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args, weights = fixture(experts=16, full_schedule=True)
            save_checkpoint(root, args, weights)
            del weights
            gc.collect()
            torch.cuda.empty_cache()
            directory = root / "cache"
            directory.mkdir()
            self.run_checkpoint(root, directory, [[3, 4, 5], [6, 7]], full=False)

    @unittest.skipUnless(
        os.environ.get("DSV41_TEST_FULL_ENGINE") == "1",
        "full checkpoint execution requires explicit host/device admission",
    )
    def test_full_nas_checkpoint_prefill_liveness(self):
        from transformers import AutoTokenizer

        root = Path(os.environ["DSV41_TEST_CHECKPOINT"]).resolve()
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
        self.assertGreater(total, 180 * 2**30)
        self.assertLessEqual(total, 192 * 2**30)
        self.assertGreater(shutil.disk_usage(storage).free, total + 64 * 2**30)
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
        self.assertTrue(all(0 < len(prompt) <= 32 for prompt in prompts))
        print(
            f"full-engine admission bytes={total} available_ram={memory['MemAvailable']} prompts={prompts}",
            flush=True,
        )
        with tempfile.TemporaryDirectory(
            prefix="v41-full-engine-", dir=storage
        ) as directory:
            self.run_checkpoint(root, directory, prompts, full=True)


if __name__ == "__main__":
    unittest.main()
