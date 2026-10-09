"""Checkpoint-to-NormalEngine generation; full NAS execution is opt-in."""

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
from contextlib import nullcontext
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
from rtp_llm.models_py.modules.dsv4.fp8.test.v41_compressor_precision import (
    compressor_bf16,
)
from rtp_llm.models_py.modules.dsv4.fp8.test.v41_mhc_precision import mhc_fp32
from rtp_llm.models_py.modules.dsv4.fp8.test.v41_reference_precision import (
    reference_precision,
)
from rtp_llm.models_py.modules.dsv4.fp8.test.v41_sparse_precision import (
    sparse_sink_div_rn,
)
from rtp_llm.models_py.modules.dsv4.fp8.test.v41_tp8_projection import tp8_projection
from rtp_llm.models_py.modules.dsv4.v41_engram_host import (
    _live_leases,
    _locked,
    _PinnedSharedTable,
    plan_stage_engram_tables,
)
from rtp_llm.ops.compute_ops import PyModelInitResources, PyModelOutputs
from rtp_llm.tools.dsv41_trace import capture_model_boundaries, save_model_trace
from rtp_llm.utils.model_weight import W


def run_rank(rank, port, checkpoint, directory, prompts, full, new_tokens, context=256):
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
    control = dist.new_group(backend="gloo", timeout=timeout)
    model = engine = None
    completed = threading.Event()
    stop_requested = threading.Event()
    stopped = threading.Event()
    precision = os.environ.get("DSV41_TEST_REFERENCE_PRECISION", "baseline")
    compare_tp8 = os.environ.get("DSV41_TEST_COMPARE_TP8") == "1"
    compare_mhc = os.environ.get("DSV41_TEST_COMPARE_MHC") == "1"
    compare_compressor = os.environ.get("DSV41_TEST_COMPARE_COMPRESSOR") == "1"
    compare_recipe = os.environ.get("DSV41_TEST_COMPARE_COMPRESSOR_RECIPE") == "1"
    compare_sparse = os.environ.get("DSV41_TEST_COMPARE_SPARSE_SINK") == "1"
    paired = (
        compare_tp8
        or compare_mhc
        or compare_compressor
        or compare_recipe
        or compare_sparse
    )
    profiles = [precision, "tp8_projection"] if compare_tp8 else [precision]
    if compare_mhc:
        profiles.append("mhc_fp32")
        if compare_tp8:
            profiles.append("tp8_projection_mhc_fp32")
    base_profiles = profiles.copy()
    if compare_compressor:
        profiles += [
            "compressor_bf16" if profile == "baseline" else f"{profile}_compressor_bf16"
            for profile in base_profiles
        ]
    if compare_recipe:
        profiles += [
            (
                "compressor_recipe"
                if profile == "baseline"
                else f"{profile}_compressor_recipe"
            )
            for profile in base_profiles
        ]
    current_profile = [precision]
    if compare_sparse:
        profiles += [
            "tp8_projection_mhc_fp32_compressor_recipe_sparse_sink_div_rn_prefill",
            "tp8_projection_mhc_fp32_compressor_recipe_sparse_sink_div_rn",
        ]
    trace_steps = dict.fromkeys(profiles, 0)
    try:
        if paired and precision != "baseline":
            raise ValueError("Paired comparison requires unpatched production math")
        if compare_sparse and not (compare_tp8 and compare_mhc and compare_recipe):
            raise ValueError(
                "Sparse comparison requires TP8, mHC and compressor recipe controls"
            )
        with reference_precision(precision), torch.inference_mode():
            assert torch.cuda.get_device_capability() == (12, 0)
            if full:
                assert torch.cuda.mem_get_info()[0] > 60 * 2**30
            config, kv_config = load_configuration(
                Path(checkpoint), max_seq_len=context
            )
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
            register = _PinnedSharedTable.register_cuda
            calls = {"initialize": 0, "active": 0, "idle": 0}
            observations = {}
            sequences = {}
            hashes = {}
            outputs_by_prompt = {}

            def checked_register(table, device):
                began = time.monotonic()
                print(
                    f"rank={rank} Engram registration start key={table.source.key} "
                    f"bytes={table.source.nbytes}",
                    flush=True,
                )
                register(table, device)
                print(
                    f"rank={rank} Engram registration complete key={table.source.key} "
                    f"seconds={time.monotonic() - began:.2f}",
                    flush=True,
                )

            def checked_initialize(resource):
                assert isinstance(resource, PyModelInitResources)
                assert resource.kv_cache is not None and not model._materialized
                calls["initialize"] += 1
                began = time.monotonic()
                result = initialize(resource)
                print(
                    f"rank={rank} native initialization seconds={time.monotonic() - began:.2f} "
                    f"allocated={torch.cuda.memory_allocated()} precision={precision}",
                    flush=True,
                )
                return result

            def checked_forward(request, fmha_impl=None):
                stop = torch.tensor([int(stop_requested.is_set())], device="cuda")
                dist.all_reduce(stop, op=dist.ReduceOp.MIN)
                if stop.item():
                    engine.request_stop()
                    stopped.set()
                trace_directory = os.environ.get("DSV41_TEST_TRACE_DIR")
                profile = current_profile[0]
                recipe = "compressor_recipe" in profile
                sparse = "sparse_sink_div_rn" in profile
                sparse_decode = sparse and not profile.endswith("_prefill")
                tracing = (
                    bool(trace_directory) and rank == 0 and trace_steps[profile] < 3
                )
                with (
                    capture_model_boundaries(model.v4) if tracing else nullcontext({})
                ) as snapshots, tp8_projection(
                    profile.startswith("tp8_projection")
                ), mhc_fp32(
                    "mhc_fp32" in profile
                ), compressor_bf16(
                    profile.endswith("compressor_bf16") or recipe,
                    fused=recipe,
                    hca_projection=recipe,
                ), sparse_sink_div_rn(
                    sparse,
                    decode=sparse_decode,
                ):
                    hidden = forward(request, fmha_impl).hidden_states
                if tracing and hidden.numel():
                    snapshots["hidden"] = hidden.detach().cpu()
                    snapshots["logits_fp32"] = torch.nn.functional.linear(
                        hidden[-1:].float(), head.float()
                    ).cpu()
                    snapshots["logits_bf16_head"] = (
                        torch.nn.functional.linear(
                            hidden[-1:].bfloat16(), head.bfloat16()
                        )
                        .float()
                        .cpu()
                    )
                    trace_root = Path(trace_directory)
                    if not full:
                        trace_root /= f"bounded-context{context}-tokens{new_tokens}"
                    if paired:
                        trace_root /= profile
                    save_model_trace(
                        trace_root / f"native-prompt0-step{trace_steps[profile]}.pt",
                        snapshots,
                        {
                            "input_ids": request.input_ids.tolist(),
                            "cache_spans": request.v41_cache_spans.tolist(),
                            "precision_profile": profile,
                            "compressor_fused": recipe,
                            "hca_bf16_projection": recipe,
                            "sparse_sink_div_rn_prefill": sparse,
                            "sparse_sink_div_rn_decode": sparse_decode,
                        },
                    )
                    trace_steps[profile] += 1
                assert hidden.shape == (request.input_ids.numel(), config.hidden_size)
                assert bool(hidden.isfinite().all())
                if request.input_ids.numel():
                    attn = next(iter(request.attention_inputs.values()))
                    assert request.v41_cache_spans.shape == (1, 6)
                    request_id, _, _, _, start, count = request.v41_cache_spans[
                        0
                    ].tolist()
                    ids = request.input_ids.tolist()
                    if attn.is_prefill:
                        assert start == 0 and count == len(ids)
                        assert request_id not in sequences
                        sequences[request_id] = ids.copy()
                        hashes[request_id] = []
                        history = []
                    else:
                        history = sequences[request_id].copy()
                        assert start == len(history) and count == 1
                        assert attn.sequence_lengths.tolist() == [start]
                        sequences[request_id].extend(ids)
                    windows = []
                    for token in ids:
                        window = [token, *reversed(history[-3:])]
                        windows.append(window + [-1] * (4 - len(window)))
                        history.append(token)
                    assert request.engram_token_windows.tolist() == windows
                    key = (profile, tuple(sequences[request_id]))
                    value = hidden.cpu()
                    if key in observations:
                        torch.testing.assert_close(
                            value, observations[key], rtol=0.03, atol=0.05
                        )
                    observations[key] = value
                    hashes[request_id].append(
                        hashlib.sha256(
                            value.view(torch.uint8).numpy().tobytes()
                        ).hexdigest()
                    )
                    calls["active"] += 1
                    if len(hashes[request_id]) == new_tokens:
                        completed.set()
                else:
                    calls["idle"] += 1
                return PyModelOutputs(hidden)

            def fail_fast_forward(request, fmha_impl=None):
                try:
                    return checked_forward(request, fmha_impl)
                except BaseException:
                    # Native loop retries would hide the original test failure
                    # and leave peers waiting in the next collective round.
                    traceback.print_exc()
                    sys.stderr.flush()
                    os._exit(1)

            with patch.object(model, "initialize", checked_initialize), patch.object(
                model, "forward", fail_fast_forward
            ), patch.object(_PinnedSharedTable, "register_cuda", checked_register):
                engine = native_ops.NativeEngine(
                    model, config, parallelism, head, pdfusion=new_tokens > 1
                )
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
                for iteration in range(4 * len(profiles)):
                    phase = iteration % 4
                    profile = profiles[iteration // 4]
                    current_profile[0] = profile
                    active = phase in (1, 2) or rank == (0 if phase == 0 else 7)
                    requests = [prompts[phase % 2]] if active else []
                    completed.clear()
                    engine.enqueue(requests, new_tokens)
                    if active:
                        assert completed.wait(
                            300
                        ), "generation forwards did not complete"
                    result = engine.outputs()
                    assert result["steps"] == [new_tokens] * len(requests)
                    assert len(result["tokens"]) == len(requests)
                    for prompt, tokens in zip(requests, result["tokens"]):
                        assert tokens[: len(prompt)] == prompt
                        assert len(tokens) == len(prompt) + new_tokens
                        assert all(0 <= token < config.vocab_size for token in tokens)
                        if not full:
                            assert 127 not in tokens, "fixture sampled its image token"
                        request_id = len(sequences)
                        assert sequences[request_id] == tokens[:-1]
                        assert len(hashes[request_id]) == new_tokens
                        value = (tokens, hashes[request_id])
                        key = (profile, tuple(prompt))
                        if key in outputs_by_prompt:
                            assert value == outputs_by_prompt[key]
                        outputs_by_prompt[key] = value
                        print(
                            f"rank={rank} phase={phase} precision={profile} "
                            f"output_tokens={tokens[len(prompt):]} "
                            f"hidden_sha256={hashes[request_id]}",
                            flush=True,
                        )
                    dist.barrier(group=control)
                peers = [None] * 8
                dist.all_gather_object(peers, outputs_by_prompt, group=control)
                for profile in profiles:
                    for prompt in prompts:
                        key = (profile, tuple(prompt))
                        results = [peer[key] for peer in peers if key in peer]
                        assert len(results) == 8
                        assert all(value == results[0] for value in results)
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
                    f"checkpoint-to-native-engine full_nas={full} new_tokens={new_tokens} "
                    f"context={context} precision={profiles} generation liveness PASS",
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
    def run_checkpoint(
        self, checkpoint, directory, prompts, *, full, new_tokens=1, context=256
    ):
        self.assertEqual(torch.cuda.device_count(), 8)
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        mp.spawn(
            run_rank,
            args=(
                port,
                str(checkpoint),
                str(directory),
                prompts,
                full,
                new_tokens,
                context,
            ),
            nprocs=8,
            join=True,
        )

    def run_bounded_checkpoint(self, new_tokens, context=256):
        torch.manual_seed(41990)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args, weights = fixture(experts=16, full_schedule=True, max_seq_len=context)
            # Greedy ties choose the lower ID. Keep the synthetic image token
            # from interrupting text-only generation under diagnostic math.
            head = weights.global_weights[W.lm_head]
            head[127].copy_(head[126])
            save_checkpoint(root, args, weights)
            del weights
            gc.collect()
            torch.cuda.empty_cache()
            directory = root / "cache"
            directory.mkdir()
            prompts = (
                [
                    [(start + i) % 127 for i in range(count)]
                    for start, count in ((3, 253), (6, 503))
                ]
                if context > 256
                else [[3, 4, 5], [6, 7]]
            )
            self.run_checkpoint(
                root,
                directory,
                prompts,
                full=False,
                new_tokens=new_tokens,
                context=context,
            )

    def test_bounded_checkpoint_cold_engine(self):
        self.run_bounded_checkpoint(1)

    def test_bounded_checkpoint_cold_decode(self):
        self.run_bounded_checkpoint(9)

    def test_two_page_bounded_checkpoint_cold_decode(self):
        self.run_bounded_checkpoint(9, context=512)

    @unittest.skipUnless(
        os.environ.get("DSV41_TEST_FULL_ENGINE") == "1",
        "full checkpoint execution requires explicit host/device admission",
    )
    def test_full_nas_checkpoint_prefill_liveness(self):
        self.run_full_checkpoint(1)

    @unittest.skipUnless(
        os.environ.get("DSV41_TEST_FULL_ENGINE") == "1",
        "full checkpoint execution requires explicit host/device admission",
    )
    def test_full_nas_checkpoint_decode_liveness(self):
        self.run_full_checkpoint(9)

    @unittest.skipUnless(
        os.environ.get("DSV41_TEST_FULL_ENGINE") == "1"
        and os.environ.get("DSV41_TEST_TRACE_DIR"),
        "full checkpoint trace requires explicit admission and a trace directory",
    )
    def test_full_nas_checkpoint_accuracy_trace(self):
        self.run_full_checkpoint(3)

    def run_full_checkpoint(self, new_tokens):
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
        reuse = os.environ.get("DSV41_TEST_REUSE_ENGRAM_DIR")
        if reuse:
            reused_directory = Path(reuse).resolve(strict=True)
            self.assertEqual(reused_directory.parent, storage)
            self.assertTrue(reused_directory.name.startswith("v41-full-engine-"))
            for source in sources.values():
                source.check_sources()
                path = reused_directory / f"rtp_llm_engram_{os.getuid()}_{source.key}"
                with _locked(Path(str(path) + ".lock")):
                    self.assertFalse(
                        _live_leases(path), "reused tables have active owners"
                    )
                    self.assertFalse(path.is_symlink())
                    self.assertEqual(path.stat().st_size, source.nbytes)
                    self.assertEqual(
                        Path(str(path) + ".json").read_bytes(), source.manifest
                    )
            directory_context = nullcontext(str(reused_directory))
        else:
            directory_context = tempfile.TemporaryDirectory(
                prefix="v41-full-engine-", dir=storage
            )
        reserve_gib = int(os.environ.get("DSV41_TEST_STORAGE_RESERVE_GIB", "64"))
        self.assertGreaterEqual(reserve_gib, 32)
        free_storage = shutil.disk_usage(storage).free
        self.assertGreater(free_storage, (0 if reuse else total) + reserve_gib * 2**30)
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
            f"full-engine admission bytes={total} available_ram={memory['MemAvailable']} "
            f"free_storage={free_storage} reserve_gib={reserve_gib} "
            f"reuse_tables={bool(reuse)} prompts={prompts}",
            flush=True,
        )
        with directory_context as directory:
            self.run_checkpoint(
                root, directory, prompts, full=True, new_tokens=new_tokens
            )


if __name__ == "__main__":
    unittest.main()
