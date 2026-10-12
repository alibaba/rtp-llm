"""Eight-rank native V4.1 worker prefill, including idle DP participants."""

import os
import socket
import tempfile
import traceback
import unittest
from contextlib import ExitStack
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from rtp_llm.models_py.model_desc.deepseek_v4_model import Dsv4SharedRuntimeBufferStore
from rtp_llm.models_py.model_desc.deepseek_v41_model import DeepSeekV41Model
from rtp_llm.models_py.modules.dsv4.fp8.test.test_v41_transformer_sm120 import fixture
from rtp_llm.models_py.modules.dsv4.fp8.test.test_v41_worker_sm120 import (
    checkpoint,
    inputs,
    model_config,
    resources,
    tagged_cache,
)
from rtp_llm.ops import ParallelismConfig, RoleType
from rtp_llm.ops.compute_ops import PyAttentionInputs, PyModelInputs
from rtp_llm.utils.model_weight import W


def idle_inputs(cache):
    tagged = {}
    for tag in cache.group_tags:
        attn = PyAttentionInputs()
        attn.is_prefill = True
        attn.input_lengths = torch.empty(0, dtype=torch.int32)
        attn.prefix_lengths = torch.empty(0, dtype=torch.int32)
        attn.cu_seqlens = torch.zeros(1, dtype=torch.int32)
        attn.kv_cache_kernel_block_id_device = torch.empty(
            (0, 1), device="cuda", dtype=torch.int32
        )
        tagged[tag] = attn
    return PyModelInputs(
        input_ids=torch.empty(0, device="cuda", dtype=torch.int32),
        attention_inputs=tagged,
        engram_token_windows=torch.empty((0, 4), dtype=torch.int32),
    )


def assert_released(model):
    for layer in model.v4.layers:
        assert layer.engram_hashes is None
        assert layer.engram_token_mask is None
        assert layer.attn_hc.pre_mix_out is None
        assert layer.ffn_hc.pre_mix_out is None
        assert layer.attn._prefill_meta_shared is None


def make_workers(
    rank, config, gen_num_per_cycle=0, *, initialize_model=True, max_seq_len=256
):
    torch.manual_seed(41901)
    args, full = fixture(
        experts=16,
        full_schedule=config["num_hidden_layers"] == 40,
        max_seq_len=max_seq_len,
    )
    shard = SimpleNamespace(
        global_weights={
            key: value.clone() for key, value in full.global_weights.items()
        },
        weights=[
            {key: value.clone() for key, value in layer.items()}
            for layer in full.weights
        ],
    )
    for layer in shard.weights:
        for projection in ("w1", "w2", "w3"):
            for kind in ("w", "s"):
                key = getattr(W, f"v4_routed_{projection}_{kind}")
                layer[key] = layer[key][rank * 2 : (rank + 1) * 2].contiguous()
    parallelism = ParallelismConfig()
    parallelism.role_type = RoleType.PREFILL
    for field in ("world_size", "dp_size", "ep_size", "local_world_size"):
        setattr(parallelism, field, 8)
    for field in ("world_rank", "dp_rank", "ep_rank", "local_rank"):
        setattr(parallelism, field, rank)
    model = DeepSeekV41Model(
        model_config(args, config, gen_num_per_cycle), parallelism, shard, None, 2
    )
    if initialize_model:
        with patch.dict(
            os.environ, {"DSV41_ENGRAM_HOST_BYTES": "0" if rank == 0 else str(2**20)}
        ):
            try:
                model.initialize(resources())
            except (ValueError, RuntimeError):
                pass
            else:
                raise AssertionError("a peer's failed host admission was ignored")
        assert hasattr(model, "weight") and model.v4 is None
        assert not model._engram_tables
        directory = (
            Path(config["checkpoint_path"]).resolve()
            if rank == 0
            else model._engram_cache_dir
        )
        with patch.object(model, "_engram_cache_dir", directory):
            try:
                model.initialize(resources())
            except RuntimeError as error:
                assert "peer rejected" in str(error)
            else:
                raise AssertionError(
                    "mismatched Engram cache directories were admitted"
                )
        assert hasattr(model, "weight") and model.v4 is None
        assert not model._engram_tables
        model.initialize(resources())
        model.initialize(resources(tagged_cache(model)))
    local = ParallelismConfig()
    local.role_type = RoleType.PREFILL
    reference = DeepSeekV41Model(
        model_config(args, config, gen_num_per_cycle), local, full, None, 2
    )
    reference.initialize(resources())
    reference.initialize(resources(tagged_cache(reference)))
    return model, reference


def check_round(model, reference, rank, lengths, phase, *, prefix=0):
    n = sum(lengths)
    before = {key: value.clone() for key, value in model.kv_cache.storage.items()}
    if n:
        ids = [(3 + phase * 11 + rank * 7 + token) % 128 for token in range(n)]
        history = [[(phase + rank + i) % 128 for i in range(3)] for _ in lengths]
        request = inputs(model.kv_cache, ids, lengths, [prefix] * len(lengths), history)
        oracle_request = inputs(
            reference.kv_cache, ids, lengths, [prefix] * len(lengths), history
        )
        expected = reference.forward(oracle_request).hidden_states.clone()
    else:
        request = idle_inputs(model.kv_cache)
    with ExitStack() as stack:
        if not n:
            for layer in model.v4.layers:
                for module, method in (
                    (layer.attn, "forward"),
                    (layer.attn_hc, "pre"),
                    (layer.ffn_hc, "pre"),
                ):
                    stack.enter_context(
                        patch.object(
                            module,
                            method,
                            side_effect=AssertionError(
                                "idle rank ran token-local compute"
                            ),
                        )
                    )
            stack.enter_context(
                patch.object(
                    model._engram_hash_state,
                    "hash_token_windows",
                    side_effect=AssertionError("idle rank hashed history"),
                )
            )
        actual = model.forward(request).hidden_states.clone()
    assert actual.shape == (n, 512)
    assert bool(torch.isfinite(actual).all())
    assert_released(model)
    if n:
        torch.testing.assert_close(actual, expected, rtol=0.03, atol=0.05)
    else:
        for key, value in before.items():
            torch.testing.assert_close(
                value, model.kv_cache.storage[key], rtol=0, atol=0
            )
    dist.barrier()


def check_rejection(model, rank):
    for variant in ("history", "lengths", "mode", "schedule", "cache", "warmup"):
        before = {key: value.clone() for key, value in model.kv_cache.storage.items()}
        request = inputs(model.kv_cache) if rank == 0 else idle_inputs(model.kv_cache)
        with ExitStack() as stack:
            if rank == 0:
                if variant == "history":
                    request = PyModelInputs(
                        input_ids=request.input_ids,
                        attention_inputs=request.attention_inputs,
                    )
                elif variant in ("lengths", "mode"):
                    for attn in request.attention_inputs.values():
                        if variant == "lengths":
                            attn.input_lengths = torch.tensor([3, 0], dtype=torch.int32)
                        else:
                            attn.is_prefill = False
                elif variant == "schedule":
                    stack.enter_context(
                        patch.dict(os.environ, {"DSV4_MOE_CHUNK_PREFILL": "0"})
                    )
                elif variant == "cache":
                    request.attention_inputs[
                        "indexer_kv"
                    ].kv_cache_kernel_block_id_device.fill_(99)
                else:
                    stack.enter_context(patch.object(model, "kv_cache", None))
            try:
                model.forward(request)
            except (ValueError, RuntimeError, NotImplementedError):
                pass
            else:
                raise AssertionError(f"rank {rank} admitted {variant} disagreement")
        assert model._worker_state == "ready"
        assert_released(model)
        for key, value in before.items():
            torch.testing.assert_close(
                value, model.kv_cache.storage[key], rtol=0, atol=0
            )


def run_rank(rank, port, config):
    torch.cuda.set_device(rank)
    torch.set_num_threads(1)
    os.environ.update(
        {
            "NCCL_P2P_DISABLE": "1",
            "DSV41_ENGRAM_HOST_BYTES": str(2**20),
            "DSV4_MOE_STRATEGY": "auto",
            "DSV4_MOE_CHUNK_PREFILL": "1",
            "DSV4_MOE_CHUNK_TOKENS": "4",
            "DSV4_MOE_EXTENT_COUNT_FUSE": "1",
            "MOEDBG": "0",
        }
    )
    Dsv4SharedRuntimeBufferStore._reset_for_test()
    dist.init_process_group(
        "nccl",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=8,
        timeout=timedelta(seconds=120),
    )
    model = reference = None
    try:
        with torch.inference_mode():
            model, reference = make_workers(rank, config)
            table_path = model._engram_tables[1]._path
            local_inode = torch.tensor(
                [table_path.stat().st_ino], device="cuda", dtype=torch.int64
            )
            inodes = torch.empty(8, device="cuda", dtype=torch.int64)
            dist.all_gather_into_tensor(inodes, local_inode)
            assert bool((inodes == inodes[0]).all())
            for phase, lengths in enumerate(
                (
                    (),
                    (3, 2) if rank == 0 else (),
                    (rank + 1,),
                    (2, 1) if rank % 2 else (),
                    (1,),
                )
            ):
                model.kv_cache, reference.kv_cache = tagged_cache(model), tagged_cache(
                    reference
                )
                check_round(model, reference, rank, lengths, phase)
            # An idle rank must preserve earlier cache state until it resumes.
            check_round(model, reference, rank, () if rank % 2 else (2,), 5, prefix=1)
            check_round(model, reference, rank, (1,), 6, prefix=1 if rank % 2 else 3)
            check_rejection(model, rank)
            model.kv_cache, reference.kv_cache = tagged_cache(model), tagged_cache(
                reference
            )
            check_round(model, reference, rank, (2,) if rank == 7 else (), 7)
            with patch.object(
                model.v4.layers[2],
                "forward",
                side_effect=RuntimeError("injected DP block failure"),
            ):
                try:
                    model.forward(inputs(model.kv_cache))
                except RuntimeError as error:
                    assert "injected DP block failure" in str(error)
                else:
                    raise AssertionError("DP block failure was ignored")
            assert model._worker_state == "failed"
            assert_released(model)
            try:
                model.forward(inputs(model.kv_cache))
            except RuntimeError as error:
                assert "worker is failed" in str(error)
            else:
                raise AssertionError("failed DP worker was reused")
            reference.close()
            if rank:
                model.close()
            dist.barrier()
            if rank == 0:
                assert table_path.exists()
                model.close()
                assert not table_path.exists()
                print(
                    "V4.1 DP8 worker: changing idle ranks, EP1 oracle, history/cache lifecycle and peer rejection PASS",
                    flush=True,
                )
            dist.barrier()
    except BaseException:
        traceback.print_exc()
        raise
    finally:
        if reference is not None:
            reference.close()
        if model is not None:
            model.close()
        dist.destroy_process_group()


class V41DPWorkerSM120Test(unittest.TestCase):
    def test_eight_rank_prefill_workers(self):
        self.assertEqual(torch.cuda.device_count(), 8)
        with tempfile.TemporaryDirectory() as root, socket.socket() as sock:
            torch.manual_seed(41900)
            config = checkpoint(Path(root))
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
            sock.close()
            mp.spawn(run_rank, args=(port, config), nprocs=8, join=True)


if __name__ == "__main__":
    unittest.main()
