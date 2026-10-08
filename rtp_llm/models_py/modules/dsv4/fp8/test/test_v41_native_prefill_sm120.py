"""Bounded DP8 prefill through the native allocator, wrapper and executor."""

import hashlib
import importlib.metadata
import os
import socket
import tempfile
import unittest
from datetime import timedelta
from pathlib import Path
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from rtp_llm.cpp.normal_engine.test.libth_v41_native_prefill_test import NativePrefill
from rtp_llm.models_py.model_desc.deepseek_v4_model import Dsv4SharedRuntimeBufferStore
from rtp_llm.models_py.modules.dsv4.fp8.test.test_v41_dp_worker_sm120 import (
    assert_released,
    make_workers,
)
from rtp_llm.models_py.modules.dsv4.fp8.test.test_v41_worker_sm120 import (
    checkpoint,
    inputs,
    model_config,
    tagged_cache,
)


def cache_snapshot(cache):
    return {
        (layer, group.tag): group.kv_cache_base.clone()
        for layer in range(cache.layer_count)
        for group in cache.get_layer_cache_groups(layer)
    }


def run_rank(rank, port, config):
    torch.cuda.set_device(rank)
    torch.set_num_threads(1)
    expected_version = os.environ.get("DSV41_TEST_DEEPGEMM_VERSION")
    expected_hash = os.environ.get("DSV41_TEST_DEEPGEMM_SHA256")
    if expected_version or expected_hash:
        import deep_gemm

        version = importlib.metadata.version("deep-gemm")
        binary = Path(deep_gemm._C.__file__)
        digest = hashlib.sha256(binary.read_bytes()).hexdigest()
        if expected_version:
            assert version == expected_version, (version, expected_version)
        if expected_hash:
            assert digest == expected_hash, (str(binary), digest, expected_hash)
        if rank == 0:
            print(f"DeepGEMM {version}: {binary} sha256={digest}", flush=True)
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
        timeout=timedelta(seconds=180),
    )
    model = reference = native = None
    try:
        with torch.inference_mode():
            model, reference = make_workers(rank, config)
            torch.manual_seed(41950)
            head = torch.randn(128, 512, device="cuda", dtype=torch.bfloat16)
            native = NativePrefill(
                model,
                model_config(model._v4_args, config),
                model.parallelism_config,
                head,
            )
            assert type(model.kv_cache).__name__ == "KVCache"
            for phase, lengths in enumerate(
                (
                    (),
                    (3, 2) if rank == 0 else (),
                    (rank + 1,),
                    (),
                    (2, 1) if rank % 2 else (),
                    (1,),
                    (2,) if rank == 7 else (),
                )
            ):
                requests = [
                    [(rank * 7 + phase * 13 + row * 3 + i) % 128 for i in range(n)]
                    for row, n in enumerate(lengths)
                ]
                before = cache_snapshot(model.kv_cache)
                reference.kv_cache = tagged_cache(reference)
                observed = {}
                forward = model.forward

                def checked_forward(request, fmha_impl=None):
                    observed["calls"] = observed.get("calls", 0) + 1
                    assert set(request.attention_inputs) == set(
                        model.kv_cache.group_tags
                    )
                    for attn in request.attention_inputs.values():
                        assert attn.is_prefill
                        assert attn.input_lengths.tolist() == list(lengths)
                        assert attn.prefix_lengths.tolist() == [0] * len(lengths)
                        assert attn.kv_cache_kernel_block_id_device.shape[0] == len(
                            lengths
                        )
                    assert request.v41_cache_spans.shape == (len(lengths), 6)
                    ids = [token for row in requests for token in row]
                    assert request.input_ids.tolist() == ids
                    assert request.engram_token_windows.shape == (len(ids), 4)
                    if ids:
                        oracle_request = inputs(reference.kv_cache, ids, lengths)
                        torch.testing.assert_close(
                            request.engram_token_windows.cpu(),
                            oracle_request.engram_token_windows.cpu(),
                            rtol=0,
                            atol=0,
                        )
                        expected = reference.forward(
                            oracle_request
                        ).hidden_states.clone()
                    actual = forward(request, fmha_impl).hidden_states.clone()
                    if ids:
                        torch.testing.assert_close(
                            actual, expected, rtol=0.03, atol=0.05
                        )
                        last = torch.tensor(lengths, device="cuda").cumsum(0) - 1
                        observed["tokens"] = (
                            (actual[last].float() @ head.float().T).argmax(-1).tolist()
                        )
                    else:
                        assert actual.shape == (0, 512)
                        observed["tokens"] = []
                    from rtp_llm.ops.compute_ops import PyModelOutputs

                    return PyModelOutputs(actual)

                votes = []

                def reduce_any_prefill(local):
                    votes.append(local)
                    value = torch.tensor([int(local)], device="cuda", dtype=torch.int32)
                    dist.all_reduce(value, op=dist.ReduceOp.MAX)
                    return bool(value.item())

                with patch.object(model, "forward", checked_forward):
                    try:
                        result = native.run(requests, reduce_any_prefill)
                    except Exception as error:
                        print(
                            f"native prefill rank={rank} phase={phase}: {error}",
                            flush=True,
                        )
                        raise
                assert votes == [bool(lengths)]
                assert observed["calls"] == 1
                assert result["tokens"] == [
                    row + [token] for row, token in zip(requests, observed["tokens"])
                ]
                assert result["steps"] == ([1] * len(lengths) if lengths else [0])
                assert result["free_before"] == result["free_after"]
                assert_released(model)
                if not lengths:
                    after = cache_snapshot(model.kv_cache)
                    for key in before:
                        torch.testing.assert_close(
                            before[key], after[key], rtol=0, atol=0
                        )
                dist.barrier()
            if rank == 0:
                print(
                    "V4.1 native DP8 prefill: allocator, gather, wrapper, MoE and sampler PASS",
                    flush=True,
                )
    finally:
        del native
        if reference is not None:
            reference.close()
        if model is not None:
            model.close()
        dist.destroy_process_group()


class V41NativePrefillSM120Test(unittest.TestCase):
    def test_eight_rank_native_executor(self):
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
