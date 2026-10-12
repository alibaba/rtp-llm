"""Portable rotary origin agreement across bounded DP8/EP8 workers."""

import json
import os
import socket
import tempfile
import traceback
import unittest
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from rtp_llm.models_py.model_desc.deepseek_v4_model import Dsv4SharedRuntimeBufferStore
from rtp_llm.models_py.modules.dsv4.fp8.test.test_v41_dp_worker_sm120 import (
    assert_released,
    check_round,
    idle_inputs,
    make_workers,
)
from rtp_llm.models_py.modules.dsv4.fp8.test.test_v41_worker_sm120 import (
    checkpoint,
    inputs,
    resources,
    tagged_cache,
)


def weight_snapshot(model):
    return [
        {key: value.clone() for key, value in mapping.items()}
        for mapping in (model.weight.global_weights, *model.weight.weights)
    ]


def require_weights(model, before):
    for mapping, expected in zip(
        (model.weight.global_weights, *model.weight.weights), before
    ):
        assert mapping.keys() == expected.keys()
        for key, value in expected.items():
            assert torch.equal(value.view(torch.uint8), mapping[key].view(torch.uint8))


def emit(marker, **values):
    print(marker + " " + json.dumps(values), flush=True)


@torch.inference_mode()
def run_rank(rank, port, config):
    torch.cuda.set_device(rank)
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.float32)
    os.environ.update(
        {
            "NCCL_P2P_DISABLE": "1",
            "DSV41_ENGRAM_HOST_BYTES": str(2**20),
            "DSV41_ENGRAM_CACHE_DIR": "/dev/shm",
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
    model = reference = None
    try:
        policy = "cpu" if rank == 0 else "cuda"
        model, reference = make_workers(
            rank, dict(config, rotary_compute_device=policy), initialize_model=False
        )
        before = weight_snapshot(model)
        wrapper = model.weight
        try:
            model.initialize(resources())
        except RuntimeError as error:
            assert "peer rejected request or model schedule" in str(error)
        else:
            raise AssertionError("mixed rotary origins entered materialization")
        assert model.weight is wrapper and model.v4 is None
        assert model._worker_state == "new" and not model._materialized
        assert not model._engram_tables
        require_weights(model, before)
        emit(
            "DSV41_ROTARY_DP_COLD_REJECT",
            rank=rank,
            policy=policy,
            weights_exact=True,
            tables_empty=True,
        )
        del before, wrapper
        reference.close()
        model.close()
        reference = model = None
        dist.barrier()

        model, reference = make_workers(
            rank, dict(config, rotary_compute_device="cuda"), initialize_model=False
        )
        assert model.initialize(resources())
        assert model.initialize(resources(tagged_cache(model)))
        for layer, peer in zip(model.v4.layers, reference.v4.layers):
            assert layer.attn._rotary_compute_device == "cuda"
            assert layer.attn.freqs_cis.device == torch.device("cuda", rank)
            assert layer.attn.freqs_cis is peer.attn.freqs_cis
        for phase, lengths in enumerate(
            ((3, 2) if rank == 0 else (), (rank + 1,), (2, 1) if rank % 2 else ())
        ):
            model.kv_cache, reference.kv_cache = tagged_cache(model), tagged_cache(
                reference
            )
            check_round(model, reference, rank, lengths, phase)
            emit(
                "DSV41_ROTARY_DP_ROUND",
                rank=rank,
                phase=phase,
                rows=sum(lengths),
                device=rank,
                oracle_rtol=0.03,
                oracle_atol=0.05,
            )

        before = {key: value.clone() for key, value in model.kv_cache.storage.items()}
        request = inputs(model.kv_cache) if rank == 0 else idle_inputs(model.kv_cache)
        if rank == 0:
            model._v4_args.v41_config["rotary_compute_device"] = "cpu"
        try:
            try:
                model.forward(request)
            except (ValueError, RuntimeError) as error:
                assert (
                    "rotary origin changed" in str(error)
                    if rank == 0
                    else "peer rejected" in str(error)
                )
            else:
                raise AssertionError("one-rank rotary mutation entered layers")
        finally:
            if rank == 0:
                model._v4_args.v41_config["rotary_compute_device"] = "cuda"
        assert model._worker_state == "ready"
        assert_released(model)
        for key, value in before.items():
            assert torch.equal(value, model.kv_cache.storage[key])
        emit(
            "DSV41_ROTARY_DP_MUTATION_REJECT",
            rank=rank,
            actor=0,
            cache_exact=True,
            ready=True,
        )
        model.kv_cache, reference.kv_cache = tagged_cache(model), tagged_cache(
            reference
        )
        check_round(model, reference, rank, (2,) if rank == 7 else (), 3)
        emit("DSV41_ROTARY_DP_RESUME", rank=rank, active=rank == 7, ready=True)
        del before, request
        reference.close()
        model.close()
        assert not model._engram_tables and not reference._engram_tables
        dist.barrier()
        emit("DSV41_ROTARY_DP_RELEASE", rank=rank, tables_empty=True)
    except BaseException:
        traceback.print_exc()
        raise
    finally:
        if reference is not None:
            reference.close()
        if model is not None:
            model.close()
        Dsv4SharedRuntimeBufferStore._reset_for_test()
        dist.destroy_process_group()


class RotaryDPAgreementSM120Test(unittest.TestCase):
    def test_eight_rank_cold_policy_rejection_and_ready_worker_recovery(self):
        self.assertEqual(torch.cuda.device_count(), 8)
        for rank in range(8):
            self.assertEqual(torch.cuda.get_device_capability(rank), (12, 0))
        with tempfile.TemporaryDirectory() as directory, socket.socket() as sock:
            torch.manual_seed(41900)
            config = checkpoint(Path(directory))
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
            sock.close()
            mp.spawn(run_rank, args=(port, config), nprocs=8, join=True)


if __name__ == "__main__":
    unittest.main()
