"""Maintained short sparse/frequency composition on eight changing DP lanes."""

import json
import os
import socket
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

from rtp_llm.models_py.model_desc.deepseek_v4_model import Dsv4SharedRuntimeBufferStore
from rtp_llm.models_py.modules.dsv4.fp8.test.test_v41_dp_decode_sm120 import idle_decode
from rtp_llm.models_py.modules.dsv4.fp8.test.test_v41_dp_worker_sm120 import (
    assert_released,
    check_round,
    idle_inputs,
    make_workers,
)
from rtp_llm.models_py.modules.dsv4.fp8.test.test_v41_worker_sm120 import (
    checkpoint,
    decode_inputs,
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
        policy = None if rank == 0 else "short_preserved"
        model, reference = make_workers(
            rank,
            dict(config, rotary_compute_device="cuda", sparse_attention_mode=policy),
            initialize_model=False,
        )
        before = weight_snapshot(model)
        wrapper = model.weight
        try:
            model.initialize(resources())
        except RuntimeError as error:
            assert "peer rejected request or model schedule" in str(error)
        else:
            raise AssertionError("mixed sparse modes entered materialization")
        assert model.weight is wrapper and model.v4 is None
        assert model._worker_state == "new" and not model._materialized
        assert not model._engram_tables
        require_weights(model, before)
        emit(
            "DSV41_SHORT_DP_COLD_REJECT",
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
            rank,
            dict(
                config,
                rotary_compute_device="cuda",
                sparse_attention_mode="short_preserved",
            ),
            initialize_model=False,
        )
        assert model.initialize(resources())
        assert model.initialize(resources(tagged_cache(model)))
        for layer, peer in zip(model.v4.layers, reference.v4.layers):
            assert layer.attn._rotary_compute_device == "cuda"
            assert layer.attn._sparse_attention_mode == "short_preserved"
            assert layer.attn.freqs_cis.device == torch.device("cuda", rank)
            assert layer.attn.freqs_cis is peer.attn.freqs_cis
        for phase, lengths in enumerate(
            ((3,) if rank == 0 else (), (rank + 1,), (3,) if rank % 2 else (), (3,))
        ):
            model.kv_cache, reference.kv_cache = tagged_cache(model), tagged_cache(
                reference
            )
            check_round(model, reference, rank, lengths, phase)
            emit(
                "DSV41_SHORT_DP_PREFILL",
                rank=rank,
                phase=phase,
                rows=sum(lengths),
                device=rank,
                oracle_rtol=0.03,
                oracle_atol=0.05,
            )

        start = 3
        prompt = [(3 + 3 * 11 + rank * 7 + token) % 128 for token in range(3)]
        history = prompt[::-1]
        for step in range(12):
            active = (
                step in (2, 5, 7, 8, 9, 10, 11)
                or (step == 1 and rank == 0)
                or (step == 3 and rank % 2 == 0)
                or (step == 4 and rank % 2 == 1)
                or (step == 6 and rank == 7)
            )
            before = {
                key: value.clone() for key, value in model.kv_cache.storage.items()
            }
            token = (31 + rank * 3 + step) % 128
            if active:
                request = decode_inputs(
                    model.kv_cache, [token], [start], [history], prompt_lengths=[3]
                )
                oracle = decode_inputs(
                    reference.kv_cache, [token], [start], [history], prompt_lengths=[3]
                )
                expected = reference.forward(oracle).hidden_states.clone()
            else:
                request = idle_decode(model.kv_cache)
            with ExitStack() as stack:
                if not active:
                    for layer in model.v4.layers:
                        for module, method in (
                            (layer.attn, "forward_decode"),
                            (layer.attn_hc, "pre"),
                            (layer.ffn_hc, "pre"),
                        ):
                            stack.enter_context(
                                patch.object(
                                    module,
                                    method,
                                    side_effect=AssertionError(
                                        "idle lane ran token compute"
                                    ),
                                )
                            )
                    stack.enter_context(
                        patch.object(
                            model._engram_hash_state,
                            "hash_token_windows",
                            side_effect=AssertionError("idle lane hashed history"),
                        )
                    )
                actual = model.forward(request).hidden_states.clone()
            assert actual.shape == (int(active), 512) and bool(
                torch.isfinite(actual).all()
            )
            assert model._worker_state == "ready"
            assert_released(model)
            if active:
                torch.testing.assert_close(actual, expected, rtol=0.03, atol=0.05)
                start += 1
                history = [token, *history[:2]]
            else:
                for key, value in before.items():
                    assert torch.equal(value, model.kv_cache.storage[key])
            emit(
                "DSV41_SHORT_DP_DECODE",
                rank=rank,
                step=step,
                active=active,
                start=start,
                oracle_rtol=0.03,
                oracle_atol=0.05,
                idle_cache_exact=not active,
            )
            dist.barrier()

        for variant in ("mutation", "batch", "prefix", "position"):
            before = {
                key: value.clone() for key, value in model.kv_cache.storage.items()
            }
            request = (
                idle_decode(model.kv_cache)
                if variant == "position"
                else idle_inputs(model.kv_cache)
            )
            if rank == 0:
                if variant == "position":
                    request = decode_inputs(
                        model.kv_cache, [8], [128], [[5, 4, 3]], prompt_lengths=[3]
                    )
                elif variant == "batch":
                    request = inputs(model.kv_cache)
                else:
                    request = inputs(
                        model.kv_cache,
                        ids=[3],
                        lengths=[1],
                        prefixes=[1] if variant == "prefix" else None,
                    )
                if variant == "mutation":
                    model._v4_args.v41_config["sparse_attention_mode"] = None
            try:
                try:
                    model.forward(request)
                except (ValueError, RuntimeError, NotImplementedError) as error:
                    assert (
                        ("short sparse" in str(error) or "mode changed" in str(error))
                        if rank == 0
                        else "peer rejected" in str(error)
                    )
                else:
                    raise AssertionError(
                        "one-rank short sparse rejection entered layers"
                    )
            finally:
                if rank == 0 and variant == "mutation":
                    model._v4_args.v41_config["sparse_attention_mode"] = (
                        "short_preserved"
                    )
            assert model._worker_state == "ready"
            assert_released(model)
            for key, value in before.items():
                assert torch.equal(value, model.kv_cache.storage[key])
            emit(
                "DSV41_SHORT_DP_REJECT",
                rank=rank,
                actor=0,
                variant=variant,
                cache_exact=True,
                ready=True,
            )
            dist.barrier()
        model.kv_cache, reference.kv_cache = tagged_cache(model), tagged_cache(
            reference
        )
        check_round(model, reference, rank, (2,) if rank == 7 else (), 4)
        emit("DSV41_SHORT_DP_RESUME", rank=rank, active=rank == 7, ready=True)
        del before, request
        reference.close()
        model.close()
        assert not model._engram_tables and not reference._engram_tables
        dist.barrier()
        emit("DSV41_SHORT_DP_RELEASE", rank=rank, tables_empty=True)
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


class ShortSparseNodeSM120Test(unittest.TestCase):
    def test_eight_rank_composition_prefill_decode_and_recovery(self):
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
