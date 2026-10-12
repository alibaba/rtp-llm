"""Eager DP8/EP8 worker decode after real prefill, with bounded weights."""

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
from rtp_llm.models_py.modules.dsv4.fp8.test.test_v41_dp_worker_sm120 import (
    assert_released,
    idle_inputs,
    make_workers,
)
from rtp_llm.models_py.modules.dsv4.fp8.test.test_v41_worker_sm120 import (
    checkpoint,
    context_tables,
    decode_inputs,
    inputs,
    tagged_cache,
)


def idle_decode(cache):
    request = idle_inputs(cache)
    for attn in request.attention_inputs.values():
        attn.is_prefill = False
        attn.sequence_lengths = torch.empty(0, dtype=torch.int32)
    return request


def check_rejections(model, rank, starts, history, tables=None):
    for variant in ("phase", "positions", "history", "cache", "later_page"):
        saved = {key: value.clone() for key, value in model.kv_cache.storage.items()}
        request = idle_decode(model.kv_cache)
        if rank == 0:
            request = decode_inputs(
                model.kv_cache, [8, 9], starts, history, tables=tables
            )
            if variant == "positions":
                for attn in request.attention_inputs.values():
                    attn.sequence_lengths = torch.tensor(
                        [model._v4_args.max_seq_len, starts[1]], dtype=torch.int32
                    )
            elif variant == "history":
                request.engram_token_windows.fill_(128)
            elif variant == "cache":
                request.attention_inputs[
                    "indexer_kv"
                ].kv_cache_kernel_block_id_device.fill_(99)
            elif variant == "later_page":
                request.attention_inputs["indexer_kv"].kv_cache_kernel_block_id_device[
                    1, -1
                ] = 0
        elif variant == "phase":
            request = idle_inputs(model.kv_cache)
        try:
            model.forward(request)
        except (ValueError, RuntimeError, NotImplementedError):
            pass
        else:
            raise AssertionError(f"rank {rank} admitted {variant} disagreement")
        assert model._worker_state == "ready"
        assert_released(model)
        for key, value in saved.items():
            torch.testing.assert_close(
                value, model.kv_cache.storage[key], rtol=0, atol=0
            )
    dist.barrier()


def run_rank(rank, port, config, context=256):
    torch.cuda.set_device(rank)
    torch.set_num_threads(1)
    os.environ.update(
        {
            "NCCL_P2P_DISABLE": "1",
            "DSV41_ENGRAM_HOST_BYTES": str(2**20),
            "DSV4_MOE_STRATEGY": "auto",
            "DSV4_MOE_CHUNK_PREFILL": "1",
            "DSV4_MOE_CHUNK_TOKENS": "32" if context > 256 else "1",
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
        with torch.inference_mode():
            model, reference = make_workers(rank, config, max_seq_len=context)
            paths = [table._path for table in model._engram_tables.values()]
            tables = None
            if context > 256:
                model.kv_cache = tagged_cache(model, pages=8)
                reference.kv_cache = tagged_cache(reference, pages=8)
                tables = context_tables(model.kv_cache)
            starts = [253, 502] if context > 256 else [3, 5]
            prompt_lengths = starts.copy()
            prompt = [(3 + rank * 7 + token) % 128 for token in range(sum(starts))]
            actual = model.forward(
                inputs(model.kv_cache, prompt, starts, tables=tables)
            ).hidden_states
            expected = reference.forward(
                inputs(reference.kv_cache, prompt, starts, tables=tables)
            ).hidden_states
            torch.testing.assert_close(actual, expected, rtol=0.03, atol=0.05)
            history = [
                list(reversed(prompt[starts[0] - 3 : starts[0]])),
                list(reversed(prompt[-3:])),
            ]
            check_rejections(model, rank, starts, history, tables)
            for step in range(12):
                active = (
                    step in (2, 5, 7, 8, 9, 10, 11)
                    or (step == 1 and rank == 0)
                    or (step == 3 and rank % 2 == 0)
                    or (step == 4 and rank % 2 == 1)
                    or (step == 6 and rank == 7)
                )
                count = (1 if rank == 7 and step % 2 else 2) if active else 0
                saved = {
                    key: value.clone() for key, value in model.kv_cache.storage.items()
                }
                ids = [
                    (31 + rank * 3 + step + request) % 128 for request in range(count)
                ]
                order = list(reversed(range(count))) if step % 2 else list(range(count))
                if active:
                    request = decode_inputs(
                        model.kv_cache,
                        ids,
                        starts[:count],
                        history[:count],
                        order,
                        prompt_lengths=prompt_lengths[:count],
                        tables=tables,
                    )
                    oracle_request = decode_inputs(
                        reference.kv_cache,
                        ids,
                        starts[:count],
                        history[:count],
                        order,
                        prompt_lengths=prompt_lengths[:count],
                        tables=tables,
                    )
                    expected = reference.forward(oracle_request).hidden_states.clone()
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
                                            "idle rank ran token-local decode"
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
                assert actual.shape == (count, 512) and bool(
                    torch.isfinite(actual).all()
                )
                assert_released(model)
                if active:
                    torch.testing.assert_close(actual, expected, rtol=0.03, atol=0.05)
                    for request_id in range(count):
                        starts[request_id] += 1
                        history[request_id] = [
                            ids[request_id],
                            *history[request_id][:2],
                        ]
                else:
                    for key, value in saved.items():
                        torch.testing.assert_close(
                            value, model.kv_cache.storage[key], rtol=0, atol=0
                        )
                dist.barrier()
            reference.close()
            if rank:
                model.close()
            dist.barrier()
            if rank == 0:
                assert all(path.exists() for path in paths)
                model.close()
                assert all(not path.exists() for path in paths)
                print(
                    f"V4.1 DP8 decode layers={config['num_hidden_layers']} context={context}: prefill, eight continuations, changing idle ranks, EP1 parity and teardown PASS",
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


class V41DPDecodeSM120Test(unittest.TestCase):
    def run_case(self, full_schedule, context=256):
        self.assertEqual(torch.cuda.device_count(), 8)
        with tempfile.TemporaryDirectory() as root, socket.socket() as sock:
            torch.manual_seed(41900)
            config = checkpoint(Path(root), full_schedule=full_schedule)
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
            sock.close()
            mp.spawn(run_rank, args=(port, config, context), nprocs=8, join=True)

    def test_six_layer_decode(self):
        self.run_case(False)

    def test_complete_layer_schedule_decode(self):
        self.run_case(True)

    def test_two_page_complete_layer_schedule_decode(self):
        self.run_case(True, context=512)


if __name__ == "__main__":
    unittest.main()
