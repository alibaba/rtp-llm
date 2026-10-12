"""Bounded native DP8 prefill, repeated decode, real LM head and sampling."""

import os
import socket
import tempfile
import traceback
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
from rtp_llm.models_py.modules.dsv4.fp8.test.test_v41_native_prefill_sm120 import (
    cache_snapshot,
)
from rtp_llm.models_py.modules.dsv4.fp8.test.test_v41_worker_sm120 import (
    checkpoint,
    context_tables,
    decode_inputs,
    inputs,
    model_config,
    tagged_cache,
)
from rtp_llm.ops.compute_ops import PyModelOutputs


def run_rank(rank, port, config, context=256):
    torch.cuda.set_device(rank)
    torch.set_num_threads(1)
    os.environ.update(
        {
            "NCCL_P2P_DISABLE": "1",
            "DSV41_ENGRAM_HOST_BYTES": str(2**20),
            "DSV4_MOE_STRATEGY": "auto",
            "DSV4_MOE_CHUNK_PREFILL": "1",
            "DSV4_MOE_CHUNK_TOKENS": "32" if context > 256 else "4",
            "DSV4_MOE_EXTENT_COUNT_FUSE": "1",
            "RTP_LLM_STREAM_ASYNC": "0",
            "RTP_LLM_DEVICE_INPUT": "0",
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
            model, reference = make_workers(rank, config, max_seq_len=context)
            torch.manual_seed(41950)
            head = torch.randn(128, 512, device="cuda", dtype=torch.bfloat16)
            native = NativePrefill(
                model,
                model_config(model._v4_args, config),
                model.parallelism_config,
                head,
            )
            assert type(model.kv_cache).__name__ == "KVCache"
            phases = (
                (
                    (253, 503) if rank == 0 else (),
                    (254 + rank % 2, 500) if rank % 2 else (254,),
                    (503,) if rank == 7 else (),
                )
                if context > 256
                else (
                    (3, 5) if rank == 0 else (),
                    (3 + rank, 125) if rank % 2 else (3 + rank,),
                    (5,) if rank == 7 else (),
                )
            )
            for phase, lengths in enumerate(phases):
                sequences = [
                    [(rank * 7 + phase * 13 + row * 3 + i) % 128 for i in range(n)]
                    for row, n in enumerate(lengths)
                ]
                requests = [row.copy() for row in sequences]
                before = cache_snapshot(model.kv_cache)
                reference.kv_cache = tagged_cache(
                    reference, pages=8 if context > 256 else 4
                )
                tables = context_tables(reference.kv_cache) if context > 256 else None
                previous_pages = {}
                saw_growth = False
                step = 0
                forward = model.forward

                def checked_forward(request, fmha_impl=None):
                    nonlocal step, saw_growth
                    expected_ids = (
                        [token for row in sequences for token in row]
                        if step == 0
                        else [row[-1] for row in sequences]
                    )
                    assert request.input_ids.tolist() == expected_ids
                    assert request.v41_cache_spans.shape == (len(lengths), 6)
                    for tag, attn in request.attention_inputs.items():
                        assert attn.is_prefill == (step == 0)
                        assert attn.input_lengths.tolist() == list(lengths)
                        assert attn.sequence_lengths.tolist() == (
                            [] if step == 0 else [len(row) - 1 for row in sequences]
                        )
                        assert attn.kv_cache_kernel_block_id_device.shape[0] == len(
                            lengths
                        )
                        pages = attn.kv_cache_kernel_block_id_device.tolist()
                        retained = tag in ("swa_kv", "csa_state")
                        for row, sequence in enumerate(sequences):
                            needed = 1 if retained else (len(sequence) + 255) // 256
                            current = pages[row][:needed]
                            assert len(current) == needed and all(
                                page > 0 for page in current
                            )
                            assert len(set(current)) == needed
                            old = previous_pages.get((tag, row), [])
                            assert current[: len(old)] == old
                            if old and len(current) > len(old):
                                assert not retained
                                saw_growth = True
                            previous_pages[tag, row] = current
                            if retained:
                                assert len(pages[row]) == 1
                    if lengths:
                        if step == 0:
                            oracle_request = inputs(
                                reference.kv_cache, expected_ids, lengths, tables=tables
                            )
                        else:
                            history = [
                                list(reversed(row[max(0, len(row) - 4) : -1]))
                                for row in sequences
                            ]
                            history = [row + [-1] * (3 - len(row)) for row in history]
                            oracle_request = decode_inputs(
                                reference.kv_cache,
                                expected_ids,
                                [len(row) - 1 for row in sequences],
                                history,
                                prompt_lengths=lengths,
                                tables=tables,
                            )
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
                    if lengths:
                        torch.testing.assert_close(
                            actual, expected, rtol=0.03, atol=0.05
                        )
                        selected = (
                            torch.tensor(lengths, device="cuda").cumsum(0) - 1
                            if step == 0
                            else torch.arange(len(lengths), device="cuda")
                        )
                        samples = (
                            (actual[selected].float() @ head.float().T)
                            .argmax(-1)
                            .tolist()
                        )
                        for row, token in zip(sequences, samples):
                            row.append(token)
                    else:
                        assert actual.shape == (0, 512)
                    step += 1
                    return PyModelOutputs(actual)

                with patch.object(model, "forward", checked_forward):
                    result = native.run_decode(requests, 9)
                assert step == 9
                assert result["tokens"] == sequences
                assert result["free_before"] == result["free_after"]
                if context > 256 and lengths and lengths[0] < 256:
                    assert (
                        saw_growth
                    ), "native GLOBAL/index pages never grew across token 256"
                assert_released(model)
                if not lengths:
                    after = cache_snapshot(model.kv_cache)
                    for key, value in before.items():
                        torch.testing.assert_close(value, after[key], rtol=0, atol=0)
                dist.barrier()
            if rank == 0:
                print(
                    f"V4.1 native DP8 full-schedule decode context={context}: real prefill, eight decode steps, Engram history, stable page growth, EP1 parity, sampling, idle phase and reclamation PASS",
                    flush=True,
                )
    except BaseException:
        traceback.print_exc()
        raise
    finally:
        del native
        if reference is not None:
            reference.close()
        if model is not None:
            model.close()
        dist.destroy_process_group()


class V41NativeDecodeSM120Test(unittest.TestCase):
    def run_case(self, context):
        self.assertEqual(torch.cuda.device_count(), 8)
        with tempfile.TemporaryDirectory() as root, socket.socket() as sock:
            torch.manual_seed(41900)
            config = checkpoint(Path(root), full_schedule=True)
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
            sock.close()
            mp.spawn(run_rank, args=(port, config, context), nprocs=8, join=True)

    def test_full_schedule_eight_rank_native_decode(self):
        self.run_case(256)

    def test_two_page_full_schedule_eight_rank_native_decode(self):
        self.run_case(512)


if __name__ == "__main__":
    unittest.main()
