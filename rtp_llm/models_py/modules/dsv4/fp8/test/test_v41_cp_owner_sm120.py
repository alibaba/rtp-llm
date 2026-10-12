"""Maintained eager attention with two isolated PP-local CP4 groups on SM120."""

import json
import os
import socket
import unittest
from contextlib import ExitStack
from dataclasses import replace
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from rtp_llm.models_py.distributed.collective_torch import (
    Group,
    _get_group,
    destroy_distributed_environment,
    init_distributed_environment,
)
from rtp_llm.models_py.modules.dsv4.cp import build_cp_context
from rtp_llm.models_py.modules.dsv4.fp8.attention import bind_attn_cache
from rtp_llm.models_py.modules.dsv4.fp8.test.test_v41_attention_owner_sm120 import (
    NativeTaggedCache,
    make_layer,
    prepare,
)
from rtp_llm.models_py.modules.dsv4.prefill_workspace import PrefillWorkspace
from rtp_llm.ops import NcclCommConfig, ParallelismConfig


def zigzag(lengths, prefixes, rank, device):
    """Independent per-request inverse map matching native padded CP inputs."""
    chunks = [2 * ((n + 7) // 8) for n in lengths]
    local_rows = sum(chunks)
    padding, inverse, local_indices, full_restore = [], [], [], []
    local_offset = padded_offset = full_offset = 0
    for size, chunk in zip(lengths, chunks):
        padded, half = 4 * chunk, chunk // 2
        request_inverse = [0] * padded
        for peer in range(4):
            positions = [
                *range(peer * half, (peer + 1) * half),
                *range(padded - (peer + 1) * half, padded - peer * half),
            ]
            for i, position in enumerate(positions):
                request_inverse[position] = peer * local_rows + local_offset + i
            if peer == rank:
                # Padded values remain distinct; only real rows may be stored.
                local_indices.extend(full_offset + min(p, size - 1) for p in positions)
        padding.extend([1] * size + [0] * (padded - size))
        inverse.extend(request_inverse)
        full_restore.extend(request_inverse[:size])
        local_offset += chunk
        padded_offset += padded
        full_offset += size
    info = SimpleNamespace(
        prefill_qkv_padding_mask=torch.tensor(padding, dtype=torch.int32),
        prefill_qkv_restore_indice=torch.tensor(inverse, dtype=torch.int32),
        prefill_actual_input_lengths_cpu=torch.tensor(lengths, dtype=torch.int32),
        prefill_cp_chunk_lengths=torch.tensor(chunks, dtype=torch.int32),
    )
    cp = build_cp_context(
        info,
        4,
        rank,
        local_rows,
        device,
        position_offset=torch.tensor(prefixes, device=device, dtype=torch.int64),
        position_offset_cpu=torch.tensor(prefixes, dtype=torch.int64),
        kv_cache_sharded=False,
    )
    torch.testing.assert_close(
        cp.unpad_restore,
        torch.tensor(full_restore, device=device, dtype=torch.int64),
        rtol=0,
        atol=0,
    )
    return cp, torch.tensor(local_indices, device=device), chunks


def restore(value, cp, group):
    gathered = torch.empty(
        (4 * len(value), *value.shape[1:]), dtype=value.dtype, device=value.device
    )
    dist.all_gather_into_tensor(gathered, value.contiguous(), group=group)
    return gathered.index_select(0, cp.unpad_restore)


def page_tables(cache):
    return {
        tag: torch.tensor(
            [[1], [2]] if tag in ("swa_kv", "csa_state") else [[1, 3], [2, 4]],
            device="cuda",
            dtype=torch.int32,
        )
        for tag in cache.layouts
    }


def prepare_cp(layer, x, cache, tables, cp, chunks):
    lens = torch.tensor(chunks, device=x.device, dtype=torch.int32)
    cu = torch.tensor([0, chunks[0], sum(chunks)], device=x.device, dtype=torch.int32)
    with bind_attn_cache(layer, cache, tables, cp_ctx=cp):
        meta = layer._build_shared_prefill_meta(
            x,
            cp.global_positions,
            sp_per_req=cp.prefix_lengths,
            cu_seqlens=cu,
            batch_size=2,
            input_lengths=lens,
            prefix_lengths=cp.prefix_lengths,
            position_ids=cp.global_positions,
            req_id_per_token=cp.req_id_per_token,
            max_seqlen_q=max(chunks),
        )
    layer._set_prefill_meta_shared(
        meta._replace(
            workspace=PrefillWorkspace(
                x.device,
                q_rows=len(x),
                q_dim=layer.n_heads * layer.head_dim,
                reserve_cp=False,
            )
        )
    )


def require_rejection(layer, cache, tables, cp, chunks):
    """Bad local admission cannot mutate native bytes or enter a collective."""
    lens = torch.tensor(chunks, device="cuda", dtype=torch.int32)
    before = {k: v.clone() for k, v in cache.storage.items()}
    cases = (
        replace(cp, cp_size=2),
        replace(cp, kv_cache_sharded=True),
        replace(cp, cp_rank=(cp.cp_rank + 1) % 4),
        replace(cp, input_lengths_global=None),
        replace(cp, prefix_lengths=cp.prefix_lengths + 1),
        replace(
            cp,
            input_lengths_global=torch.tensor(
                [513, 513], device="cuda", dtype=torch.int32
            ),
            seq_len_full=1026,
            unpad_restore=torch.zeros(1026, device="cuda", dtype=torch.int64),
        ),
    )
    for invalid in cases:
        with bind_attn_cache(layer, cache, tables, cp_ctx=invalid):
            try:
                layer._require_qualified_prefill(lens, cp.prefix_lengths, cp_on=True)
            except (ValueError, NotImplementedError):
                pass
            else:
                raise AssertionError("unsupported CP metadata was admitted")
        for key, value in before.items():
            torch.testing.assert_close(cache.storage[key], value, rtol=0, atol=0)


def run_family(rank, pp_rank, group, source_id, ratio):
    torch.manual_seed(41400 + pp_rank * 100 + source_id)
    reference_shared = {}
    reference = [make_layer(source_id, ratio, reference_shared, gen=0, dim=5120)]
    if ratio:
        reference.append(
            make_layer(
                source_id + 1, ratio, reference_shared, source_id, gen=0, dim=5120
            )
        )
    torch.manual_seed(41400 + pp_rank * 100 + source_id)
    shared = {}
    owners = [make_layer(source_id, ratio, shared, gen=0, dim=5120)]
    if ratio:
        owners.append(
            make_layer(source_id + 1, ratio, shared, source_id, gen=0, dim=5120)
        )
    cache, oracle = NativeTaggedCache(owners, gen=0, pages=6), NativeTaggedCache(
        reference, gen=0, pages=6
    )
    tables = page_tables(cache)
    prefixes = [0, 0]
    rounds = ([1, 1], [4, 3], [252, 253], [255, 255])
    for step, lengths in enumerate(rounds):
        rng = torch.Generator().manual_seed(41500 + 100 * pp_rank + source_id + step)
        full = (
            (torch.randn(sum(lengths), 5120, generator=rng) * 0.025).bfloat16().cuda()
        )
        cp, indices, chunks = zigzag(lengths, prefixes, rank % 4, full.device)
        local = full.index_select(0, indices)
        # Distinct padding rows exercise masks rather than zero-input luck.
        local[~cp.local_is_real] = (rank + 1) / 32
        if step == 0:
            assert bool(cp.local_is_real.any()) == (rank % 4 == 0)
            require_rejection(owners[0], cache, tables, cp, chunks)
        with ExitStack() as bindings:
            for layer in owners:
                bindings.enter_context(bind_attn_cache(layer, cache, tables, cp_ctx=cp))
            for index, (layer, control) in enumerate(zip(owners, reference)):
                positions, _, _ = prepare(
                    control, full, oracle, lengths, prefixes, tables
                )
                with bind_attn_cache(control, oracle, tables, cp_ctx=None):
                    expected = control._forward_prefill(full, positions).clone()
                prepare_cp(layer, local, cache, tables, cp, chunks)
                actual = layer._forward_prefill(local, cp.global_positions).clone()
                restored = restore(actual, cp, group)
                torch.testing.assert_close(restored, expected, rtol=0, atol=0)
                assert bool(torch.isfinite(restored).all())
                if ratio:
                    selected = restore(shared["topk"][source_id], cp, group)
                    torch.testing.assert_close(
                        selected, reference_shared["topk"][source_id], rtol=0, atol=0
                    )
                    for actual_values, expected_values in zip(
                        shared["global"][source_id],
                        reference_shared["global"][source_id],
                    ):
                        for a, b in zip(actual_values, expected_values):
                            torch.testing.assert_close(a, b, rtol=0, atol=0)
                # Whole allocations include reserved pages and stride gaps.
                for key, value in cache.storage.items():
                    if key[0] <= index:
                        torch.testing.assert_close(
                            value, oracle.storage[key], rtol=0, atol=0
                        )
        for layer in (*owners, *reference):
            layer._set_prefill_meta_shared(None)
            assert layer._cp_ctx is None
        assert shared["global"] == {} and shared["topk"] == {}
        prefixes = [p + n for p, n in zip(prefixes, lengths)]
        print(
            "DSV41_CP_OWNER "
            + json.dumps(
                {
                    "rank": rank,
                    "pp_rank": pp_rank,
                    "source": source_id,
                    "ratio": ratio,
                    "step": step,
                    "ends": prefixes,
                    "bytes_exact": True,
                    "outputs_exact": True,
                }
            ),
            flush=True,
        )


def run_rank(rank, port):
    torch.cuda.set_device(rank)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    os.environ.update({"NCCL_P2P_DISABLE": "1", "RTP_V41_BATCHED_PRODUCER": "0"})
    dist.init_process_group(
        "nccl",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=8,
        timeout=timedelta(seconds=120),
    )
    cfg = ParallelismConfig()
    cfg.world_size = cfg.local_world_size = 8
    cfg.world_rank = cfg.local_rank = rank
    cfg.tp_size, cfg.dp_size, cfg.pp_size = 4, 1, 2
    try:
        init_distributed_environment(cfg, NcclCommConfig(nccl_ip="127.0.0.1"), port)
        group = _get_group(Group.TP)
        pp_rank = rank // 4
        assert tuple(dist.get_process_group_ranks(group)) == tuple(
            range(pp_rank * 4, (pp_rank + 1) * 4)
        )
        assert dist.get_rank(group) == rank % 4
        with torch.inference_mode():
            for source, ratio in ((0, 0), (2, 2), (20, 1)):
                run_family(rank, pp_rank, group, source, ratio)
        dist.barrier()
    finally:
        destroy_distributed_environment()


class V41CPOwnerSM120Test(unittest.TestCase):
    def test_two_pp_local_cp4_groups_match_cp1_owners_and_native_bytes(self):
        self.assertGreaterEqual(torch.cuda.device_count(), 8)
        for rank in range(8):
            self.assertEqual(torch.cuda.get_device_capability(rank), (12, 0))
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        with patch.dict(os.environ, {"RTP_V41_BATCHED_PRODUCER": "0"}):
            mp.spawn(run_rank, args=(port,), nprocs=8, join=True)


if __name__ == "__main__":
    unittest.main()
