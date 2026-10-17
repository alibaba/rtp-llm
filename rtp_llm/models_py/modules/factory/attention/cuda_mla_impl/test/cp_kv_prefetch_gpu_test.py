"""Eight-rank real FP8 cache, host staging, ring reuse and NCCL event coverage."""

import json
import os
import socket
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def run_case(rank, prefixes, lengths, producer):
    from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.cp_kv_prefetch import (
        CrossLayerKvPrefetch,
        prefix_plan,
    )
    from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.flashmla_sparse_cp_impl import (
        SparseMlaFp8CPOp,
    )
    from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.layered_prefill_cache import (
        ContiguousPrefillCache,
    )
    from rtp_llm.ops.compute_ops import LayerKVCache

    size, owner, page, layers = 8, 512, 64, 8
    totals = [p + q for p, q in zip(prefixes, lengths)]
    # Independent scalar ownership/reference: cache pages are deliberately
    # permuted and sparse, then the real staging cache compacts them again.
    owned = [[i for i in range(n) if i // owner % size == rank] for n in totals]
    pages = [max(1, (len(rows) + page - 1) // page) for rows in owned]
    physical = list(range(2 * sum(pages) + 1, 1, -2))
    table = torch.full((len(totals), max(pages)), -1, dtype=torch.int32, device="cpu")
    token_slots, new_slots, logical_rows = [], [], []
    offset = logical_start = 0
    for batch, (p, q, n, rows, count) in enumerate(
        zip(prefixes, lengths, totals, owned, pages)
    ):
        table[batch, :count] = torch.tensor(
            physical[offset : offset + count], device="cpu"
        )
        lookup = {
            position: int(table[batch, local // page]) * page + local % page
            for local, position in enumerate(rows)
        }
        for position in rows:
            token_slots.append(lookup[position])
            logical_rows.append(logical_start + position)
        new_slots.extend(lookup.get(position, -1) for position in range(p, p + q))
        logical_start += n
        offset += count
    physical_pages = max(physical) + 1
    slots_cpu = torch.tensor(new_slots, dtype=torch.int64, device="cpu")
    all_owned = torch.tensor(token_slots, dtype=torch.int64, device="cpu")
    logical_rows = torch.tensor(logical_rows, dtype=torch.int64, device="cpu")
    fresh_owned = slots_cpu[slots_cpu >= 0]
    backing, expected, fresh_bytes = [], [], []
    for layer in range(layers):
        ids = torch.arange(sum(totals), dtype=torch.int64, device="cpu")
        values = (
            (ids[:, None] + torch.arange(512, device="cpu") + 3 * layer) % 31 - 15
        ).to(torch.float8_e4m3fn)
        scales = (
            1 + (ids[:, None] + torch.arange(4, device="cpu") + layer) % 4
        ).float()
        rope = ((ids[:, None] + torch.arange(64, device="cpu") + 7 * layer) % 257).to(
            torch.bfloat16
        )
        encoded = torch.cat(
            (
                values.view(torch.uint8),
                scales.view(torch.uint8),
                rope.view(torch.uint8),
            ),
            dim=1,
        )
        decoded = torch.cat(
            (
                (values.float().reshape(-1, 4, 128) * scales[:, :, None])
                .reshape(-1, 512)
                .bfloat16(),
                rope,
            ),
            dim=1,
        )
        source = torch.zeros(
            (physical_pages, page, 656), dtype=torch.uint8, device="cpu"
        )
        source.view(-1, 656)[all_owned] = encoded[logical_rows]
        fresh_bytes.append(source.view(-1, 656)[fresh_owned].cuda())
        # A prefetched future layer must never read these not-yet-written rows.
        source.view(-1, 656)[fresh_owned] = 255
        backing.append(source.cuda() if layer < 2 else source.pin_memory())
        expected.append(decoded.cuda())
    plan = prefix_plan(totals, [0] * len(totals), size, rank, owner)
    op = SimpleNamespace(
        block_table=table.cuda(),
        kv_cache_sharded=True,
        sharded_slot_mapping=slots_cpu.cuda(),
        write_cache_store_impl=None,
        _gather=SimpleNamespace(batch_size=len(totals)),
        sharded_local_kv_lens=torch.tensor(
            [((n + size * owner - 1) // (size * owner)) * owner for n in totals],
            dtype=torch.int32,
        ),
        sharded_actual_local_kv_lens=plan["history_lengths"].cuda(),
        sharded_actual_workspace_starts=plan["history_starts"].cuda(),
        sharded_actual_total_local_kv_len=plan["history_actual"],
        sharded_total_local_kv_len=plan["history_padded"],
        sharded_kv_restore_indices=plan["history_restore"].cuda(),
        cp_info=SimpleNamespace(
            prefill_actual_input_lengths_cpu=torch.tensor(lengths, device="cpu")
        ),
        _allocate_fused_kv=lambda: torch.empty(
            (sum(totals), 576), dtype=torch.bfloat16
        ),
    )
    op._pack_sharded_kv_cache = (
        lambda cache, out: SparseMlaFp8CPOp._pack_sharded_kv_cache(op, cache, out)
    )
    inputs = SimpleNamespace(
        is_prefill=True,
        cache_store_inputs=None,
        kv_cache_kernel_block_id_host=table,
        kv_cache_block_id_host=table,
        prefix_lengths=torch.tensor(prefixes, device="cpu"),
    )
    op.attn_inputs = inputs
    fmha = SimpleNamespace(
        fmha_impl=op,
        attn_inputs=inputs,
        _cp_size=size,
        _cp_rank=rank,
        _kv_owner_tokens_per_block=owner,
        _kv_overlap_stream=producer,
        write_cache_store_impl=None,
    )
    fmha.begin_cross_layer_prefetch = lambda cache: CrossLayerKvPrefetch(fmha, cache)
    cache = ContiguousPrefillCache(
        backing, [i // 2 for i in range(layers)], physical_pages * page, 2, 2
    )
    cache.begin_forward(fmha)
    pipeline = cache.kv_prefetch
    assert pipeline.registered == (
        os.environ.get("RTP_LLM_CP_KV_REGISTERED", "0") == "1"
    )
    fmha._cross_layer_prefetch = pipeline
    compute = torch.cuda.current_stream()
    results, timing = [], []
    # Both fused slots must own distinct memory for the whole forward.
    assert pipeline.buffers[0].data_ptr() != pipeline.buffers[1].data_ptr()
    try:
        for layer in range(layers):
            cache.before_layer(layer)
            kv_cache = LayerKVCache()
            kv_cache.kv_cache_base = backing[layer]
            with cache.attention(layer, fmha, kv_cache) as target:
                slots = op.sharded_slot_mapping
                target.kv_cache_base.view(-1, 656).index_copy_(
                    0, slots[slots >= 0], fresh_bytes[layer]
                )
                fused, ready = pipeline.prepare(layer, target)
                torch.cuda._sleep(100_000)  # independent Q work
                compute.wait_event(ready)
                pipeline.consume(layer)
                results.append(fused.clone())  # attention reads the current slot
                torch.cuda._sleep(200_000)
                pipeline.release(layer)
            torch.cuda._sleep(100_000)  # O projection window
            cache.wait_prefetch_complete()
            # Assert actual device ordering, not just host call ordering.
            before_moe = torch.cuda.Event(enable_timing=True)
            before_moe.record(compute)
            timing.append((pipeline.latest_ready, before_moe))
            torch.cuda._sleep(200_000)  # excluded MoE interval
        cache.end_forward()
    except BaseException:
        cache.end_forward(failed=True)
        raise
    torch.cuda.synchronize()
    for producer_end, before_moe in timing:
        assert (
            producer_end.elapsed_time(before_moe) >= 0
        ), "history crossed the MoE fence"
    for layer, result in enumerate(results):
        torch.testing.assert_close(
            result, expected[layer], rtol=0, atol=0, msg=f"rank={rank}, layer={layer}"
        )
    # Verify writeback has kept every newly generated FP8 byte, including
    # partial pages shared by immutable prefix and new tokens.
    for layer, source in enumerate(backing):
        torch.testing.assert_close(
            source.cpu().view(-1, 656)[fresh_owned],
            fresh_bytes[layer].cpu(),
            rtol=0,
            atol=0,
        )
    assert fmha._cross_layer_prefetch is None
    return {
        "prefixes": prefixes,
        "new_lengths": lengths,
        "layers": layers,
        "zero_owned_new": not bool(len(fresh_owned)),
        "bit_exact": True,
        "registered": pipeline.registered,
    }


def check_registered_buffer_growth(rank, producer):
    from rtp_llm.models_py.distributed import pynccl_cp

    before = torch.cuda.memory_reserved()
    # Several small capacity increases must not retain all old private pools.
    # This reproduces the long-pressure failure without loading model weights.
    previous_capacity = 0
    previous_ptr = None
    for rows in range(8192, 65537, 8192):
        stats = torch.cuda.memory_stats()
        pair = pynccl_cp.registered_kv_overlap_buffers(
            rows, 576, torch.bfloat16, torch.device("cuda", rank), dist.group.WORLD
        )
        with torch.cuda.stream(producer):
            pair[0].zero_()
            pair[1].zero_()
        producer.synchronize()
        capacity = pynccl_cp._OVERLAP_BUFFERS[
            (
                *pynccl_cp._comm_key(dist.group.WORLD, pair[0].device),
                576,
                torch.bfloat16,
            )
        ].rows
        assert rows <= capacity < 2 * rows
        if rows <= previous_capacity:
            assert pair[0].data_ptr() == previous_ptr
            for counter in (
                "num_device_alloc",
                "num_device_free",
                "num_sync_all_streams",
            ):
                assert torch.cuda.memory_stats()[counter] == stats[counter], counter
        previous_capacity, previous_ptr = capacity, pair[0].data_ptr()
        del pair
    after = torch.cuda.memory_reserved()
    active_bytes = 65536 * 576 * 2 * 9
    assert after <= before + active_bytes + 16 * 1024**2, (
        "old registered history pools retained memory after growth",
        before,
        after,
        active_bytes,
    )
    Path(os.environ["TEST_UNDECLARED_OUTPUTS_DIR"]).joinpath(
        f"registered_growth_rank{rank}.json"
    ).write_text(
        json.dumps(dict(before=before, after=after, active_bytes=active_bytes))
    )


def _indexer_reference(input, group):
    output = torch.empty(
        (dist.get_world_size(group) * input.size(0), input.size(1)),
        dtype=torch.uint8,
        device=input.device,
    )
    dist.all_gather_into_tensor(output, input.view(torch.uint8), group=group)
    return output.view(input.dtype)


def check_registered_indexer(rank, producer):
    from rtp_llm.models_py.distributed import pynccl_cp

    compute = torch.cuda.current_stream()
    hlocal, houtput = pynccl_cp.registered_kv_overlap_buffers(
        8192, 576, torch.bfloat16, torch.device("cuda", rank), dist.group.WORLD
    )
    results = []
    for role, width, dtype in (
        ("indexer_k_quant", 128, torch.float8_e4m3fn),
        ("indexer_k_quant", 128, torch.float8_e5m2),
        ("indexer_k_packed", 129, torch.uint8),
    ):
        for iteration, rows in enumerate((1, 257, 8192, 32768, 1025, 65536, 4096)):
            data = torch.randint(
                0, 256, (rows, width), dtype=torch.uint8, device="cuda"
            ).view(dtype)
            with torch.cuda.stream(producer):
                hlocal.fill_(rank + iteration)
                pynccl_cp.all_gather_kv_overlap(
                    hlocal, dist.group.WORLD, output=houtput
                )
            actual = pynccl_cp.all_gather(data, group=dist.group.WORLD, role=role)
            key = (
                *pynccl_cp._comm_key(dist.group.WORLD, data.device),
                role,
                width,
                dtype,
            )
            assert (
                actual.data_ptr()
                == pynccl_cp._INDEXER_BUFFERS[key][1].gathered.data_ptr()
            )
            reference = _indexer_reference(data, dist.group.WORLD)
            assert torch.equal(actual.view(torch.uint8), reference.view(torch.uint8))
            snapshot = actual.clone()
            pynccl_cp.all_gather(
                torch.zeros_like(data), group=dist.group.WORLD, role=role
            )
            assert torch.equal(snapshot.view(torch.uint8), reference.view(torch.uint8))
            compute.wait_stream(producer)
            for peer in range(8):
                assert bool(
                    (houtput[peer * 8192 : (peer + 1) * 8192] == peer + iteration).all()
                )
            results.append(dict(role=role, rows=rows, dtype=str(dtype), bitexact=True))
            del actual, reference, snapshot, data
    # A caller on a different stream must not overwrite the persistent slot.
    data = torch.full((257, 129), rank, dtype=torch.uint8, device="cuda")
    other = torch.cuda.Stream()
    other.wait_stream(compute)
    with torch.cuda.stream(other):
        assert (
            pynccl_cp._registered_indexer_all_gather(
                data, dist.group.WORLD, "indexer_k_packed"
            )
            is None
        )
        actual = pynccl_cp.all_gather(
            data, group=dist.group.WORLD, role="indexer_k_packed"
        )
    compute.wait_stream(other)
    reference = _indexer_reference(data, dist.group.WORLD)
    assert torch.equal(actual, reference)
    del actual, reference, data
    # Preallocate input to isolate private-pool capacity changes from inputs.
    data = torch.zeros((131072, 128), dtype=torch.uint8, device="cuda")
    torch.cuda.synchronize()
    before = torch.cuda.memory_reserved()
    previous_capacity = 0
    previous_ptr = None
    for rows in range(8192, 131073, 8192):
        stats = torch.cuda.memory_stats()
        actual = pynccl_cp.all_gather(
            data[:rows], group=dist.group.WORLD, role="indexer_k_quant"
        )
        compute.synchronize()
        key = (
            *pynccl_cp._comm_key(dist.group.WORLD, data.device),
            "indexer_k_quant",
            128,
            torch.uint8,
        )
        capacity = pynccl_cp._INDEXER_BUFFERS[key][1].rows
        assert rows <= capacity < 2 * rows
        if rows <= previous_capacity:
            assert actual.data_ptr() == previous_ptr
            for counter in (
                "num_device_alloc",
                "num_device_free",
                "num_sync_all_streams",
            ):
                assert torch.cuda.memory_stats()[counter] == stats[counter], counter
        previous_capacity, previous_ptr = capacity, actual.data_ptr()
        del actual
    after = torch.cuda.memory_reserved()
    assert after <= before + 131072 * 128 * 9 + 16 * 1024**2, (before, after)
    Path(os.environ["TEST_UNDECLARED_OUTPUTS_DIR"]).joinpath(
        f"indexer_rank{rank}.json"
    ).write_text(
        json.dumps(
            dict(
                cases=results,
                stream_fallback=True,
                memory=dict(before=before, after=after),
            ),
            indent=2,
        )
    )
    del hlocal, houtput, data


def check_registered_failure_cleanup(rank):
    from rtp_llm.models_py.distributed import pynccl_cp

    device = torch.device("cuda", rank)
    lib, comm = pynccl_cp._COMMS[pynccl_cp._comm_key(dist.group.WORLD, device)]
    register = lib.window_register
    calls = 0

    def fail_second(*args):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("injected window registration failure")
        return register(*args)

    before = torch.cuda.memory_reserved()
    with patch.object(lib, "window_register", side_effect=fail_second):
        try:
            pynccl_cp._RegisteredKvBuffers(
                dist.group.WORLD, device, 1, 129, torch.uint8, lib, comm
            )
        except RuntimeError as error:
            assert str(error) == "injected window registration failure"
        else:
            raise AssertionError("registration failure was not propagated")
    after = torch.cuda.memory_reserved()
    assert after == before, (before, after)
    Path(os.environ["TEST_UNDECLARED_OUTPUTS_DIR"]).joinpath(
        f"registered_cleanup_rank{rank}.json"
    ).write_text(json.dumps(dict(before=before, after=after, register_calls=calls)))


def worker(rank, port):
    torch.cuda.set_device(rank)
    torch.set_default_device(f"cuda:{rank}")
    dist.init_process_group(
        "nccl",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=8,
        timeout=timedelta(seconds=180),
    )
    from rtp_llm.models_py.distributed import collective_torch, pynccl_cp

    collective_torch._group_map[collective_torch.Group.DP_AND_TP] = dist.group.WORLD
    collective_torch._initialized = True
    collective_torch._parallelism_config = SimpleNamespace(
        tp_size=8, dp_size=1, world_size=8
    )
    producer = pynccl_cp.warmup_kv_overlap(dist.group.WORLD)
    pynccl_cp.warmup(dist.group.WORLD)
    cases = [
        ([1, 511, 4097, 0], [6, 1025, 13, 0]),
        ([0, 64, 513], [257, 0, 1]),
        ([4096, 512, 8192], [257, 1, 0]),
    ]
    real_event = torch.cuda.Event

    def timed_event(*args, **kwargs):
        kwargs["enable_timing"] = True
        return real_event(*args, **kwargs)

    with patch.object(torch.cuda, "Event", side_effect=timed_event):
        results = []
        for direct in (False, True):
            pynccl_cp._OPT_ENABLED = direct
            for index_opt in (False, True):
                with patch.dict(
                    os.environ, {"RTP_LLM_CP_KV_INDEX_OPT": str(int(index_opt))}
                ):
                    for p, q in cases:
                        result = run_case(rank, p, q, producer)
                        result["direct_new_rows_nccl"] = direct
                        result["index_opt"] = index_opt
                        results.append(result)
    Path(os.environ["TEST_UNDECLARED_OUTPUTS_DIR"]).joinpath(
        f"prefetch_rank{rank}.json"
    ).write_text(json.dumps(results, indent=2))
    if os.environ.get("RTP_LLM_CP_KV_REGISTERED", "0") == "1":
        check_registered_buffer_growth(rank, producer)
    if os.environ.get("RTP_LLM_CP_INDEXER_REGISTERED", "0") == "1":
        check_registered_indexer(rank, producer)
        check_registered_failure_cleanup(rank)
    dist.barrier()
    pynccl_cp.destroy()
    dist.destroy_process_group()


if __name__ == "__main__":
    if torch.cuda.device_count() < 8:
        raise RuntimeError("requires eight allocated GPUs")
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    mp.spawn(worker, args=(port,), nprocs=8, join=True)
