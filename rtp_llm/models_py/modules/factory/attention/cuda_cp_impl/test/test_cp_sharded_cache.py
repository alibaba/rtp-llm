"""Real two-rank CP FULL attention with block-round-robin cache storage."""

import socket
import unittest

import torch
import torch.multiprocessing as mp

from rtp_llm.models_py.distributed.collective_torch import (
    destroy_distributed_environment,
    init_distributed_environment,
)
from rtp_llm.models_py.modules.factory.attention.cuda_cp_impl.prefill_mha.allgather_cp_impl import (
    PCPAllGatherAttnOp,
)
from rtp_llm.models_py.modules.factory.attention.cuda_cp_impl.prefill_mha.allgather_overlap_impl import (
    PCPAllGatherOverlapAttnOp,
)
from rtp_llm.models_py.modules.factory.attention.cuda_cp_impl.prefill_mha.alltoall_cp_impl import (
    PCPAll2AllAttnOp,
)
from rtp_llm.models_py.modules.factory.attention.cuda_cp_impl.prefill_mha.cp_cache import (
    CPPagedCacheGeometry,
    save_full_checkpoints,
)
from rtp_llm.models_py.modules.factory.attention.cuda_cp_impl.test.cp_test_utils import (
    build_cp_attn_inputs,
    make_configs,
    zigzag_positions_for_rank,
)
from rtp_llm.ops import NcclCommConfig
from rtp_llm.ops.compute_ops import LayerKVCache


def _reference(q, kv, positions, prefix):
    heads, dim = q.shape[1:]
    kv_heads = kv.shape[2]
    keys = kv[:, 0].float().repeat_interleave(heads // kv_heads, dim=1).transpose(0, 1)
    values = (
        kv[:, 1].float().repeat_interleave(heads // kv_heads, dim=1).transpose(0, 1)
    )
    scores = q[positions].float().transpose(0, 1) @ keys.transpose(-1, -2) / dim**0.5
    invalid = (
        torch.arange(kv.shape[0], device=q.device)[None, :]
        > (prefix + positions)[:, None]
    )
    scores.masked_fill_(invalid, float("-inf"))
    return (scores.softmax(-1) @ values).transpose(0, 1)


def _execute(
    rank,
    op_class,
    cfg,
    parallel,
    prefixes,
    suffixes,
    kvs,
    queries,
    table,
    cache,
    save_positions=None,
    save_blocks=None,
):
    device = cache.kv_cache_base.device
    padded = [(n + 3) // 4 * 4 for n in suffixes]
    inputs = build_cp_attn_inputs(
        [p + n for p, n in zip(prefixes, suffixes)],
        [n // 2 for n in padded],
        2,
        cfg.kernel_tokens_per_block,
        prefixes,
        device,
    )
    inputs.kv_cache_kernel_block_id = table
    if save_positions is not None:
        inputs.cache_save_positions = save_positions
    if save_blocks is not None:
        inputs.cache_save_block_ids = save_blocks
    inputs.context_parallel_info.prefill_qkv_padding_mask = torch.cat(
        [torch.arange(pad, device=device) < n for n, pad in zip(suffixes, padded)]
    ).to(torch.int32)
    qkv_parts, shuffle, masks, references = [], [], [], []
    for prefix, n, pad, kv, q in zip(prefixes, suffixes, padded, kvs, queries):
        positions = zigzag_positions_for_rank(pad, 2, rank)
        shuffle.extend(positions)
        pos = torch.tensor(positions, device=device)
        valid = pos < n
        idx = pos[valid]
        masks.append(valid)
        local = torch.zeros(
            len(positions),
            (cfg.head_num + 2 * cfg.kv_head_num) * cfg.size_per_head,
            device=device,
            dtype=q.dtype,
        )
        local[valid] = torch.cat(
            (
                q[idx].flatten(1),
                kv[prefix + idx, 0].flatten(1),
                kv[prefix + idx, 1].flatten(1),
            ),
            dim=-1,
        )
        qkv_parts.append(local)
        references.append(_reference(q, kv, idx, prefix))
    inputs.context_parallel_info.prefill_shuffle_indices = torch.tensor(
        shuffle, dtype=torch.int32, device=device
    )
    op = op_class(cfg, inputs, parallel, backend="fa2")
    geometry = (
        CPPagedCacheGeometry(cfg, inputs, parallel)
        if parallel.prefill_cp_config.kv_cache_sharded
        else None
    )
    op.cache_geometry = geometry
    params = op.prepare(inputs)
    attention_cache = geometry.materialize(cache) if geometry else cache
    if geometry:
        for batch, prefix in enumerate(prefixes):
            for pos in range(prefix):
                page = int(
                    geometry.block_ids[batch, pos // cfg.kernel_tokens_per_block]
                )
                assert torch.equal(
                    attention_cache.kv_cache_base[
                        page, :, :, pos % cfg.kernel_tokens_per_block
                    ],
                    kvs[batch][pos],
                )
    output = op.forward(torch.cat(qkv_parts), attention_cache, params)
    if geometry:
        geometry.commit_suffix(attention_cache, cache)
    save_full_checkpoints(cache, inputs, cfg, parallel)
    torch.testing.assert_close(
        output[torch.cat(masks)].float(), torch.cat(references), rtol=1e-2, atol=1e-2
    )


def _run_case(rank, op_class, sharded, prefixes, suffixes):
    device = torch.device("cuda", rank)
    base, kernel, heads, kv_heads, dim = 256, 128, 16, 2, 256
    cfg, parallel = make_configs(heads, kv_heads, dim, base, 2, rank)
    cfg.kernel_tokens_per_block = kernel
    parallel.prefill_cp_config.kv_cache_sharded = sharded
    lengths = [p + n for p, n in zip(prefixes, suffixes)]
    size, owner = (2, rank) if sharded else (1, 0)
    # Reserve four continuation tokens, with real base-block-aligned physical
    # IDs. Kernel subpages of each permuted physical block remain contiguous.
    counts = [(n + 4 + base * size - 1) // (base * size) for n in lengths]
    ppb = base // kernel
    table = torch.full((len(lengths), max(counts) * ppb), -1, dtype=torch.int32)
    permutation = (
        torch.randperm(sum(counts), generator=torch.Generator().manual_seed(42 + rank))
        + 1
    )
    cursor = 0
    for batch, count in enumerate(counts):
        ids = permutation[cursor : cursor + count]
        table[batch, : count * ppb] = (ids[:, None] * ppb + torch.arange(ppb)).flatten()
        cursor += count
    save_start = sum(counts) + 1
    save_blocks = torch.arange(
        save_start, save_start + len(lengths) * 4, dtype=torch.int32
    ).reshape(len(lengths), 4)
    # P=N-1 and P=N prove later KV cannot leak into an earlier save. Future
    # points and slots at/below G are never written even with valid block IDs.
    save_positions = torch.tensor(
        [[n - 1, n, n + 10, -1] for n in lengths], dtype=torch.int64
    )
    storage = torch.full(
        ((save_start + len(lengths) * 4) * ppb, 2, kv_heads, kernel, dim),
        32.0,
        dtype=torch.bfloat16,
        device=device,
    )
    cache = LayerKVCache(storage, kernel, 0, "full")
    rng = torch.Generator().manual_seed(601)
    kvs, queries = [], []
    for batch, (prefix, suffix, length) in enumerate(zip(prefixes, suffixes, lengths)):
        kv = torch.randn(length, 2, kv_heads, dim, generator=rng).to(
            device=device, dtype=torch.bfloat16
        )
        q = torch.randn(suffix, heads, dim, generator=rng).to(
            device=device, dtype=torch.bfloat16
        )
        kvs.append(kv)
        queries.append(q)
        for pos in range(prefix):
            if (pos // base) % size != owner:
                continue
            local = (pos // (base * size)) * base + pos % base
            storage[int(table[batch, local // kernel]), :, :, local % kernel] = kv[pos]
    before = storage.clone()
    _execute(
        rank,
        op_class,
        cfg,
        parallel,
        prefixes,
        suffixes,
        kvs,
        queries,
        table,
        cache,
        save_positions,
        save_blocks,
    )
    expected = before.clone()
    for batch, (prefix, length) in enumerate(zip(prefixes, lengths)):
        for pos in range(prefix, length):
            if (pos // base) % size != owner:
                continue
            local = (pos // (base * size)) * base + pos % base
            expected[int(table[batch, local // kernel]), :, :, local % kernel] = kvs[
                batch
            ][pos]
        for point, dst in zip(
            save_positions[batch].tolist(), save_blocks[batch].tolist()
        ):
            if point <= prefix or point > length:
                continue
            global_start = (((point - 1) // base) // size * size + owner) * base
            for offset in range(max(0, min(base, point - global_start))):
                expected[dst * ppb + offset // kernel, :, :, offset % kernel] = kvs[
                    batch
                ][global_start + offset]
    assert torch.equal(storage.view(torch.uint8), expected.view(torch.uint8))
    saved_before = storage[save_start * ppb :].clone()
    # Restore the actual selected final checkpoint into poisoned runtime tails
    # and perform a second true CP attention call, rather than only comparing
    # serialized saved bytes. Original prefix pages remain in the same cache.
    for batch, length in enumerate(lengths):
        local_slot = ((length - 1) // base) // size
        dst = int(save_blocks[batch, 1])
        for offset in range(ppb):
            physical = int(table[batch, local_slot * ppb + offset])
            storage[physical].fill_(-32)
            storage[physical].copy_(storage[dst * ppb + offset])
        extension = torch.randn(4, 2, kv_heads, dim, generator=rng).to(
            device=device, dtype=torch.bfloat16
        )
        kvs[batch] = torch.cat((kvs[batch], extension))
        queries[batch] = torch.randn(4, heads, dim, generator=rng).to(
            device=device, dtype=torch.bfloat16
        )
    _execute(
        rank,
        op_class,
        cfg,
        parallel,
        lengths,
        [4] * len(lengths),
        kvs,
        queries,
        table,
        cache,
    )
    assert torch.equal(
        storage[save_start * ppb :].view(torch.uint8), saved_before.view(torch.uint8)
    )
    print(
        f"rank={rank} backend={op_class.__name__} sharded={sharded} prefixes={prefixes} suffixes={suffixes}: O, storage, selected-save/restore PASS",
        flush=True,
    )


def _worker(rank, port):
    torch.cuda.set_device(rank)
    _, parallel = make_configs(cp_size=2, cp_rank=rank)
    parallel.world_size = 2
    parallel.world_rank = rank
    parallel.local_rank = rank
    parallel.dp_size = 1
    nccl = NcclCommConfig(
        nccl_ip="127.0.0.1",
        tp_nccl_port=port + 1,
        dp_tp_nccl_port=port + 2,
        ffn_tp_nccl_port=port + 3,
    )
    init_distributed_environment(
        parallel, nccl, port, timeout=180
    )
    try:
        for op in (PCPAllGatherAttnOp, PCPAllGatherOverlapAttnOp, PCPAll2AllAttnOp):
            for sharded in (False, True):
                for prefixes, suffixes in (
                    ([0], [256]),
                    ([0, 1, 256, 1300], [16, 20, 32, 700]),
                    ([0, 256, 1300], [1, 17, 701]),
                ):
                    _run_case(rank, op, sharded, prefixes, suffixes)
    finally:
        destroy_distributed_environment()


class CPShardedCacheTest(unittest.TestCase):
    def test_all_cp_backends_with_real_two_rank_collectives(self):
        self.assertGreaterEqual(torch.cuda.device_count(), 2)
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        mp.spawn(_worker, args=(port,), nprocs=2, join=True)

    def test_save_full_checkpoints_on_side_stream_observes_only_committed_kv(self):
        """Forward-window gate invariant for the checkpoint save stream.

        save_full_checkpoints(save_stream=...) must observe the current
        stream's committed runtime KV only (never partial writes of the same
        layer), and once the terminal save event completes the saved blocks
        must hold the committed bytes. This mirrors the engine contract:
        PyWrappedModel::forward returns only after this event, and the owner
        publishes only after that return.
        """
        if not torch.cuda.is_available():
            self.skipTest("requires CUDA")
        device = torch.device("cuda", torch.cuda.current_device())
        base, kernel, kv_heads, dim = 4, 4, 2, 32
        cfg, parallel = make_configs(4, kv_heads, dim, base, 1, 0)
        cfg.kernel_tokens_per_block = kernel
        parallel.prefill_cp_config.kv_cache_sharded = False
        ppb = base // kernel
        # sequence_lengths=[4] (full lengths), cp_chunk_lengths=[4] (single
        # rank owns everything at cp_size=1), prefix=0.
        inputs = build_cp_attn_inputs([4], [4], 1, kernel, [0], device)
        # One runtime block holds the four suffix tokens; a private block is
        # the save destination. Negative slots prove no spurious writes.
        table = torch.tensor([[0, 1]], dtype=torch.int32)
        inputs.kv_cache_kernel_block_id = table
        inputs.cache_save_positions = torch.tensor([[4]], dtype=torch.int64)
        inputs.cache_save_block_ids = torch.tensor([[1]], dtype=torch.int32)
        storage = torch.zeros(3 * ppb, 2, kv_heads, kernel, dim, device=device)
        cache = LayerKVCache(storage, kernel, 0, "full")

        # Commit this "layer's" runtime KV on the current stream. storage[0]
        # is one kernel page: [2 (K/V), kv_heads, kernel, dim].
        runtime_payload = torch.randn(2, kv_heads, kernel, dim, device=device)
        storage[0] = runtime_payload

        save_stream = torch.cuda.Stream()
        save_event = torch.cuda.Event()
        layer_done = torch.cuda.Event()
        layer_done.record()
        save_stream.wait_event(layer_done)
        save_full_checkpoints(cache, inputs, cfg, parallel, save_stream=save_stream)
        save_event.record(save_stream)

        # The next layer owns distinct runtime storage; it must not overwrite
        # the source being read by the outstanding checkpoint copy.
        next_layer_payload = torch.randn(2, kv_heads, kernel, dim, device=device)
        storage[2 * ppb] = next_layer_payload

        # The completion event is the publication gate: once it fires, the
        # committed bytes are in place independently of the next layer's write.
        save_event.synchronize()
        torch.testing.assert_close(
            storage[ppb].float(), runtime_payload.float(), rtol=0, atol=0
        )
        torch.testing.assert_close(
            storage[2 * ppb].float(), next_layer_payload.float(), rtol=0, atol=0
        )


if __name__ == "__main__":
    unittest.main()
