"""Shared cache-slot and graph checks for the real Mega attention tests."""

import torch

from rtp_llm.models_py.modules.dsv4.fp8.decode.decode_attn_metadata import (
    allocate_decode_metadata_fp8,
    update_decode_metadata_in_place_fp8,
)
from rtp_llm.models_py.modules.dsv4.fp8.decode.mega_csa_weights import HC, MQA_SPLIT_KV
from rtp_llm.models_py.modules.dsv4.kv_cache_utils import SWA_KV
from rtp_llm.ops.compute_ops import LayerKVCache
from rtp_llm.test.utils.numeric_util import calc_diff


class TaggedKVCache:
    """Python-owned test pools exposing the native per-layer tag interface."""

    def __init__(self, tensors: dict[str, torch.Tensor], tokens_per_block: int):
        self.layers = {
            tag: LayerKVCache(tensor, tokens_per_block, 0, group_id, tag)
            for group_id, (tag, tensor) in enumerate(tensors.items())
        }

    def get_layer_cache(self, layer_id: int, tag: str) -> LayerKVCache:
        assert layer_id == 0
        return self.layers[tag]

    def get_seq_size_per_block(self, tag: str) -> int:
        return self.layers[tag].seq_size_per_block

    def get_kernel_seq_size_per_block(self, tag: str) -> int:
        return self.get_seq_size_per_block(tag)


def slots_from_block_table(block_table: torch.Tensor, entries: int) -> torch.Tensor:
    offsets = torch.arange(entries, dtype=torch.int64, device=block_table.device)
    return (block_table.to(torch.int64).unsqueeze(-1) * entries + offsets).flatten()


@torch.inference_mode()
def check_dynamic_graph_replays(test, make_pools, fill_context) -> None:
    attn = test.block.attn
    ratio = int(attn.compress_ratio)
    buckets = []
    for batch_size, q_len in ((1, 1), (2, 3)):
        pools = make_pools(test.device, batch_size=batch_size)
        reference_pools = make_pools(test.device, batch_size=batch_size)
        metadata = allocate_decode_metadata_fp8(
            max_batch_size=batch_size,
            q_len=q_len,
            window_size=attn.window_size,
            head_dim=attn.head_dim,
            max_seq_len=pools.max_seq_len,
            compress_ratios=[ratio],
            index_topk=attn.indexer.index_topk if attn.indexer is not None else 1024,
            device=test.device,
            paged_pool_specs={
                kind: (
                    pools.entries_per_block[kind],
                    pools.tokens_per_block[kind],
                    table.shape[1],
                )
                for kind, table in pools.block_tables.items()
            },
        )
        metadata.is_cuda_graph = True
        position = torch.full(
            (batch_size,), ratio - 1, dtype=torch.int32, device=test.device
        )
        update_decode_metadata_in_place_fp8(
            metadata,
            position,
            forbid_realloc=True,
            paged_block_tables=pools.block_tables,
            paged_pool_entries_per_block=pools.entries_per_block,
            paged_pool_tokens_per_block=pools.tokens_per_block,
            capture_full_width_lengths=True,
        )
        graph_input = torch.randn(
            batch_size, q_len, HC, attn.dim, dtype=torch.bfloat16, device=test.device
        ).mul_(0.05)
        graph_work = torch.empty_like(graph_input)
        for _ in range(2):
            test.runtime.begin_decode(metadata)
            test._forward_mega(graph_input.clone(), metadata, pools)
        torch.cuda.synchronize(test.device)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            test.runtime.begin_decode(metadata)
            graph_work.copy_(graph_input)
            output = test._forward_mega(graph_work, metadata, pools)
        buckets.append(
            (graph, graph_input, graph_work, output, metadata, pools, reference_pools)
        )

    # Alternate captured shapes, then change requests at the same stable addresses.
    for replay in range(2):
        for graph, graph_input, _, output, metadata, pools, reference_pools in buckets:
            batch_size, q_len = graph_input.shape[:2]
            label = f"Graph ratio={ratio} B={batch_size} S={q_len} replay={replay}"
            with test.subTest(label=label):
                for pool in (pools, reference_pools):
                    pool.reset()
                    fill_context(pool, test.device, seed=71 + replay)
                    swa_history = pool.tensors[SWA_KV][pool.block_tables[SWA_KV].long()]
                    test.assertTrue(swa_history.flatten(2).any(2).all().item(), label)
                    if batch_size > 1:
                        test.assertFalse(
                            torch.equal(swa_history[0], swa_history[1]), label
                        )
                    if replay:
                        for table in pool.block_tables.values():
                            table.copy_(table.flip(0).flip(1))
                # Cross MQA scheduling and cache-page boundaries between replays.
                position = (
                    ratio
                    * (
                        MQA_SPLIT_KV * (replay + 1) + 1
                        if ratio == 4
                        else 1 + replay * 2
                    )
                    - 1
                )
                update_decode_metadata_in_place_fp8(
                    metadata,
                    torch.full(
                        (batch_size,), position, dtype=torch.int32, device=test.device
                    ),
                    forbid_realloc=True,
                    paged_block_tables=pools.block_tables,
                    paged_pool_entries_per_block=pools.entries_per_block,
                    paged_pool_tokens_per_block=pools.tokens_per_block,
                )
                hidden = torch.randn_like(graph_input).mul_(0.05)
                graph_input.copy_(hidden)
                reference, reference_metadata = test._run_reference_step(
                    position, hidden.clone(), reference_pools
                )
                graph.replay()
                torch.cuda.synchronize(test.device)
                test.assertTrue(torch.isfinite(output).all().item(), label)
                test.assertLess(
                    calc_diff(output.float(), reference.float()), 1.0e-3, label
                )
                test._assert_written_pools_match(
                    metadata,
                    reference_metadata,
                    pools,
                    reference_pools,
                    label=label,
                    **({"expect_boundary_write": True} if ratio == 128 else {}),
                )
