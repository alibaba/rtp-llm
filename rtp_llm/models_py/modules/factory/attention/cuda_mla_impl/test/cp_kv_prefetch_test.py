import socket
import unittest
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from rtp_llm.models_py.distributed import pynccl_cp
from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.cp_kv_prefetch import (
    CrossLayerKvPrefetch,
    build_prefix_plan,
)


def _check_same_layer(rank, group, device, stream):
    from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.flashmla_sparse_cp_impl import (
        SparseMlaCpImpl,
        SparseMlaFp8CPOp,
    )
    from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.mla_kv_cache_write_op import (
        MlaKVCacheWriteOp,
    )
    from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.rope_emb_new import (
        NewMlaRotaryEmbeddingOp,
    )
    from rtp_llm.ops import KvCacheDataType

    def ints(values):
        return torch.tensor(values, dtype=torch.int32, device=device)

    impl = SparseMlaCpImpl.__new__(SparseMlaCpImpl)
    op = SparseMlaFp8CPOp.__new__(SparseMlaFp8CPOp)
    impl.fmha_impl = op
    impl._glm53_prefill_workspace = object()
    impl._kv_cache_sharded = True
    impl._cp_size = 2
    impl.attn_configs = SimpleNamespace(q_lora_rank=1)
    impl.rope_params = SimpleNamespace()
    half_cos = torch.zeros(4, 32, device=device)
    half_sin = torch.ones(4, 32, device=device)
    impl.rope_impl = NewMlaRotaryEmbeddingOp(
        torch.cat((half_cos, half_sin), dim=-1), is_neox_style=True
    )

    op.kv_cache_sharded = True
    op.use_cuda_graph = False
    op._gather = SimpleNamespace(batch_size=1, total_kv_len=4)
    op.block_table = ints([[1]])
    op.kv_lora_rank = 512
    op.qk_rope_head_dim = 64
    op.full_rope_pos_ids = ints([rank * 2, rank * 2 + 1])
    op.kv_restore_unpad_indices = torch.arange(4, device=device)
    op.sharded_kv_restore_indices = torch.arange(4, device=device)
    op.sharded_local_kv_lens = ints([2])
    op.sharded_actual_local_kv_lens = ints([2])
    op.sharded_actual_workspace_starts = ints([0])
    op.sharded_total_local_kv_len = 2
    op.sharded_actual_total_local_kv_len = 2
    op.sharded_slot_mapping = torch.tensor(
        [2, 3, -1, -1] if rank == 0 else [-1, -1, 2, 3],
        dtype=torch.int64,
        device=device,
    )
    op.kv_cache_write_op = MlaKVCacheWriteOp(KvCacheDataType.FP8)
    op.mla_params = impl.rope_params
    published = []
    op.write_cache_store_impl = published.append
    op.total_local_ids = torch.empty(0, dtype=torch.int64, device=device)
    op.attn_inputs = SimpleNamespace(is_prefill=True)

    compressed = torch.zeros(2, 512, dtype=torch.bfloat16, device=device)
    key = torch.arange(64, dtype=torch.bfloat16, device=device).expand(2, -1).clone()
    key.add_(rank * 5)
    reference = key.clone()
    q = torch.zeros(2, 1, 64, dtype=key.dtype, device=device)
    impl.rope_impl.forward(q, reference, impl.rope_params, op.full_rope_pos_ids)
    expected = torch.empty(4, 64, dtype=reference.dtype, device=device)
    dist.all_gather_into_tensor(expected, reference, group=group)

    cache = SimpleNamespace(
        kv_cache_base=torch.empty(4, 2, 656, dtype=torch.uint8, device=device)
    )
    pynccl_cp.configure(False)
    assert not impl.overlap_kv_projections
    pynccl_cp.configure(True)
    assert impl.overlap_cache_before_q
    with impl.prefetch_kv_projections(compressed, key):
        producer, producer_ready = impl._overlapped_projection
        assert producer.shape == (4, 576)
        impl.prepare_cache_before_q(0, cache, 64)
        fused, ready, publish = impl._prepared_kv
        assert ready is not producer_ready and publish
        assert not published
        # An empty query shard still consumes its producer event and publishes
        # after Indexer, without launching another history collective.
        op.forward(q, compressed, key, None, None, cache, prepared_kv=impl._prepared_kv)
        assert published == [cache]
        torch.testing.assert_close(producer[:, 512:], expected, rtol=0, atol=0)
        torch.testing.assert_close(fused[:, :512], torch.zeros_like(fused[:, :512]))
        torch.testing.assert_close(fused[:, 512:], expected, rtol=0, atol=0)
        impl.wait_kv_overlap_before_moe()
    assert impl._overlapped_projection is None and impl._prepared_kv is None
    assert impl._latest_kv_wait is None
    assert impl.kv_overlap_stream().cuda_stream == stream.cuda_stream


def _worker(rank, port):
    device = torch.device("cuda", rank)
    torch.cuda.set_device(device)
    dist.init_process_group(
        "nccl", init_method=f"tcp://127.0.0.1:{port}", world_size=2, rank=rank
    )
    try:
        group = dist.group.WORLD
        stream = pynccl_cp.warmup_kv_overlap(group, device)
        compute = torch.cuda.current_stream(device)
        _check_same_layer(rank, group, device, stream)
        cases = [
            ([5, 2, 0], [3, 4, 2]),
            ([2], [2]),  # Rank one has neither history nor current owned rows.
        ]
        for prefixes, lengths in cases:
            source = torch.zeros(64, 2, 656, dtype=torch.uint8, device=device)
            slots = []
            table = torch.full((len(prefixes), 8), -1, dtype=torch.int32)
            old, new = [], []
            expected = []
            for row, (prefix, count) in enumerate(zip(prefixes, lengths)):
                for pos in range(prefix + count):
                    owner = (pos // 4) % 2
                    value = row * 32 + pos + 1
                    expected.append(value)
                    if owner != rank:
                        if pos >= prefix:
                            slots.append(-1)
                        continue
                    local = (pos // 8) * 4 + pos % 4
                    page = row * 8 + local // 2
                    slot = page * 2 + local % 2
                    table[row, local // 2] = page
                    if pos < prefix:
                        old.append((page, slot % 2, value))
                    else:
                        new.append((page, slot % 2, value))
                        slots.append(slot)
            table = table.to(device)

            def write(entries):
                for page, offset, value in entries:
                    source[page, offset, 512:528].view(torch.float32).fill_(1)
                    source[page, offset, 528:].view(torch.bfloat16).fill_(value)

            write(old)
            staged = SimpleNamespace(
                backing=[source, source],
                borrow_next_layer=lambda layer, _table: (source, table, None),
            )
            op = SimpleNamespace(
                block_table=table,
                attn_inputs=SimpleNamespace(prefix_lengths=torch.tensor(prefixes)),
                cp_info=SimpleNamespace(
                    prefill_actual_input_lengths_cpu=torch.tensor(lengths)
                ),
                prefill_cp_size=2,
                prefill_cp_rank=rank,
                kv_owner_tokens_per_block=4,
                kv_lora_rank=512,
                qk_rope_head_dim=64,
            )
            pipeline = CrossLayerKvPrefetch(op, staged, stream, group)
            stream.wait_stream(compute)
            with torch.cuda.stream(stream):
                pipeline.prefetch(1)
            compute.wait_event(pipeline.pending[2])
            write(new)
            logical_slots = torch.tensor(slots, dtype=torch.int64, device=device)
            stream.wait_stream(compute)
            with torch.cuda.stream(stream):
                result = pipeline.finish_layer(
                    1, SimpleNamespace(kv_cache_base=source), logical_slots
                )
            compute.wait_stream(stream)
            expected_rope = torch.tensor(expected, dtype=torch.bfloat16, device=device)
            torch.testing.assert_close(
                result[:, :512], torch.zeros_like(result[:, :512])
            )
            torch.testing.assert_close(
                result[:, 512:], expected_rope[:, None].expand(-1, 64)
            )
    finally:
        pynccl_cp.destroy()
        dist.destroy_process_group()


class CpHistoryPlanTest(unittest.TestCase):
    def test_round_robin_ownership_and_rank_major_patch(self):
        for rank in range(2):
            plan = build_prefix_plan([5, 2, 0], [3, 4, 2], 2, rank, 4)
            self.assertEqual(
                plan.history_actual, ([4, 2, 0] if rank == 0 else [1, 0, 0])
            )
            self.assertEqual(plan.history_padded, [4, 4, 0])
            self.assertEqual(plan.history_restore.tolist(), [0, 1, 2, 3, 8, 4, 5])
            self.assertEqual(plan.history_dest.tolist(), [0, 1, 2, 3, 4, 8, 9])
            self.assertEqual(plan.new_dest.tolist(), [5, 6, 7, 10, 11, 12, 13, 14, 15])
            self.assertEqual(
                plan.new_local_rows.tolist(),
                [3, 4, 7, 8] if rank == 0 else [0, 1, 2, 5, 6],
            )
            self.assertEqual(plan.new_restore.tolist(), [5, 6, 7, 0, 1, 8, 9, 2, 3])

    def test_empty_local_owner_keeps_collective_length(self):
        plan = build_prefix_plan([2], [2], 2, 1, 4)
        self.assertEqual(plan.history_actual, [0])
        self.assertEqual(plan.history_padded, [4])
        self.assertEqual(plan.new_local_rows.numel(), 0)
        self.assertEqual(plan.new_padded, 2)

    @unittest.skipUnless(torch.cuda.device_count() >= 2, "two CUDA GPUs required")
    def test_native_fp8_history_and_fresh_rows_on_two_ranks(self):
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            port = listener.getsockname()[1]
        mp.spawn(_worker, args=(port,), nprocs=2, join=True)


if __name__ == "__main__":
    unittest.main()
