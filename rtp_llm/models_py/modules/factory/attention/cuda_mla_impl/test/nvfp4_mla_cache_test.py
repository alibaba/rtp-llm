"""Execute NVFP4 MLA writes, selected-row reads and the real FlashMLA consumer."""

from datetime import timedelta
from types import SimpleNamespace
from unittest import TestCase, main, skipUnless
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from flash_mla import flash_mla_with_kvcache, get_mla_metadata

from rtp_llm.models_py.distributed import collective_torch
from rtp_llm.models_py.kernels.cuda.nvfp4_kv import (
    FP8_MLA_ROW_BYTES,
    MLA_ROW_BYTES,
    nvfp4_mla_rows_to_fp8,
    pack_nvfp4_rows,
)
from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl import (
    flashmla_sparse_cp_impl,
    flashmla_sparse_impl,
)
from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.nvfp4_mla_cache import (
    gather_padded_nvfp4_mla_rows,
    gather_selected_mla_fp8,
    write_nvfp4_mla_cache,
)
from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.pinned_mla_cache import (
    PinnedMlaWorkingSet,
)
from rtp_llm.test.utils.port_util import PortManager


@skipUnless(torch.cuda.is_available(), "requires a CUDA device")
class Nvfp4MlaCacheTest(TestCase):
    def setUp(self):
        torch.manual_seed(2026)
        self.device = torch.device("cuda", torch.cuda.current_device())

    def _rows(self, n):
        nope = torch.randn((n, 512), device=self.device, dtype=torch.bfloat16)
        rope = torch.randn((n, 64), device=self.device, dtype=torch.bfloat16)
        return nope, rope

    def _plan(self, op, query_request_ids, block_table, attn_inputs=None):
        params = SimpleNamespace(
            batch_indice_h=query_request_ids.cpu(), batch_indice_d=query_request_ids
        )
        op.plan(params, block_table, attn_inputs)

    def _ref_attention(self, q, fp8_pages, physical, scale):
        result, _ = flash_mla_with_kvcache(
            q=q.unsqueeze(0),
            k_cache=fp8_pages,
            block_table=None,
            cache_seqlens=None,
            head_dim_v=512,
            tile_scheduler_metadata=get_mla_metadata()[0],
            is_fp8_kvcache=True,
            indices=physical.unsqueeze(0),
            softmax_scale=scale,
        )
        return result.squeeze(0)

    def test_write_and_select_respect_paged_strides_and_padding(self):
        backing = torch.full(
            (3, 65, MLA_ROW_BYTES), 0xAA, device=self.device, dtype=torch.uint8
        )
        cache = backing[:, :64]
        nope, rope = self._rows(5)
        slots = torch.tensor([0, 63, 65, 191, -1], device=self.device)
        write_nvfp4_mla_cache(nope, rope, cache, slots)
        self.assertTrue(torch.all(backing[:, 64] == 0xAA).item())
        self.assertTrue(torch.all(backing[0, 1] == 0xAA).item())
        expected_pack = pack_nvfp4_rows(nope, rope)
        torch.testing.assert_close(
            torch.stack([cache[0, 0], cache[0, 63], cache[1, 1], cache[2, 63]]),
            expected_pack[:4],
        )
        physical = torch.tensor(
            [[191, 0, 63, -1], [63, 65, 191, 193]], device=self.device
        )
        fp8, remap = gather_selected_mla_fp8(cache, physical)
        torch.testing.assert_close(
            remap,
            torch.tensor(
                [[0, 1, 2, -1], [4, 5, 6, -1]], device=self.device, dtype=torch.int32
            ),
        )
        selected_src = expected_pack[[3, 0, 1, 1, 2, 3]]
        expected = nvfp4_mla_rows_to_fp8(selected_src)
        got = fp8.view(-1, FP8_MLA_ROW_BYTES)[[0, 1, 2, 4, 5, 6]]
        self._compare_fp8_rows(got, expected)

    def _compare_fp8_rows(self, got, expected):
        torch.testing.assert_close(got[..., 528:], expected[..., 528:])
        torch.testing.assert_close(
            got[..., 512:528].contiguous().view(torch.float32),
            expected[..., 512:528].contiguous().view(torch.float32),
            rtol=1e-6,
            atol=0,
        )
        torch.testing.assert_close(
            got[..., :512].contiguous().view(torch.float8_e4m3fn).float(),
            expected[..., :512].contiguous().view(torch.float8_e4m3fn).float(),
        )

    def test_sparse_prefill_decode_and_graph_use_selected_fp8(self):
        nope, rope = self._rows(192)
        cache = pack_nvfp4_rows(nope, rope).view(3, 64, MLA_ROW_BYTES)
        fp8_pages = nvfp4_mla_rows_to_fp8(cache).unsqueeze(2)
        table = torch.tensor([[2, 0], [1, 2]], dtype=torch.int32, device=self.device)
        row_to_request = torch.tensor(
            [0, 0, 1, 1], dtype=torch.int32, device=self.device
        )
        q = torch.randn((4, 64, 576), dtype=torch.bfloat16, device=self.device)
        topk = torch.arange(64, dtype=torch.int32, device=self.device).repeat(4, 1)
        topk[::2] += 51
        topk[:, 2] = -1
        topk[3].fill_(-1)
        op = flashmla_sparse_impl.SparseMlaFp8Op(
            64,
            512,
            64,
            128,
            64,
            1.0,
            64,
            use_cuda_graph=True,
            use_nvfp4_cache=True,
        )
        self._plan(op, row_to_request, table, SimpleNamespace(is_prefill=True))
        physical = op._convert_topk_indices_to_global(topk).squeeze(1)
        expected = self._ref_attention(q, fp8_pages, physical, op.scale)

        def forward():
            return op.forward(q, cache, topk)

        got = forward()
        torch.testing.assert_close(got, expected, rtol=0.02, atol=0.01)

        # Force multiple bounded chunks, with both full and tail chunks.
        with patch.object(flashmla_sparse_impl, "MAX_NVFP4_MLA_SELECTED_ROWS", 192):
            torch.testing.assert_close(forward(), expected, rtol=0.02, atol=0.01)

        warmup = torch.cuda.Stream()
        warmup.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(warmup):
            forward()
        torch.cuda.current_stream().wait_stream(warmup)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=warmup):
            captured = forward()

        # A replay must use updated selections rather than captured topk values.
        topk[:] = 3
        topk[1, 5:17] = -1
        physical = op._convert_topk_indices_to_global(topk).squeeze(1)
        expected = self._ref_attention(q, fp8_pages, physical, op.scale)
        graph.replay()
        torch.testing.assert_close(captured, expected, rtol=0.02, atol=0.01)

    def test_fp16_activation_and_pinned_working_set(self):
        nope, rope = self._rows(128)
        packed = pack_nvfp4_rows(nope, rope).view(2, 64, MLA_ROW_BYTES)
        pinned = torch.empty(packed.shape, dtype=torch.uint8, pin_memory=True)
        pinned.copy_(packed)
        work = PinnedMlaWorkingSet([pinned], 64, 64, self.device)
        logical = torch.full((1, 1, 64), -1, dtype=torch.int32, device=self.device)
        logical[0, 0, :3] = torch.tensor([127, 29, 103], device=self.device)
        physical = work.begin(logical)

        new_nope, new_rope = self._rows(1)
        scratch = torch.empty(
            (1, 1, MLA_ROW_BYTES), dtype=torch.uint8, device=self.device
        )
        write_nvfp4_mla_cache(
            new_nope.half(),
            new_rope.half(),
            scratch,
            torch.zeros((1,), dtype=torch.int64, device=self.device),
        )
        work.write(
            0,
            torch.tensor([29], dtype=torch.int32, device=self.device),
            scratch.flatten(0, 1),
        )
        packed[0, 29] = pack_nvfp4_rows(new_nope.half(), new_rope.half().bfloat16())[0]
        fp8 = nvfp4_mla_rows_to_fp8(packed).unsqueeze(2)
        q = torch.randn((1, 64, 576), dtype=torch.float16, device=self.device)
        op = flashmla_sparse_impl.SparseMlaFp8Op(
            64, 512, 64, 128, 64, 1.0, 64, use_nvfp4_cache=True
        )
        got = op.forward(q, work.layer_cache(0), logical, physical_indices=physical)
        want = self._ref_attention(q.bfloat16(), fp8, logical.squeeze(1), op.scale)
        self.assertEqual(got.dtype, torch.float16)
        torch.testing.assert_close(got, want.half(), rtol=0.02, atol=0.01)

    def test_cp_packed_padding_and_rank_major_index_mapping(self):
        nope, rope = self._rows(16)
        rows = pack_nvfp4_rows(nope, rope)
        cache = rows.view(8, 2, MLA_ROW_BYTES)
        local = gather_padded_nvfp4_mla_rows(
            cache,
            torch.tensor([[4, 6], [7, -1]], dtype=torch.int32, device=self.device),
            torch.tensor([3, 1], dtype=torch.int32, device=self.device),
            torch.tensor([4, 2], dtype=torch.int32, device=self.device),
            torch.tensor([0, 4], dtype=torch.int32, device=self.device),
            total_rows=6,
        )
        torch.testing.assert_close(local[[0, 1, 2, 4]], rows[[8, 9, 12, 14]])
        self.assertTrue(torch.all(local[[3, 5]] == 0).item())
        remote_rows = rows[:6]
        combined = torch.cat((local, remote_rows)).unsqueeze(1)
        op = flashmla_sparse_cp_impl.SparseMlaFp8CPOp(
            64, 512, 64, 128, 64, 1.0, 64, use_nvfp4_cache=True
        )
        op.precomputed_kv_offsets = torch.tensor(
            [0, 5], device=self.device, dtype=torch.int32
        )
        op.sharded_kv_restore_indices = torch.tensor(
            [0, 1, 2, 6, 7, 4, 10], device=self.device, dtype=torch.int32
        )
        q = torch.randn((2, 64, 576), device=self.device, dtype=torch.bfloat16)
        topk = torch.full((2, 64), -1, dtype=torch.int32, device=self.device)
        topk[0, :3] = torch.tensor([4, 0, 3], device=self.device)
        topk[1, :2] = torch.tensor([1, 0], device=self.device)
        expected_ids = topk.clone()
        expected_ids[0, :3] = torch.tensor([7, 0, 6], device=self.device)
        expected_ids[1, :2] = torch.tensor([10, 4], device=self.device)
        fp8_pages = torch.zeros(
            (1, 64, 1, FP8_MLA_ROW_BYTES), device=self.device, dtype=torch.uint8
        )
        fp8_pages[0, :12] = nvfp4_mla_rows_to_fp8(combined)
        want = self._ref_attention(q, fp8_pages, expected_ids, op.scale)
        result = op._attend_nvfp4_cp(q, None, topk, combined, None)
        torch.testing.assert_close(result, want, rtol=0.02, atol=0.01)


def _cp_nvfp4_worker(rank: int, port: int) -> None:
    device = torch.device("cuda", rank)
    torch.cuda.set_device(device)
    dist.init_process_group(
        "nccl",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=120),
        device_id=device,
    )
    try:
        # Exercise real NCCL, with the test world process group as the TP group.
        with patch.object(
            collective_torch, "_get_group", return_value=dist.group.WORLD
        ):
            op = flashmla_sparse_cp_impl.SparseMlaFp8CPOp(
                64, 512, 64, 128, 64, 1.0, 64, use_nvfp4_cache=True
            )
            op.attn_inputs = SimpleNamespace(is_prefill=True, cache_store_inputs=None)
            op.kv_cache_sharded = True
            op.block_table = torch.tensor([[1, 0]], dtype=torch.int32, device=device)
            op.sharded_actual_local_kv_lens = torch.tensor(
                [5 if rank == 0 else 3], dtype=torch.int32, device=device
            )
            op.sharded_local_kv_lens = torch.tensor(
                [8], dtype=torch.int32, device=device
            )
            op.sharded_workspace_starts = torch.tensor(
                [0], dtype=torch.int32, device=device
            )
            op.sharded_total_local_kv_len = 8
            op.sharded_kv_restore_indices = torch.tensor(
                [0, 1, 2, 3, 8, 9, 10, 4], dtype=torch.int32, device=device
            )
            op.total_local_ids = torch.arange(int(rank == 0), device=device)
            op.total_local_ids_is_identity = True
            op.precomputed_kv_offsets = torch.zeros(
                int(rank == 0), dtype=torch.int32, device=device
            )

            def page_rows(owner):
                generator = torch.Generator(device).manual_seed(320 + owner)
                nope = torch.randn(
                    (8, 512), generator=generator, device=device, dtype=torch.bfloat16
                )
                rope = torch.randn(
                    (8, 64), generator=generator, device=device, dtype=torch.bfloat16
                )
                return pack_nvfp4_rows(nope, rope)

            cache = SimpleNamespace(
                kv_cache_base=page_rows(rank).view(2, 4, MLA_ROW_BYTES)
            )
            q = torch.randn(
                (int(rank == 0), 64, 576), dtype=torch.bfloat16, device=device
            )
            topk = torch.full(
                (int(rank == 0), 64), -1, dtype=torch.int32, device=device
            )
            if rank == 0:
                topk[0, :4] = torch.tensor([7, 4, 0, 6], device=device)

            def forward():
                return op.forward(
                    q,
                    None,
                    None,
                    topk,
                    op.total_local_ids,
                    kv_cache=cache,
                    prepared_kv=cache.kv_cache_base,
                )

            def reference():
                expected_packed = torch.cat(
                    (page_rows(0)[[0, 4]], page_rows(1)[[4, 6]])
                )
                fp8 = torch.zeros(
                    (1, 64, 1, FP8_MLA_ROW_BYTES), dtype=torch.uint8, device=device
                )
                # Selected physical rows in top-k order: rank0/0, rank1/4,
                # rank0/4, rank1/6.
                fp8[0, :4, 0] = nvfp4_mla_rows_to_fp8(expected_packed[[0, 2, 1, 3]])
                reference_indices = torch.full_like(topk, -1)
                reference_indices[0, :4] = torch.arange(
                    4, device=device, dtype=torch.int32
                )
                expected, _ = flash_mla_with_kvcache(
                    q=q.unsqueeze(0),
                    k_cache=fp8,
                    block_table=None,
                    cache_seqlens=None,
                    head_dim_v=512,
                    tile_scheduler_metadata=get_mla_metadata()[0],
                    is_fp8_kvcache=True,
                    indices=reference_indices.unsqueeze(0),
                    softmax_scale=op.scale,
                )
                return expected.squeeze(0)

            got = forward()
            if rank == 0:
                torch.testing.assert_close(got, reference(), atol=0.01, rtol=0.02)
            else:
                assert got is None

            # Rank without queries must still participate during graph replay.
            stream = torch.cuda.Stream(device=device)
            stream.wait_stream(torch.cuda.current_stream(device))
            with torch.cuda.stream(stream):
                forward()
            stream.synchronize()
            dist.barrier()
            graph = torch.cuda.CUDAGraph()
            try:
                with torch.cuda.graph(graph, stream=stream):
                    got = forward()
                graph.replay()
                if rank == 0:
                    torch.testing.assert_close(got, reference(), atol=0.01, rtol=0.02)
                else:
                    assert got is None
                torch.cuda.synchronize(device)
            finally:
                # NCCL cannot tear down a communicator still held by a graph.
                del graph
    finally:
        dist.destroy_process_group()


class Nvfp4MlaCpCollectiveTest(TestCase):
    @skipUnless(torch.cuda.device_count() >= 2, "requires two CUDA devices")
    def test_sharded_prefill_collective_with_empty_query_rank(self):
        ports, locks = PortManager().get_consecutive_ports(1)
        try:
            mp.spawn(_cp_nvfp4_worker, args=(ports[0],), nprocs=2, join=True)
        finally:
            for lock in locks:
                lock.__exit__(None, None, None)


if __name__ == "__main__":
    main()
