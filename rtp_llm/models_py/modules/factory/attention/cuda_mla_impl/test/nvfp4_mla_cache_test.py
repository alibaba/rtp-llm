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
        # Exercise (0x3f9c, 0x3f6a) boundary where div_rn vs PyTorch CUDA's
        # tensor / 2688.0 reciprocal multiply flips E4M3 block scale 0x7b vs 0x7a.
        nope[3].zero_()
        nope[3, :16] = 1.21875
        nope[3, 16:32] = 0.9140625
        rope[3].zero_()
        rope[3, 0:32:2] = 1.21875
        rope[3, 32:64:2] = 0.9140625
        slots = torch.tensor([0, 63, 65, 191, -1], device=self.device)
        write_nvfp4_mla_cache(nope, rope, cache, slots)
        self.assertTrue(torch.all(backing[:, 64] == 0xAA).item())
        self.assertTrue(torch.all(backing[0, 1] == 0xAA).item())
        expected_pack = pack_nvfp4_rows(nope, rope)
        torch.testing.assert_close(
            torch.stack([cache[0, 0], cache[0, 63], cache[1, 1], cache[2, 63]]),
            expected_pack[:4],
            rtol=0,
            atol=0,
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

    def test_interleaved_rope_matches_training_deinterleaved_nvfp4_blocks(self):
        from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.rope_emb_new import (
            NewMlaRotaryEmbeddingOp,
        )

        tokens, heads = 6, 4
        inv_freq = 1.0 / (
            8_000_000.0 ** (torch.arange(0, 64, 2, dtype=torch.float32, device=self.device) / 64.0)
        )
        pos = torch.arange(tokens, dtype=torch.int32, device=self.device)
        freqs = torch.outer(pos.float(), inv_freq)
        cos_sin_cache = torch.cat((torch.cos(freqs), torch.sin(freqs)), dim=-1).contiguous()

        q_raw = torch.randn((tokens, heads, 64), device=self.device, dtype=torch.bfloat16)
        k_raw = torch.randn((tokens, 64), device=self.device, dtype=torch.bfloat16)
        # Exaggerate odd vs even pre-RoPE channels so block-16 NVFP4 grouping is decisive.
        k_raw[:, 0::2] *= 0.05
        k_raw[:, 1::2] *= 32.0
        nope = torch.randn((tokens, 512), device=self.device, dtype=torch.bfloat16)

        # 1. Exact bit-level parity against FP32-arithmetic half-split RoPE (matching FlashInfer's FP32 sin/cos rotation).
        cos_fp32 = torch.cat((torch.cos(freqs), torch.cos(freqs)), dim=-1)
        sin_fp32 = torch.cat((torch.sin(freqs), torch.sin(freqs)), dim=-1)

        def ref_rope_fp32(x):
            x_de = torch.cat((x[..., 0::2], x[..., 1::2]), dim=-1)
            x_rot = torch.cat((-x_de[..., 32:], x_de[..., :32]), dim=-1)
            c = cos_fp32.unsqueeze(1) if x.ndim == 3 else cos_fp32
            s = sin_fp32.unsqueeze(1) if x.ndim == 3 else sin_fp32
            return (x_de.float() * c + x_rot.float() * s).to(x.dtype)

        q_ref = ref_rope_fp32(q_raw)
        k_ref = ref_rope_fp32(k_raw)
        k_ref_q = pack_nvfp4_rows(nope, k_ref, deinterleave_rope=False)
        k_ref_unpacked = nvfp4_mla_rows_to_fp8(k_ref_q, deinterleave_rope=False)[
            :, 528:
        ].view(torch.bfloat16)
        ref_dots = torch.einsum("thd,sd->ths", q_ref.float(), k_ref_unpacked.float())

        # RTP-LLM runtime path: FlashInfer interleave=True -> write_nvfp4_mla_cache -> gather_selected_mla_fp8.
        q_rt = q_raw.clone()
        k_rt = k_raw.clone()
        rope_op = NewMlaRotaryEmbeddingOp(cos_sin_cache=cos_sin_cache, is_neox_style=False)
        rope_op.forward(q_rt, k_rt, SimpleNamespace(positions_d=pos))
        cache = torch.zeros((1, 64, MLA_ROW_BYTES), dtype=torch.uint8, device=self.device)
        write_nvfp4_mla_cache(
            nope,
            k_rt,
            cache,
            torch.arange(tokens, dtype=torch.int64, device=self.device),
            deinterleave_rope=True,
        )
        fp8_staged, _ = gather_selected_mla_fp8(
            cache,
            torch.arange(tokens, dtype=torch.int32, device=self.device).unsqueeze(0),
            deinterleave_rope=True,
        )
        k_rt_unpacked = fp8_staged[0, :tokens, 0, 528:].contiguous().view(torch.bfloat16)
        torch.testing.assert_close(
            torch.cat((k_rt_unpacked[:, 0::2], k_rt_unpacked[:, 1::2]), dim=-1),
            k_ref_unpacked,
            rtol=0,
            atol=0,
        )
        rt_dots = torch.einsum("thd,sd->ths", q_rt.float(), k_rt_unpacked.float())
        torch.testing.assert_close(rt_dots, ref_dots, rtol=1e-5, atol=1e-5)

        # 2. Also verify alignment against Megatron-Bridge BF16-arithmetic RoPE and confirm
        # deinterleave_rope=True strictly prevents cross-channel NVFP4 block scale pollution.
        cos_bf16 = cos_fp32.to(torch.bfloat16)
        sin_bf16 = sin_fp32.to(torch.bfloat16)

        def ref_rope_bf16(x):
            x_de = torch.cat((x[..., 0::2], x[..., 1::2]), dim=-1)
            x_rot = torch.cat((-x_de[..., 32:], x_de[..., :32]), dim=-1)
            c = cos_bf16.unsqueeze(1) if x.ndim == 3 else cos_bf16
            s = sin_bf16.unsqueeze(1) if x.ndim == 3 else sin_bf16
            return x_de * c + x_rot * s

        q_mb = ref_rope_bf16(q_raw)
        k_mb = ref_rope_bf16(k_raw)
        k_mb_unpacked = nvfp4_mla_rows_to_fp8(
            pack_nvfp4_rows(nope, k_mb, deinterleave_rope=False), deinterleave_rope=False
        )[:, 528:].view(torch.bfloat16)
        mb_dots = torch.einsum("thd,sd->ths", q_mb.float(), k_mb_unpacked.float())

        wrong_staged = nvfp4_mla_rows_to_fp8(
            pack_nvfp4_rows(nope, k_rt, deinterleave_rope=False), deinterleave_rope=False
        )[:, 528:].view(torch.bfloat16)
        wrong_dots = torch.einsum("thd,sd->ths", q_rt.float(), wrong_staged.float())

        err_deinterleaved = (rt_dots - mb_dots).norm().item()
        err_interleaved_bug = (wrong_dots - mb_dots).norm().item()
        cos_sim = torch.nn.functional.cosine_similarity(
            rt_dots.flatten(), mb_dots.flatten(), dim=0
        ).item()
        self.assertGreater(cos_sim, 0.999)
        self.assertLess(err_deinterleaved, err_interleaved_bug * 0.5)

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

        op_no_deinterleave = flashmla_sparse_cp_impl.SparseMlaFp8CPOp(
            64,
            512,
            64,
            128,
            64,
            1.0,
            64,
            use_nvfp4_cache=True,
            deinterleave_rope=False,
        )
        self.assertFalse(op_no_deinterleave.deinterleave_rope)
        self.assertTrue(op.deinterleave_rope)


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
