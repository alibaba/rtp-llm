"""Correctness and CUDA Graph tests for TokenSpeed MLA decode."""

import os
from types import SimpleNamespace
from unittest import TestCase, main, mock, skipUnless

import torch

from rtp_llm.ops import KvCacheDataType, RopeConfig
from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.mla_kv_cache_write_op import (
    MlaKVCacheWriteOp,
)
from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.paged_mla_decode import (
    MLA_DECODE_KERNEL_ENV,
    PagedMlaDecodeMetadata,
)
from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.tokenspeed_mla_impl import (
    _TOKENSPEED_MLA_API,
    TokenSpeedMlaDecodeImpl,
    TokenSpeedMlaDecodeOp,
    tokenspeed_mla_kernel_supported,
)
from rtp_llm.utils.model_weight import W


def _is_blackwell() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 10


RUN_KERNEL = _is_blackwell() and _TOKENSPEED_MLA_API is not None
SKIP_REASON = "requires Blackwell GPU and tokenspeed-mla"


class TokenSpeedMlaDependencyTest(TestCase):
    def test_declared_blackwell_target_has_tokenspeed_api(self):
        self.assertTrue(torch.cuda.is_available(), "SM100 test target requires CUDA")
        self.assertTrue(_is_blackwell(), "test target requires SM100 or SM103")
        self.assertIsNotNone(
            _TOKENSPEED_MLA_API,
            "tokenspeed-mla must be present; the kernel tests must not pass by skip",
        )


class K3Tp8Geometry:
    """Kimi K3 geometry after attention TP sharding."""

    num_heads = 12
    kv_lora_rank = 512
    qk_rope_head_dim = 64
    qk_nope_head_dim = 128
    v_head_dim = 128
    page_size = 64

    @property
    def head_dim_qk(self) -> int:
        return self.kv_lora_rank + self.qk_rope_head_dim

    @property
    def scale(self) -> float:
        return (self.qk_nope_head_dim + self.qk_rope_head_dim) ** -0.5


class FakeMlaParams:
    def __init__(self, kv_lens, page_indptr, page_indices, device="cuda"):
        batch_size = len(kv_lens)
        self.qo_indptr_h = torch.arange(batch_size + 1, dtype=torch.int32)
        self.kvlen_h = torch.tensor(kv_lens, dtype=torch.int32)
        self.kvlen_d = self.kvlen_h.to(device=device)
        self.decode_page_indptr_h = torch.tensor(page_indptr, dtype=torch.int32)
        self.decode_page_indptr_d = self.decode_page_indptr_h.to(device=device)
        self.page_indice_d = torch.tensor(
            page_indices, dtype=torch.int32, device=device
        )


class FakeLayerKVCache:
    def __init__(self, kv_cache_base: torch.Tensor):
        self.kv_cache_base = kv_cache_base


def build_kv_layout(geo, kv_lens, num_pages, seed=0, page_stride_elems=None):
    torch.manual_seed(seed)
    logical_page_elems = geo.page_size * geo.head_dim_qk
    page_stride_elems = page_stride_elems or logical_page_elems
    if page_stride_elems < logical_page_elems:
        raise ValueError(
            f"page stride {page_stride_elems} is smaller than {logical_page_elems}"
        )
    if page_stride_elems == logical_page_elems:
        kv_cache = torch.empty(
            num_pages,
            geo.page_size,
            geo.head_dim_qk,
            dtype=torch.bfloat16,
            device="cuda",
        )
    else:
        storage = torch.empty(
            (num_pages - 1) * page_stride_elems + logical_page_elems,
            dtype=torch.bfloat16,
            device="cuda",
        )
        kv_cache = storage.as_strided(
            (num_pages, geo.page_size, geo.head_dim_qk),
            (page_stride_elems, geo.head_dim_qk, 1),
        )
    kv_cache.copy_(torch.randn_like(kv_cache) * 0.1)
    page_indptr = [0]
    page_indices = []
    next_page = 0
    for kv_len in kv_lens:
        num_blocks = (kv_len + geo.page_size - 1) // geo.page_size
        page_indices.extend(range(next_page, next_page + num_blocks))
        next_page += num_blocks
        page_indptr.append(len(page_indices))
    assert next_page <= num_pages
    return kv_cache, page_indptr, page_indices


def reference_mla_decode(
    q_nope,
    q_pe,
    kc_weight,
    vc_weight,
    kv_cache,
    kv_lens,
    page_indptr,
    page_indices,
    geo,
):
    q_latent = torch.bmm(q_nope.transpose(0, 1).float(), kc_weight.float()).transpose(
        0, 1
    )
    outputs = []
    for batch_id, kv_len in enumerate(kv_lens):
        pages = page_indices[page_indptr[batch_id] : page_indptr[batch_id + 1]]
        tokens = kv_cache[pages].reshape(-1, geo.head_dim_qk)[:kv_len]
        compressed_kv = tokens[:, : geo.kv_lora_rank].float()
        rope = tokens[:, geo.kv_lora_rank :].float()
        scores = (
            q_latent[batch_id] @ compressed_kv.T + q_pe[batch_id].float() @ rope.T
        ) * geo.scale
        outputs.append(torch.softmax(scores, dim=-1) @ compressed_kv)
    attention = torch.stack(outputs).to(torch.bfloat16)
    return (
        torch.bmm(attention.transpose(0, 1).float(), vc_weight.float())
        .transpose(0, 1)
        .to(torch.bfloat16)
    )


def make_op(geo, max_bs=0, max_context_len=0, is_cuda_graph=False):
    torch.manual_seed(42)
    kc_weight = (
        torch.randn(
            geo.num_heads, geo.qk_nope_head_dim, geo.kv_lora_rank, device="cuda"
        )
        * 0.02
    ).to(torch.bfloat16)
    vc_weight = (
        torch.randn(geo.num_heads, geo.kv_lora_rank, geo.v_head_dim, device="cuda")
        * 0.02
    ).to(torch.bfloat16)
    op = TokenSpeedMlaDecodeOp(
        geo.num_heads,
        geo.kv_lora_rank,
        geo.qk_rope_head_dim,
        geo.qk_nope_head_dim,
        geo.page_size,
        1.0,
        [{W.mla_kc: kc_weight, W.mla_vc: vc_weight}],
        max_bs=max_bs,
        max_context_len=max_context_len,
        is_cuda_graph=is_cuda_graph,
    )
    return op, kc_weight, vc_weight


def run_case(test, geo, kv_lens, num_pages):
    kv_cache, page_indptr, page_indices = build_kv_layout(geo, kv_lens, num_pages)
    op, kc_weight, vc_weight = make_op(geo)
    batch_size = len(kv_lens)
    q_nope = (
        torch.randn(batch_size, geo.num_heads, geo.qk_nope_head_dim, device="cuda")
        * 0.5
    ).to(torch.bfloat16)
    q_pe = (
        torch.randn(batch_size, geo.num_heads, geo.qk_rope_head_dim, device="cuda")
        * 0.5
    ).to(torch.bfloat16)
    op.plan(FakeMlaParams(kv_lens, page_indptr, page_indices))
    actual = op.forward(q_nope, q_pe, FakeLayerKVCache(kv_cache), 0)
    expected = reference_mla_decode(
        q_nope,
        q_pe,
        kc_weight,
        vc_weight,
        kv_cache,
        kv_lens,
        page_indptr,
        page_indices,
        geo,
    )
    relative_error = (
        (actual.float() - expected.float()).abs().max() / expected.float().abs().max()
    ).item()
    test.assertLess(relative_error, 2e-2)
    return op


class PagedMlaDecodeMetadataTest(TestCase):
    def test_keeps_physical_page_ids_without_expansion(self):
        metadata = PagedMlaDecodeMetadata(64, 64, 0, 0, False, torch.device("cpu"))
        params = FakeMlaParams([65, 129], [0, 2, 5], [3, 5, 9, 10, 11], device="cpu")
        metadata.plan(params)
        self.assertEqual(metadata.padded_blocks, 3)
        torch.testing.assert_close(
            metadata.block_tables,
            torch.tensor([[3, 5, 0], [9, 10, 11]], dtype=torch.int32),
            rtol=0,
            atol=0,
        )

    def test_graph_refresh_preserves_metadata_addresses(self):
        metadata = PagedMlaDecodeMetadata(64, 64, 2, 192, True, torch.device("cpu"))
        params = FakeMlaParams([64, 65], [0, 1, 3], [0, 3, 4], device="cpu")
        metadata.plan(params)
        table_ptr = metadata.block_tables.data_ptr()
        lengths_ptr = metadata.seq_lens.data_ptr()
        physical_table = torch.tensor([[7, 8, 9], [11, 12, 13]], dtype=torch.int32)
        metadata.refresh_cuda_graph(
            physical_table, torch.tensor([65, 129], dtype=torch.int32)
        )
        self.assertEqual(metadata.block_tables.data_ptr(), table_ptr)
        self.assertEqual(metadata.seq_lens.data_ptr(), lengths_ptr)
        torch.testing.assert_close(
            metadata.block_tables,
            torch.tensor([[7, 8, 0], [11, 12, 13]], dtype=torch.int32),
            rtol=0,
            atol=0,
        )

    def test_zero_length_rows_do_not_alias_live_page_ids(self):
        metadata = PagedMlaDecodeMetadata(64, 64, 0, 0, False, torch.device("cpu"))
        params = FakeMlaParams(
            [0, 65, 0], [0, 0, 2, 2], [7, 9], device="cpu"
        )
        metadata.plan(params)
        torch.testing.assert_close(
            metadata.block_tables,
            torch.tensor([[0, 0], [7, 9], [0, 0]], dtype=torch.int32),
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            metadata.seq_lens,
            torch.tensor([0, 65, 0], dtype=torch.int32),
            rtol=0,
            atol=0,
        )


@skipUnless(RUN_KERNEL, SKIP_REASON)
class TokenSpeedMlaDecodeOpTest(TestCase):
    def setUp(self):
        torch.cuda.set_device(0)
        self.geo = K3Tp8Geometry()

    def test_k3_tp8_single_request(self):
        run_case(self, self.geo, [384], num_pages=8)

    def test_k3_tp8_variable_batch(self):
        op = run_case(self, self.geo, [65, 512, 129, 1000], num_pages=32)
        self.assertEqual(op._padded_blocks, 16)

    def test_cuda_graph_group_refresh_across_page_boundaries(self):
        geo = self.geo
        batch_size = 2
        max_context_len = 192
        blocks_per_request = max_context_len // geo.page_size
        op, kc_weight, vc_weight = make_op(
            geo,
            max_bs=batch_size,
            max_context_len=max_context_len,
            is_cuda_graph=True,
        )
        kv_cache, _, _ = build_kv_layout(
            geo, [max_context_len] * batch_size, batch_size * blocks_per_request, 7
        )
        q_nope = (
            torch.randn(batch_size, geo.num_heads, geo.qk_nope_head_dim, device="cuda")
            * 0.5
        ).to(torch.bfloat16)
        q_pe = (
            torch.randn(batch_size, geo.num_heads, geo.qk_rope_head_dim, device="cuda")
            * 0.5
        ).to(torch.bfloat16)
        physical_table = torch.tensor(
            [[0, 1, 2], [3, 4, 5]], dtype=torch.int32, device="cuda"
        )

        def compact_layout(kv_lens):
            page_indptr = [0]
            page_indices = []
            for batch_id, kv_len in enumerate(kv_lens):
                live_blocks = (kv_len + geo.page_size - 1) // geo.page_size
                page_indices.extend(
                    range(
                        batch_id * blocks_per_request,
                        batch_id * blocks_per_request + live_blocks,
                    )
                )
                page_indptr.append(len(page_indices))
            return page_indptr, page_indices

        initial_lens = [64, 65]
        initial_indptr, initial_indices = compact_layout(initial_lens)
        op.plan(FakeMlaParams(initial_lens, initial_indptr, initial_indices))
        graph_lengths = torch.tensor(initial_lens, dtype=torch.int32, device="cuda")
        op._metadata.refresh_cuda_graph(physical_table, graph_lengths)
        op.forward(q_nope, q_pe, FakeLayerKVCache(kv_cache), 0)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            op._metadata.refresh_cuda_graph(physical_table, graph_lengths)
            output = op.forward(q_nope, q_pe, FakeLayerKVCache(kv_cache), 0)

        replay_lens = [65, 129]
        graph_lengths.copy_(torch.tensor(replay_lens, dtype=torch.int32, device="cuda"))
        graph.replay()
        torch.cuda.synchronize()
        replay_indptr, replay_indices = compact_layout(replay_lens)
        expected = reference_mla_decode(
            q_nope,
            q_pe,
            kc_weight,
            vc_weight,
            kv_cache,
            replay_lens,
            replay_indptr,
            replay_indices,
            geo,
        )
        relative_error = (
            (output.float() - expected.float()).abs().max()
            / expected.float().abs().max()
        ).item()
        self.assertLess(relative_error, 2e-2)

    def test_cuda_graph_page128_across_long_sequence_boundaries(self):
        geo = self.geo
        geo.page_size = 128
        batch_size = 1
        max_context_len = 65536
        tested_lens = (4095, 4096, 4097, 12287, 12288, 12289)
        live_pages = (max(tested_lens) + geo.page_size - 1) // geo.page_size
        graph_pages = max_context_len // geo.page_size
        op, kc_weight, vc_weight = make_op(
            geo,
            max_bs=batch_size,
            max_context_len=max_context_len,
            is_cuda_graph=True,
        )
        # RTP reserves zero as an empty block-table entry and physical pages are
        # not generally contiguous. Use odd positive IDs so the test exercises
        # the same indirection instead of accidentally treating cache offsets as
        # page-table positions.
        physical_page_ids = list(range(1, 2 * live_pages, 2))
        # K3 HybridCache's 4096-token physical slot is sized by the larger KDA
        # state. Its 32 MLA kernel pages therefore have a 101760-element BF16
        # stride instead of the compact 128 * 576 stride.
        hybrid_page_stride_elems = 6512640 // 2 // 32
        kv_cache, _, _ = build_kv_layout(
            geo,
            [2 * max(tested_lens)],
            2 * live_pages,
            seed=19,
            page_stride_elems=hybrid_page_stride_elems,
        )
        self.assertEqual(kv_cache.stride(0), hybrid_page_stride_elems)
        q_nope = (
            torch.randn(batch_size, geo.num_heads, geo.qk_nope_head_dim, device="cuda")
            * 0.5
        ).to(torch.bfloat16)
        q_pe = (
            torch.randn(batch_size, geo.num_heads, geo.qk_rope_head_dim, device="cuda")
            * 0.5
        ).to(torch.bfloat16)
        physical_table = torch.zeros(
            (batch_size, graph_pages), dtype=torch.int32, device="cuda"
        )
        initially_materialized_pages = (
            tested_lens[0] + geo.page_size - 1
        ) // geo.page_size
        physical_table[0, :initially_materialized_pages] = torch.tensor(
            physical_page_ids[:initially_materialized_pages],
            dtype=torch.int32,
            device="cuda",
        )

        def compact_layout(kv_len):
            num_pages = (kv_len + geo.page_size - 1) // geo.page_size
            return [0, num_pages], physical_page_ids[:num_pages]

        initial_len = tested_lens[0]
        initial_indptr, initial_indices = compact_layout(initial_len)
        op.plan(FakeMlaParams([initial_len], initial_indptr, initial_indices))
        graph_lengths = torch.tensor(
            [initial_len], dtype=torch.int32, device="cuda"
        )
        op._metadata.refresh_cuda_graph(physical_table, graph_lengths)
        op.forward(q_nope, q_pe, FakeLayerKVCache(kv_cache), 0)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            op._metadata.refresh_cuda_graph(physical_table, graph_lengths)
            output = op.forward(q_nope, q_pe, FakeLayerKVCache(kv_cache), 0)

        for replay_len in tested_lens:
            replay_pages = (replay_len + geo.page_size - 1) // geo.page_size
            physical_table[0, :replay_pages].copy_(
                torch.tensor(
                    physical_page_ids[:replay_pages],
                    dtype=torch.int32,
                    device="cuda",
                )
            )
            graph_lengths.copy_(
                torch.tensor([replay_len], dtype=torch.int32, device="cuda")
            )
            graph.replay()
            torch.cuda.synchronize()
            replay_indptr, replay_indices = compact_layout(replay_len)
            expected = reference_mla_decode(
                q_nope,
                q_pe,
                kc_weight,
                vc_weight,
                kv_cache,
                [replay_len],
                replay_indptr,
                replay_indices,
                geo,
            )
            self.assertTrue(torch.isfinite(output).all(), f"length={replay_len}")
            relative_error = (
                (output.float() - expected.float()).abs().max()
                / expected.float().abs().max()
            ).item()
            self.assertLess(
                relative_error, 2e-2, f"length={replay_len}, rel_err={relative_error}"
            )

    def test_cuda_graph_page128_refreshes_multiple_groups_before_kernel(self):
        geo = self.geo
        geo.page_size = 128
        batch_size = 1
        group_count = 3
        max_context_len = 65536
        old_len = 12288
        new_len = 12289
        live_pages = (new_len + geo.page_size - 1) // geo.page_size
        graph_pages = max_context_len // geo.page_size
        total_pages = group_count * live_pages + 1
        op, kc_weight, vc_weight = make_op(
            geo,
            max_bs=batch_size,
            max_context_len=max_context_len,
            is_cuda_graph=True,
        )
        kv_cache, _, _ = build_kv_layout(
            geo, [total_pages * geo.page_size], total_pages, seed=29
        )
        q_nope = (
            torch.randn(batch_size, geo.num_heads, geo.qk_nope_head_dim, device="cuda")
            * 0.5
        ).to(torch.bfloat16)
        q_pe = (
            torch.randn(batch_size, geo.num_heads, geo.qk_rope_head_dim, device="cuda")
            * 0.5
        ).to(torch.bfloat16)

        group_page_ids = [
            list(range(1 + group_id * live_pages, 1 + (group_id + 1) * live_pages))
            for group_id in range(group_count)
        ]
        old_pages = old_len // geo.page_size
        group_tables = torch.zeros(
            (group_count, batch_size, graph_pages),
            dtype=torch.int32,
            device="cuda",
        )
        for group_id, page_ids in enumerate(group_page_ids):
            group_tables[group_id, 0, :old_pages] = torch.tensor(
                page_ids[:old_pages], dtype=torch.int32, device="cuda"
            )

        op.plan(FakeMlaParams([old_len], [0, old_pages], group_page_ids[0][:old_pages]))
        graph_lengths = torch.tensor([old_len], dtype=torch.int32, device="cuda")
        op._metadata.refresh_cuda_graph(group_tables[0], graph_lengths)
        op.forward(q_nope, q_pe, FakeLayerKVCache(kv_cache), 0)

        graph_outputs = []
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for group_id in range(group_count):
                op._metadata.refresh_cuda_graph(group_tables[group_id], graph_lengths)
                graph_outputs.append(
                    op.forward(q_nope, q_pe, FakeLayerKVCache(kv_cache), 0)
                )

        for group_id, page_ids in enumerate(group_page_ids):
            group_tables[group_id, 0, old_pages] = page_ids[old_pages]
        graph_lengths.fill_(new_len)
        graph.replay()
        torch.cuda.synchronize()

        for group_id, output in enumerate(graph_outputs):
            expected = reference_mla_decode(
                q_nope,
                q_pe,
                kc_weight,
                vc_weight,
                kv_cache,
                [new_len],
                [0, live_pages],
                group_page_ids[group_id],
                geo,
            )
            self.assertTrue(torch.isfinite(output).all(), f"group={group_id}")
            relative_error = (
                (output.float() - expected.float()).abs().max()
                / expected.float().abs().max()
            ).item()
            self.assertLess(
                relative_error,
                2e-2,
                f"group={group_id}, rel_err={relative_error}",
            )

    def test_cuda_graph_masks_unwritten_new_page_after_graph_kv_write(self):
        geo = self.geo
        geo.page_size = 128
        batch_size = 1
        old_len = 12288
        new_len = old_len + 1
        # Kimi K3 has 93 KV layers. Round up so a shared TokenSpeed workspace
        # is exercised for at least one full model decode step.
        layer_count = 96
        old_pages = old_len // geo.page_size
        live_pages = old_pages + 1
        max_context_len = 65536
        graph_pages = max_context_len // geo.page_size
        op, kc_weight, vc_weight = make_op(
            geo,
            max_bs=batch_size,
            max_context_len=max_context_len,
            is_cuda_graph=True,
        )

        physical_page_ids = list(range(1, 2 * live_pages, 2))
        hybrid_page_stride_elems = 6512640 // 2 // 32
        kv_cache, _, _ = build_kv_layout(
            geo,
            [2 * new_len],
            2 * live_pages,
            seed=37,
            page_stride_elems=hybrid_page_stride_elems,
        )
        new_page_id = physical_page_ids[-1]
        kv_cache[new_page_id].fill_(float("nan"))
        new_kv = torch.randn(
            geo.head_dim_qk, dtype=torch.bfloat16, device="cuda"
        )
        write_op = MlaKVCacheWriteOp(
            KvCacheDataType.BASE, clear_page_on_boundary=True
        )
        write_slot_mapping = torch.tensor(
            [physical_page_ids[0] * geo.page_size + 1],
            dtype=torch.int64,
            device="cuda",
        )
        write_params = SimpleNamespace(slot_mapping=write_slot_mapping)
        q_nope = (
            torch.randn(batch_size, geo.num_heads, geo.qk_nope_head_dim, device="cuda")
            * 0.5
        ).to(torch.bfloat16)
        q_pe = (
            torch.randn(batch_size, geo.num_heads, geo.qk_rope_head_dim, device="cuda")
            * 0.5
        ).to(torch.bfloat16)
        physical_table = torch.zeros(
            (batch_size, graph_pages), dtype=torch.int32, device="cuda"
        )
        physical_table[0, :old_pages] = torch.tensor(
            physical_page_ids[:old_pages], dtype=torch.int32, device="cuda"
        )
        graph_lengths = torch.tensor([old_len], dtype=torch.int32, device="cuda")
        op.plan(
            FakeMlaParams(
                [old_len], [0, old_pages], physical_page_ids[:old_pages]
            )
        )
        op._metadata.refresh_cuda_graph(physical_table, graph_lengths)
        op.forward(q_nope, q_pe, FakeLayerKVCache(kv_cache), 0)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            write_op.forward(
                new_kv[: geo.kv_lora_rank].unsqueeze(0),
                new_kv[geo.kv_lora_rank :].unsqueeze(0),
                FakeLayerKVCache(kv_cache),
                write_params,
            )
            op._metadata.refresh_cuda_graph(physical_table, graph_lengths)
            output = q_nope
            for _ in range(layer_count):
                output = op.forward(output, q_pe, FakeLayerKVCache(kv_cache), 0)

        kv_cache[new_page_id].fill_(float("nan"))
        write_slot_mapping.fill_(new_page_id * geo.page_size)
        physical_table[0, old_pages] = new_page_id
        graph_lengths.fill_(new_len)
        graph.replay()
        torch.cuda.synchronize()

        expected = q_nope
        for _ in range(layer_count):
            expected = reference_mla_decode(
                expected,
                q_pe,
                kc_weight,
                vc_weight,
                kv_cache,
                [new_len],
                [0, live_pages],
                physical_page_ids,
                geo,
            )
        self.assertTrue(torch.isfinite(expected).all())
        self.assertTrue(torch.isfinite(output).all())
        relative_error = (
            (output.float() - expected.float()).abs().max()
            / expected.float().abs().max()
        ).item()
        self.assertLess(relative_error, 2e-2, f"rel_err={relative_error}")


class TokenSpeedMlaDecodeSupportTest(TestCase):
    def _configs(self):
        from rtp_llm.ops import AttentionConfigs, KvCacheDataType

        configs = AttentionConfigs()
        configs.use_mla = True
        configs.is_sparse = False
        configs.kv_cache_dtype = KvCacheDataType.BASE
        configs.head_num = 12
        configs.kv_lora_rank = 512
        configs.rope_head_dim = 64
        configs.kernel_tokens_per_block = 64
        return configs

    def test_static_geometry_uses_local_heads(self):
        self.assertTrue(tokenspeed_mla_kernel_supported(12, 512, 64, 64))
        self.assertTrue(tokenspeed_mla_kernel_supported(12, 512, 64, 128))
        self.assertTrue(tokenspeed_mla_kernel_supported(96, 512, 64, 64))
        self.assertFalse(tokenspeed_mla_kernel_supported(0, 512, 64, 64))
        self.assertFalse(tokenspeed_mla_kernel_supported(129, 512, 64, 64))

    def test_disabled_by_default(self):
        with mock.patch.dict(os.environ):
            os.environ.pop(MLA_DECODE_KERNEL_ENV, None)
            self.assertFalse(
                TokenSpeedMlaDecodeImpl.support(
                    self._configs(), SimpleNamespace(is_prefill=False)
                )
            )

    def test_env_gating(self):
        with mock.patch.dict(
            os.environ, {MLA_DECODE_KERNEL_ENV: "tokenspeed_mla"}
        ), mock.patch(
            "rtp_llm.models_py.modules.factory.attention.cuda_mla_impl."
            "tokenspeed_mla_impl._TOKENSPEED_MLA_API",
            object(),
        ), mock.patch(
            "rtp_llm.models_py.modules.factory.attention.cuda_mla_impl."
            "tokenspeed_mla_impl._is_tokenspeed_blackwell",
            return_value=True,
        ):
            self.assertTrue(
                TokenSpeedMlaDecodeImpl.support(
                    self._configs(), SimpleNamespace(is_prefill=False)
                )
            )

    def test_explicit_selection_fails_when_package_is_missing(self):
        with mock.patch.dict(
            os.environ, {MLA_DECODE_KERNEL_ENV: "tokenspeed_mla"}
        ), mock.patch(
            "rtp_llm.models_py.modules.factory.attention.cuda_mla_impl."
            "tokenspeed_mla_impl._TOKENSPEED_MLA_API",
            None,
        ):
            with self.assertRaisesRegex(RuntimeError, "requires tokenspeed-mla"):
                TokenSpeedMlaDecodeImpl.support(
                    self._configs(), SimpleNamespace(is_prefill=False)
                )

    def test_rejects_prefill_sparse_arch_and_geometry(self):
        with mock.patch.dict(
            os.environ, {MLA_DECODE_KERNEL_ENV: "tokenspeed_mla"}
        ), mock.patch(
            "rtp_llm.models_py.modules.factory.attention.cuda_mla_impl."
            "tokenspeed_mla_impl._TOKENSPEED_MLA_API",
            object(),
        ):
            with mock.patch(
                "rtp_llm.models_py.modules.factory.attention.cuda_mla_impl."
                "tokenspeed_mla_impl._is_tokenspeed_blackwell",
                return_value=True,
            ):
                self.assertFalse(
                    TokenSpeedMlaDecodeImpl.support(
                        self._configs(), SimpleNamespace(is_prefill=True)
                    )
                )
                configs = self._configs()
                configs.is_sparse = True
                with self.assertRaisesRegex(RuntimeError, "does not support sparse MLA"):
                    TokenSpeedMlaDecodeImpl.support(
                        configs, SimpleNamespace(is_prefill=False)
                    )
                configs = self._configs()
                configs.kernel_tokens_per_block = 96
                with self.assertRaisesRegex(RuntimeError, "does not support geometry"):
                    TokenSpeedMlaDecodeImpl.support(
                        configs, SimpleNamespace(is_prefill=False)
                    )
            with mock.patch(
                "rtp_llm.models_py.modules.factory.attention.cuda_mla_impl."
                "tokenspeed_mla_impl._is_tokenspeed_blackwell",
                return_value=False,
            ):
                with self.assertRaisesRegex(RuntimeError, "requires SM100 or SM103"):
                    TokenSpeedMlaDecodeImpl.support(
                        self._configs(), SimpleNamespace(is_prefill=False)
                    )

    def test_impl_clears_new_pages_only_for_cuda_graph(self):
        configs = self._configs()
        configs.nope_head_dim = 128
        configs.softmax_extra_scale = 1.0
        configs.rope_config = RopeConfig()
        configs.rope_config.is_neox_style = False
        attn_inputs = SimpleNamespace(sequence_lengths=torch.empty(0))

        module = (
            "rtp_llm.models_py.modules.factory.attention.cuda_mla_impl."
            "tokenspeed_mla_impl"
        )
        with mock.patch(f"{module}.TokenSpeedMlaDecodeOp"), mock.patch(
            f"{module}.NewMlaRotaryEmbeddingOp"
        ), mock.patch(f"{module}.MlaKVCacheWriteOp") as write_op_cls, mock.patch(
            f"{module}.MlaFlashInferImplBase.__init__", return_value=None
        ):
            for is_cuda_graph in (False, True):
                TokenSpeedMlaDecodeImpl(
                    configs,
                    attn_inputs,
                    weights=[],
                    cos_sin_cache=torch.empty(0),
                    is_cuda_graph=is_cuda_graph,
                )
                self.assertEqual(
                    write_op_cls.call_args.kwargs["clear_page_on_boundary"],
                    is_cuda_graph,
                )


if __name__ == "__main__":
    main()
