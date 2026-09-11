"""Correctness and CUDA Graph tests for TokenSpeed MLA decode."""

import math
import os
from types import SimpleNamespace
from unittest import TestCase, main, mock, skipUnless

import torch

from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl import (
    tokenspeed_mla_impl,
)
from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.flashinfer_mla_wrapper import (
    decode_query_length,
)
from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.mla_kv_cache_write_op import (
    MlaKVCacheWriteOp,
)
from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.tokenspeed_mla_impl import (
    TokenSpeedMlaDecodeImpl,
    TokenSpeedMlaDecodeOp,
    _load_tokenspeed_mla,
    _TokenSpeedDecodeMetadata,
    tokenspeed_mla_kernel_supported,
)
from rtp_llm.ops import KvCacheDataType, RopeConfig
from rtp_llm.ops.compute_ops import rtp_llm_ops
from rtp_llm.utils.model_weight import W


def _is_blackwell() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 10


RUN_KERNEL = _is_blackwell() and _load_tokenspeed_mla()
SKIP_REASON = "requires Blackwell GPU and tokenspeed-mla"


class TokenSpeedMlaDependencyTest(TestCase):
    def test_declared_blackwell_target_has_tokenspeed_api(self):
        self.assertTrue(torch.cuda.is_available(), "SM100 test target requires CUDA")
        self.assertTrue(_is_blackwell(), "test target requires SM100 or SM103")
        self.assertIsNotNone(
            tokenspeed_mla_impl._TOKENSPEED_MLA_API,
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
    batch_size = len(kv_lens)
    if q_nope.size(0) % batch_size != 0:
        raise ValueError("query tokens must be divisible by batch size")
    q_len = q_nope.size(0) // batch_size
    q_nope = q_nope.view(batch_size, q_len, geo.num_heads, geo.qk_nope_head_dim)
    q_pe = q_pe.view(batch_size, q_len, geo.num_heads, geo.qk_rope_head_dim)
    q_latent = torch.einsum("bqhd,hdl->bqhl", q_nope.float(), kc_weight.float())
    outputs = []
    for batch_id, kv_len in enumerate(kv_lens):
        pages = page_indices[page_indptr[batch_id] : page_indptr[batch_id + 1]]
        tokens = kv_cache[pages].reshape(-1, geo.head_dim_qk)[:kv_len]
        compressed_kv = tokens[:, : geo.kv_lora_rank].float()
        rope = tokens[:, geo.kv_lora_rank :].float()
        scores = torch.einsum(
            "qhl,kl->qhk", q_latent[batch_id], compressed_kv
        ) + torch.einsum("qhr,kr->qhk", q_pe[batch_id].float(), rope)
        scores *= geo.scale
        query_positions = kv_len - q_len + torch.arange(q_len, device=scores.device)
        key_positions = torch.arange(kv_len, device=scores.device)
        scores.masked_fill_(
            key_positions.view(1, 1, -1) > query_positions.view(-1, 1, 1),
            float("-inf"),
        )
        outputs.append(
            torch.einsum("qhk,kl->qhl", torch.softmax(scores, -1), compressed_kv)
        )
    attention = torch.stack(outputs).to(q_nope.dtype)
    return (
        torch.einsum("bqhl,hlv->bqhv", attention.float(), vc_weight.float())
        .reshape(batch_size * q_len, geo.num_heads, geo.v_head_dim)
        .to(q_nope.dtype)
    )


def make_op(geo, max_bs=0, max_q_len=1, max_context_len=0, is_cuda_graph=False):
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
        max_q_len=max_q_len,
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


def run_multi_query_case(test, geo, kv_lens, q_len, is_cuda_graph):
    batch_size = len(kv_lens)
    num_pages = sum((kv_len + geo.page_size - 1) // geo.page_size for kv_len in kv_lens)
    kv_cache, page_indptr, page_indices = build_kv_layout(
        geo, kv_lens, num_pages, seed=53
    )
    op, kc_weight, vc_weight = make_op(
        geo,
        max_bs=batch_size if is_cuda_graph else 0,
        max_q_len=q_len,
        max_context_len=max(kv_lens),
        is_cuda_graph=is_cuda_graph,
    )
    q_nope = (
        torch.randn(
            batch_size * q_len,
            geo.num_heads,
            geo.qk_nope_head_dim,
            device="cuda",
        )
        * 0.5
    ).to(torch.bfloat16)
    q_pe = (
        torch.randn(
            batch_size * q_len,
            geo.num_heads,
            geo.qk_rope_head_dim,
            device="cuda",
        )
        * 0.5
    ).to(torch.bfloat16)
    op.plan(FakeMlaParams(kv_lens, page_indptr, page_indices))
    if is_cuda_graph:
        op.forward(q_nope, q_pe, FakeLayerKVCache(kv_cache), 0)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual = op.forward(q_nope, q_pe, FakeLayerKVCache(kv_cache), 0)
        graph.replay()
        torch.cuda.synchronize()
    else:
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


class TokenSpeedDecodeMetadataTest(TestCase):
    def test_keeps_physical_page_ids_without_expansion(self):
        metadata = _TokenSpeedDecodeMetadata(64, 0, 0, False, torch.device("cpu"))
        params = FakeMlaParams([65, 129], [0, 2, 5], [3, 5, 9, 10, 11], device="cpu")
        metadata.plan(params)
        self.assertEqual(metadata.padded_blocks, 3)
        torch.testing.assert_close(
            metadata.block_tables,
            torch.tensor([[3, 5, 0], [9, 10, 11]], dtype=torch.int32),
            rtol=0,
            atol=0,
        )

    def test_zero_length_rows_do_not_alias_live_page_ids(self):
        metadata = _TokenSpeedDecodeMetadata(64, 0, 0, False, torch.device("cpu"))
        params = FakeMlaParams([0, 65, 0], [0, 0, 2, 2], [7, 9], device="cpu")
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

    def test_workspace_can_be_shared_by_serial_graph_instances(self):
        first, _, _ = make_op(
            self.geo,
            max_bs=1,
            max_q_len=1,
            max_context_len=128,
            is_cuda_graph=True,
        )
        second, _, _ = make_op(
            self.geo,
            max_bs=1,
            max_q_len=1,
            max_context_len=128,
            is_cuda_graph=True,
        )
        self.assertNotEqual(first._workspace.data_ptr(), second._workspace.data_ptr())
        second.bind_cuda_graph_workspace(first._workspace)
        self.assertEqual(first._workspace.data_ptr(), second._workspace.data_ptr())

    def test_graph_workspace_reserves_speculative_query_upper_bound(self):
        with mock.patch.dict(os.environ, {"GEN_NUM_PER_CIRCLE": "3"}):
            decode, _, _ = make_op(
                self.geo,
                max_bs=1,
                max_q_len=1,
                max_context_len=128,
                is_cuda_graph=True,
            )
            verify, _, _ = make_op(
                self.geo,
                max_bs=1,
                max_q_len=4,
                max_context_len=128,
                is_cuda_graph=True,
            )
        self.assertEqual(
            decode._workspace_storage.numel(), verify._workspace_storage.numel()
        )
        verify.bind_cuda_graph_workspace(decode._workspace_storage)
        self.assertEqual(
            decode._workspace_storage.data_ptr(),
            verify._workspace_storage.data_ptr(),
        )
        # Kernel-facing views retain the exact size required by each q_len,
        # while sharing the fixed backing allocation and base address.
        self.assertNotEqual(decode._workspace.numel(), verify._workspace.numel())
        self.assertEqual(decode._workspace.data_ptr(), verify._workspace.data_ptr())

    def test_hybrid_model_mla_weights_need_not_be_on_layer_zero(self):
        base_op, kc_weight, _ = make_op(self.geo)
        op = TokenSpeedMlaDecodeOp(
            self.geo.num_heads,
            self.geo.kv_lora_rank,
            self.geo.qk_rope_head_dim,
            self.geo.qk_nope_head_dim,
            self.geo.page_size,
            1.0,
            [{}, base_op.weights[0]],
        )
        self.assertEqual(op._dtype, kc_weight.dtype)

        q_nope = torch.randn(
            1,
            self.geo.num_heads,
            self.geo.qk_nope_head_dim,
            dtype=torch.bfloat16,
            device="cuda",
        )
        q_pe = torch.randn(
            1,
            self.geo.num_heads,
            self.geo.qk_rope_head_dim,
            dtype=torch.bfloat16,
            device="cuda",
        )
        absorbed = op._absorb_query(q_nope, q_pe, layer_id=1)
        self.assertEqual(
            tuple(absorbed.shape),
            (1, self.geo.num_heads, self.geo.head_dim_qk),
        )
        with self.assertRaisesRegex(RuntimeError, "layer 0"):
            op._absorb_query(q_nope, q_pe, layer_id=0)

    def test_k3_tp8_variable_batch(self):
        op = run_case(self, self.geo, [65, 512, 129, 1000], num_pages=32)
        self.assertEqual(op._padded_blocks, 16)

    def test_eager_q_len_is_delegated_to_tokenspeed(self):
        run_multi_query_case(self, self.geo, [384, 513], q_len=5, is_cuda_graph=False)

    def test_cuda_graph_supports_captured_q_len_greater_than_one(self):
        run_multi_query_case(self, self.geo, [384, 513], q_len=5, is_cuda_graph=True)

    def test_target_verify_graph_writes_all_tokens_before_attention(self):
        geo = self.geo
        geo.page_size = 128
        batch_size = 2
        q_len = 5
        prefix_lengths = torch.tensor([126, 255], dtype=torch.int32)
        final_kv_lens = (prefix_lengths + q_len).tolist()
        block_table_host = torch.tensor([[0, 1, 0], [2, 3, 4]], dtype=torch.int32)
        page_indices = [0, 1, 2, 3, 4]
        page_indptr = [0, 2, 5]

        kv_cache, _, _ = build_kv_layout(geo, final_kv_lens, len(page_indices), seed=61)
        # These pages are newly allocated during target verify.  Their tails
        # must be cleared before any of the q_len writes execute.
        kv_cache[1].fill_(float("nan"))
        kv_cache[4].fill_(float("nan"))
        initial_cache = kv_cache.clone()

        compressed_kv = (
            torch.randn(batch_size * q_len, geo.kv_lora_rank, device="cuda") * 0.1
        ).to(torch.bfloat16)
        k_pe = (
            torch.randn(batch_size * q_len, geo.qk_rope_head_dim, device="cuda") * 0.1
        ).to(torch.bfloat16)
        q_nope = (
            torch.randn(
                batch_size * q_len,
                geo.num_heads,
                geo.qk_nope_head_dim,
                device="cuda",
            )
            * 0.5
        ).to(torch.bfloat16)
        q_pe = (
            torch.randn(
                batch_size * q_len,
                geo.num_heads,
                geo.qk_rope_head_dim,
                device="cuda",
            )
            * 0.5
        ).to(torch.bfloat16)

        params = rtp_llm_ops.FlashInferMlaAttnParams()
        params.fill_params(
            prefix_lengths,
            torch.empty(0, dtype=torch.int32),
            torch.full((batch_size,), q_len, dtype=torch.int32),
            block_table_host,
            geo.page_size,
            False,
        )
        op, kc_weight, vc_weight = make_op(
            geo,
            max_bs=batch_size,
            max_q_len=q_len,
            max_context_len=384,
            is_cuda_graph=True,
        )
        op.plan(params)
        write_op = MlaKVCacheWriteOp(KvCacheDataType.BASE, clear_page_on_boundary=True)

        def graph_forward():
            write_op.forward(
                compressed_kv,
                k_pe,
                FakeLayerKVCache(kv_cache),
                params,
            )
            return op.forward(q_nope, q_pe, FakeLayerKVCache(kv_cache), 0)

        # Warm every kernel before capture, then restore the live cache image.
        graph_forward()
        kv_cache.copy_(initial_cache)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual = graph_forward()
        kv_cache.copy_(initial_cache)
        graph.replay()
        torch.cuda.synchronize()

        expected_slots = torch.tensor(
            [126, 127, 128, 129, 130, 511, 512, 513, 514, 515],
            dtype=torch.int64,
            device="cuda",
        )
        torch.testing.assert_close(params.slot_mapping, expected_slots, rtol=0, atol=0)
        written = torch.cat((compressed_kv, k_pe), dim=1)
        flat_cache = kv_cache.view(-1, geo.head_dim_qk)
        torch.testing.assert_close(flat_cache[expected_slots], written)
        torch.testing.assert_close(kv_cache[1, 3:], torch.zeros_like(kv_cache[1, 3:]))
        torch.testing.assert_close(kv_cache[4, 4:], torch.zeros_like(kv_cache[4, 4:]))

        expected = reference_mla_decode(
            q_nope,
            q_pe,
            kc_weight,
            vc_weight,
            kv_cache,
            final_kv_lens,
            page_indptr,
            page_indices,
            geo,
        )
        relative_error = (
            (actual.float() - expected.float()).abs().max()
            / expected.float().abs().max()
        ).item()
        self.assertLess(relative_error, 2e-2)

    def test_cuda_graph_replan_across_page_boundaries(self):
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
        op.forward(q_nope, q_pe, FakeLayerKVCache(kv_cache), 0)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = op.forward(q_nope, q_pe, FakeLayerKVCache(kv_cache), 0)

        replay_lens = [65, 129]
        replay_indptr, replay_indices = compact_layout(replay_lens)
        op.plan(FakeMlaParams(replay_lens, replay_indptr, replay_indices))
        graph.replay()
        torch.cuda.synchronize()
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

        def compact_layout(kv_len):
            num_pages = (kv_len + geo.page_size - 1) // geo.page_size
            return [0, num_pages], physical_page_ids[:num_pages]

        initial_len = tested_lens[0]
        initial_indptr, initial_indices = compact_layout(initial_len)
        op.plan(FakeMlaParams([initial_len], initial_indptr, initial_indices))
        op.forward(q_nope, q_pe, FakeLayerKVCache(kv_cache), 0)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = op.forward(q_nope, q_pe, FakeLayerKVCache(kv_cache), 0)

        for replay_len in tested_lens:
            replay_indptr, replay_indices = compact_layout(replay_len)
            op.plan(FakeMlaParams([replay_len], replay_indptr, replay_indices))
            graph.replay()
            torch.cuda.synchronize()
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
        new_kv = torch.randn(geo.head_dim_qk, dtype=torch.bfloat16, device="cuda")
        write_op = MlaKVCacheWriteOp(KvCacheDataType.BASE, clear_page_on_boundary=True)
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
        op.plan(FakeMlaParams([old_len], [0, old_pages], physical_page_ids[:old_pages]))
        op.forward(q_nope, q_pe, FakeLayerKVCache(kv_cache), 0)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            write_op.forward(
                new_kv[: geo.kv_lora_rank].unsqueeze(0),
                new_kv[geo.kv_lora_rank :].unsqueeze(0),
                FakeLayerKVCache(kv_cache),
                write_params,
            )
            output = q_nope
            for _ in range(layer_count):
                output = op.forward(output, q_pe, FakeLayerKVCache(kv_cache), 0)

        kv_cache[new_page_id].fill_(float("nan"))
        write_slot_mapping.fill_(new_page_id * geo.page_size)
        op.plan(FakeMlaParams([new_len], [0, live_pages], physical_page_ids))
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
    module = (
        "rtp_llm.models_py.modules.factory.attention.cuda_mla_impl."
        "tokenspeed_mla_impl"
    )

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

    def _inputs(self, prompt_lengths=(1,), is_prefill=False, is_target_verify=False):
        return SimpleNamespace(
            is_prefill=is_prefill,
            is_target_verify=is_target_verify,
            input_lengths=torch.tensor(prompt_lengths, dtype=torch.int32),
            input_lengths_host=torch.tensor(prompt_lengths, dtype=torch.int32),
            total_tokens=sum(prompt_lengths) if is_target_verify else 0,
        )

    def test_normal_decode_ignores_original_prompt_lengths(self):
        self.assertEqual(
            decode_query_length(self._inputs(prompt_lengths=(17, 311))),
            1,
        )

    def test_target_verify_reads_uniform_host_query_length(self):
        self.assertEqual(
            decode_query_length(
                self._inputs(
                    prompt_lengths=(4, 4),
                    is_prefill=True,
                    is_target_verify=True,
                )
            ),
            4,
        )

    def test_target_verify_cuda_graph_capture_uses_host_query_length(self):
        attn_inputs = self._inputs(
            prompt_lengths=(4, 4),
            is_prefill=True,
            is_target_verify=True,
        )
        # CudaGraphRunner capture descriptors historically did not publish
        # total_tokens. The pinned host descriptor is sufficient on its own.
        attn_inputs.total_tokens = 0
        self.assertEqual(decode_query_length(attn_inputs), 4)

    def test_target_verify_without_host_uses_packed_query_shape(self):
        attn_inputs = self._inputs(
            prompt_lengths=(4, 4),
            is_prefill=True,
            is_target_verify=True,
        )
        attn_inputs.input_lengths_host = None
        self.assertEqual(decode_query_length(attn_inputs), 4)

    def test_target_verify_without_host_rejects_missing_packed_query_shape(self):
        attn_inputs = self._inputs(
            prompt_lengths=(4, 4),
            is_prefill=True,
            is_target_verify=True,
        )
        attn_inputs.input_lengths_host = None
        attn_inputs.total_tokens = 0
        with self.assertRaisesRegex(RuntimeError, "positive rectangular query shape"):
            decode_query_length(attn_inputs)

    def test_target_verify_rejects_nonuniform_host_query_lengths(self):
        attn_inputs = self._inputs(
            prompt_lengths=(4, 3),
            is_prefill=True,
            is_target_verify=True,
        )
        attn_inputs.total_tokens = 8
        with self.assertRaisesRegex(RuntimeError, "uniform host query lengths"):
            decode_query_length(attn_inputs)

    def test_target_verify_rejects_stale_host_query_lengths(self):
        attn_inputs = self._inputs(
            prompt_lengths=(4, 4),
            is_prefill=True,
            is_target_verify=True,
        )
        attn_inputs.input_lengths_host = torch.tensor([17, 311], dtype=torch.int32)
        with self.assertRaisesRegex(RuntimeError, "uniform host query lengths"):
            decode_query_length(attn_inputs)

    def test_target_verify_rejects_host_and_packed_shape_mismatch(self):
        attn_inputs = self._inputs(
            prompt_lengths=(4, 4),
            is_prefill=True,
            is_target_verify=True,
        )
        attn_inputs.total_tokens = 10
        with self.assertRaisesRegex(RuntimeError, "do not match the packed query"):
            decode_query_length(attn_inputs)

    def test_capability_is_delegated_to_tokenspeed(self):
        checker = mock.Mock()
        with mock.patch(
            f"{self.module}._tokenspeed_compute_capability", return_value=(10, 3)
        ), mock.patch(
            f"{self.module}._load_tokenspeed_mla", return_value=True
        ), mock.patch(
            f"{self.module}._TOKENSPEED_CAN_IMPLEMENT", checker
        ):
            self.assertTrue(
                tokenspeed_mla_kernel_supported(
                    96, 512, 64, 128, q_len=17, dtype=torch.float16
                )
            )
        checker.assert_called_once_with(
            torch_dtype=torch.float16,
            page_size=128,
            num_heads=96,
            seq_len_q=17,
            kv_lora_rank=512,
            qk_rope_head_dim=64,
            is_persistent=False,
            is_var_seq=True,
            is_var_split_kv=False,
            compute_capability=(10, 3),
        )

    def test_optional_dependency_abi_error_is_reported_as_unavailable(self):
        with mock.patch(f"{self.module}._TOKENSPEED_MLA_API", None), mock.patch(
            f"{self.module}._TOKENSPEED_GET_NUM_SM", None
        ), mock.patch(f"{self.module}._TOKENSPEED_CAN_IMPLEMENT", None), mock.patch(
            f"{self.module}._TOKENSPEED_IMPORT_ERROR", None
        ), mock.patch(
            f"{self.module}._TOKENSPEED_IMPORT_ATTEMPTED", False
        ), mock.patch(
            f"{self.module}._ensure_tokenspeed_cutlass_compat",
            side_effect=RuntimeError("CuTe ABI mismatch"),
        ):
            self.assertFalse(_load_tokenspeed_mla())
            self.assertIsInstance(
                tokenspeed_mla_impl._TOKENSPEED_IMPORT_ERROR, RuntimeError
            )

    def test_capability_rejection_is_not_reimplemented_in_rtp(self):
        with mock.patch(
            f"{self.module}._tokenspeed_compute_capability", return_value=(10, 0)
        ), mock.patch(
            f"{self.module}._load_tokenspeed_mla", return_value=True
        ), mock.patch(
            f"{self.module}._TOKENSPEED_CAN_IMPLEMENT",
            side_effect=ValueError("unsupported by TokenSpeed"),
        ):
            self.assertFalse(tokenspeed_mla_kernel_supported(12, 777, 48, 96, q_len=9))

    def test_prefers_tokenspeed_on_supported_blackwell(self):
        with mock.patch(
            f"{self.module}._is_tokenspeed_blackwell", return_value=True
        ), mock.patch(
            f"{self.module}._load_tokenspeed_mla", return_value=True
        ), mock.patch(
            f"{self.module}.tokenspeed_mla_kernel_supported", return_value=True
        ) as capability:
            self.assertTrue(
                TokenSpeedMlaDecodeImpl.support(
                    self._configs(), self._inputs(prompt_lengths=(8, 8))
                )
            )
        self.assertEqual(capability.call_args.args[4], 1)

    def test_falls_back_on_other_arch_or_missing_dependency(self):
        with mock.patch(f"{self.module}._is_tokenspeed_blackwell", return_value=False):
            self.assertFalse(
                TokenSpeedMlaDecodeImpl.support(self._configs(), self._inputs())
            )
        with mock.patch(
            f"{self.module}._is_tokenspeed_blackwell", return_value=True
        ), mock.patch(f"{self.module}._load_tokenspeed_mla", return_value=False):
            self.assertFalse(
                TokenSpeedMlaDecodeImpl.support(self._configs(), self._inputs())
            )

    def test_falls_back_when_tokenspeed_rejects_runtime_shape(self):
        with mock.patch(
            f"{self.module}._is_tokenspeed_blackwell", return_value=True
        ), mock.patch(
            f"{self.module}._load_tokenspeed_mla", return_value=True
        ), mock.patch(
            f"{self.module}.tokenspeed_mla_kernel_supported",
            return_value=False,
        ):
            self.assertFalse(
                TokenSpeedMlaDecodeImpl.support(
                    self._configs(), self._inputs(prompt_lengths=(9, 9))
                )
            )

    def test_prompt_lengths_do_not_change_decode_query_shape(self):
        with mock.patch(
            f"{self.module}._is_tokenspeed_blackwell", return_value=True
        ), mock.patch(
            f"{self.module}._load_tokenspeed_mla", return_value=True
        ), mock.patch(
            f"{self.module}.tokenspeed_mla_kernel_supported", return_value=True
        ) as capability:
            self.assertTrue(
                TokenSpeedMlaDecodeImpl.support(
                    self._configs(), self._inputs(prompt_lengths=(17, 311))
                )
            )
        self.assertEqual(capability.call_args.args[4], 1)

    def test_mtp_draft_decode_uses_one_query_per_step(self):
        with mock.patch(
            f"{self.module}._is_tokenspeed_blackwell", return_value=True
        ), mock.patch(
            f"{self.module}._load_tokenspeed_mla", return_value=True
        ), mock.patch(
            f"{self.module}.tokenspeed_mla_kernel_supported", return_value=True
        ) as capability:
            self.assertTrue(
                TokenSpeedMlaDecodeImpl.support(
                    self._configs(),
                    self._inputs(
                        prompt_lengths=(101, 307),
                        is_prefill=False,
                        is_target_verify=False,
                    ),
                )
            )
        self.assertEqual(capability.call_args.args[4], 1)

    def test_mtp_target_verify_uses_propose_plus_one_query_tokens(self):
        with mock.patch(
            f"{self.module}._is_tokenspeed_blackwell", return_value=True
        ), mock.patch(
            f"{self.module}._load_tokenspeed_mla", return_value=True
        ), mock.patch(
            f"{self.module}.tokenspeed_mla_kernel_supported", return_value=True
        ) as capability:
            self.assertTrue(
                TokenSpeedMlaDecodeImpl.support(
                    self._configs(),
                    self._inputs(
                        prompt_lengths=(4, 4),
                        is_prefill=True,
                        is_target_verify=True,
                    ),
                )
            )
        self.assertEqual(capability.call_args.args[4], 4)

    def test_prefill_is_never_selected(self):
        self.assertFalse(
            TokenSpeedMlaDecodeImpl.support(
                self._configs(), self._inputs(is_prefill=True)
            )
        )

    def test_impl_clears_new_pages_only_for_cuda_graph(self):
        configs = self._configs()
        configs.nope_head_dim = 128
        configs.softmax_extra_scale = 1.0
        configs.rope_config = RopeConfig()
        configs.rope_config.is_neox_style = False
        attn_inputs = SimpleNamespace(
            sequence_lengths=torch.zeros(1, dtype=torch.int32, device="cuda"),
            input_lengths=torch.ones(1, dtype=torch.int32, device="cuda"),
            kv_cache_kernel_block_id_device=torch.zeros(
                (1, 1), dtype=torch.int32, device="cuda"
            ),
        )

        with mock.patch(f"{self.module}.TokenSpeedMlaDecodeOp"), mock.patch(
            f"{self.module}.NewMlaRotaryEmbeddingOp"
        ), mock.patch(f"{self.module}.MlaKVCacheWriteOp") as write_op_cls, mock.patch(
            f"{self.module}.MlaFlashInferImplBase.__init__", return_value=None
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

    def test_impl_sizes_graph_buffers_from_captured_query_shape(self):
        configs = self._configs()
        configs.nope_head_dim = 128
        configs.softmax_extra_scale = 1.0
        configs.rope_config = RopeConfig()
        configs.rope_config.is_neox_style = False
        attn_inputs = SimpleNamespace(
            sequence_lengths=torch.zeros(2, dtype=torch.int32, device="cuda"),
            input_lengths=torch.ones(2, dtype=torch.int32, device="cuda"),
            input_lengths_host=torch.tensor([5, 5], dtype=torch.int32),
            kv_cache_kernel_block_id_device=torch.zeros(
                (2, 1), dtype=torch.int32, device="cuda"
            ),
        )

        with mock.patch(
            f"{self.module}.TokenSpeedMlaDecodeOp"
        ) as decode_op_cls, mock.patch(
            f"{self.module}.NewMlaRotaryEmbeddingOp"
        ), mock.patch(
            f"{self.module}.MlaKVCacheWriteOp"
        ), mock.patch(
            f"{self.module}.MlaFlashInferImplBase.__init__", return_value=None
        ):
            TokenSpeedMlaDecodeImpl(
                configs,
                attn_inputs,
                weights=[],
                cos_sin_cache=torch.empty(0),
                is_cuda_graph=True,
            )

        self.assertEqual(decode_op_cls.call_args.kwargs["max_bs"], 2)
        self.assertEqual(decode_op_cls.call_args.kwargs["max_q_len"], 1)

    def test_impl_sizes_target_verify_graph_from_actual_query_shape(self):
        configs = self._configs()
        configs.nope_head_dim = 128
        configs.softmax_extra_scale = 1.0
        configs.rope_config = RopeConfig()
        configs.rope_config.is_neox_style = False
        attn_inputs = SimpleNamespace(
            is_prefill=True,
            is_target_verify=True,
            sequence_lengths=torch.empty(0, dtype=torch.int32, device="cuda"),
            input_lengths=torch.tensor([4, 4], dtype=torch.int32, device="cuda"),
            kv_cache_kernel_block_id_device=torch.zeros(
                (2, 1), dtype=torch.int32, device="cuda"
            ),
            total_tokens=8,
        )

        with mock.patch(
            f"{self.module}.TokenSpeedMlaDecodeOp"
        ) as decode_op_cls, mock.patch(
            f"{self.module}.NewMlaRotaryEmbeddingOp"
        ), mock.patch(
            f"{self.module}.MlaKVCacheWriteOp"
        ), mock.patch(
            f"{self.module}.MlaFlashInferImplBase.__init__", return_value=None
        ):
            TokenSpeedMlaDecodeImpl(
                configs,
                attn_inputs,
                weights=[],
                cos_sin_cache=torch.empty(0),
                is_cuda_graph=True,
            )

        self.assertEqual(decode_op_cls.call_args.kwargs["max_bs"], 2)
        self.assertEqual(decode_op_cls.call_args.kwargs["max_q_len"], 4)


    @skipUnless(RUN_KERNEL, SKIP_REASON)
    def test_swa_draft_factory_keeps_compute_parallelism(self):
        from rtp_llm.cpp.cuda_graph.tests.libtest_cuda_graph_runner import CudaGraphRunner
        from rtp_llm.config.model_config import ModelConfig
        from rtp_llm.model_loader.model_weight_info import ModelWeights
        from rtp_llm.models_py.model_desc.module_base import GptModelBase
        from rtp_llm.models_py.modules.factory.attention.attn_factory import AttnImplFactory
        from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.sliding_window_mla_decode import (
            SlidingWindowMlaDecodeImpl,
        )
        from rtp_llm.ops import CPRotateMethod, HybridAttentionType, ParallelismConfig, RoleType
        from rtp_llm.ops.compute_ops import LayerKVCache, PyAttentionInputs, PyModelInputs, PyModelOutputs

        torch.manual_seed(73)
        page, window, batch_capacity = 4096, 2048, 8
        config = ModelConfig()
        config.num_layers, config.max_seq_len = 1, page * 2
        config.quant_config = None
        config.attn_config = self._configs()
        config.attn_config.head_num = 96
        config.attn_config.kv_head_num = 1
        config.attn_config.nope_head_dim = 128
        config.attn_config.tokens_per_block = page
        config.attn_config.kernel_tokens_per_block = 128
        config.attn_config.sliding_window = window
        config.hybrid_attention_config.enable_hybrid_attention = True
        config.hybrid_attention_config.enable_independent_kv_cache_pools = True
        config.hybrid_attention_config.hybrid_attention_types = [
            HybridAttentionType.SLIDING_WINDOW
        ]
        parallel = ParallelismConfig()
        parallel.tp_size = parallel.world_size = 8
        parallel.role_type = RoleType.DECODE
        parallel.decode_cp_kv_cache_sharded = True
        for queries in (1, 7):
            for method in (CPRotateMethod.DISABLED, CPRotateMethod.ALL_GATHER):
                with self.subTest(queries=queries, method=method):
                    parallel.prefill_cp_config.method = method
                    heads = 96 // parallel.get_attn_tp_size()
                    weights = ModelWeights(2, "cuda", torch.bfloat16)
                    kc = torch.randn(heads, 128, 512, device="cuda", dtype=torch.bfloat16) * .02
                    vc = torch.randn(heads, 512, 128, device="cuda", dtype=torch.bfloat16) * .02
                    weights.weights[0] = {W.mla_kc: kc, W.mla_vc: vc}
                    weights.weights[1] = {W.mla_kc: kc, W.mla_vc: vc}
                    # Identity RoPE still executes the real kernel; this test's
                    # independent oracle concerns physical pages and the window.
                    rope = torch.zeros(page * 2, 64, device="cuda")
                    rope[:, :32] = 1
                    weights.set_global_weight(W.rope_cos_sin_cache, rope)
                    inputs = PyAttentionInputs()
                    inputs.is_prefill = inputs.is_mtp_draft_update = queries > 1
                    inputs.total_tokens = queries
                    inputs.input_lengths_host = torch.tensor([queries], dtype=torch.int32)
                    inputs.input_lengths = inputs.input_lengths_host.cuda()
                    inputs.prefix_lengths_host = torch.tensor(
                        [17] if queries > 1 else [], dtype=torch.int32
                    )
                    inputs.prefix_lengths = inputs.prefix_lengths_host.cuda()
                    inputs.sequence_lengths_host = torch.tensor(
                        [17] if queries == 1 else [], dtype=torch.int32
                    )
                    inputs.sequence_lengths = inputs.sequence_lengths_host.cuda()
                    table = torch.zeros((1, 32), dtype=torch.int32)
                    table[0, :2] = torch.tensor([3, 1])
                    inputs.kv_cache_kernel_block_id_host = table
                    inputs.kv_cache_kernel_block_id_device = table.cuda()
                    cache = LayerKVCache()
                    cache.kv_cache_base = torch.empty(
                        ((2 * batch_capacity + 3) * 32, 128, 576), dtype=torch.bfloat16, device="cuda"
                    )
                    q = torch.empty(queries, heads, 192, dtype=torch.bfloat16, device="cuda")
                    appended = torch.empty(queries, 512, dtype=torch.bfloat16, device="cuda")
                    # Eagle3 leaves k_pe as a slice of fused Q/K/V+gate output.
                    fused = torch.empty(
                        queries, 2112 + heads * 128, dtype=torch.bfloat16, device="cuda"
                    )
                    kpe = fused[:, 2048:2112]

                    def reference(query, new_kv, new_kpe, start, physical):
                        flat = cache.kv_cache_base.view(-1, 576)
                        # Oracle walks physical P pages without using the tested
                        # sparse index converter or the native planner's slots.
                        ordered = torch.cat([
                            flat[p * page : (p + 1) * page] for p in physical
                        ])
                        ordered[start : start + queries, :512] = new_kv
                        ordered[start : start + queries, 512:] = new_kpe
                        absorbed = torch.bmm(query[..., :128].transpose(0, 1), kc).transpose(0, 1)
                        projected = torch.cat((absorbed, query[..., 128:]), -1).float()
                        expected = []
                        for j in range(queries):
                            end = start + j + 1
                            values = ordered[max(0, end - window) : end].float()
                            score = projected[j] @ values.T * (192 ** -.5)
                            expected.append(score.softmax(-1) @ values[:, :512])
                        expected = torch.stack(expected).to(torch.bfloat16)
                        output = torch.bmm(expected.transpose(0, 1), vc).transpose(0, 1)
                        return output, ordered, start, physical

                    def prepare(step):
                        start = (17, page - 3, page + 19, page * 2 - queries)[step]
                        physical = (3, 1) if step != 1 else (1, 3)
                        inputs.prefix_lengths_host.fill_(start)
                        inputs.sequence_lengths_host.fill_(start)
                        inputs.prefix_lengths.copy_(inputs.prefix_lengths_host)
                        inputs.sequence_lengths.copy_(inputs.sequence_lengths_host)
                        table[0, :2] = torch.tensor(physical)
                        inputs.kv_cache_kernel_block_id_device.copy_(table)
                        cache.kv_cache_base.normal_()
                        q.normal_()
                        appended.normal_()
                        kpe.normal_()
                        return reference(q, appended, kpe, start, physical)

                    # Match native draft/verify capture: reserve the largest
                    # legal prefix before replaying shorter live histories.
                    expected = prepare(3)
                    impl = AttnImplFactory.get_fmha_impl(
                        config, parallel, weights, inputs, is_cuda_graph=True
                    )
                    self.assertIsInstance(impl, SlidingWindowMlaDecodeImpl)
                    self.assertEqual(impl.fmha_impl.num_heads, heads)

                    def invoke():
                        return impl.forward(q, appended, kpe, cache, 0)

                    def check(output, reference):
                        expected_output, ordered, start, physical = reference
                        torch.testing.assert_close(output, expected_output, atol=.015, rtol=.03)
                        normalized_error = (
                            (output.float() - expected_output.float()).abs().max()
                            / expected_output.float().abs().max()
                        )
                        self.assertLess(normalized_error.item(), .02)
                        positions = torch.arange(start, start + queries, device="cuda")
                        pages = torch.tensor(physical, device="cuda")
                        slots = pages[positions // page] * page + positions % page
                        torch.testing.assert_close(
                            cache.kv_cache_base.view(-1, 576)[slots],
                            ordered[start : start + queries],
                            atol=0,
                            rtol=0,
                        )

                    # A shared Impl must select the requested layer's group for
                    # both the KV writer and the attention reader.
                    actual_table = inputs.kv_cache_kernel_block_id_device
                    wrong_table = torch.zeros_like(actual_table)
                    inputs.kv_cache_kernel_block_id_device_by_group = [wrong_table, actual_table]
                    inputs.kv_cache_layer_to_group_host = torch.tensor([0, 1], dtype=torch.int32)
                    inputs.kv_cache_kernel_block_id_device = wrong_table
                    check(impl.forward(q, appended, kpe, cache, 1), expected)
                    self.assertEqual(impl.fmha_params.block_table.data_ptr(), actual_table.data_ptr())
                    inputs.kv_cache_kernel_block_id_device_by_group = []
                    inputs.kv_cache_layer_to_group_host = torch.empty(0, dtype=torch.int32)

                    check(invoke(), expected)
                    stream = torch.cuda.Stream()
                    stream.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(stream):
                        invoke()
                    torch.cuda.current_stream().wait_stream(stream)
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph, stream=stream):
                        captured = invoke()
                    positions_ptr = impl.fmha_params.positions_d.data_ptr()
                    slots_ptr = impl.fmha_params.slot_mapping.data_ptr()
                    for step in range(3):
                        expected = prepare(step)
                        impl.prepare_cuda_graph(inputs)
                        self.assertEqual(impl.fmha_params.positions_d.data_ptr(), positions_ptr)
                        self.assertEqual(impl.fmha_params.slot_mapping.data_ptr(), slots_ptr)
                        graph.replay()
                        check(captured, expected)
                    graph.reset()

                    query_width = heads * 192
                    hidden_width = query_width + 576

                    class RunnerModel(GptModelBase):
                        def forward(self, model_inputs, fmha_impl=None):
                            if fmha_impl is None:
                                fmha_impl = self.prepare_fmha_impl(model_inputs)
                            hidden = model_inputs.input_hiddens.to(torch.bfloat16)
                            graph_q = hidden[:, :query_width].contiguous().view(-1, heads, 192)
                            graph_ckv = hidden[:, query_width : query_width + 512].contiguous()
                            graph_fused = torch.empty(
                                hidden.shape[0], 2112 + heads * 128,
                                dtype=torch.bfloat16, device=hidden.device,
                            )
                            graph_kpe = graph_fused[:, 2048:2112]
                            graph_kpe.copy_(hidden[:, query_width + 512 :])
                            output = fmha_impl.forward(graph_q, graph_ckv, graph_kpe, cache, 0)
                            return PyModelOutputs(
                                torch.nn.functional.pad(output.flatten(1), (0, hidden_width - heads * 128)),
                                fmha_impl.fmha_params,
                            )

                    runner_model = RunnerModel(config, parallel, weights, max_generate_batch_size=batch_capacity)
                    runner = CudaGraphRunner()
                    try:
                        if queries == 1:
                            runner.init_decode(
                                runner_model, hidden_size=hidden_width,
                                max_seq_len=page * 2, tokens_per_block=page,
                                kernel_tokens_per_block=128, decode_capture_batch_sizes=[1, batch_capacity],
                                max_context_batch_size=batch_capacity,
                            )
                        else:
                            runner.init_prefill(
                                runner_model, hidden_size=hidden_width,
                                max_context_batch_size=batch_capacity, max_seq_len=page * 2,
                                tokens_per_block=page, kernel_tokens_per_block=128,
                                prefill_capture_seq_lens=[], num_tokens_per_bs=queries,
                                is_mtp_draft_update=True,
                            )
                        for step in range(3):
                            expected = prepare(step)
                            batch = 1 if step == 1 else batch_capacity
                            expected_batch = [expected]
                            hidden_batch = [torch.cat((q.flatten(1), appended, kpe), -1)]
                            for request in range(1, batch):
                                scale = -0.5 if request % 2 else 0.5
                                expected_batch.append(reference(
                                    q * scale, appended * scale, kpe * scale,
                                    page + 103 + 17 * request, (2 * request + 3, 2 * request + 2),
                                ))
                                hidden_batch.append(hidden_batch[0] * scale)
                            starts = [item[2] for item in expected_batch]
                            native_table = torch.zeros(batch, table.shape[1], dtype=torch.int32)
                            for request, item in enumerate(expected_batch):
                                native_table[request, :2] = torch.tensor(item[3])
                            inputs.total_tokens = batch * queries
                            inputs.input_lengths_host = torch.full((batch,), queries, dtype=torch.int32).pin_memory()
                            inputs.prefix_lengths_host = torch.tensor(starts if queries > 1 else [], dtype=torch.int32).pin_memory()
                            inputs.sequence_lengths_host = torch.tensor(starts if queries == 1 else [], dtype=torch.int32).pin_memory()
                            inputs.input_lengths = inputs.input_lengths_host.cuda()
                            inputs.prefix_lengths = inputs.prefix_lengths_host.cuda()
                            inputs.sequence_lengths = inputs.sequence_lengths_host.cuda()
                            inputs.sequence_lengths_plus_1_d = torch.tensor(
                                [start + 1 for start in starts], dtype=torch.int32, device="cuda"
                            )
                            inputs.cu_seqlens = torch.arange(batch + 1, dtype=torch.int32, device="cuda") * queries
                            inputs.cu_seqlens_host = inputs.cu_seqlens.cpu().pin_memory()
                            inputs.decode_cu_seqlens_d = inputs.cu_seqlens
                            inputs.cu_kv_seqlens = torch.tensor(
                                [0] + [start + queries for start in starts], dtype=torch.int32, device="cuda"
                            ).cumsum(0, dtype=torch.int32)
                            inputs.kv_cache_kernel_block_id_host = native_table.pin_memory()
                            inputs.kv_cache_kernel_block_id_device = native_table.cuda()
                            replay = PyModelInputs()
                            replay.input_ids = torch.arange(batch * queries, dtype=torch.int32, device="cuda")
                            replay.input_hiddens = torch.cat(hidden_batch, 0).to(
                                torch.float16 if queries == 1 else torch.bfloat16
                            )
                            replay.attention_inputs = inputs
                            padding_page = cache.kv_cache_base[:32].clone() if queries > 1 else None
                            self.assertTrue(runner.canRun(replay))
                            if step == 1:
                                runner.prepareAttentionInputs(replay)
                            output = runner.forward(replay).hidden_states
                            if padding_page is not None:
                                torch.testing.assert_close(cache.kv_cache_base[:32], padding_page, atol=0, rtol=0)
                            for request, expected in enumerate(expected_batch):
                                check(output[request * queries : (request + 1) * queries, :heads * 128].view(queries, heads, 128).to(torch.bfloat16), expected)
                    finally:
                        del runner


@skipUnless(RUN_KERNEL, SKIP_REASON)
class TokenSpeedPageRrKernelTest(TestCase):
    def _run_history(self, batch, queries, dtype, pdl, head_major=False):
        from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.tokenspeed_mla_page_rr import (
            tokenspeed_mla_page_rr_decode,
        )

        torch.manual_seed(104)
        heads, latent, rope, page, width = 96, 512, 64, 128, 4
        # Preserve allocator padding and latent/RoPE slice strides in the ABI.
        cache_storage = (
            torch.randn((16, page, latent + rope + 16), device="cuda", dtype=dtype)
            * 0.1
        )
        kv = cache_storage[..., : latent + rope]
        q_storage = (
            torch.randn(
                (batch, queries, heads, latent + rope + 16), device="cuda", dtype=dtype
            )
            * 0.1
        )
        query = q_storage[..., : latent + rope]
        if head_major:
            # A head-major projection/all-gather can expose the kernel's
            # [B,Q,H,D] query as a view, without a token-major reorder copy.
            q_storage = query.permute(2, 0, 1, 3).contiguous()
            query = q_storage.permute(1, 2, 0, 3)
        table_storage = torch.full(
            (batch, width + 3), -1, device="cuda", dtype=torch.int32
        )
        table = table_storage[:, :width]
        table.copy_(
            torch.tensor(
                [[7, 2, 11, 0], [3, 9, 1, 6]], device="cuda", dtype=torch.int32
            ).repeat(batch // 2, 1)
        )
        query_block_tables = torch.empty(
            (batch * queries, width), device="cuda", dtype=torch.int32
        )
        query_block_tables.view(batch, queries, width).copy_(table[:, None, :])
        lengths = torch.tensor(
            ([0] * queries + list(range(126, 126 + queries))) * (batch // 2),
            device="cuda",
            dtype=torch.int32,
        ).view(batch, queries)
        workspace = torch.empty(
            torch.cuda.get_device_properties(0).multi_processor_count
            * heads
            * (latent + 1)
            * 4,
            device="cuda",
            dtype=torch.int8,
        )
        output_storage = torch.full(
            (batch * queries * heads * latent + 32,), 17, device="cuda", dtype=dtype
        )
        out = output_storage[:-32].view(batch, queries, heads, latent)
        scale, output_scale = 192**-0.5, -1.25

        def invoke(block_tables=query_block_tables):
            return tokenspeed_mla_page_rr_decode(
                query,
                kv,
                workspace,
                latent,
                rope,
                block_tables,
                lengths,
                width * page,
                scale,
                output_scale=output_scale,
                out=out,
                enable_pdl=pdl,
            )

        def check(result):
            self.assertIs(result[0], out)
            self.assertEqual(result[1].shape, (batch, queries, heads))
            for b in range(batch):
                for q in range(queries):
                    n = int(lengths[b, q])
                    keys = kv[table[b].long()].reshape(-1, latent + rope)[:n].float()
                    scores = query[b, q].float() @ keys.T * scale
                    expected = (scores.softmax(-1) @ keys[:, :latent]) * output_scale
                    torch.testing.assert_close(
                        result[0][b, q].float(), expected, atol=3e-3, rtol=1e-2
                    )
                    torch.testing.assert_close(
                        result[1][b, q],
                        scores.logsumexp(-1) / math.log(2),
                        atol=1e-3,
                        rtol=1e-3,
                    )
            self.assertTrue(torch.all(output_storage[-32:] == 17))
            self.assertTrue(torch.all(table_storage[:, width:] == -1))

        # NaNs make an unwritten empty output observable, rather than allowing
        # accidental zeroed allocator contents to satisfy the merge identity.
        with self.assertRaisesRegex(ValueError, "query page tables"):
            invoke(query_block_tables[:-1])
        out.fill_(float("nan"))
        result = invoke()
        check(result)
        for _ in range(3):
            invoke()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = invoke()
        for replay in range(3):
            table.copy_(torch.roll(table.clone(), shifts=1, dims=1))
            query_block_tables.view(batch, queries, width).copy_(table[:, None, :])
            q_storage.mul_(0.9)
            if replay == 1:
                lengths.zero_()
            else:
                lengths.add_(1)
            out.fill_(float("nan"))
            graph.replay()
            check(captured)

    def test_shared_pages_mtp_graph_output_contract(self):
        for dtype in (torch.bfloat16, torch.float16):
            for pdl in (False, True):
                with self.subTest(dtype=dtype, pdl=pdl):
                    self._run_history(2, 7, dtype, pdl)

    def test_shared_pages_without_split_workspace(self):
        self._run_history(80, 1, torch.bfloat16, False)

    def test_shared_pages_head_major_query_view(self):
        self._run_history(2, 7, torch.bfloat16, True, head_major=True)

    def test_page_rr_live_metadata_and_native_writer_history(self):
        from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.page_rr_mla_metadata import (
            PageRRMlaDecodeMetadata,
        )
        from rtp_llm.ops.compute_ops import PyAttentionInputs

        page, kernel_page, cp = 4096, 128, 8
        latent, rope = 512, 64
        for mode in ("decode", "verify", "draft_update"):
            for rank in (0, 1):
                with self.subTest(mode=mode, rank=rank):
                    queries = 1 if mode == "decode" else 7
                    inputs = PyAttentionInputs()
                    inputs.is_prefill = mode != "decode"
                    inputs.is_target_verify = mode == "verify"
                    inputs.is_mtp_draft_update = mode == "draft_update"
                    inputs.is_cuda_graph = True
                    inputs.total_tokens = 2 * queries
                    inputs.input_lengths = torch.full(
                        (2,), queries, device="cuda", dtype=torch.int32
                    )
                    inputs.prefix_lengths = torch.empty_like(inputs.input_lengths)
                    inputs.sequence_lengths = torch.empty_like(inputs.input_lengths)
                    inputs.sequence_lengths_plus_1_d = torch.empty_like(
                        inputs.input_lengths
                    )
                    # Kernel tables are views of allocator P-pages, with K-page
                    # expansion and row padding, not independent invented K ids.
                    physical = [[3, 1], [4, 2]]
                    table_storage = torch.full(
                        (2, 69), -1, device="cuda", dtype=torch.int32
                    )
                    inputs.kv_cache_kernel_block_id_device = table_storage[:, :64]
                    metadata = PageRRMlaDecodeMetadata(page, kernel_page, cp, rank)
                    cache = torch.randn(
                        (5 * page // kernel_page, kernel_page, latent + rope),
                        device="cuda",
                        dtype=torch.bfloat16,
                    )
                    append = torch.randn(
                        (2 * queries, latent + rope),
                        device="cuda",
                        dtype=torch.bfloat16,
                    )
                    writer = MlaKVCacheWriteOp(KvCacheDataType.BASE)

                    def prepare(starts, capture=False):
                        inputs.prefix_lengths.copy_(
                            torch.tensor(starts, device="cuda", dtype=torch.int32)
                        )
                        inputs.sequence_lengths.copy_(inputs.prefix_lengths)
                        inputs.sequence_lengths_plus_1_d.copy_(
                            inputs.prefix_lengths + 1
                        )
                        if capture and mode == "decode":
                            # Match CudaGraphRunner's synthetic q1 initialization.
                            inputs.sequence_lengths_plus_1_d.zero_()
                        expanded = [
                            [
                                p * (page // kernel_page) + k
                                for p in row
                                for k in range(page // kernel_page)
                            ]
                            for row in physical
                        ]
                        inputs.kv_cache_kernel_block_id_device.copy_(
                            torch.tensor(expanded, device="cuda", dtype=torch.int32)
                        )
                        metadata.prepare(
                            inputs, forbid_realloc=metadata.positions_d is not None
                        )

                    def write():
                        writer.forward(
                            append[:, :latent],
                            append[:, latent:],
                            FakeLayerKVCache(cache),
                            metadata,
                        )

                    def check(starts, before):
                        expected_positions, expected_lengths, expected_slots = (
                            [],
                            [],
                            [],
                        )
                        expected = before.view(-1, latent + rope)
                        for b, start in enumerate(starts):
                            for q in range(queries):
                                position = start + q
                                expected_positions.append(position)
                                # Independent count in global token coordinates;
                                # do not reuse the producer's interval formula.
                                expected_lengths.append(
                                    sum(
                                        t // page % cp == rank
                                        for t in range(position + 1)
                                    )
                                )
                                slot = -1
                                if position // page % cp == rank:
                                    physical_page = physical[b][position // (page * cp)]
                                    slot = physical_page * page + position % page
                                    expected[slot].copy_(append[b * queries + q])
                                expected_slots.append(slot)
                        self.assertEqual(
                            metadata.positions_d.tolist(), expected_positions
                        )
                        self.assertEqual(
                            metadata.local_causal_lens.flatten().tolist(),
                            expected_lengths,
                        )
                        self.assertEqual(metadata.slot_mapping.tolist(), expected_slots)
                        torch.testing.assert_close(
                            metadata.query_block_tables,
                            inputs.kv_cache_kernel_block_id_device.repeat_interleave(
                                queries, dim=0
                            ),
                            atol=0,
                            rtol=0,
                        )
                        torch.testing.assert_close(cache, before, atol=0, rtol=0)
                        self.assertTrue(torch.all(table_storage[:, 64:] == -1))

                    starts = [32765, 4093]
                    prepare(starts, capture=True)
                    pointers = tuple(
                        t.data_ptr()
                        for t in (
                            metadata.positions_d,
                            metadata.slot_mapping,
                            metadata.local_causal_lens,
                            metadata.query_block_tables,
                        )
                    )
                    before = cache.clone()
                    write()
                    check(starts, before)
                    for _ in range(3):
                        write()
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        write()
                    for starts in ([32768, 4096], [1, 32768], [32767, 4095]):
                        physical = [list(reversed(row)) for row in physical]
                        append.mul_(0.9)
                        prepare(starts)
                        before = cache.clone()
                        graph.replay()
                        check(starts, before)
                        self.assertEqual(
                            pointers,
                            tuple(
                                t.data_ptr()
                                for t in (
                                    metadata.positions_d,
                                    metadata.slot_mapping,
                                    metadata.local_causal_lens,
                                    metadata.query_block_tables,
                                )
                            ),
                        )

                    original_table = inputs.kv_cache_kernel_block_id_device
                    inputs.kv_cache_kernel_block_id_device = original_table[:, :-1]
                    try:
                        with self.assertRaisesRegex(
                            ValueError, "query page table shape"
                        ):
                            metadata.prepare(inputs, forbid_realloc=True)
                    finally:
                        inputs.kv_cache_kernel_block_id_device = original_table
                    metadata.prepare(inputs, forbid_realloc=True)


if __name__ == "__main__":
    main()
