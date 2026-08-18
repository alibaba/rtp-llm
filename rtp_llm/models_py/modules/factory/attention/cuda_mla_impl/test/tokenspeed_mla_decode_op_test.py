"""Correctness and CUDA Graph tests for TokenSpeed MLA decode."""

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
    MLA_DECODE_KERNEL_ENV,
    TokenSpeedMlaDecodeImpl,
    TokenSpeedMlaDecodeOp,
    _get_mla_decode_kernel,
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

    def test_auto_prefers_tokenspeed_on_supported_blackwell(self):
        with mock.patch.dict(os.environ):
            os.environ.pop(MLA_DECODE_KERNEL_ENV, None)
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

    def test_auto_falls_back_on_other_arch_or_missing_dependency(self):
        with mock.patch.dict(os.environ):
            os.environ.pop(MLA_DECODE_KERNEL_ENV, None)
            with mock.patch(
                f"{self.module}._is_tokenspeed_blackwell", return_value=False
            ):
                self.assertFalse(
                    TokenSpeedMlaDecodeImpl.support(self._configs(), self._inputs())
                )
            with mock.patch(
                f"{self.module}._is_tokenspeed_blackwell", return_value=True
            ), mock.patch(f"{self.module}._load_tokenspeed_mla", return_value=False):
                self.assertFalse(
                    TokenSpeedMlaDecodeImpl.support(self._configs(), self._inputs())
                )

    def test_flashinfer_selection_skips_tokenspeed(self):
        with mock.patch.dict(
            os.environ, {MLA_DECODE_KERNEL_ENV: "flashinfer"}
        ), mock.patch(f"{self.module}._load_tokenspeed_mla") as loader:
            self.assertFalse(
                TokenSpeedMlaDecodeImpl.support(self._configs(), self._inputs())
            )
            loader.assert_not_called()

    def test_selector_rejects_unknown_backend(self):
        with mock.patch.dict(os.environ, {MLA_DECODE_KERNEL_ENV: "tokenspeed"}):
            with self.assertRaisesRegex(RuntimeError, "expected one of"):
                _get_mla_decode_kernel()

    def test_explicit_tokenspeed_is_strict(self):
        with mock.patch.dict(
            os.environ, {MLA_DECODE_KERNEL_ENV: "tokenspeed_mla"}
        ), mock.patch(f"{self.module}._is_tokenspeed_blackwell", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "requires SM100 or SM103"):
                TokenSpeedMlaDecodeImpl.support(self._configs(), self._inputs())

        with mock.patch.dict(
            os.environ, {MLA_DECODE_KERNEL_ENV: "tokenspeed_mla"}
        ), mock.patch(
            f"{self.module}._is_tokenspeed_blackwell", return_value=True
        ), mock.patch(
            f"{self.module}._load_tokenspeed_mla", return_value=False
        ):
            with self.assertRaisesRegex(RuntimeError, "tokenspeed-mla dependency"):
                TokenSpeedMlaDecodeImpl.support(self._configs(), self._inputs())

    def test_auto_falls_back_when_tokenspeed_rejects_runtime_shape(self):
        with mock.patch.dict(os.environ):
            os.environ.pop(MLA_DECODE_KERNEL_ENV, None)
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

    def test_explicit_selection_reports_unsupported_kernel_shape(self):
        with mock.patch.dict(
            os.environ, {MLA_DECODE_KERNEL_ENV: "tokenspeed_mla"}
        ), mock.patch(
            f"{self.module}._is_tokenspeed_blackwell", return_value=True
        ), mock.patch(
            f"{self.module}._load_tokenspeed_mla", return_value=True
        ), mock.patch(
            f"{self.module}.tokenspeed_mla_kernel_supported", return_value=False
        ):
            with self.assertRaisesRegex(RuntimeError, "does not support"):
                TokenSpeedMlaDecodeImpl.support(
                    self._configs(), self._inputs(prompt_lengths=(9, 9))
                )

    def test_prompt_lengths_do_not_change_decode_query_shape(self):
        with mock.patch.dict(
            os.environ, {MLA_DECODE_KERNEL_ENV: "tokenspeed_mla"}
        ), mock.patch(
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
        with mock.patch.dict(
            os.environ, {MLA_DECODE_KERNEL_ENV: "tokenspeed_mla"}
        ), mock.patch(
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
        with mock.patch.dict(
            os.environ, {MLA_DECODE_KERNEL_ENV: "tokenspeed_mla"}
        ), mock.patch(
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
        with mock.patch.dict(os.environ, {MLA_DECODE_KERNEL_ENV: "tokenspeed_mla"}):
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


if __name__ == "__main__":
    main()
