"""
Correctness tests for TrtllmGenMlaDecodeOp (FlashInfer trtllm-gen MLA decode).

The op is compared against a pure PyTorch reference implementing absorbed MLA
decode attention with the same absorb/output projections, covering:
- variable batch sizes and sequence lengths (including non page-aligned)
- block table padding alignment required by the trtllm-gen kernel
- CUDA graph capture/replay with changing kv lengths
- impl selection gating (env switch, architecture, page size)

Usage:
    python trtllm_gen_mla_decode_op_test.py
    python -m unittest trtllm_gen_mla_decode_op_test
"""

import os
from types import SimpleNamespace
from unittest import TestCase, main, mock, skipUnless

import torch

from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.mla_kv_cache_write_op import (
    MlaKVCacheWriteOp,
)
from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.paged_mla_decode import (
    MLA_DECODE_KERNEL_ENV,
)
from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.trtllm_gen_mla_impl import (
    _TRTLLM_BLOCK_ALIGNMENT_TOKENS,
    _TRTLLM_MLA_API,
    TrtllmGenMlaDecodeImpl,
    TrtllmGenMlaDecodeOp,
    trtllm_gen_dispatch_num_heads,
    trtllm_gen_kernel_supported,
)
from rtp_llm.ops import KvCacheDataType, RopeConfig
from rtp_llm.utils.model_weight import W


def _is_blackwell() -> bool:
    if not torch.cuda.is_available():
        return False
    return torch.cuda.get_device_capability()[0] == 10


SKIP_REASON = "requires Blackwell GPU and flashinfer trtllm-gen MLA API"
RUN_KERNEL = (
    torch.cuda.is_available() and _is_blackwell() and _TRTLLM_MLA_API is not None
)


class TrtllmGenMlaDependencyTest(TestCase):
    def test_declared_blackwell_target_has_trtllm_gen_api(self):
        self.assertTrue(torch.cuda.is_available(), "SM100 test target requires CUDA")
        self.assertTrue(_is_blackwell(), "test target requires SM100 or SM103")
        self.assertIsNotNone(
            _TRTLLM_MLA_API,
            "FlashInfer runtime must expose the trtllm-gen MLA API",
        )


class K3Geometry:
    """Kimi K3 MLA geometry after TP8 attention sharding."""

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


class K3Tp4Geometry(K3Geometry):
    """Kimi K3 MLA geometry after TP4 attention sharding."""

    num_heads = 24


class FakeMlaParams:
    """Duck-typed stand-in for rtp_llm_ops.FlashInferMlaAttnParams."""

    def __init__(self, kv_lens, page_indptr, page_indices):
        batch_size = len(kv_lens)
        self.qo_indptr_h = torch.arange(0, batch_size + 1, dtype=torch.int32)
        self.kvlen_h = torch.tensor(kv_lens, dtype=torch.int32)
        self.kvlen_d = self.kvlen_h.to(device="cuda")
        self.decode_page_indptr_h = torch.tensor(page_indptr, dtype=torch.int32)
        self.decode_page_indptr_d = self.decode_page_indptr_h.to(device="cuda")
        self.page_indice_d = torch.tensor(
            page_indices, dtype=torch.int32, device="cuda"
        )


class FakeLayerKVCache:
    def __init__(self, kv_cache_base: torch.Tensor):
        self.kv_cache_base = kv_cache_base


def build_kv_layout(kv_lens, page_size, num_pages, kv_dim, seed=0):
    """Build a paged kv cache plus matching block tables for each request."""
    torch.manual_seed(seed)
    kv_cache = (
        torch.randn(num_pages, page_size, kv_dim, dtype=torch.bfloat16, device="cuda")
        * 0.1
    )
    page_indptr = [0]
    page_indices = []
    next_page = 0
    for kv_len in kv_lens:
        num_blocks = (kv_len + page_size - 1) // page_size
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
    """Pure torch absorbed MLA decode reference."""
    batch_size = len(kv_lens)
    q_lat = torch.bmm(q_nope.transpose(0, 1).float(), kc_weight.float()).transpose(0, 1)
    outputs = []
    for b in range(batch_size):
        kv_len = kv_lens[b]
        pages = page_indices[page_indptr[b] : page_indptr[b + 1]]
        tokens = kv_cache[pages].reshape(-1, geo.head_dim_qk)[:kv_len]
        ckv = tokens[:, : geo.kv_lora_rank].float()
        kpe = tokens[:, geo.kv_lora_rank :].float()
        scores = (q_lat[b] @ ckv.T + q_pe[b].float() @ kpe.T) * geo.scale
        probs = torch.softmax(scores, dim=-1)
        outputs.append(probs @ ckv)
    attn_out = torch.stack(outputs).to(torch.bfloat16)  # [B, H, kv_lora_rank]
    final = torch.bmm(attn_out.transpose(0, 1).float(), vc_weight.float()).transpose(
        0, 1
    )
    return final.to(torch.bfloat16)


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
    weights = [{W.mla_kc: kc_weight, W.mla_vc: vc_weight}]
    op = TrtllmGenMlaDecodeOp(
        geo.num_heads,
        geo.kv_lora_rank,
        geo.qk_rope_head_dim,
        geo.qk_nope_head_dim,
        geo.page_size,
        1.0,  # softmax_extra_scale
        weights,
        max_bs=max_bs,
        max_context_len=max_context_len,
        is_cuda_graph=is_cuda_graph,
    )
    return op, kc_weight, vc_weight


def run_case(test, geo, kv_lens, num_pages):
    kv_cache, page_indptr, page_indices = build_kv_layout(
        kv_lens, geo.page_size, num_pages, geo.head_dim_qk
    )
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

    params = FakeMlaParams(kv_lens, page_indptr, page_indices)
    op.plan(params)
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
    rel_err = (
        actual.float() - expected.float()
    ).abs().max().item() / expected.float().abs().max().item()
    test.assertLess(
        rel_err,
        2e-2,
        f"rel_err={rel_err:.4e} for kv_lens={kv_lens}",
    )
    return op


@skipUnless(RUN_KERNEL, SKIP_REASON)
class TrtllmGenMlaDecodeOpTest(TestCase):
    def setUp(self):
        torch.cuda.set_device(0)
        self.geo = K3Geometry()

    def test_single_request(self):
        run_case(self, self.geo, [384], num_pages=8)

    def test_non_page_aligned_lengths(self):
        run_case(self, self.geo, [363, 400, 65], num_pages=16)

    def test_batch(self):
        run_case(self, self.geo, [512, 1024, 2048, 129], num_pages=64)

    def test_single_page(self):
        run_case(self, self.geo, [50, 64], num_pages=4)

    def test_block_alignment_padding(self):
        geo = self.geo
        kv_lens = [65]  # 2 blocks needed, already aligned to 128 tokens
        op = run_case(self, geo, kv_lens, num_pages=4)
        align_blocks = _TRTLLM_BLOCK_ALIGNMENT_TOKENS // geo.page_size
        self.assertEqual(op._padded_blocks % align_blocks, 0)
        # 65 tokens need 2 blocks; padding must not shrink below that
        self.assertGreaterEqual(op._padded_blocks, 2)

    def test_odd_block_count_padded(self):
        geo = self.geo
        # 192 tokens = 3 blocks -> must be padded to 4 blocks (256 tokens)
        op = run_case(self, geo, [192], num_pages=8)
        align_blocks = _TRTLLM_BLOCK_ALIGNMENT_TOKENS // geo.page_size
        self.assertEqual(op._padded_blocks % align_blocks, 0)
        self.assertGreaterEqual(op._padded_blocks, 3)

    def test_long_context(self):
        run_case(self, self.geo, [8192, 8000], num_pages=258)

    def test_explicit_backend_executes_trtllm_gen(self):
        op = run_case(self, self.geo, [1024] * 8, num_pages=128)
        self.assertEqual(op.backend_name, "trtllm_gen")
        self.assertEqual(op._kernel, "trtllm")

    def test_tp4_head_padding(self):
        geo = K3Tp4Geometry()
        op = run_case(self, geo, [384, 513], num_pages=16)
        self.assertEqual(op.num_heads, 24)
        self.assertEqual(op._dispatch_num_heads, 32)


@skipUnless(RUN_KERNEL, SKIP_REASON)
class TrtllmGenMlaDecodeCudaGraphTest(TestCase):
    def setUp(self):
        torch.cuda.set_device(0)
        self.geo = K3Geometry()

    def test_capture_and_replay_with_growing_kv(self):
        geo = self.geo
        max_bs = 4
        max_context_len = 2048
        num_pages = max_bs * (max_context_len // geo.page_size) + 8

        op, kc_weight, vc_weight = make_op(
            geo, max_bs=max_bs, max_context_len=max_context_len, is_cuda_graph=True
        )
        self.assertEqual(op._kernel, "trtllm")

        kv_cache, _, _ = build_kv_layout(
            [max_context_len] * max_bs,
            geo.page_size,
            num_pages,
            geo.head_dim_qk,
            seed=7,
        )
        q_nope = (
            torch.randn(max_bs, geo.num_heads, geo.qk_nope_head_dim, device="cuda")
            * 0.5
        ).to(torch.bfloat16)
        q_pe = (
            torch.randn(max_bs, geo.num_heads, geo.qk_rope_head_dim, device="cuda")
            * 0.5
        ).to(torch.bfloat16)

        def plan_for(kv_lens):
            _, page_indptr, page_indices = build_kv_layout(
                kv_lens, geo.page_size, num_pages, geo.head_dim_qk, seed=7
            )
            op.plan(FakeMlaParams(kv_lens, page_indptr, page_indices))

        kv_lens_0 = [256, 320, 192, 400]
        plan_for(kv_lens_0)
        op.forward(q_nope, q_pe, FakeLayerKVCache(kv_cache), 0)  # warmup before capture

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = op.forward(q_nope, q_pe, FakeLayerKVCache(kv_cache), 0)

        # Replay with grown kv lengths written into the same kv cache pages.
        kv_lens_1 = [512, 700, 300, 896]
        plan_for(kv_lens_1)
        graph.replay()
        torch.cuda.synchronize()

        _, page_indptr_1, page_indices_1 = build_kv_layout(
            kv_lens_1, geo.page_size, num_pages, geo.head_dim_qk, seed=7
        )
        expected = reference_mla_decode(
            q_nope,
            q_pe,
            kc_weight,
            vc_weight,
            kv_cache,
            kv_lens_1,
            page_indptr_1,
            page_indices_1,
            geo,
        )
        rel_err = (
            output.float() - expected.float()
        ).abs().max().item() / expected.float().abs().max().item()
        self.assertLess(rel_err, 2e-2, f"cuda graph replay rel_err={rel_err:.4e}")

    def test_group_refresh_includes_current_token_across_page_boundaries(self):
        geo = self.geo
        max_bs = 2
        max_context_len = 256
        blocks_per_request = max_context_len // geo.page_size
        num_pages = max_bs * blocks_per_request
        op, kc_weight, vc_weight = make_op(
            geo, max_bs=max_bs, max_context_len=max_context_len, is_cuda_graph=True
        )

        torch.manual_seed(17)
        kv_cache = (
            torch.randn(
                num_pages,
                geo.page_size,
                geo.head_dim_qk,
                dtype=torch.bfloat16,
                device="cuda",
            )
            * 0.1
        )
        q_nope = (
            torch.randn(max_bs, geo.num_heads, geo.qk_nope_head_dim, device="cuda")
            * 0.5
        ).to(torch.bfloat16)
        q_pe = (
            torch.randn(max_bs, geo.num_heads, geo.qk_rope_head_dim, device="cuda")
            * 0.5
        ).to(torch.bfloat16)

        block_table = torch.stack(
            [
                torch.arange(
                    batch_id * blocks_per_request,
                    (batch_id + 1) * blocks_per_request,
                    dtype=torch.int32,
                    device="cuda",
                )
                for batch_id in range(max_bs)
            ]
        )

        def compact_layout(kv_lens):
            page_indptr = [0]
            page_indices = []
            for batch_id, kv_len in enumerate(kv_lens):
                live_blocks = (kv_len + geo.page_size - 1) // geo.page_size
                start = batch_id * blocks_per_request
                page_indices.extend(range(start, start + live_blocks))
                page_indptr.append(len(page_indices))
            return page_indptr, page_indices

        initial_lens = [64, 65]
        initial_indptr, initial_indices = compact_layout(initial_lens)
        op.plan(FakeMlaParams(initial_lens, initial_indptr, initial_indices))
        graph_lengths = torch.tensor(initial_lens, dtype=torch.int32, device="cuda")
        op._refresh_graph_block_tables(block_table, graph_lengths)
        op.forward(q_nope, q_pe, FakeLayerKVCache(kv_cache), 0)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            op._refresh_graph_block_tables(block_table, graph_lengths)
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
        rel_err = (
            output.float() - expected.float()
        ).abs().max().item() / expected.float().abs().max().item()
        self.assertLess(rel_err, 2e-2, f"group refresh rel_err={rel_err:.4e}")
        torch.testing.assert_close(op._seq_lens[:max_bs], graph_lengths, rtol=0, atol=0)

    def test_new_page_is_cleared_before_graph_decode(self):
        geo = self.geo
        old_len = geo.page_size
        new_len = old_len + 1
        op, kc_weight, vc_weight = make_op(
            geo, max_bs=1, max_context_len=2 * geo.page_size, is_cuda_graph=True
        )
        kv_cache, _, _ = build_kv_layout(
            [2 * geo.page_size], geo.page_size, 2, geo.head_dim_qk, seed=31
        )
        kv_cache[1].fill_(float("nan"))
        new_kv = torch.randn(geo.head_dim_qk, dtype=torch.bfloat16, device="cuda")
        write_op = MlaKVCacheWriteOp(
            KvCacheDataType.BASE, clear_page_on_boundary=True
        )
        slot_mapping = torch.tensor([1], dtype=torch.int64, device="cuda")
        write_params = SimpleNamespace(slot_mapping=slot_mapping)
        block_table = torch.tensor([[0, 0]], dtype=torch.int32, device="cuda")
        graph_lengths = torch.tensor([old_len], dtype=torch.int32, device="cuda")
        q_nope = (
            torch.randn(1, geo.num_heads, geo.qk_nope_head_dim, device="cuda") * 0.5
        ).to(torch.bfloat16)
        q_pe = (
            torch.randn(1, geo.num_heads, geo.qk_rope_head_dim, device="cuda") * 0.5
        ).to(torch.bfloat16)

        op.plan(FakeMlaParams([old_len], [0, 1], [0]))
        op._refresh_graph_block_tables(block_table, graph_lengths)
        op.forward(q_nope, q_pe, FakeLayerKVCache(kv_cache), 0)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            write_op.forward(
                new_kv[: geo.kv_lora_rank].unsqueeze(0),
                new_kv[geo.kv_lora_rank :].unsqueeze(0),
                FakeLayerKVCache(kv_cache),
                write_params,
            )
            op._refresh_graph_block_tables(block_table, graph_lengths)
            output = op.forward(q_nope, q_pe, FakeLayerKVCache(kv_cache), 0)

        kv_cache[1].fill_(float("nan"))
        slot_mapping.fill_(geo.page_size)
        block_table[0, 1] = 1
        graph_lengths.fill_(new_len)
        graph.replay()
        torch.cuda.synchronize()

        torch.testing.assert_close(kv_cache[1, 0], new_kv, rtol=0, atol=0)
        self.assertEqual(torch.count_nonzero(kv_cache[1, 1:]).item(), 0)
        expected = reference_mla_decode(
            q_nope,
            q_pe,
            kc_weight,
            vc_weight,
            kv_cache,
            [new_len],
            [0, 2],
            [0, 1],
            geo,
        )
        self.assertTrue(torch.isfinite(output).all())
        rel_err = (
            (output.float() - expected.float()).abs().max()
            / expected.float().abs().max()
        ).item()
        self.assertLess(rel_err, 2e-2, f"new-page rel_err={rel_err:.4e}")


class TrtllmGenMlaDecodeSupportTest(TestCase):
    def _attn_configs(self):
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

    def _attn_inputs(self, is_prefill=False):
        return SimpleNamespace(is_prefill=is_prefill)

    def test_disabled_by_default(self):
        with mock.patch.dict(os.environ):
            os.environ.pop(MLA_DECODE_KERNEL_ENV, None)
            self.assertFalse(
                TrtllmGenMlaDecodeImpl.support(
                    self._attn_configs(), self._attn_inputs()
                )
            )

    def test_env_gating(self):
        with mock.patch.dict(
            os.environ, {MLA_DECODE_KERNEL_ENV: "trtllm_gen"}
        ), mock.patch(
            "rtp_llm.models_py.modules.factory.attention.cuda_mla_impl."
            "trtllm_gen_mla_impl._TRTLLM_MLA_API",
            object(),
        ), mock.patch(
            "rtp_llm.models_py.modules.factory.attention.cuda_mla_impl."
            "trtllm_gen_mla_impl._is_blackwell",
            return_value=True,
        ):
            self.assertTrue(
                TrtllmGenMlaDecodeImpl.support(
                    self._attn_configs(), self._attn_inputs()
                )
            )

    def test_explicit_selection_fails_when_package_is_missing(self):
        with mock.patch.dict(
            os.environ, {MLA_DECODE_KERNEL_ENV: "trtllm_gen"}
        ), mock.patch(
            "rtp_llm.models_py.modules.factory.attention.cuda_mla_impl."
            "trtllm_gen_mla_impl._TRTLLM_MLA_API",
            None,
        ):
            with self.assertRaisesRegex(RuntimeError, "requires flashinfer.mla"):
                TrtllmGenMlaDecodeImpl.support(
                    self._attn_configs(), self._attn_inputs()
                )

    def test_rejects_prefill_and_sparse(self):
        with mock.patch.dict(
            os.environ, {MLA_DECODE_KERNEL_ENV: "trtllm_gen"}
        ), mock.patch(
            "rtp_llm.models_py.modules.factory.attention.cuda_mla_impl."
            "trtllm_gen_mla_impl._TRTLLM_MLA_API",
            object(),
        ), mock.patch(
            "rtp_llm.models_py.modules.factory.attention.cuda_mla_impl."
            "trtllm_gen_mla_impl._is_blackwell",
            return_value=True,
        ):
            self.assertFalse(
                TrtllmGenMlaDecodeImpl.support(
                    self._attn_configs(), self._attn_inputs(is_prefill=True)
                )
            )
            configs = self._attn_configs()
            configs.is_sparse = True
            with self.assertRaisesRegex(RuntimeError, "does not support sparse MLA"):
                TrtllmGenMlaDecodeImpl.support(configs, self._attn_inputs())

    def test_page_size_matches_upstream_support(self):
        with mock.patch.dict(
            os.environ, {MLA_DECODE_KERNEL_ENV: "trtllm_gen"}
        ), mock.patch(
            "rtp_llm.models_py.modules.factory.attention.cuda_mla_impl."
            "trtllm_gen_mla_impl._TRTLLM_MLA_API",
            object(),
        ), mock.patch(
            "rtp_llm.models_py.modules.factory.attention.cuda_mla_impl."
            "trtllm_gen_mla_impl._is_blackwell",
            return_value=True,
        ):
            configs = self._attn_configs()
            for page_size in (32, 64):
                configs.kernel_tokens_per_block = page_size
                self.assertTrue(
                    TrtllmGenMlaDecodeImpl.support(configs, self._attn_inputs())
                )
            for page_size in (16, 128, 256):
                configs.kernel_tokens_per_block = page_size
                with self.assertRaisesRegex(RuntimeError, "page sizes 32 or 64"):
                    TrtllmGenMlaDecodeImpl.support(configs, self._attn_inputs())

    def test_rejects_unsupported_arch_and_geometry(self):
        with mock.patch.dict(
            os.environ, {MLA_DECODE_KERNEL_ENV: "trtllm_gen"}
        ), mock.patch(
            "rtp_llm.models_py.modules.factory.attention.cuda_mla_impl."
            "trtllm_gen_mla_impl._TRTLLM_MLA_API",
            object(),
        ):
            with mock.patch(
                "rtp_llm.models_py.modules.factory.attention.cuda_mla_impl."
                "trtllm_gen_mla_impl._is_blackwell",
                return_value=False,
            ):
                with self.assertRaisesRegex(RuntimeError, "requires SM100 or SM103"):
                    TrtllmGenMlaDecodeImpl.support(
                        self._attn_configs(), self._attn_inputs()
                    )
            configs = self._attn_configs()
            configs.head_num = 129
            with self.assertRaisesRegex(RuntimeError, "does not support geometry"):
                TrtllmGenMlaDecodeImpl.support(configs, self._attn_inputs())

    def test_impl_clears_new_pages_only_for_cuda_graph(self):
        configs = self._attn_configs()
        configs.nope_head_dim = 128
        configs.softmax_extra_scale = 1.0
        configs.rope_config = RopeConfig()
        configs.rope_config.is_neox_style = False
        attn_inputs = SimpleNamespace(sequence_lengths=torch.empty(0))

        module = (
            "rtp_llm.models_py.modules.factory.attention.cuda_mla_impl."
            "trtllm_gen_mla_impl"
        )
        with mock.patch(f"{module}.TrtllmGenMlaDecodeOp"), mock.patch(
            f"{module}.NewMlaRotaryEmbeddingOp"
        ), mock.patch(f"{module}.MlaKVCacheWriteOp") as write_op_cls, mock.patch(
            f"{module}.MlaFlashInferImplBase.__init__", return_value=None
        ):
            for is_cuda_graph in (False, True):
                TrtllmGenMlaDecodeImpl(
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


class TrtllmGenKernelDispatchTest(TestCase):
    """Dispatch and head-padding checks (pure Python, no GPU required)."""

    NUM_SMS = 148

    def test_aligned_heads_do_not_need_padding(self):
        dispatch = trtllm_gen_dispatch_num_heads
        for num_heads in (8, 12, 16, 32, 64, 128):
            self.assertEqual(
                dispatch(num_heads, 128, 8192, self.NUM_SMS), num_heads
            )

    def test_k3_tp4_pads_to_next_executable_tile(self):
        dispatch = trtllm_gen_dispatch_num_heads
        self.assertEqual(dispatch(24, 1, 8192, self.NUM_SMS), 32)
        self.assertEqual(dispatch(24, 128, 8192, self.NUM_SMS), 32)
        self.assertTrue(trtllm_gen_kernel_supported(24, 128, 8192, self.NUM_SMS))

    def test_wide_heads_pad_only_when_keeps_kernel_needs_it(self):
        dispatch = trtllm_gen_dispatch_num_heads
        self.assertEqual(dispatch(96, 1, 16384, self.NUM_SMS), 96)
        self.assertEqual(dispatch(96, 2, 16384, self.NUM_SMS), 128)
        self.assertEqual(dispatch(96, 6, 4096, self.NUM_SMS), 96)
        self.assertEqual(dispatch(96, 8, 4096, self.NUM_SMS), 128)
        self.assertEqual(dispatch(96, 24, 1024, self.NUM_SMS), 96)
        self.assertEqual(dispatch(96, 32, 1024, self.NUM_SMS), 128)

    def test_cuda_graph_keeps_max_capture_dispatch_heads(self):
        op = object.__new__(TrtllmGenMlaDecodeOp)
        op.num_heads = 96
        op.token_per_block = 64
        op.use_cuda_graph = True
        op._max_context_len = 16384
        op._num_sms = self.NUM_SMS
        op._dispatch_num_heads = 128
        op._attn_output = object()
        op._kernel = "trtllm"
        op._ensure_trtllm_ready = mock.Mock()
        op._metadata = SimpleNamespace(
            plan=mock.Mock(),
            block_tables=None,
            seq_lens=None,
            column_indices=None,
            batch_size=1,
            padded_blocks=128,
            max_seq_len=16384,
        )
        params = SimpleNamespace(
            qo_indptr_h=torch.tensor([0, 1], dtype=torch.int32),
            kvlen_h=torch.tensor([16384], dtype=torch.int32),
        )

        op.plan(params)

        self.assertEqual(
            trtllm_gen_dispatch_num_heads(96, 1, 16384, self.NUM_SMS), 96
        )
        self.assertEqual(op._dispatch_num_heads, 128)
        op._metadata.plan.assert_called_once_with(params)

        empty_params = SimpleNamespace(
            qo_indptr_h=torch.tensor([0], dtype=torch.int32),
            kvlen_h=torch.empty(0, dtype=torch.int32),
        )
        with self.assertRaisesRegex(RuntimeError, "does not support"):
            op.plan(empty_params)
        op._metadata.plan.assert_called_once_with(params)

    def test_invalid_shapes_rejected(self):
        self.assertFalse(trtllm_gen_kernel_supported(0, 8, 1024, self.NUM_SMS))
        self.assertFalse(trtllm_gen_kernel_supported(129, 8, 1024, self.NUM_SMS))
        self.assertFalse(trtllm_gen_kernel_supported(96, 0, 1024, self.NUM_SMS))
        self.assertFalse(trtllm_gen_kernel_supported(96, 8, 0, self.NUM_SMS))
        self.assertFalse(trtllm_gen_kernel_supported(96, 8, 1024, 0))
        self.assertIsNone(
            trtllm_gen_dispatch_num_heads(129, 8, 1024, self.NUM_SMS)
        )


if __name__ == "__main__":
    main()
