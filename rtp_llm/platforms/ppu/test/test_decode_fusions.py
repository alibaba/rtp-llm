"""Bitwise old/new boundaries, including changing inputs in CUDA Graph replay."""

import dataclasses
import importlib
import unittest

import torch

from rtp_llm.platforms.ppu.kernels.cuda.ppu_fp4_indexer import topk_decode
from rtp_llm.platforms.ppu.kernels.cuda.ppu_sglang_permute import fused_permute
from rtp_llm.platforms.ppu.kernels.ppu_decode_fused_quant import (
    inverse_rope_fp8_permute,
    swiglu_fp8,
)
from rtp_llm.platforms.ppu.kernels.ppu_gate_padding import gate_with_padding
from rtp_llm.platforms.ppu.kernels.ppu_hash_gate import hash_gate_with_padding
from rtp_llm.platforms.ppu.kernels.ppu_hc_sum import hc_post_sum
from rtp_llm.platforms.ppu.kernels.ppu_inverse_rope import inverse_rope_inplace
from rtp_llm.platforms.ppu.kernels.ppu_qkv_norm import normalize_decode_qkv
from rtp_llm.platforms.ppu.kernels.ppu_query_slice import prepare_indexer_query
from rtp_llm.platforms.ppu.kernels.ppu_shared_swiglu import silu_mul_merged_bf16
from rtp_llm.platforms.ppu.models.dsv4.decode_query import slice_compressor_query
from rtp_llm.platforms.ppu.modules.fused_moe.mxfp4_low_latency import pad_topk
from rtp_llm.platforms.ppu.modules.linear.fp8_linear import quantize_ppu_fp8_activation


@dataclasses.dataclass(frozen=True)
class QueryMetadata:
    positions: torch.Tensor
    b_idx: torch.Tensor
    state_slots: torch.Tensor
    kv_slots: torch.Tensor
    token_to_req: torch.Tensor
    compressed_lens_per_token: torch.Tensor = None
    is_batched: bool = True
    seq_start_per_req: torch.Tensor = None
    cu_seq_per_req: torch.Tensor = None
    has_prefix: bool = True


@unittest.skipUnless(
    torch.cuda.is_available() and torch.cuda.get_device_name() == "ZW-M890P",
    "requires a PPU M890P",
)
class DecodeFusionsTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(890424)

    @torch.inference_mode()
    def test_hc_pre_quantized(self):
        from rtp_llm.models_py.modules.base.cuda.norm import RMSNorm
        from rtp_llm.platforms.ppu.models.dsv4.ppu_hc import PpuHCUnit

        unit = PpuHCUnit(
            torch.randn((24, 16384), device="cuda") / 128,
            torch.randn(24, device="cuda"),
            torch.ones(3, device="cuda"),
            options={"DSV4_MHC_PRE_GEMM_BACKEND": "deepgemm_deterministic"},
            dim=4096,
            hc_mult=4,
            hc_sinkhorn_iters=20,
            norm_eps=1e-6,
            hc_eps=1e-6,
            allow_graph=True,
            fuse_prenorm=True,
            fuse_norm=True,
            fuse_pre_quant=True,
        )
        norm = RMSNorm(torch.rand(4096, device="cuda", dtype=torch.bfloat16), 1e-6)
        for b, q in ((1, 1), (8, 1), (8, 4), (32, 4), (128, 4)):
            x = torch.randn((b, q, 4, 4096), device="cuda", dtype=torch.bfloat16)

            def old():
                y, post, comb = unit.pre_norm(x, norm, tp_size=1, tp_rank=0)
                quantized = quantize_ppu_fp8_activation(
                    y.reshape(-1, 4096), quantization="v2_column"
                )
                return y, post, comb, quantized

            for std in (0.0, 1e-30, 1.0, 1e8):
                x.normal_().mul_(std)
                self.replay(old, lambda: unit.pre_norm_quantized(x, norm), x.normal_)

    @torch.inference_mode()
    def test_qkv_swa_cache(self):
        from rtp_llm.models_py.modules.dsv4.fp8._swa_kv_insert_triton import (
            quantize_and_insert_k_cache,
        )

        for b, q in ((1, 1), (8, 1), (8, 4), (32, 4), (128, 4)):
            rows = b * q
            x = torch.randn((b, q, 1536), device="cuda", dtype=torch.bfloat16)
            qw = torch.randn(1024, device="cuda", dtype=torch.bfloat16)
            kw = torch.randn(512, device="cuda", dtype=torch.bfloat16)
            angle = torch.randn((rows, 32), device="cuda")
            freq = torch.polar(torch.ones_like(angle), angle)
            for block_size in (1, 64, 256):
                blocks = (rows + block_size - 1) // block_size + 1
                stride = block_size * 584 + 64
                raw0 = torch.full(
                    (blocks * stride,), 133, device="cuda", dtype=torch.uint8
                )
                raw1 = raw0.clone()
                cache0 = raw0.as_strided((blocks, block_size, 584), (stride, 584, 1))
                cache1 = raw1.as_strided((blocks, block_size, 584), (stride, 584, 1))
                slots = torch.arange(rows, device="cuda")
                slots[::3] = -1

                def old():
                    y = normalize_decode_qkv(x, qw, kw, freq, 1e-6, quantize_q=True)
                    quantize_and_insert_k_cache(y[1].view(-1, 512), cache0, slots)
                    return y, raw0

                def new():
                    y = normalize_decode_qkv(
                        x,
                        qw,
                        kw,
                        freq,
                        1e-6,
                        quantize_q=True,
                        swa_cache=cache1,
                        slot_mapping=slots,
                    )
                    return y, raw1

                def mutate():
                    x.normal_()
                    slots.copy_(torch.randperm(rows, device="cuda"))
                    slots[1::3] = -1
                    raw0.fill_(133)
                    raw1.fill_(133)

                for std in (0.0, 1e-30, 1.0, 1e8):
                    mutate()
                    x.mul_(std)
                    self.replay(old, new, mutate)

    def same(self, actual, expected):
        if torch.is_tensor(expected):
            self.assertEqual(actual.shape, expected.shape)
            self.assertEqual(actual.dtype, expected.dtype)
            # Include FP8 payload, signed zero and NaN bits in this comparison.
            # Singleton strided views may be considered contiguous by PyTorch;
            # materialize a flat unit-stride buffer before reinterpreting bytes.
            a = torch.empty(actual.numel(), device=actual.device, dtype=actual.dtype)
            e = torch.empty_like(a)
            a.copy_(actual.reshape(-1))
            e.copy_(expected.reshape(-1))
            self.assertTrue(
                torch.equal(
                    a.view(torch.uint8),
                    e.view(torch.uint8),
                )
            )
        elif dataclasses.is_dataclass(expected):
            for field in dataclasses.fields(expected):
                a, e = getattr(actual, field.name), getattr(expected, field.name)
                if torch.is_tensor(e):
                    self.assertEqual(a.stride(), e.stride())
                self.same(a, e)
        elif isinstance(expected, (tuple, list)):
            self.assertEqual(len(actual), len(expected))
            for a, e in zip(actual, expected):
                self.same(a, e)
        else:
            self.assertEqual(actual, expected)

    def replay(self, old, new, mutate):
        expected = old()
        self.same(new(), expected)
        for _ in range(3):
            new()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual = new()
        for _ in range(3):
            mutate()
            expected = old()
            graph.replay()
            self.same(actual, expected)

    @torch.inference_mode()
    def test_quantization_boundaries(self):
        for b, q in ((1, 1), (3, 2), (8, 4), (32, 4)):
            x = torch.randn((b, q, 4096), device="cuda", dtype=torch.bfloat16)
            for layout in ("v2_column", "v2_row"):
                for limit in (0.0, 7.0):
                    with self.subTest(b=b, q=q, layout=layout, limit=limit):
                        self.replay(
                            lambda: quantize_ppu_fp8_activation(
                                silu_mul_merged_bf16(x, limit).reshape(-1, 2048),
                                quantization=layout,
                            ),
                            lambda: swiglu_fp8(x, limit, quantization=layout),
                            lambda: x.normal_(std=10),
                        )
            o = torch.randn((b, q, 64, 512), device="cuda", dtype=torch.bfloat16)
            for per_batch in (False, True):
                angle = torch.randn((b if per_batch else b * q, 32), device="cuda")
                freq = torch.polar(torch.ones_like(angle), angle)

                def old_inverse():
                    y = o.clone()
                    inverse_rope_inplace(y, freq, 64)
                    payload, scale = quantize_ppu_fp8_activation(
                        y.reshape(-1, 4096), quantization="v2_row"
                    )
                    return fused_permute(
                        payload.view(b * q, 8, 4096), scale.view(b * q, 8, 32)
                    )

                for column_major in (False, True):
                    self.replay(
                        old_inverse,
                        lambda: inverse_rope_fp8_permute(
                            o, freq, 64, 8, column_major_scales=column_major
                        ),
                        o.normal_,
                    )

    @torch.inference_mode()
    def test_inverse_rope_consumer_scale_layout(self):
        from deep_gemm.jit_kernels.utils import get_col_major_tma_aligned_tensor

        for m in (1, 3, 8, 17, 32, 127):
            for groups in (1, 2, 4, 8):
                x = torch.randn(
                    (m, 1, groups * 8, 512), device="cuda", dtype=torch.bfloat16
                )
                angle = torch.randn((m, 32), device="cuda")
                freq = torch.polar(torch.ones_like(angle), angle)

                def old():
                    return inverse_rope_fp8_permute(x, freq, 64, groups)

                def new():
                    return inverse_rope_fp8_permute(
                        x, freq, 64, groups, column_major_scales=True
                    )

                self.replay(old, new, x.normal_)
                _, scales = new()
                self.assertEqual(scales.stride(), (m * 32, 1, m))
                self.assertIs(get_col_major_tma_aligned_tensor(scales), scales)

    @torch.inference_mode()
    def test_attention_projection_tiles(self):
        from rtp_llm.platforms.ppu.models.dsv4.ppu_wo_a import PpuWoAFp8Linear
        from rtp_llm.platforms.ppu.modules.linear.fp8_linear import PpuFp8Linear

        for n, k, groups in ((32768, 1024, 1), (4096, 8192, 1), (8192, 4096, 8)):
            weight = torch.randn((n, k), device="cuda", dtype=torch.bfloat16).to(
                torch.float8_e4m3fn
            )
            scale = torch.ones((n // 128, k // 128), device="cuda")
            if groups == 1:
                linear = PpuFp8Linear(
                    weight, scale, scale_is_prepared=True, quantization="v2_column"
                )
            else:
                linear = PpuWoAFp8Linear(
                    weight,
                    scale.to(torch.float8_e8m0fnu),
                    groups=groups,
                    k_local=k,
                    sglang_layout=True,
                    quantization="v2_row",
                )
            # Include unqualified M values to exercise the normal fallback.
            for m in (1, 8, 16, 17, 24, 32, 33):
                shape = (m, k) if groups == 1 else (m, groups, k)
                x = torch.randn(shape, device="cuda", dtype=torch.bfloat16)

                def old():
                    linear._decode_attention_gemm = False
                    return linear(x)

                def new():
                    linear._decode_attention_gemm = True
                    return linear(x)

                for magnitude in (1e-30, 1.0, 1e8):
                    x.normal_().mul_(magnitude)
                    self.replay(old, new, x.normal_)

    @torch.inference_mode()
    def test_qkv_norm_quant(self):
        for b, q in ((1, 1), (3, 2), (8, 4), (32, 4)):
            x = torch.randn((b, q, 1536), device="cuda", dtype=torch.bfloat16)
            qw = torch.randn(1024, device="cuda", dtype=torch.bfloat16)
            kw = torch.randn(512, device="cuda", dtype=torch.bfloat16)
            angle = torch.randn((b * q, 32), device="cuda")
            freq = torch.polar(torch.ones_like(angle), angle)

            def old():
                qr, kv = normalize_decode_qkv(x, qw, kw, freq, 1e-6)
                return (
                    qr,
                    kv,
                    quantize_ppu_fp8_activation(
                        qr.reshape(-1, 1024), quantization="v2_column"
                    ),
                )

            self.replay(
                old,
                lambda: normalize_decode_qkv(x, qw, kw, freq, 1e-6, quantize_q=True),
                x.normal_,
            )

    @torch.inference_mode()
    def test_routes_and_inactive_mask(self):
        from rtp_llm.models_py.modules.factory.fused_moe.utils.fp8_fp4.gate import (
            _select_routes_with_nonfinite_fallback,
        )
        from rtp_llm.models_py.triton_kernels.moe.gate_fused import (
            fused_sqrtsoftplus_gate,
        )

        for hashed in (False, True):
            for m in (1, 8, 32, 128):
                for k in (1, 6, 16):
                    scores = torch.randn((m, 256), device="cuda")
                    bias = torch.randn(256, device="cuda")
                    ids = torch.randint(-1024, 1024, (m,), device="cuda")
                    table = torch.randint(0, 256, (1024, k), device="cuda")
                    mask = torch.ones(m, dtype=torch.bool, device="cuda")

                    def old():
                        if hashed:
                            active = torch.nn.functional.softplus(scores).sqrt()
                            w, i = _select_routes_with_nonfinite_fallback(
                                active,
                                active,
                                k,
                                2.5,
                                True,
                                table[ids].long(),
                                router_logits=scores,
                            )
                        else:
                            w, i = fused_sqrtsoftplus_gate(
                                scores, bias, k, route_scale=2.5
                            )
                        i, w = pad_topk(i, w, mask)
                        return w, i

                    def new():
                        if hashed:
                            return hash_gate_with_padding(
                                scores, ids, table, 2.5, active_token_mask=mask
                            )
                        return gate_with_padding(
                            scores, bias, k, 2.5, active_token_mask=mask
                        )

                    def mutate():
                        scores.normal_(std=100)
                        scores[0, 0] = float("nan")
                        mask.logical_not_()
                        ids.random_(-1024, 1024)

                    with self.subTest(hashed=hashed, m=m, k=k):
                        self.replay(old, new, mutate)

    @torch.inference_mode()
    def test_hc_sum_alias_and_rounding(self):
        from rtp_llm.models_py.modules.dsv4 import tilelang_kernels  # noqa: F401
        from rtp_llm.models_py.triton_kernels.moe.shared_expert import (
            fused_moe_epilogue,
        )

        old_post = importlib.import_module(
            "rtp_llm.models_py.3rdparty.tile_kernels.mhc.post_kernel"
        ).mhc_post_fwd
        for b, q in ((1, 1), (8, 1), (8, 4), (32, 4)):
            r = torch.randn((b, q, 4096), device="cuda", dtype=torch.bfloat16)
            s = torch.randn_like(r)
            res = torch.randn((b, q, 4, 4096), device="cuda", dtype=torch.bfloat16)
            post = torch.rand((b, q, 4, 1), device="cuda")
            comb = torch.rand((b, q, 4, 4), device="cuda")

            def old():
                y = fused_moe_epilogue(
                    r.reshape(-1, 4096), s.reshape(-1, 4096), torch.bfloat16
                ).view_as(r)
                return old_post(y, res.clone(), post, comb, enable_pdl=False)

            def new():
                out = hc_post_sum(r, s, res, post, comb)
                self.assertEqual(out.data_ptr(), res.data_ptr())
                return out

            def mutate():
                r.normal_()
                s.normal_()
                res.normal_()
                post.uniform_()
                comb.uniform_()

            self.replay(old, new, mutate)

    @torch.inference_mode()
    def test_query_metadata_layout(self):
        for b in (0, 1, 8, 32):
            width = 4
            x = torch.randn((b, width, 4096), device="cuda", dtype=torch.bfloat16)
            qr = torch.randn((b, width, 1024), device="cuda", dtype=torch.bfloat16)
            start = torch.arange(b, device="cuda", dtype=torch.int32) + 32768
            pos = (start[:, None] + torch.arange(width, device="cuda")).reshape(-1)
            req = torch.arange(b, device="cuda").repeat_interleave(width)
            slots = torch.arange(b * width, device="cuda") + 256
            for copy_q in (False, True):
                for has_length in (False, True):
                    meta = QueryMetadata(
                        pos,
                        req,
                        slots,
                        slots + 32,
                        req,
                        (pos + 1) // 4 if has_length else None,
                    )
                    for index in range(width):

                        def old():
                            return (
                                x[:, index : index + 1].contiguous(),
                                (
                                    qr[:, index : index + 1].contiguous()
                                    if copy_q
                                    else None
                                ),
                                start + index,
                                slice_compressor_query(meta, b, width, index),
                            )

                        def new():
                            return prepare_indexer_query(
                                x, qr if copy_q else None, start, meta, index
                            )

                        def mutate():
                            x.normal_()
                            qr.normal_()
                            start.add_(1)
                            pos.add_(1)
                            slots.add_(7)

                        self.replay(old, new, mutate)

    @torch.inference_mode()
    def test_c4_state_ring_and_cache(self):
        from rtp_llm.platforms.ppu.kernels.cuda.ppu_fp4_indexer import (
            compress4_decode,
            compress4_decode_store,
            norm_rope_store,
        )

        for m in (1, 8, 32):
            state = torch.randn((m * 3, 4, 512), device="cuda")
            reference_state = state.clone()
            cache = torch.zeros((m, 4352), device="cuda", dtype=torch.uint8)
            reference_cache = cache.clone()
            x = torch.randn((m, 512), device="cuda")
            ape = torch.randn((8, 128), device="cuda")
            weight = torch.rand(128, device="cuda")
            angle = torch.randn((64, 32), device="cuda")
            freqs = torch.view_as_real(
                torch.polar(torch.ones_like(angle), angle)
            ).flatten(-2)
            plan = torch.zeros((m, 4), device="cuda", dtype=torch.int32)
            plan[:, 1] = -1
            slots = torch.full((m,), -1, device="cuda", dtype=torch.int32)

            def old():
                y = compress4_decode(reference_state, x, ape, plan.view(torch.uint8))
                norm_rope_store(
                    y,
                    plan.view(torch.uint8),
                    weight,
                    1e-6,
                    freqs,
                    slots,
                    reference_cache,
                    64,
                    is_decode=True,
                )

            def new():
                self.assertTrue(
                    compress4_decode_store(
                        state,
                        x,
                        ape,
                        plan.view(torch.uint8),
                        weight,
                        1e-6,
                        freqs,
                        slots,
                        cache,
                        64,
                    )
                )

            for _ in range(3):
                old()
                new()
            old_graph, new_graph = torch.cuda.CUDAGraph(), torch.cuda.CUDAGraph()
            with torch.cuda.graph(old_graph):
                old()
            with torch.cuda.graph(new_graph):
                new()
            for seq in range(1, 18):
                current = ((seq - 1) // 4) % 3
                rows = [
                    [
                        seq,
                        r * 12 + (seq - 1) % 12,
                        r * 3 + (current + 2) % 3,
                        r * 3 + current,
                    ]
                    for r in range(m)
                ]
                if m > 1:
                    rows[-1][1] = -1
                plan.copy_(torch.tensor(rows, device="cuda", dtype=torch.int32))
                slots.copy_(
                    torch.arange(m, device="cuda", dtype=torch.int32) * 64 + seq // 4
                )
                if m > 1:
                    slots[-1] = -1
                x.normal_()
                old_graph.replay()
                new_graph.replay()
                self.same(state, reference_state)
                self.same(cache, reference_cache)
            self.assertFalse(
                compress4_decode_store(
                    state,
                    x,
                    ape,
                    plan.view(torch.uint8),
                    weight,
                    1e-6,
                    freqs,
                    slots,
                    cache,
                    128,
                )
            )

    @torch.inference_mode()
    def test_c4_decode_queries_ring_and_cache(self):
        from rtp_llm.platforms.ppu.kernels.cuda.ppu_fp4_indexer import (
            compress4_decode_store,
            project_compress4_decode_queries_store,
        )
        from rtp_llm.platforms.ppu.kernels.ppu_fp4_indexer_cache import (
            build_decode_plan,
        )

        for batch, width, entries in (
            (1, 2, 8),
            (3, 3, 12),
            (8, 4, 12),
            (17, 2, 12),
            (32, 4, 8),
        ):
            for dtype_mask in (0, 31, 26):
                dtypes = [
                    torch.int64 if dtype_mask & (1 << i) else torch.int32
                    for i in range(5)
                ]
                # Permuted physical pages and a padded block-table stride.
                pages = torch.randperm(batch * 3, device="cuda").reshape(batch, 3)
                table = torch.zeros((batch, 5), device="cuda", dtype=dtypes[4])
                table[:, :3] = pages
                table = table[:, :3]
                state = torch.randn((batch * 3 * entries, 512), device="cuda")
                ref_state = state.clone()
                cache = torch.zeros((batch, 4352), device="cuda", dtype=torch.uint8)
                ref_cache = cache.clone()
                x = torch.randn(
                    (batch, width, 4096), device="cuda", dtype=torch.bfloat16
                )
                projection = (
                    torch.randn((512, 4096), device="cuda", dtype=torch.bfloat16) / 64
                )
                ape = torch.randn((8, 128), device="cuda")
                weight = torch.rand(128, device="cuda")
                angle = torch.randn((128, 32), device="cuda")
                freqs = torch.view_as_real(
                    torch.polar(torch.ones_like(angle), angle)
                ).flatten(-2)
                values = [
                    torch.zeros(batch * width, device="cuda", dtype=d)
                    for d in dtypes[:4]
                ]
                values[2].fill_(-1)
                values[3].fill_(-1)
                meta = QueryMetadata(*values, token_to_req=values[1])
                starts = torch.zeros(batch, device="cuda", dtype=torch.int32)

                def old():
                    for index in range(width):
                        prepared = prepare_indexer_query(x, None, starts, meta, index)
                        if prepared is None:
                            step_x = x[:, index : index + 1].contiguous()
                            step = slice_compressor_query(meta, batch, width, index)
                        else:
                            step_x, _, _, step = prepared
                        projected = torch.mm(
                            step_x.reshape(batch, 4096),
                            projection.t(),
                            out_dtype=torch.float32,
                        )
                        plan, slots = build_decode_plan(step, table, entries, 16)
                        self.assertTrue(
                            compress4_decode_store(
                                ref_state,
                                projected,
                                ape,
                                plan,
                                weight,
                                1e-6,
                                freqs,
                                slots,
                                ref_cache,
                                64,
                            )
                        )

                def new():
                    self.assertTrue(
                        project_compress4_decode_queries_store(
                            state,
                            x,
                            projection,
                            ape,
                            weight,
                            1e-6,
                            freqs,
                            meta,
                            table,
                            entries,
                            16,
                            cache,
                            64,
                        )
                    )

                old()
                new()
                old_graph, new_graph = torch.cuda.CUDAGraph(), torch.cuda.CUDAGraph()
                with torch.cuda.graph(old_graph):
                    old()
                with torch.cuda.graph(new_graph):
                    new()
                requests = (
                    torch.arange(batch, device="cuda")
                    .flip(0)[:, None]
                    .expand(-1, width)
                )
                for start in (0, 1, 3, 7, 11, 15, 23, 31, 47, 63):
                    pos = (
                        start
                        + torch.arange(width, device="cuda")[None, :]
                        + torch.arange(batch, device="cuda")[:, None] % 4
                    )
                    physical = table[requests, (pos // 16) % 3].long()
                    writes = physical * entries + pos % entries
                    slots = requests * 64 + pos // 4
                    if batch > 1:
                        # Independent inactive state, position, and cache slots.
                        writes[-1, 0] = -1
                        pos[0, -1] = -1
                        slots[0, 0] = -1
                    for target, source in zip(values, (pos, requests, writes, slots)):
                        target.copy_(source.flatten())
                    x.normal_()
                    old_graph.replay()
                    new_graph.replay()
                    self.same(state, ref_state)
                    self.same(cache, ref_cache)
                before = state.clone(), cache.clone()
                self.assertFalse(
                    project_compress4_decode_queries_store(
                        state,
                        x,
                        projection,
                        ape,
                        weight,
                        1e-6,
                        freqs,
                        meta,
                        table,
                        entries,
                        16,
                        cache,
                        128,
                    )
                )
                self.same(state, before[0])
                self.same(cache, before[1])

    @torch.inference_mode()
    def test_hc_head_preserves_reference_projection(self):
        from rtp_llm.models_py.modules.dsv4.hc.fallback_impl import FallbackHCHead
        from rtp_llm.platforms.ppu.models.dsv4.ppu_hc_head import PpuDecodeHCHead

        fn = torch.randn((4, 16384), device="cuda") * 0.01
        base = torch.randn(4, device="cuda")
        scale = torch.ones(1, device="cuda")
        kwargs = dict(dim=4096, hc_mult=4, norm_eps=1e-6, hc_eps=1e-6)
        old, new = FallbackHCHead(fn, base, scale, **kwargs), PpuDecodeHCHead(
            fn, base, scale, **kwargs
        )
        for b, q in ((1, 1), (3, 1), (8, 4), (32, 4)):
            x = torch.randn((b, q, 4, 4096), device="cuda", dtype=torch.bfloat16)
            self.replay(lambda: old.head(x), lambda: new.head(x), x.normal_)
        # Unsupported width and oversized batches keep the fallback implementation.
        x = torch.empty((0, 4, 4096), device="cuda", dtype=torch.bfloat16)
        self.same(new.head(x), old.head(x))

    @torch.inference_mode()
    def test_topk_ordering(self):
        for k in (512, 1024):
            for n in (256, 8192, 9216, 20000, 40000):
                scores = torch.randn((8, n), device="cuda")
                lengths = torch.randint(
                    max(0, n - 511), n + 1, (8,), device="cuda", dtype=torch.int32
                )
                out = torch.empty((8, k), device="cuda", dtype=torch.int32)
                reference = torch.empty_like(out)
                self.replay(
                    lambda: topk_decode(scores, lengths, reference),
                    lambda: topk_decode(scores, lengths, out, canonical_in_kernel=True),
                    scores.normal_,
                )
                scores.zero_()
                self.same(
                    topk_decode(scores, lengths, out, canonical_in_kernel=True),
                    topk_decode(scores, lengths, reference),
                )


if __name__ == "__main__":
    unittest.main()
