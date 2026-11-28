"""Parity, CUDA Graph audit, and microbenchmark tests for Track B-2 decode small-ops."""

from __future__ import annotations

import ctypes
import statistics
import unittest
from contextlib import contextmanager
from typing import Callable
from unittest.mock import patch

import torch

from rtp_llm.models_py.kernels.cuda.fp8_kernel import (
    sgl_per_token_group_quant_fp8,
)
from rtp_llm.models_py.kernels.cuda.quant_layouts import dequantize_fp8_weight
from rtp_llm.models_py.modules.base.cuda.indexer_op import _unpack_ue8m0_scale
from rtp_llm.models_py.modules.dsv4._fused_rmsnorm_rope_triton import (
    fused_rmsnorm_rope,
)
from rtp_llm.models_py.modules.dsv4.decode.indexer_decode_op import (
    IndexerDecodeV4Op,
    _fast_path_available,
)
from rtp_llm.models_py.modules.dsv4.fp8.attention import AttentionFP8
from rtp_llm.models_py.modules.dsv4.fp8.decode.compute_qkv import (
    DecodeQKV,
    decode_compute_qkv,
)
from rtp_llm.models_py.modules.dsv4.fp8.decode.mega_csa_runtime import MegaCSARuntime
from rtp_llm.models_py.modules.dsv4.fp8.decode.mega_csa_weights import (
    FLASH_GEOMETRY,
    HEAD_DIM,
    INDEX_HEAD_DIM,
    INDEX_HEADS,
    PRO_GEOMETRY,
    ROPE_DIM,
    CSAGeometry,
)
from rtp_llm.models_py.modules.dsv4.fp8.decode.output_proj import decode_output_proj
from rtp_llm.models_py.modules.dsv4.indexer_topk import select_indexer_topk
from rtp_llm.test.utils.numeric_util import calc_diff
from rtp_llm.utils.model_weight import W

try:
    import deep_gemm
except Exception:
    deep_gemm = None  # type: ignore[assignment]

_O_LORA_RANK = 1024
_MAX_SEQ_LEN = 4096
_BS_DECODE_GRID = (1, 4, 16, 32, 64, 96, 128)
_BS_SPEC_GRID = (1, 4, 16, 32)


def _ue8m0_scale(
    shape: tuple[int, int], device: torch.device, seed: int
) -> torch.Tensor:
    rows = torch.arange(shape[0], dtype=torch.int64, device=device).unsqueeze(1)
    cols = torch.arange(shape[1], dtype=torch.int64, device=device).unsqueeze(0)
    exponents = 124 + (seed + rows * 3 + cols * 5) % 7
    return exponents.to(torch.uint8).view(torch.float8_e8m0fnu)


def _make_weights(
    device: torch.device, g: CSAGeometry, *, seed: int = 20261128
) -> dict[str, torch.Tensor]:
    gen = torch.Generator(device=device).manual_seed(seed)

    def _bf16(shape: tuple[int, ...], scale: float = 0.02) -> torch.Tensor:
        return torch.randn(
            shape, generator=gen, device=device, dtype=torch.bfloat16
        ).mul_(scale)

    def _fp8(shape: tuple[int, ...], scale: float = 0.02) -> torch.Tensor:
        return _bf16(shape, scale=scale).to(torch.float8_e4m3fn)

    o_group_input = g.main_heads * HEAD_DIM // g.o_groups
    return {
        W.v4_attn_wq_a_w: _fp8((g.q_lora_rank, g.dim)),
        W.v4_attn_wq_a_s: _ue8m0_scale((g.q_lora_rank // 128, g.dim // 128), device, 0),
        W.v4_attn_wkv_w: _fp8((HEAD_DIM, g.dim)),
        W.v4_attn_wkv_s: _ue8m0_scale((HEAD_DIM // 128, g.dim // 128), device, 1),
        W.v4_attn_wq_b_w: _fp8((g.main_heads * HEAD_DIM, g.q_lora_rank), scale=0.03),
        W.v4_attn_wq_b_s: _ue8m0_scale(
            (g.main_heads * HEAD_DIM // 128, g.q_lora_rank // 128), device, 2
        ),
        W.v4_indexer_wq_b_w: _fp8(
            (INDEX_HEADS * INDEX_HEAD_DIM, g.q_lora_rank), scale=0.03
        ),
        W.v4_indexer_wq_b_s: _ue8m0_scale(
            (INDEX_HEADS * INDEX_HEAD_DIM // 128, g.q_lora_rank // 128), device, 3
        ),
        W.v4_compressor_wkv: _bf16((2 * HEAD_DIM, g.dim)),
        W.v4_compressor_wgate: _bf16((2 * HEAD_DIM, g.dim)),
        W.v4_indexer_compressor_wkv: _bf16((2 * INDEX_HEAD_DIM, g.dim)),
        W.v4_indexer_compressor_wgate: _bf16((2 * INDEX_HEAD_DIM, g.dim)),
        W.v4_indexer_weights_proj_w: _bf16((INDEX_HEADS, g.dim)),
        W.v4_attn_q_norm: torch.rand(g.q_lora_rank, generator=gen, device=device)
        .add_(0.5)
        .bfloat16(),
        W.v4_attn_kv_norm: torch.rand(HEAD_DIM, generator=gen, device=device)
        .add_(0.5)
        .bfloat16(),
        W.v4_indexer_compressor_norm: torch.rand(
            INDEX_HEAD_DIM, generator=gen, device=device
        )
        .add_(0.5)
        .bfloat16(),
        W.v4_compressor_norm: torch.rand(HEAD_DIM, generator=gen, device=device)
        .add_(0.5)
        .bfloat16(),
        W.v4_compressor_ape: _bf16((4, 2 * HEAD_DIM)),
        W.v4_indexer_compressor_ape: _bf16((4, 2 * INDEX_HEAD_DIM)),
        W.v4_attn_norm: torch.rand(g.dim, generator=gen, device=device)
        .add_(0.5)
        .bfloat16(),
        W.v4_attn_sink: torch.randn(g.main_heads, generator=gen, device=device),
        W.v4_attn_wo_a_w: _fp8((g.o_groups * _O_LORA_RANK, o_group_input), scale=0.01),
        W.v4_attn_wo_a_s: _ue8m0_scale(
            (g.o_groups * _O_LORA_RANK // 128, o_group_input // 128), device, 4
        ),
        W.v4_attn_wo_b_w: _fp8((g.dim, g.o_groups * _O_LORA_RANK), scale=0.01),
        W.v4_attn_wo_b_s: _ue8m0_scale(
            (g.dim // 128, g.o_groups * _O_LORA_RANK // 128), device, 5
        ),
    }


def _make_attention(
    device: torch.device, g: CSAGeometry, *, compress_ratio: int
) -> AttentionFP8:
    weights = _make_weights(device, g)
    attn = AttentionFP8(
        layer_id=0,
        dim=g.dim,
        n_heads=g.main_heads,
        q_lora_rank=g.q_lora_rank,
        head_dim=HEAD_DIM,
        rope_head_dim=ROPE_DIM,
        o_lora_rank=_O_LORA_RANK,
        o_groups=g.o_groups,
        window_size=128,
        compress_ratio=compress_ratio,
        compress_rope_theta=160000.0,
        rope_theta=10000.0,
        rope_factor=16.0,
        beta_fast=32,
        beta_slow=1,
        original_seq_len=_MAX_SEQ_LEN,
        max_batch_size=256,
        max_seq_len=_MAX_SEQ_LEN,
        index_n_heads=INDEX_HEADS,
        index_head_dim=INDEX_HEAD_DIM,
        index_topk=g.index_topk,
        norm_eps=1.0e-6,
        layer_weights=weights,
        tp_size=1,
        tp_rank=0,
    )
    attn.init_rope_cache(device)
    return attn


def _unfused_decode_compute_qkv_ref(
    attn: AttentionFP8,
    x: torch.Tensor,
    position_ids: torch.Tensor,
    *,
    return_freqs_cis: bool = True,
    return_qr: bool | None = None,
) -> DecodeQKV:
    """Pre-Track-B2 unfused baseline for decode_compute_qkv."""
    rd = attn.rope_head_dim
    pos = position_ids.reshape(-1)
    if pos.device != attn.freqs_cis.device:
        pos = pos.to(device=attn.freqs_cis.device)
    q_proj = attn._lin(attn.wq_a, x)
    qr = attn._rmsnorm_weighted(q_proj, attn.q_norm)
    q = attn._lin(attn.wq_b, qr).unflatten(-1, (attn.n_heads, attn.head_dim))
    q = fused_rmsnorm_rope(
        q,
        None,
        attn.freqs_cis,
        rd,
        eps=attn.eps,
        position_ids=pos,
    )
    kv_in = attn._lin(attn.wkv, x)
    kv = fused_rmsnorm_rope(
        kv_in,
        attn.kv_norm,
        attn.freqs_cis,
        rd,
        eps=attn.eps,
        position_ids=pos,
    )
    need_qr = (
        bool(return_qr)
        if return_qr is not None
        else (getattr(attn, "compress_ratio", 4) == 4)
    )
    if return_freqs_cis:
        idx = pos if pos.dtype == torch.long else pos.to(torch.long)
        freqs_cis = attn.freqs_cis.index_select(0, idx)
        if not freqs_cis.is_contiguous():
            freqs_cis = freqs_cis.contiguous()
    else:
        freqs_cis = attn.freqs_cis
    return DecodeQKV(qr=qr if need_qr else None, q=q, kv=kv, freqs_cis=freqs_cis)


def _unfused_indexer_forward_ref(
    op: IndexerDecodeV4Op,
    q_indexer: torch.Tensor,
    kv_indexer: torch.Tensor,
    weights: torch.Tensor,
    compressed_len_per_req: torch.Tensor,
    out_buffer: torch.Tensor,
) -> torch.Tensor:
    """Pre-Track-B2 unfused fast-path baseline using sglang quant + zero-pad + copy."""
    assert deep_gemm is not None
    B, q_len, H, D = q_indexer.shape
    T_max = kv_indexer.shape[1]
    device = q_indexer.device
    block_kv = op.page_block_size
    num_blocks_per_req = (T_max + block_kv - 1) // block_kv
    T_padded = num_blocks_per_req * block_kv
    num_blocks = B * num_blocks_per_req
    scales_per_token = D // op.block_size
    head_dim_with_sf = D + scales_per_token * 4
    is_ue8m0 = op.scale_fmt == "ue8m0"

    q_flat = q_indexer.contiguous().view(-1, D)
    q_fp8, q_scale = sgl_per_token_group_quant_fp8(
        q_flat,
        group_size=op.block_size,
        eps=1e-4,
        column_major_scales=True,
        scale_tma_aligned=True,
        scale_ue8m0=is_ue8m0,
    )
    q_fp8_in = q_fp8.view(B, q_len, H, D)
    if is_ue8m0:
        q_scale = _unpack_ue8m0_scale(q_scale)
    q_scale = q_scale.view(B * q_len, H, 1)
    scaled_weights = (
        weights.contiguous().view(B * q_len, H, 1).float()
        * q_scale.float()
        * op.softmax_scale
    ).view(B * q_len, H)

    if T_padded != T_max:
        kv_padded = torch.zeros((B, T_padded, D), dtype=kv_indexer.dtype, device=device)
        kv_padded[:, :T_max].copy_(kv_indexer)
    else:
        kv_padded = kv_indexer.contiguous()

    k_flat = kv_padded.view(-1, D)
    k_fp8, k_scale = sgl_per_token_group_quant_fp8(
        k_flat,
        group_size=op.block_size,
        eps=1e-4,
        column_major_scales=is_ue8m0,
        scale_tma_aligned=is_ue8m0,
        scale_ue8m0=is_ue8m0,
    )
    if is_ue8m0:
        k_scale = _unpack_ue8m0_scale(k_scale)

    kv_cache_u8 = torch.empty(
        (num_blocks, block_kv, 1, head_dim_with_sf),
        dtype=torch.uint8,
        device=device,
    )
    kv_cache_blocks = kv_cache_u8.view(num_blocks, block_kv * head_dim_with_sf)
    kv_cache_blocks[:, : block_kv * D].copy_(
        k_fp8.view(num_blocks, block_kv * D).view(torch.uint8)
    )
    k_scale_3d = k_scale.view(num_blocks, block_kv, scales_per_token)
    if not k_scale_3d.is_contiguous() or k_scale_3d.stride(-1) != 1:
        k_scale_3d = k_scale_3d.clone(memory_format=torch.contiguous_format)
    kv_cache_blocks[:, block_kv * D :].copy_(
        k_scale_3d.view(torch.uint8).view(num_blocks, block_kv * scales_per_token * 4)
    )

    block_table = torch.arange(num_blocks, dtype=torch.int32, device=device).view(
        B, num_blocks_per_req
    )
    kv_lens_1d = compressed_len_per_req.to(torch.int32).contiguous()
    context_lens = (
        kv_lens_1d.view(B, 1)
        if q_len == 1
        else kv_lens_1d.view(B, 1).expand(B, q_len).contiguous()
    )
    schedule_meta = deep_gemm.get_paged_mqa_logits_metadata(
        context_lens, block_kv, deep_gemm.get_num_sms()
    )
    logits = deep_gemm.fp8_paged_mqa_logits(
        q_fp8_in,
        kv_cache_u8,
        scaled_weights,
        context_lens,
        block_table,
        schedule_meta,
        T_padded,
        clean_logits=False,
    ).view(B, q_len, T_padded)
    idxs = select_indexer_topk(
        logits,
        op.index_topk,
        lengths=context_lens.reshape(-1),
    )
    out_buffer.copy_(idxs)
    return out_buffer


def _inspect_cuda_graph_nodes(graph: torch.cuda.CUDAGraph) -> dict[str, int]:
    """Count raw CUDA graph nodes by type via libcudart."""
    libcudart = ctypes.CDLL("libcudart.so")
    cuda_graph_fn = libcudart.cudaGraphGetNodes
    cuda_graph_fn.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_size_t),
    ]
    cuda_graph_fn.restype = ctypes.c_int

    node_type_fn = libcudart.cudaGraphNodeGetType
    node_type_fn.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int)]
    node_type_fn.restype = ctypes.c_int

    raw_graph = ctypes.c_void_p(graph.raw_cuda_graph())
    num_nodes = ctypes.c_size_t(0)
    rc = cuda_graph_fn(raw_graph, None, ctypes.byref(num_nodes))
    if rc != 0:
        raise RuntimeError(f"cudaGraphGetNodes count failed with rc={rc}")
    n = int(num_nodes.value)
    counts = {"total": n, "kernel": 0, "memcpy": 0, "memset": 0, "other": 0}
    if n == 0:
        return counts
    nodes = (ctypes.c_void_p * n)()
    rc = cuda_graph_fn(raw_graph, nodes, ctypes.byref(num_nodes))
    if rc != 0:
        raise RuntimeError(f"cudaGraphGetNodes array failed with rc={rc}")
    ntype = ctypes.c_int(-1)
    for i in range(n):
        rc = node_type_fn(nodes[i], ctypes.byref(ntype))
        if rc != 0:
            raise RuntimeError(f"cudaGraphNodeGetType failed with rc={rc}")
        v = int(ntype.value)
        if v == 0:
            counts["kernel"] += 1
        elif v == 1:
            counts["memcpy"] += 1
        elif v == 2:
            counts["memset"] += 1
        else:
            counts["other"] += 1
    return counts


def _capture_graph(
    fn: Callable[[], object],
    *,
    warmup: int = 3,
    repeats_in_graph: int = 1,
    inspect_nodes: bool = True,
) -> tuple[torch.cuda.CUDAGraph, object, dict[str, int]]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph(keep_graph=inspect_nodes)
    out = None
    with torch.cuda.graph(graph):
        for _ in range(repeats_in_graph):
            out = fn()
    torch.cuda.synchronize()
    nodes = _inspect_cuda_graph_nodes(graph) if inspect_nodes else {}
    return graph, out, nodes


def _bench_graph_us(
    fn: Callable[[], object], *, repeats_in_graph: int = 10, iters: int = 40
) -> float:
    graph, _, _ = _capture_graph(
        fn, warmup=4, repeats_in_graph=repeats_in_graph, inspect_nodes=False
    )
    for _ in range(5):
        graph.replay()
    torch.cuda.synchronize()
    samples = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end) * 1000.0 / repeats_in_graph)
    return float(statistics.median(samples))


@contextmanager
def _capture_cpu_ops(fn: Callable[[], object]):
    for _ in range(2):
        fn()
    torch.cuda.synchronize()
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU],
        record_shapes=False,
    ) as prof:
        fn()
    torch.cuda.synchronize()
    yield {evt.key for evt in prof.key_averages()}


class DecodeSmallOpsFusionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA is required")
        cap = torch.cuda.get_device_capability()
        if cap not in ((10, 0), (10, 3)):
            raise unittest.SkipTest(f"DSv4 FP8 UE8M0 requires SM100/SM103, got {cap}")
        cls.device = torch.device("cuda", torch.cuda.current_device())
        cls.flash_csa = _make_attention(cls.device, FLASH_GEOMETRY, compress_ratio=4)
        cls.flash_hca = _make_attention(cls.device, FLASH_GEOMETRY, compress_ratio=128)
        cls.flash_swa = _make_attention(cls.device, FLASH_GEOMETRY, compress_ratio=0)
        cls.pro_csa = _make_attention(cls.device, PRO_GEOMETRY, compress_ratio=4)
        cls.pro_swa = _make_attention(cls.device, PRO_GEOMETRY, compress_ratio=0)

    def _assert_compute_qkv_matches(
        self,
        attn: AttentionFP8,
        bsz: int,
        q_len: int,
        *,
        return_freqs_cis: bool = True,
        return_qr: bool | None = None,
    ) -> None:
        torch.manual_seed(1000 + bsz * 13 + q_len * 7 + int(attn.compress_ratio))
        x = torch.randn(
            bsz, q_len, attn.dim, dtype=torch.bfloat16, device=self.device
        ).mul_(0.25)
        pos = torch.randint(
            0, _MAX_SEQ_LEN, (bsz * q_len,), dtype=torch.int32, device=self.device
        )
        ref = _unfused_decode_compute_qkv_ref(
            attn,
            x,
            pos,
            return_freqs_cis=return_freqs_cis,
            return_qr=return_qr,
        )
        got = decode_compute_qkv(
            attn,
            x,
            pos,
            return_freqs_cis=return_freqs_cis,
            return_qr=return_qr,
        )
        self.assertEqual(tuple(got.q.shape), tuple(ref.q.shape))
        self.assertEqual(tuple(got.kv.shape), tuple(ref.kv.shape))
        self.assertTrue(got.q.is_contiguous())
        self.assertTrue(got.kv.is_contiguous())
        self.assertTrue(got.freqs_cis.is_contiguous())
        if ref.qr is None:
            self.assertIsNone(got.qr)
        else:
            self.assertIsNotNone(got.qr)
            assert got.qr is not None
            self.assertTrue(got.qr.is_contiguous())
            self.assertLess(calc_diff(got.qr.float(), ref.qr.float()), 1.0e-3)
        self.assertLess(calc_diff(got.q.float(), ref.q.float()), 5.0e-3)
        self.assertLess(calc_diff(got.kv.float(), ref.kv.float()), 1.0e-3)
        torch.testing.assert_close(got.freqs_cis, ref.freqs_cis, rtol=0.0, atol=0.0)

    @torch.inference_mode()
    def test_compute_qkv_parity_flash_all_modes_and_batches(self) -> None:
        for bs in _BS_DECODE_GRID:
            for attn in (self.flash_csa, self.flash_hca, self.flash_swa):
                with self.subTest(
                    geom="flash", ratio=attn.compress_ratio, bs=bs, q_len=1
                ):
                    self._assert_compute_qkv_matches(attn, bs, 1)
        for bs in _BS_SPEC_GRID:
            for attn in (self.flash_csa, self.flash_swa):
                with self.subTest(
                    geom="flash", ratio=attn.compress_ratio, bs=bs, q_len=4
                ):
                    self._assert_compute_qkv_matches(attn, bs, 4)
        # Explicit return_qr override on SWA layer and return_freqs_cis=False
        self._assert_compute_qkv_matches(
            self.flash_swa, 16, 1, return_freqs_cis=False, return_qr=True
        )
        self._assert_compute_qkv_matches(
            self.flash_csa, 8, 4, return_freqs_cis=False, return_qr=False
        )

    @torch.inference_mode()
    def test_compute_qkv_parity_pro_geometry(self) -> None:
        for bs, q_len in ((1, 1), (32, 1), (128, 1), (16, 4)):
            for attn in (self.pro_csa, self.pro_swa):
                with self.subTest(
                    geom="pro", ratio=attn.compress_ratio, bs=bs, q_len=q_len
                ):
                    self._assert_compute_qkv_matches(attn, bs, q_len)

    @torch.inference_mode()
    def test_compute_qkv_freqs_gather_int32_int64_strided_and_empty(self) -> None:
        x = torch.randn(
            4, 1, self.flash_csa.dim, dtype=torch.bfloat16, device=self.device
        ).mul_(0.2)
        pos_full = torch.tensor(
            [17, 999, 1024, 0, 5, 3, 2048, 7], dtype=torch.int32, device=self.device
        )
        pos_strided = pos_full[::2]
        self.assertFalse(pos_strided.is_contiguous())
        got32 = decode_compute_qkv(
            self.flash_csa, x, pos_strided, return_freqs_cis=True
        )
        got64 = decode_compute_qkv(
            self.flash_csa, x, pos_strided.to(torch.int64), return_freqs_cis=True
        )
        ref = _unfused_decode_compute_qkv_ref(
            self.flash_csa, x, pos_strided, return_freqs_cis=True
        )
        torch.testing.assert_close(got32.freqs_cis, ref.freqs_cis, rtol=0.0, atol=0.0)
        torch.testing.assert_close(got64.freqs_cis, ref.freqs_cis, rtol=0.0, atol=0.0)
        self.assertTrue(got32.freqs_cis.is_contiguous())

        # Empty batch guard
        x_empty = torch.empty(
            0, 1, self.flash_swa.dim, dtype=torch.bfloat16, device=self.device
        )
        pos_empty = torch.empty(0, dtype=torch.int32, device=self.device)
        got_empty = decode_compute_qkv(self.flash_swa, x_empty, pos_empty)
        self.assertEqual(
            tuple(got_empty.q.shape), (0, 1, self.flash_swa.n_heads, HEAD_DIM)
        )
        self.assertEqual(tuple(got_empty.freqs_cis.shape), (0, ROPE_DIM // 2))

    @torch.inference_mode()
    def test_compute_qkv_x_quant_dedup_and_override_guards(self) -> None:
        attn = _make_attention(self.device, FLASH_GEOMETRY, compress_ratio=4)
        x = torch.randn(8, 1, attn.dim, dtype=torch.bfloat16, device=self.device).mul_(
            0.2
        )
        pos = torch.arange(8, dtype=torch.int32, device=self.device)

        orig_q_quant = attn.wq_a.quantize_input
        orig_kv_quant = attn.wkv.quantize_input
        counts = {"wq_a_quant": 0, "wkv_quant": 0}

        def count_q_quant(*args, **kwargs):
            counts["wq_a_quant"] += 1
            return orig_q_quant(*args, **kwargs)

        def count_kv_quant(*args, **kwargs):
            counts["wkv_quant"] += 1
            return orig_kv_quant(*args, **kwargs)

        with (
            patch.object(attn.wq_a, "quantize_input", side_effect=count_q_quant),
            patch.object(attn.wkv, "quantize_input", side_effect=count_kv_quant),
        ):
            decode_compute_qkv(attn, x, pos)
        self.assertEqual(counts["wq_a_quant"], 1)
        self.assertEqual(counts["wkv_quant"], 0)

        # Instance-overridden _lin / _rmsnorm_weighted must bypass quantized reuse/fusion.
        lin_calls = []

        def custom_lin(layer, inp, out=None):
            lin_calls.append(layer)
            return AttentionFP8._lin(attn, layer, inp, out=out)

        attn._lin = custom_lin  # type: ignore[method-assign]
        decode_compute_qkv(attn, x, pos)
        self.assertEqual(len(lin_calls), 3)
        del attn.__dict__["_lin"]

        norm_calls = []

        def custom_norm(inp, weight):
            norm_calls.append(tuple(inp.shape))
            return AttentionFP8._rmsnorm_weighted(attn, inp, weight)

        attn._rmsnorm_weighted = custom_norm  # type: ignore[method-assign]
        decode_compute_qkv(attn, x, pos)
        self.assertEqual(len(norm_calls), 1)
        del attn.__dict__["_rmsnorm_weighted"]

    @torch.inference_mode()
    def test_compute_qkv_cuda_graph_audit_and_replay_parity(self) -> None:
        for ratio, attn, bs, q_len in (
            (4, self.flash_csa, 1, 1),
            (4, self.flash_csa, 32, 1),
            (4, self.flash_csa, 8, 4),
            (0, self.flash_swa, 32, 1),
        ):
            with self.subTest(ratio=ratio, bs=bs, q_len=q_len):
                x = torch.randn(
                    bs, q_len, attn.dim, dtype=torch.bfloat16, device=self.device
                ).mul_(0.2)
                pos = torch.randint(
                    0, 1024, (bs * q_len,), dtype=torch.int32, device=self.device
                )
                with _capture_cpu_ops(lambda: decode_compute_qkv(attn, x, pos)) as ops:
                    self.assertNotIn("aten::index_select", ops)

                _, ref_out, ref_nodes = _capture_graph(
                    lambda: _unfused_decode_compute_qkv_ref(attn, x, pos)
                )
                fused_graph, fused_out, fused_nodes = _capture_graph(
                    lambda: decode_compute_qkv(attn, x, pos)
                )
                print(
                    f"[AUDIT compute_qkv ratio={ratio} bs={bs} q_len={q_len}] "
                    f"unfused={ref_nodes} -> fused={fused_nodes}"
                )
                self.assertEqual(fused_nodes["memset"], 0)
                self.assertEqual(fused_nodes["memcpy"], 0)
                self.assertLess(fused_nodes["kernel"], ref_nodes["kernel"])

                # Mutate graph inputs in-place at stable addresses and replay.
                new_x = torch.randn_like(x).mul_(0.3)
                new_pos = torch.randint(
                    1024, 3000, (bs * q_len,), dtype=torch.int32, device=self.device
                )
                x.copy_(new_x)
                pos.copy_(new_pos)
                fused_graph.replay()
                torch.cuda.synchronize()
                expected = _unfused_decode_compute_qkv_ref(attn, new_x, new_pos)
                self.assertLess(
                    calc_diff(fused_out.q.float(), expected.q.float()), 5.0e-3
                )
                self.assertLess(
                    calc_diff(fused_out.kv.float(), expected.kv.float()), 1.0e-3
                )
                torch.testing.assert_close(
                    fused_out.freqs_cis, expected.freqs_cis, rtol=0.0, atol=0.0
                )
                if expected.qr is not None:
                    self.assertIsNotNone(fused_out.qr)
                    self.assertLess(
                        calc_diff(fused_out.qr.float(), expected.qr.float()), 1.0e-3
                    )

    @torch.inference_mode()
    def test_indexer_decode_op_parity_and_edge_cases(self) -> None:
        if not _fast_path_available(self.device, 64):
            self.skipTest("deep_gemm fp8_paged_mqa_logits unavailable")

        H, D, K, block_kv = INDEX_HEADS, INDEX_HEAD_DIM, 512, 64
        op_ue8m0 = IndexerDecodeV4Op(
            H,
            D,
            K,
            softmax_scale=1.0 / (D**0.5),
            scale_fmt="ue8m0",
            page_block_size=block_kv,
        )
        op_fp32 = IndexerDecodeV4Op(
            H,
            D,
            K,
            softmax_scale=1.0 / (D**0.5),
            scale_fmt="fp32",
            page_block_size=block_kv,
        )

        def _check_case(op: IndexerDecodeV4Op, bs: int, q_len: int, t_max: int) -> None:
            torch.manual_seed(4000 + bs * 17 + q_len * 3 + t_max)
            q = torch.randn(
                bs, q_len, H, D, dtype=torch.bfloat16, device=self.device
            ).mul_(0.2)
            kv = torch.randn(
                bs, t_max, D, dtype=torch.bfloat16, device=self.device
            ).mul_(0.2)
            w = torch.randn(
                bs, q_len, H, dtype=torch.bfloat16, device=self.device
            ).mul_(0.5)
            lens = torch.randint(
                max(K + 8, t_max // 2),
                t_max + 1,
                (bs,),
                dtype=torch.int32,
                device=self.device,
            )

            # Dirty CUDA caching allocator blocks right before calling _forward_fast
            # to prove padded tail blocks >= T_max are explicitly zeroed without torch.zeros.
            num_blocks_per_req = (t_max + block_kv - 1) // block_kv
            dirty = torch.full(
                (bs * num_blocks_per_req, block_kv, 1, D + 4),
                0x5A,
                dtype=torch.uint8,
                device=self.device,
            )
            del dirty

            out_fused = torch.full(
                (bs, q_len, K), -9, dtype=torch.int32, device=self.device
            )
            out_unfused = torch.full(
                (bs, q_len, K), -9, dtype=torch.int32, device=self.device
            )

            op.forward(q, kv, w, lens, out_fused, force_reference=False)
            _unfused_indexer_forward_ref(op, q, kv, w, lens, out_unfused)

            # Compare fused Triton quant against unfused sglang baseline (near-exact TopK sets)
            # and verify no index >= lens[b] was ever selected from padded tail.
            min_unfused_overlap = 1.0
            for b in range(bs):
                valid_len = int(lens[b].item())
                for s in range(q_len):
                    got = {int(v) for v in out_fused[b, s].tolist() if int(v) >= 0}
                    want_unfused = {
                        int(v) for v in out_unfused[b, s].tolist() if int(v) >= 0
                    }
                    self.assertTrue(all(0 <= idx < valid_len for idx in got))
                    overlap = len(got & want_unfused) / max(len(want_unfused), 1)
                    min_unfused_overlap = min(min_unfused_overlap, overlap)
            self.assertGreaterEqual(
                min_unfused_overlap,
                0.985,
                f"Low overlap vs unfused sglang baseline at bs={bs}, q_len={q_len}, T={t_max}",
            )

            # Signal-bearing parity vs BF16 reference oracle (realistic non-flat score margin).
            q_sig = q.clone()
            q_sig[..., :4].add_(0.25)
            kv_sig = kv.clone()
            kv_sig[:, :K, :4].add_(0.5)
            w_pos = w.abs().add_(0.2)
            out_sig_fused = torch.full(
                (bs, q_len, K), -9, dtype=torch.int32, device=self.device
            )
            out_sig_ref = torch.full(
                (bs, q_len, K), -9, dtype=torch.int32, device=self.device
            )
            op.forward(q_sig, kv_sig, w_pos, lens, out_sig_fused, force_reference=False)
            op.forward(q_sig, kv_sig, w_pos, lens, out_sig_ref, force_reference=True)
            min_ref_iou = 1.0
            for b in range(bs):
                for s in range(q_len):
                    got = {int(v) for v in out_sig_fused[b, s].tolist() if int(v) >= 0}
                    want = {int(v) for v in out_sig_ref[b, s].tolist() if int(v) >= 0}
                    iou = len(got & want) / max(len(got | want), 1)
                    min_ref_iou = min(min_ref_iou, iou)
            self.assertGreaterEqual(
                min_ref_iou,
                0.95,
                f"Low signal IoU vs BF16 reference at bs={bs}, q_len={q_len}, T={t_max}",
            )

        for bs in _BS_DECODE_GRID:
            with self.subTest(fmt="ue8m0", bs=bs, q_len=1, t_max=700):
                _check_case(op_ue8m0, bs, 1, 700)
        for bs in _BS_SPEC_GRID:
            with self.subTest(fmt="ue8m0", bs=bs, q_len=4, t_max=600):
                _check_case(op_ue8m0, bs, 4, 600)
        with self.subTest(fmt="fp32", bs=16, q_len=1, t_max=700):
            _check_case(op_fp32, 16, 1, 700)

        # Reference and fast-path tail-padding (-1) when T_max < K
        q_small = torch.randn(
            2, 4, H, D, dtype=torch.bfloat16, device=self.device
        ).mul_(0.1)
        kv_small = torch.randn(
            2, 120, D, dtype=torch.bfloat16, device=self.device
        ).mul_(0.1)
        w_small = torch.randn(2, 4, H, dtype=torch.bfloat16, device=self.device).mul_(
            0.2
        )
        lens_small = torch.tensor([120, 80], dtype=torch.int32, device=self.device)
        out_small = torch.full((2, 4, K), -7, dtype=torch.int32, device=self.device)
        out_small_fast = torch.full(
            (2, 4, K), -7, dtype=torch.int32, device=self.device
        )
        op_ue8m0.forward(
            q_small, kv_small, w_small, lens_small, out_small, force_reference=True
        )
        op_ue8m0.forward(
            q_small,
            kv_small,
            w_small,
            lens_small,
            out_small_fast,
            force_reference=False,
        )
        self.assertTrue((out_small[:, :, 120:] == -1).all().item())
        self.assertTrue((out_small[1, :, 80:] == -1).all().item())
        self.assertTrue((out_small_fast[:, :, 120:] == -1).all().item())
        self.assertTrue((out_small_fast[1, :, 80:] == -1).all().item())
        for b, lim in ((0, 120), (1, 80)):
            for s in range(4):
                got_set = {int(v) for v in out_small_fast[b, s].tolist() if int(v) >= 0}
                self.assertEqual(got_set, set(range(lim)))

        # Non-contiguous int64 length path
        out_strided = torch.empty((4, 1, K), dtype=torch.int32, device=self.device)
        lens_i64 = torch.tensor(
            [600, 620, 640, 660, 700, 700, 700, 700],
            dtype=torch.int64,
            device=self.device,
        )[::2]
        q4 = torch.randn(4, 1, H, D, dtype=torch.bfloat16, device=self.device).mul_(0.2)
        kv4 = torch.randn(4, 700, D, dtype=torch.bfloat16, device=self.device).mul_(0.2)
        w4 = torch.randn(4, 1, H, dtype=torch.bfloat16, device=self.device).mul_(0.5)
        op_ue8m0.forward(q4, kv4, w4, lens_i64, out_strided, force_reference=False)
        self.assertTrue((out_strided >= 0).all().item())

    @torch.inference_mode()
    def test_indexer_decode_cuda_graph_audit_and_replay_parity(self) -> None:
        if not _fast_path_available(self.device, 64):
            self.skipTest("deep_gemm fp8_paged_mqa_logits unavailable")

        H, D, K, block_kv, t_max = INDEX_HEADS, INDEX_HEAD_DIM, 512, 64, 700
        op = IndexerDecodeV4Op(
            H,
            D,
            K,
            softmax_scale=1.0 / (D**0.5),
            scale_fmt="ue8m0",
            page_block_size=block_kv,
        )
        for bs, q_len in ((32, 1), (8, 4)):
            with self.subTest(bs=bs, q_len=q_len, t_max=t_max):
                q = torch.randn(
                    bs, q_len, H, D, dtype=torch.bfloat16, device=self.device
                ).mul_(0.2)
                kv = torch.randn(
                    bs, t_max, D, dtype=torch.bfloat16, device=self.device
                ).mul_(0.2)
                w = torch.randn(
                    bs, q_len, H, dtype=torch.bfloat16, device=self.device
                ).mul_(0.5)
                lens = torch.full(
                    (bs,), t_max - 20, dtype=torch.int32, device=self.device
                )
                out_unfused = torch.empty(
                    (bs, q_len, K), dtype=torch.int32, device=self.device
                )
                out_fused = torch.empty(
                    (bs, q_len, K), dtype=torch.int32, device=self.device
                )

                _, _, ref_nodes = _capture_graph(
                    lambda: _unfused_indexer_forward_ref(
                        op, q, kv, w, lens, out_unfused
                    )
                )
                fused_graph, _, fused_nodes = _capture_graph(
                    lambda: op.forward(q, kv, w, lens, out_fused, force_reference=False)
                )
                print(
                    f"[AUDIT indexer_decode bs={bs} q_len={q_len} T={t_max}] "
                    f"unfused={ref_nodes} -> fused={fused_nodes}"
                )
                self.assertEqual(fused_nodes["memset"], 0)
                self.assertEqual(fused_nodes["memcpy"], 0)
                self.assertGreater(ref_nodes["memset"] + ref_nodes["memcpy"], 0)
                self.assertLess(fused_nodes["kernel"], ref_nodes["kernel"])

                # Replay at updated buffer contents (shorter valid lengths)
                q.copy_(torch.randn_like(q).mul_(0.3))
                kv.copy_(torch.randn_like(kv).mul_(0.3))
                w.copy_(torch.randn_like(w).mul_(0.5))
                lens.fill_(550)
                fused_graph.replay()
                torch.cuda.synchronize()
                expected = torch.empty_like(out_fused)
                _unfused_indexer_forward_ref(op, q, kv, w, lens, expected)
                for b in range(bs):
                    for s in range(q_len):
                        got_set = {int(v) for v in out_fused[b, s].tolist()}
                        exp_set = {int(v) for v in expected[b, s].tolist()}
                        self.assertTrue(all(0 <= v < 550 for v in got_set))
                        self.assertGreaterEqual(len(got_set & exp_set) / K, 0.985)

    @torch.inference_mode()
    def test_output_proj_and_runtime_smallops(self) -> None:
        # 1. Empty batch fast path skips dequantize_fp8_weight on CUDA.
        o_empty = torch.empty(
            0,
            1,
            self.flash_swa.n_heads,
            HEAD_DIM,
            dtype=torch.bfloat16,
            device=self.device,
        )
        freqs_empty = torch.empty(
            0, ROPE_DIM // 2, dtype=torch.complex64, device=self.device
        )
        with patch(
            "rtp_llm.models_py.modules.dsv4.fp8.decode.output_proj.dequantize_fp8_weight",
            side_effect=AssertionError(
                "dequantize_fp8_weight must not run for empty o"
            ),
        ):
            out_empty = decode_output_proj(
                self.flash_swa, o_empty, freqs_empty, bsz=0, q_len=1
            )
        self.assertEqual(tuple(out_empty.shape), (0, 1, self.flash_swa.dim))
        self.assertEqual(out_empty.device, o_empty.device)

        # 2. Non-empty parity vs eager inv-RoPE + dequantize_fp8_weight + einsum reference.
        from rtp_llm.models_py.modules.dsv4.rope import apply_rotary_emb_batched

        for bs, q_len in ((1, 1), (32, 1), (8, 4)):
            o = torch.randn(
                bs,
                q_len,
                self.flash_swa.n_heads,
                HEAD_DIM,
                dtype=torch.bfloat16,
                device=self.device,
            ).mul_(0.1)
            pos = torch.randint(
                0, 512, (bs * q_len,), dtype=torch.int32, device=self.device
            )
            freqs = self.flash_swa.freqs_cis.index_select(0, pos.to(torch.long))
            got = decode_output_proj(
                self.flash_swa, o.clone(), freqs, bsz=bs, q_len=q_len
            )
            ref_o = o.clone()
            rope_buf = ref_o[..., -ROPE_DIM:].reshape(
                bs * q_len, 1, self.flash_swa.n_heads, ROPE_DIM
            )
            apply_rotary_emb_batched(rope_buf, freqs, inverse=True)
            ref_o[..., -ROPE_DIM:] = rope_buf.reshape(
                bs, q_len, self.flash_swa.n_heads, ROPE_DIM
            )
            ref_o = ref_o.reshape(bs, q_len, self.flash_swa.n_groups, -1)
            wo_a_bf16 = dequantize_fp8_weight(
                self.flash_swa.wo_a_w, self.flash_swa.wo_a_s
            ).to(o.dtype)
            wo_a = wo_a_bf16.view(
                self.flash_swa.n_groups, self.flash_swa.o_lora_rank, -1
            )
            o_flat = torch.einsum("bsgd,grd->bsgr", ref_o, wo_a).flatten(2)
            ref = self.flash_swa._lin(self.flash_swa.wo_b, o_flat)
            self.assertEqual(tuple(got.shape), (bs, q_len, self.flash_swa.dim))
            self.assertTrue(torch.isfinite(got).all().item())
            self.assertLess(calc_diff(got.float(), ref.float()), 5.0e-3)

        # 3. MegaCSARuntime.num_hc_splits SM-count caching
        MegaCSARuntime._DEVICE_SM_COUNTS.clear()
        orig_props = torch.cuda.get_device_properties
        prop_calls = 0

        def count_props(dev):
            nonlocal prop_calls
            prop_calls += 1
            return orig_props(dev)

        with patch.object(torch.cuda, "get_device_properties", side_effect=count_props):
            s1 = MegaCSARuntime.num_hc_splits(1, self.device)
            s2 = MegaCSARuntime.num_hc_splits(32, self.device)
            s3 = MegaCSARuntime.num_hc_splits(
                128, torch.device("cuda", self.device.index)
            )
        self.assertEqual(prop_calls, 1)
        self.assertGreaterEqual(s1, 1)
        self.assertGreaterEqual(s2, 1)
        self.assertGreaterEqual(s3, 1)

    @torch.inference_mode()
    def test_microbenchmark_decode_smallops_report(self) -> None:
        if not _fast_path_available(self.device, 64):
            self.skipTest("deep_gemm fp8_paged_mqa_logits unavailable")

        print(
            "\n=== Track B-2 Decode SmallOps Microbenchmark (CUDAGraph Replay Median) ==="
        )
        for label, attn in (
            ("compute_qkv_flash_csa(ratio=4)", self.flash_csa),
            ("compute_qkv_flash_swa(ratio=0)", self.flash_swa),
        ):
            for bs, q_len in [(b, 1) for b in _BS_DECODE_GRID] + [
                (b, 4) for b in _BS_SPEC_GRID
            ]:
                x = torch.randn(
                    bs, q_len, attn.dim, dtype=torch.bfloat16, device=self.device
                ).mul_(0.2)
                pos = torch.randint(
                    0,
                    _MAX_SEQ_LEN,
                    (bs * q_len,),
                    dtype=torch.int32,
                    device=self.device,
                )
                _, _, base_nodes = _capture_graph(
                    lambda: _unfused_decode_compute_qkv_ref(attn, x, pos)
                )
                _, _, opt_nodes = _capture_graph(
                    lambda: decode_compute_qkv(attn, x, pos)
                )
                base_us = _bench_graph_us(
                    lambda: _unfused_decode_compute_qkv_ref(attn, x, pos)
                )
                opt_us = _bench_graph_us(lambda: decode_compute_qkv(attn, x, pos))
                speedup = base_us / max(opt_us, 1e-6)
                print(
                    f"  {label:30s} bs={bs:3d} q={q_len}: "
                    f"before={base_us:7.2f}us (k={base_nodes['kernel']:2d}) -> "
                    f"after={opt_us:7.2f}us (k={opt_nodes['kernel']:2d}) "
                    f"speedup={speedup:.2f}x"
                )

        H, D, K, block_kv, t_max = INDEX_HEADS, INDEX_HEAD_DIM, 512, 64, 700
        op = IndexerDecodeV4Op(
            H,
            D,
            K,
            softmax_scale=1.0 / (D**0.5),
            scale_fmt="ue8m0",
            page_block_size=block_kv,
        )
        for bs, q_len in [(b, 1) for b in _BS_DECODE_GRID] + [
            (b, 4) for b in _BS_SPEC_GRID
        ]:
            q = torch.randn(
                bs, q_len, H, D, dtype=torch.bfloat16, device=self.device
            ).mul_(0.2)
            kv = torch.randn(
                bs, t_max, D, dtype=torch.bfloat16, device=self.device
            ).mul_(0.2)
            w = torch.randn(
                bs, q_len, H, dtype=torch.bfloat16, device=self.device
            ).mul_(0.5)
            lens = torch.full((bs,), t_max - 17, dtype=torch.int32, device=self.device)
            out_buf = torch.empty((bs, q_len, K), dtype=torch.int32, device=self.device)

            _, _, base_nodes = _capture_graph(
                lambda: _unfused_indexer_forward_ref(op, q, kv, w, lens, out_buf)
            )
            _, _, opt_nodes = _capture_graph(
                lambda: op.forward(q, kv, w, lens, out_buf, force_reference=False)
            )
            base_us = _bench_graph_us(
                lambda: _unfused_indexer_forward_ref(op, q, kv, w, lens, out_buf)
            )
            opt_us = _bench_graph_us(
                lambda: op.forward(q, kv, w, lens, out_buf, force_reference=False)
            )
            speedup = base_us / max(opt_us, 1e-6)
            print(
                f"  {'indexer_decode_ue8m0':30s} bs={bs:3d} q={q_len}: "
                f"before={base_us:7.2f}us (k={base_nodes['kernel']:2d},mset={base_nodes['memset']},mcpy={base_nodes['memcpy']}) -> "
                f"after={opt_us:7.2f}us (k={opt_nodes['kernel']:2d},mset={opt_nodes['memset']},mcpy={opt_nodes['memcpy']}) "
                f"speedup={speedup:.2f}x"
            )


if __name__ == "__main__":
    unittest.main()
