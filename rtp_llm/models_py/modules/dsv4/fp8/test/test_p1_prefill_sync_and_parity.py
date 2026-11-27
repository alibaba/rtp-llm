"""Unit tests for next-round P1 prefill sync elimination, kernel parity, and norm+quant wiring."""

from __future__ import annotations

import contextlib
import unittest
from types import SimpleNamespace
from typing import Dict, Optional, Tuple
from unittest.mock import patch

import flashinfer
import torch

from rtp_llm.models_py.kernels.cuda.deepgemm_wrapper import (
    has_deep_gemm,
    is_deep_gemm_e8m0_used,
)
from rtp_llm.models_py.kernels.cuda.fp8_kernel import (
    requant_weight_ue8m0,
    sgl_per_token_group_quant_fp8,
)
from rtp_llm.models_py.modules.dsv4.cp import (
    build_cp_context_for_forward,
    cp_gather_last_by_request,
)
from rtp_llm.models_py.modules.dsv4.fp8._compressor_vllm_triton import (
    build_cos_sin_cache,
    run_fused_compress_kv_write,
)
from rtp_llm.models_py.modules.dsv4.fp8._swa_ops_triton import (
    _SPARSE_PREFILL_TOPK_ALIGNMENT,
    combine_topk_swa_indices,
    combine_topk_swa_indices_cp,
)
from rtp_llm.models_py.modules.dsv4.fp8.attention import (
    AttentionFP8,
    PrefillMeta,
    _v4_fp8_linear,
)
from rtp_llm.models_py.modules.dsv4.fp8.compressor import CompressorFP8
from rtp_llm.models_py.modules.dsv4.fp8.indexer import IndexerFP8
from rtp_llm.models_py.modules.dsv4.kv_cache_utils import (
    CSA_KV,
    CSA_STATE,
    HCA_KV,
    HCA_STATE,
    INDEXER_KV,
    INDEXER_STATE,
    SWA_KV,
)
from rtp_llm.models_py.modules.dsv4.prefill_workspace import PrefillWorkspace
from rtp_llm.models_py.modules.dsv4.rope import precompute_freqs_cis


@contextlib.contextmanager
def _forbid_cuda_d2h_sync():
    """Raise AssertionError if any CUDA -> host sync occurs inside the block."""
    orig_sync = torch.cuda.synchronize
    orig_item = torch.Tensor.item
    orig_tolist = torch.Tensor.tolist
    orig_cpu = torch.Tensor.cpu
    orig_to = torch.Tensor.to
    orig_nonzero = torch.nonzero
    orig_t_nonzero = torch.Tensor.nonzero

    def _sync(*args, **kwargs):
        raise AssertionError("Unexpected torch.cuda.synchronize during zero-sync path")

    def _item(self):
        if self.is_cuda:
            raise AssertionError(
                f"Unexpected CUDA .item() on {tuple(self.shape)} {self.dtype}"
            )
        return orig_item(self)

    def _tolist(self):
        if self.is_cuda:
            raise AssertionError(
                f"Unexpected CUDA .tolist() on {tuple(self.shape)} {self.dtype}"
            )
        return orig_tolist(self)

    def _cpu(self, *args, **kwargs):
        if self.is_cuda:
            raise AssertionError(
                f"Unexpected CUDA .cpu() on {tuple(self.shape)} {self.dtype}"
            )
        return orig_cpu(self, *args, **kwargs)

    def _to(self, *args, **kwargs):
        target_dev = kwargs.get("device")
        if target_dev is None and args:
            arg0 = args[0]
            if isinstance(arg0, (torch.device, str)):
                target_dev = arg0
            elif isinstance(arg0, torch.Tensor):
                target_dev = arg0.device
        if (
            self.is_cuda
            and target_dev is not None
            and torch.device(target_dev).type == "cpu"
        ):
            raise AssertionError(
                f"Unexpected CUDA .to('cpu') on {tuple(self.shape)} {self.dtype}"
            )
        return orig_to(self, *args, **kwargs)

    def _nonzero(input, *args, **kwargs):
        if isinstance(input, torch.Tensor) and input.is_cuda:
            raise AssertionError(
                f"Unexpected CUDA torch.nonzero() on {tuple(input.shape)}"
            )
        return orig_nonzero(input, *args, **kwargs)

    def _t_nonzero(self, *args, **kwargs):
        if self.is_cuda:
            raise AssertionError(
                f"Unexpected CUDA Tensor.nonzero() on {tuple(self.shape)}"
            )
        return orig_t_nonzero(self, *args, **kwargs)

    with (
        patch.object(torch.cuda, "synchronize", _sync),
        patch.object(torch.Tensor, "item", _item),
        patch.object(torch.Tensor, "tolist", _tolist),
        patch.object(torch.Tensor, "cpu", _cpu),
        patch.object(torch.Tensor, "to", _to),
        patch.object(torch, "nonzero", _nonzero),
        patch.object(torch.Tensor, "nonzero", _t_nonzero),
    ):
        yield


def _make_ue8m0_linear(
    out_features: int,
    in_features: int,
    device: torch.device,
    *,
    init_scale: float = 0.03,
):
    w_bf16 = torch.randn(out_features, in_features, dtype=torch.bfloat16, device=device)
    w_fp8 = (w_bf16 * init_scale).to(torch.float8_e4m3fn)
    w_s = torch.ones(
        out_features // 128, in_features // 128, dtype=torch.float32, device=device
    )
    wq, ws = requant_weight_ue8m0(w_fp8, w_s)
    return _v4_fp8_linear(wq, ws)


def _alloc_hca_kv_cache(
    num_blocks: int, block_size: int, device: torch.device
) -> torch.Tensor:
    ts, sd = 576, 8
    block_bytes = block_size * (ts + sd)
    flat = torch.zeros((num_blocks, block_bytes), dtype=torch.uint8, device=device)
    cache = flat.as_strided(
        size=(num_blocks, block_size, ts),
        stride=(block_bytes, ts, 1),
    )
    cache._flat_backing = flat  # type: ignore[attr-defined]
    return cache


def _dequant_584b_kv(
    cache: torch.Tensor, slots: torch.Tensor, block_size: int
) -> torch.Tensor:
    """Dequantize 584B packed FP8(448) + BF16 RoPE(64) slots back to float32 [M, 512]."""
    flat_backing = cache._flat_backing  # type: ignore[attr-defined]
    out = []
    for s in slots.tolist():
        b_idx = s // block_size
        pos_in = s % block_size
        row_fp8_u8 = flat_backing[b_idx, pos_in * 576 : (pos_in + 1) * 576]
        nope_fp8 = row_fp8_u8[:448].view(torch.float8_e4m3fn).to(torch.float32)
        scale_off = block_size * 576 + pos_in * 8
        exps = flat_backing[b_idx, scale_off : scale_off + 7].to(torch.int32) - 127
        scales = torch.exp2(exps.to(torch.float32))
        nope = (nope_fp8.view(7, 64) * scales.unsqueeze(1)).reshape(448)
        rope = row_fp8_u8[448:576].view(torch.bfloat16).to(torch.float32)
        out.append(torch.cat([nope, rope], dim=0))
    return torch.stack(out, dim=0)


def _ref_combine_noncp(
    topk_indices: torch.Tensor,
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    gather_lens: torch.Tensor,
    window_size: int,
    compress_ratio: int,
    topk: int,
    M: int,
    N: int,
    *,
    cu_cmp_offsets: Optional[torch.Tensor] = None,
    cu_swa_offsets: Optional[torch.Tensor] = None,
    dense_arange_topk: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    num_tokens = int(topk_indices.shape[0])
    aligned = (
        (topk + window_size + _SPARSE_PREFILL_TOPK_ALIGNMENT - 1)
        // _SPARSE_PREFILL_TOPK_ALIGNMENT
        * _SPARSE_PREFILL_TOPK_ALIGNMENT
    )
    out = torch.full(
        (num_tokens, aligned), -1, dtype=torch.int32, device=topk_indices.device
    )
    lens = torch.zeros(num_tokens, dtype=torch.int32, device=topk_indices.device)
    qsl = query_start_loc.tolist()
    sl = seq_lens.tolist()
    gl = gather_lens.tolist()
    cu_cmp = cu_cmp_offsets.tolist() if cu_cmp_offsets is not None else None
    cu_swa = cu_swa_offsets.tolist() if cu_swa_offsets is not None else None
    for b in range(len(sl)):
        q_start = qsl[b] - qsl[0]
        q_end = qsl[b + 1] - qsl[0]
        query_len = q_end - q_start
        prefix_len = sl[b] - query_len
        gather_start = sl[b] - gl[b]
        cmp_base = cu_cmp[b] if cu_cmp is not None else M * b
        swa_base = cu_swa[b] if cu_swa is not None else M * b + N
        for t in range(q_start, q_end):
            pos = prefix_len + (t - q_start)
            tk_len = min((pos + 1) // compress_ratio, topk)
            sw_len = min(pos + 1, window_size)
            if tk_len > 0:
                if dense_arange_topk:
                    vals = torch.arange(
                        tk_len, dtype=torch.int32, device=topk_indices.device
                    )
                else:
                    vals = topk_indices[t, :tk_len]
                out[t, :tk_len] = cmp_base + vals
            if sw_len > 0:
                swa_start_pos = pos - sw_len + 1 - gather_start
                out[t, tk_len : tk_len + sw_len] = (
                    swa_base
                    + swa_start_pos
                    + torch.arange(
                        sw_len, dtype=torch.int32, device=topk_indices.device
                    )
                )
            lens[t] = tk_len + sw_len
    return out, lens


def _ref_combine_cp(
    topk_indices: torch.Tensor,
    global_positions: torch.Tensor,
    sp_int: int,
    window_size: int,
    compress_ratio: int,
    topk: int,
    M: int,
    N: int,
    *,
    req_id_per_token: Optional[torch.Tensor] = None,
    prefix_lengths: Optional[torch.Tensor] = None,
    cu_cmp_offsets: Optional[torch.Tensor] = None,
    cu_swa_offsets: Optional[torch.Tensor] = None,
    dense_arange_topk: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    num_tokens = int(global_positions.numel())
    aligned = (
        (topk + window_size + _SPARSE_PREFILL_TOPK_ALIGNMENT - 1)
        // _SPARSE_PREFILL_TOPK_ALIGNMENT
        * _SPARSE_PREFILL_TOPK_ALIGNMENT
    )
    out = torch.full(
        (num_tokens, aligned), -1, dtype=torch.int32, device=topk_indices.device
    )
    lens = torch.zeros(num_tokens, dtype=torch.int32, device=topk_indices.device)
    gp_list = global_positions.tolist()
    req_list = (
        req_id_per_token.tolist() if req_id_per_token is not None else [0] * num_tokens
    )
    pf_list = prefix_lengths.tolist() if prefix_lengths is not None else [sp_int]
    cu_cmp = cu_cmp_offsets.tolist() if cu_cmp_offsets is not None else None
    cu_swa = cu_swa_offsets.tolist() if cu_swa_offsets is not None else None
    for t in range(num_tokens):
        b = req_list[t]
        pf = pf_list[b]
        gp = gp_list[t]
        gather_start = pf - min(pf, window_size - 1)
        cmp_base = cu_cmp[b] if cu_cmp is not None else M * b
        swa_base = cu_swa[b] if cu_swa is not None else M * b + N
        tk_len = min((gp + 1) // compress_ratio, topk)
        sw_len = min(gp + 1, window_size)
        if tk_len > 0:
            if dense_arange_topk:
                vals = torch.arange(
                    tk_len, dtype=torch.int32, device=topk_indices.device
                )
            else:
                vals = topk_indices[t, :tk_len]
            out[t, :tk_len] = cmp_base + vals
        if sw_len > 0:
            swa_start_pos = gp - sw_len + 1 - gather_start
            out[t, tk_len : tk_len + sw_len] = (
                swa_base
                + swa_start_pos
                + torch.arange(sw_len, dtype=torch.int32, device=topk_indices.device)
            )
        lens[t] = tk_len + sw_len
    return out, lens


def _build_dummy_kv_pools(
    num_blocks: int,
    block_size: int,
    head_dim: int,
    idx_hd: int,
    device: torch.device,
):
    class _LayerCache:
        def __init__(self, base: torch.Tensor, tokens_per_block: int):
            self.kv_cache_base = base
            self.seq_size_per_block = tokens_per_block

    class _KVCacheMgr:
        def __init__(self, layers_by_tag: Dict[str, _LayerCache], tpb: int):
            self._layers = layers_by_tag
            self._tpb = tpb
            self.group_tags = list(layers_by_tag.keys())

        def get_layer_cache(self, layer_id: int, tag: str):
            del layer_id
            if tag not in self._layers:
                raise RuntimeError(f"missing cache tag {tag}")
            return self._layers[tag]

        def get_seq_size_per_block(self, tag: str) -> int:
            return self._tpb

        def get_kernel_seq_size_per_block(self, tag: str) -> int:
            return self._tpb

    layers: Dict[str, _LayerCache] = {}
    bt_by_type: Dict[str, torch.Tensor] = {}

    for tag in (SWA_KV, CSA_KV, HCA_KV):
        ratio = 1 if tag == SWA_KV else (4 if tag == CSA_KV else 128)
        eb = max(1, block_size // ratio)
        raw = torch.zeros((num_blocks, eb * 584), dtype=torch.uint8, device=device)
        layers[tag] = _LayerCache(raw, block_size)

    idx_eb = max(1, block_size // 4)
    layers[INDEXER_KV] = _LayerCache(
        torch.zeros((num_blocks, idx_eb * 132), dtype=torch.uint8, device=device),
        block_size,
    )
    layers[CSA_STATE] = _LayerCache(
        torch.zeros(
            (num_blocks, block_size * 4 * head_dim), dtype=torch.float32, device=device
        ),
        block_size,
    )
    layers[HCA_STATE] = _LayerCache(
        torch.zeros(
            (num_blocks, block_size * 2 * head_dim), dtype=torch.float32, device=device
        ),
        block_size,
    )
    layers[INDEXER_STATE] = _LayerCache(
        torch.zeros(
            (num_blocks, block_size * 4 * idx_hd), dtype=torch.float32, device=device
        ),
        block_size,
    )

    bt = (
        torch.arange(1, num_blocks, dtype=torch.int32, device=device)
        .view(1, -1)
        .expand(4, -1)
        .contiguous()
    )
    for tag in (
        SWA_KV,
        CSA_KV,
        HCA_KV,
        CSA_STATE,
        HCA_STATE,
        INDEXER_KV,
        INDEXER_STATE,
    ):
        bt_by_type[tag] = bt
    return _KVCacheMgr(layers, block_size), bt_by_type


def _make_meta_attn(
    compress_ratio: int,
    device: torch.device,
    kv_cache,
    bt_by_type,
    *,
    head_dim: int = 512,
    rope_dim: int = 64,
    idx_hd: int = 128,
    window_size: int = 128,
) -> AttentionFP8:
    attn = AttentionFP8.__new__(AttentionFP8)
    torch.nn.Module.__init__(attn)
    attn.layer_id = 0
    attn.dim = 512
    attn.q_lora_rank = 1536
    attn.n_heads = 16
    attn.head_dim = head_dim
    attn.rope_head_dim = rope_dim
    attn.window_size = window_size
    attn.compress_ratio = compress_ratio
    attn.eps = 1e-6
    attn._cp_ctx = None
    attn._kv_cache = kv_cache
    attn._block_tables_by_type = bt_by_type
    attn._prefill_meta_shared = None
    attn.freqs_cis = precompute_freqs_cis(rope_dim, 4096, 0, 10000.0, 1.0, 32, 1).to(
        device
    )
    attn._pool_spec = {
        SWA_KV: (torch.uint8, 584),
        CSA_KV: (torch.uint8, 584),
        HCA_KV: (torch.uint8, 584),
        INDEXER_KV: (torch.uint8, 132),
        CSA_STATE: (torch.float32, 4 * head_dim),
        HCA_STATE: (torch.float32, 2 * head_dim),
        INDEXER_STATE: (torch.float32, 4 * idx_hd),
    }

    if compress_ratio in (4, 128):
        cmp = CompressorFP8.__new__(CompressorFP8)
        torch.nn.Module.__init__(cmp)
        cmp.head_size = head_dim
        cmp.compress_ratio = compress_ratio
        cmp.rope_head_dim = rope_dim
        cmp.rms_norm_eps = 1e-6
        cmp.overlap = compress_ratio == 4
        cmp._cp_ctx = None
        cmp._state_block_table = None
        cmp._kv_block_table = None
        cmp._kv_pool_view = None
        cmp._state_pool_view = None
        cmp._state_pool_3d = None
        cmp._state_eb = 0
        cmp._kv_eb = 0
        cmp._state_tokens_per_block = 0
        cmp._kv_tokens_per_block = 0
        cmp._kv_owner_tokens_per_block = 0
        cmp._kv_cache_sharded = False
        cmp.freqs_cis = None
        cmp._cos_sin_cache = None
        attn.compressor = cmp
    else:
        attn.compressor = None

    if compress_ratio == 4:
        idx = IndexerFP8.__new__(IndexerFP8)
        torch.nn.Module.__init__(idx)
        idx.head_dim = idx_hd
        idx.rope_head_dim = rope_dim
        idx.index_n_heads = 32
        idx.index_topk = 256
        idx.compress_ratio = 4
        idx.eps = 1e-6
        idx._cp_ctx = None
        idx.freqs_cis = None
        idx._kv_cache_t = 64
        nested = CompressorFP8.__new__(CompressorFP8)
        torch.nn.Module.__init__(nested)
        nested.head_size = idx_hd
        nested.compress_ratio = 4
        nested.rope_head_dim = rope_dim
        nested.rms_norm_eps = 1e-6
        nested.overlap = True
        nested._cp_ctx = None
        nested._state_block_table = None
        nested._kv_block_table = None
        nested._kv_pool_view = None
        nested._state_pool_view = None
        nested._state_pool_3d = None
        nested._state_eb = 0
        nested._kv_eb = 0
        nested._state_tokens_per_block = 0
        nested._kv_tokens_per_block = 0
        nested._kv_owner_tokens_per_block = 0
        nested._kv_cache_sharded = False
        nested.freqs_cis = None
        nested._cos_sin_cache = None
        idx.compressor = nested
        idx.wq_b = object()
        attn.indexer = idx
    else:
        attn.indexer = None
    return attn


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class P1PrefillSyncAndParityTest(unittest.TestCase):
    @torch.inference_mode()
    def test_prefill_meta_and_cp_zero_cuda_sync(self) -> None:
        device = torch.device("cuda")
        # 1. build_cp_context_for_forward + cp_gather_last_by_request (no CUDA sync)
        lengths_cpu = torch.tensor([6, 10, 8], dtype=torch.int32)
        chunk_cpu = torch.tensor([4, 6, 4], dtype=torch.int32)
        chunk_gpu = chunk_cpu.to(device)
        prefixes_cpu = torch.tensor([0, 4, 12], dtype=torch.int32)
        prefixes_gpu = prefixes_cpu.to(device)
        # Zigzag-like chunk_length=14 for cp_size=2 (sum([4, 6, 4]) == 14)
        padded_len = 28
        padding_mask = torch.ones(padded_len, dtype=torch.int32, device=device)
        padding_mask[6:8] = 0
        padding_mask[18:20] = 0
        restore_idx = torch.arange(padded_len, dtype=torch.int32, device=device)
        shuffle_idx = torch.arange(padded_len, dtype=torch.int32, device=device)
        cp_info = SimpleNamespace(
            prefill_qkv_padding_mask=padding_mask,
            prefill_qkv_restore_indice=restore_idx,
            prefill_shuffle_indices=shuffle_idx,
            prefill_actual_input_lengths=lengths_cpu,
            prefill_actual_input_lengths_cpu=lengths_cpu,
        )
        with _forbid_cuda_d2h_sync():
            cp_ctx = build_cp_context_for_forward(
                cp_info=cp_info,
                cp_size=2,
                cp_rank=0,
                num_tokens=14,
                device=device,
                prefix_lengths=prefixes_gpu,
                prefix_lengths_host=prefixes_cpu,
                chunk_lengths_device=chunk_gpu,
                chunk_lengths_host=chunk_cpu,
                kv_cache_sharded=True,
            )
            self.assertEqual(cp_ctx.input_lengths_global_host, (6, 10, 8))
            self.assertEqual(cp_ctx.prefix_lengths_host, (0, 4, 12))

            local_2d = torch.randn(14, 16, dtype=torch.bfloat16, device=device)
            with patch(
                "rtp_llm.models_py.modules.dsv4.cp.all_gather",
                side_effect=lambda t, group=None: t.repeat(2, *([1] * (t.dim() - 1))),
            ):
                gathered_last = cp_gather_last_by_request(local_2d, cp_ctx)
        self.assertEqual(tuple(gathered_last.shape), (3, 16))

        # 2. _build_shared_prefill_meta across all ratios (0, 4, 128), fresh & continuation
        block_size = 256
        num_blocks = 32
        kv_cache, bt_by_type = _build_dummy_kv_pools(
            num_blocks, block_size, head_dim=512, idx_hd=128, device=device
        )
        for prefix_vals in ([0, 0, 0], [64, 130, 32]):
            in_vals = [150, 260, 140]
            B = len(in_vals)
            T = sum(in_vals)
            x = torch.empty((T, 512), dtype=torch.bfloat16, device=device)
            in_t = torch.tensor(in_vals, dtype=torch.int32, device=device)
            pf_t = torch.tensor(prefix_vals, dtype=torch.int32, device=device)
            cu_t = torch.tensor(
                [0, in_vals[0], in_vals[0] + in_vals[1], T],
                dtype=torch.int32,
                device=device,
            )
            sp_t = pf_t.to(torch.int64)
            pos_parts = [
                torch.arange(p, p + l, dtype=torch.int64, device=device)
                for p, l in zip(prefix_vals, in_vals)
            ]
            pos_ids = torch.cat(pos_parts, dim=0)
            req_ids = torch.cat(
                [
                    torch.full((l,), b, dtype=torch.int32, device=device)
                    for b, l in enumerate(in_vals)
                ]
            )

            for ratio in (0, 4, 128):
                attn = _make_meta_attn(ratio, device, kv_cache, bt_by_type)
                # Baseline without host pair
                ref_meta = attn._build_shared_prefill_meta(
                    x,
                    int(prefix_vals[0]),
                    sp_per_req=sp_t,
                    cu_seqlens=cu_t,
                    batch_size=B,
                    input_lengths=in_t,
                    prefix_lengths=pf_t,
                    position_ids=pos_ids,
                    req_id_per_token=req_ids,
                    max_seqlen_q=max(in_vals),
                    any_cont=any(p > 0 for p in prefix_vals),
                )
                # Zero-sync path with _prefill_host_lengths
                attn._prefill_host_lengths = (tuple(in_vals), tuple(prefix_vals))
                try:
                    with _forbid_cuda_d2h_sync():
                        fast_meta = attn._build_shared_prefill_meta(
                            x,
                            int(prefix_vals[0]),
                            sp_per_req=sp_t,
                            cu_seqlens=cu_t,
                            batch_size=B,
                            input_lengths=in_t,
                            prefix_lengths=pf_t,
                            position_ids=pos_ids,
                            req_id_per_token=req_ids,
                            max_seqlen_q=max(in_vals),
                        )
                finally:
                    attn._prefill_host_lengths = None

                self.assertEqual(fast_meta.any_cont, ref_meta.any_cont)
                self.assertEqual(fast_meta.swa_meta.M, ref_meta.swa_meta.M)
                self.assertEqual(
                    fast_meta.swa_meta.total_ws_rows, ref_meta.swa_meta.total_ws_rows
                )
                if ratio in (4, 128):
                    wm_f = (
                        fast_meta.csa_meta.workspace_meta
                        if ratio == 4
                        else fast_meta.hca_meta.workspace_meta
                    )
                    wm_r = (
                        ref_meta.csa_meta.workspace_meta
                        if ratio == 4
                        else ref_meta.hca_meta.workspace_meta
                    )
                    self.assertEqual(wm_f.M, wm_r.M)
                    self.assertEqual(wm_f.N, wm_r.N)
                    self.assertEqual(wm_f.total_ws_rows, wm_r.total_ws_rows)
                    self.assertTrue(
                        torch.equal(wm_f.new_k_slot_in_flat, wm_r.new_k_slot_in_flat)
                    )
                    cm_f = (
                        fast_meta.csa_meta.compressor_meta
                        if ratio == 4
                        else fast_meta.hca_meta.compressor_meta
                    )
                    cm_r = (
                        ref_meta.csa_meta.compressor_meta
                        if ratio == 4
                        else ref_meta.hca_meta.compressor_meta
                    )
                    self.assertEqual(
                        cm_f.can_skip_state_cache_read, cm_r.can_skip_state_cache_read
                    )
                    self.assertIsNone(cm_f.active_state_indices)
                    self.assertTrue(
                        torch.equal(
                            cm_f.active_boundary_indices, cm_r.active_boundary_indices
                        )
                    )

    @torch.inference_mode()
    def test_hca_ratio128_split_kernel_parity(self) -> None:
        device = torch.device("cuda")
        head_dim = 512
        rope_dim = 64
        ratio = 128
        state_page = 256
        rms_w = torch.rand(head_dim, dtype=torch.bfloat16, device=device) + 0.5
        ape = torch.randn(ratio, head_dim, dtype=torch.float32, device=device) * 0.1
        freqs_cis = precompute_freqs_cis(rope_dim, 8192, 0, 10000.0, 1.0, 32, 1).to(
            device
        )
        cos_sin, _ = build_cos_sin_cache(freqs_cis)

        # Case 1: Single-req fresh & non-aligned continuation (may_use_cache=False/True)
        for m, sp_cont, may_use_cache in [
            (1024, 0, False),
            (2048, 64, True),
            (4096, 192, True),
        ]:
            kv_raw = torch.randn(m, head_dim, dtype=torch.float32, device=device) * 0.1
            score_raw = (
                torch.randn(m, head_dim, dtype=torch.float32, device=device) * 0.1
            )
            num_state_blocks = (sp_cont + m + state_page - 1) // state_page + 2
            state_cache = (
                torch.randn(
                    num_state_blocks,
                    state_page,
                    2 * head_dim,
                    dtype=torch.float32,
                    device=device,
                )
                * 0.1
            )
            state_bt = torch.arange(
                num_state_blocks, dtype=torch.int32, device=device
            ).unsqueeze(0)
            token_to_req = torch.zeros(m, dtype=torch.int32, device=device)
            pos = torch.arange(sp_cont, sp_cont + m, dtype=torch.int64, device=device)
            state_slots = (pos // state_page) * state_page + (pos % state_page)
            is_b = ((pos + 1) % ratio) == 0
            active = torch.nonzero(is_b).squeeze(-1).to(torch.int32).contiguous()
            kv_bs = max(int(active.numel()), 1)
            b_idx = torch.cumsum(is_b.to(torch.int64), dim=0) - 1
            kv_slots = torch.where(is_b, kv_bs + b_idx, torch.full_like(b_idx, -1))
            valid_slots = kv_slots[active]

            kc_split = _alloc_hca_kv_cache(3, kv_bs, device)
            kc_single = _alloc_hca_kv_cache(3, kv_bs, device)

            # New split path (active_token_indices != None)
            run_fused_compress_kv_write(
                state_cache if may_use_cache else None,
                token_to_req,
                pos,
                state_slots,
                state_bt if may_use_cache else None,
                rms_w,
                1e-6,
                cos_sin,
                kc_split,
                kv_slots,
                kv_raw,
                score_raw,
                ape,
                sp_cont,
                disable_raw_path=False,
                head_dim=head_dim,
                rope_head_dim=rope_dim,
                compress_ratio=ratio,
                overlap=False,
                state_tokens_per_block=state_page,
                may_use_cache=may_use_cache,
                active_token_indices=active,
            )
            # Original monolithic path (active_token_indices=None)
            run_fused_compress_kv_write(
                state_cache if may_use_cache else None,
                token_to_req,
                pos,
                state_slots,
                state_bt if may_use_cache else None,
                rms_w,
                1e-6,
                cos_sin,
                kc_single,
                kv_slots,
                kv_raw,
                score_raw,
                ape,
                sp_cont,
                disable_raw_path=False,
                head_dim=head_dim,
                rope_head_dim=rope_dim,
                compress_ratio=ratio,
                overlap=False,
                state_tokens_per_block=state_page,
                may_use_cache=may_use_cache,
                active_token_indices=None,
            )
            deq_split = _dequant_584b_kv(kc_split, valid_slots, kv_bs)
            deq_single = _dequant_584b_kv(kc_single, valid_slots, kv_bs)
            torch.testing.assert_close(deq_split, deq_single, atol=2e-2, rtol=2e-2)

        # Case 2: Batched B=3 varlen mixed prefixes/raw
        lengths = [300, 512, 410]
        prefixes = [64, 0, 192]
        m_tot = sum(lengths)
        kv_raw = torch.randn(m_tot, head_dim, dtype=torch.float32, device=device) * 0.1
        score_raw = (
            torch.randn(m_tot, head_dim, dtype=torch.float32, device=device) * 0.1
        )
        num_state_blocks = 32
        state_cache = (
            torch.randn(
                num_state_blocks,
                state_page,
                2 * head_dim,
                dtype=torch.float32,
                device=device,
            )
            * 0.1
        )
        state_bt = torch.arange(1, 1 + 3 * 8, dtype=torch.int32, device=device).view(
            3, 8
        )
        token_to_req = torch.cat(
            [
                torch.full((l,), b, dtype=torch.int32, device=device)
                for b, l in enumerate(lengths)
            ]
        )
        pos = torch.cat(
            [
                torch.arange(p, p + l, dtype=torch.int64, device=device)
                for p, l in zip(prefixes, lengths)
            ]
        )
        sp_per_req = torch.tensor(prefixes, dtype=torch.int64, device=device)
        cu_seq_per_req = torch.tensor(
            [0, lengths[0], lengths[0] + lengths[1], m_tot],
            dtype=torch.int64,
            device=device,
        )
        block_ids = state_bt[token_to_req.long(), pos // state_page].to(torch.int64)
        state_slots = block_ids * state_page + (pos % state_page)
        is_b = ((pos + 1) % ratio) == 0
        active = torch.nonzero(is_b).squeeze(-1).to(torch.int32).contiguous()
        kv_bs = max(int(active.numel()), 1)
        b_idx = torch.cumsum(is_b.to(torch.int64), dim=0) - 1
        kv_slots = torch.where(is_b, kv_bs + b_idx, torch.full_like(b_idx, -1))
        valid_slots = kv_slots[active]

        kc_split = _alloc_hca_kv_cache(4, kv_bs, device)
        kc_single = _alloc_hca_kv_cache(4, kv_bs, device)
        for kc, act in [(kc_split, active), (kc_single, None)]:
            run_fused_compress_kv_write(
                state_cache,
                token_to_req,
                pos,
                state_slots,
                state_bt,
                rms_w,
                1e-6,
                cos_sin,
                kc,
                kv_slots,
                kv_raw,
                score_raw,
                ape,
                prefixes[0],
                disable_raw_path=False,
                head_dim=head_dim,
                rope_head_dim=rope_dim,
                compress_ratio=ratio,
                overlap=False,
                seq_start_per_req=sp_per_req,
                cu_seq_per_req=cu_seq_per_req,
                state_tokens_per_block=state_page,
                may_use_cache=True,
                active_token_indices=act,
            )
        deq_split = _dequant_584b_kv(kc_split, valid_slots, kv_bs)
        deq_single = _dequant_584b_kv(kc_single, valid_slots, kv_bs)
        torch.testing.assert_close(deq_split, deq_single, atol=2e-2, rtol=2e-2)

    @torch.inference_mode()
    def test_combine_topk_swa_indices_and_hca_cache_parity(self) -> None:
        device = torch.device("cuda")
        # Test non-CP & CP across ragged varlen, dense_arange_topk, odd topk
        for topk, ratio, dense_arange in [
            (3, 4, False),
            (2048, 4, False),
            (9, 128, True),
        ]:
            lengths = [64, 192, 300, 128]
            prefixes = [0, 32, 128, 70]
            B = len(lengths)
            T = sum(lengths)
            win = 128
            seq_lens = torch.tensor(
                [p + l for p, l in zip(prefixes, lengths)],
                dtype=torch.int32,
                device=device,
            )
            gather_lens = torch.tensor(
                [l + min(p, win - 1) for p, l in zip(prefixes, lengths)],
                dtype=torch.int32,
                device=device,
            )
            qsl = torch.tensor([0, 64, 256, 556, T], dtype=torch.int32, device=device)
            n_cmp = [(p + l) // ratio for p, l in zip(prefixes, lengths)]
            cu_cmp_list, cu_swa_list = [], []
            acc = 0
            for nc, g in zip(n_cmp, gather_lens.tolist()):
                cu_cmp_list.append(acc)
                cu_swa_list.append(acc + nc)
                acc += nc + g
            cu_cmp = torch.tensor(cu_cmp_list, dtype=torch.int32, device=device)
            cu_swa = torch.tensor(cu_swa_list, dtype=torch.int32, device=device)

            if dense_arange:
                topk_idx = (
                    torch.arange(topk, dtype=torch.int32, device=device)
                    .view(1, topk)
                    .expand(T, topk)
                )
            else:
                topk_idx = torch.randint(
                    0, 20, (T, topk), dtype=torch.int32, device=device
                )

            got_i, got_l = combine_topk_swa_indices(
                topk_indices=topk_idx,
                query_start_loc=qsl,
                seq_lens=seq_lens,
                gather_lens=gather_lens,
                window_size=win,
                compress_ratio=ratio,
                topk=topk,
                cu_cmp_offsets=cu_cmp,
                cu_swa_offsets=cu_swa,
                dense_arange_topk=dense_arange,
            )
            ref_i, ref_l = _ref_combine_noncp(
                topk_idx,
                qsl,
                seq_lens,
                gather_lens,
                win,
                ratio,
                topk,
                0,
                0,
                cu_cmp_offsets=cu_cmp,
                cu_swa_offsets=cu_swa,
                dense_arange_topk=dense_arange,
            )
            self.assertTrue(torch.equal(got_l, ref_l))
            self.assertTrue(torch.equal(got_i, ref_i))

            req_ids = torch.cat(
                [
                    torch.full((l,), b, dtype=torch.int32, device=device)
                    for b, l in enumerate(lengths)
                ]
            )
            gpos = torch.cat(
                [
                    torch.arange(p, p + l, dtype=torch.int64, device=device)
                    for p, l in zip(prefixes, lengths)
                ]
            )
            pf_t = torch.tensor(prefixes, dtype=torch.int32, device=device)
            cp_i, cp_l = combine_topk_swa_indices_cp(
                topk_indices=topk_idx,
                global_positions=gpos,
                sp_int=0,
                window_size=win,
                compress_ratio=ratio,
                topk=topk,
                req_id_per_token=req_ids,
                prefix_lengths=pf_t,
                cu_cmp_offsets=cu_cmp,
                cu_swa_offsets=cu_swa,
                dense_arange_topk=dense_arange,
            )
            ref_cp_i, ref_cp_l = _ref_combine_cp(
                topk_idx,
                gpos,
                0,
                win,
                ratio,
                topk,
                0,
                0,
                req_id_per_token=req_ids,
                prefix_lengths=pf_t,
                cu_cmp_offsets=cu_cmp,
                cu_swa_offsets=cu_swa,
                dense_arange_topk=dense_arange,
            )
            self.assertTrue(torch.equal(cp_l, ref_cp_l))
            self.assertTrue(torch.equal(cp_i, ref_cp_i))

    @unittest.skipUnless(
        torch.cuda.is_available() and has_deep_gemm() and is_deep_gemm_e8m0_used(),
        "SM100/SM103 DeepGEMM UE8M0 required",
    )
    @torch.inference_mode()
    def test_qnorm_fused_ue8m0_quant_attention_wiring(self) -> None:
        device = torch.device("cuda")
        # 1. prefill_fused_attn_norm_input_quant parity
        attn = AttentionFP8.__new__(AttentionFP8)
        torch.nn.Module.__init__(attn)
        attn.dim = 4096
        attn.q_lora_rank = 1536
        attn.n_heads = 16
        attn.head_dim = 512
        attn.rope_head_dim = 64
        attn.window_size = 128
        attn.compress_ratio = 0
        attn.eps = 1e-6
        attn.indexer = None
        attn.wq_a = _make_ue8m0_linear(1536, 4096, device)
        attn.wkv = _make_ue8m0_linear(512, 4096, device)
        attn.wq_b = _make_ue8m0_linear(16 * 512, 1536, device)
        attn.q_norm = torch.rand(1536, dtype=torch.bfloat16, device=device) + 0.5
        attn.kv_norm = torch.rand(512, dtype=torch.bfloat16, device=device) + 0.5
        norm_w = torch.rand(4096, dtype=torch.bfloat16, device=device) + 0.5

        for m in [2, 6, 129, 256]:
            x = (
                torch.randn(m, 4096, dtype=torch.bfloat16, device=device) * 0.4
            ).contiguous()
            x_ref_in = x.clone()
            x_ref_norm = flashinfer.norm.rmsnorm(x_ref_in, norm_w, eps=1e-6)
            ref_fp8, ref_s = sgl_per_token_group_quant_fp8(
                x_ref_norm,
                group_size=128,
                eps=1e-4,
                column_major_scales=True,
                scale_tma_aligned=True,
                scale_ue8m0=True,
            )
            ref_out = attn.wq_a.forward_quantized(ref_fp8, ref_s)

            x_norm, q_pair = attn.prefill_fused_attn_norm_input_quant(x, norm_w, 1e-6)
            new_out = attn.wq_a.forward_quantized(*q_pair)
            torch.testing.assert_close(x_norm, x_ref_norm, atol=2e-2, rtol=2e-2)
            torch.testing.assert_close(new_out, ref_out, atol=5e-2, rtol=5e-2)

        # 2. _prefill_compute_qkv + _materialize_prefill_q: SWA/HCA + unpatched IndexerFP8 + fallback
        freqs = precompute_freqs_cis(64, 2048, 0, 10000.0, 1.0, 32, 1).to(device)
        ws = PrefillWorkspace(
            device=device,
            q_rows=512,
            q_dim=16 * 512,
            reserve_cp=False,
            align_bytes=1 << 20,
        )
        for m in [2, 7, 128]:
            x = (
                torch.randn(m, 4096, dtype=torch.bfloat16, device=device) * 0.4
            ).contiguous()
            common = PrefillMeta(
                seqlen=m,
                seqlen_full=m,
                rd=64,
                device=device,
                cp_ctx=None,
                cp_on=False,
                freqs_cis=freqs[:m],
                topk_idxs=torch.empty(0, device=device),
                sp_int=0,
                any_cont=False,
                row_seqlens_full=torch.tensor([m], dtype=torch.long, device=device),
                use_varlen=True,
                workspace=ws,
            )
            # Fused SWA/HCA (indexer=None -> emit_norm=False -> qr is None)
            attn.indexer = None
            qkv_fused = attn._prefill_compute_qkv(x, common)
            self.assertIsNone(qkv_fused.qr)
            self.assertIsNotNone(qkv_fused.qr_quant)
            qkv_mat = attn._materialize_prefill_q(qkv_fused, common)
            q_fused = qkv_mat.q.clone()

            # Baseline exact computation without fused norm+quant
            with patch.object(attn, "_can_use_qr_fused_quant", return_value=False):
                qkv_ref = attn._prefill_compute_qkv(x, common)
                self.assertIsNotNone(qkv_ref.qr)
                self.assertIsNone(qkv_ref.qr_quant)
                q_ref = attn._materialize_prefill_q(qkv_ref, common).q.clone()

            torch.testing.assert_close(q_fused, q_ref, atol=1e-1, rtol=1e-1)
            cos_sim = torch.nn.functional.cosine_similarity(
                q_fused.float().reshape(-1, 512),
                q_ref.float().reshape(-1, 512),
                dim=-1,
            ).min()
            self.assertGreater(float(cos_sim.item()), 0.999)

            # CSA with genuine IndexerFP8 (emit_norm=False -> qr is None, _compute_indexer_q reuses qr_quant)
            idx = IndexerFP8.__new__(IndexerFP8)
            torch.nn.Module.__init__(idx)
            idx.n_heads = 32
            idx.head_dim = 128
            idx.rope_head_dim = 64
            idx.index_n_heads = 32
            idx.index_topk = 64
            idx.compress_ratio = 4
            idx.eps = 1e-6
            idx._cp_ctx = None
            idx.wq_b = _make_ue8m0_linear(32 * 128, 1536, device)
            idx.freqs_cis = freqs
            attn.indexer = idx

            qkv_csa = attn._prefill_compute_qkv(x, common)
            self.assertIsNone(qkv_csa.qr)
            self.assertIsNotNone(qkv_csa.qr_quant)
            idx_q_new = idx._compute_indexer_q(
                qkv_csa.qr, freqs[:m], qr_quant=qkv_csa.qr_quant
            )
            with patch.object(attn, "_can_use_qr_fused_quant", return_value=False):
                qkv_csa_ref = attn._prefill_compute_qkv(x, common)
                idx_q_ref = idx._compute_indexer_q(qkv_csa_ref.qr, freqs[:m])
            idx_cos = torch.nn.functional.cosine_similarity(
                idx_q_new.float().reshape(-1, 128),
                idx_q_ref.float().reshape(-1, 128),
                dim=-1,
            ).min()
            self.assertGreater(float(idx_cos.item()), 0.999)

            # CSA with instance-patched indexer.forward -> emit_norm=True must populate qkv.qr
            idx.forward = lambda *a, **k: torch.zeros((m, 64), dtype=torch.int32, device=device)  # type: ignore[method-assign]
            qkv_patched = attn._prefill_compute_qkv(x, common)
            self.assertIsNotNone(qkv_patched.qr)
            self.assertIsNotNone(qkv_patched.qr_quant)
            self.assertEqual(tuple(qkv_patched.qr.shape), (m, 1536))

        # 3. Unaligned 16B weight pointer falls back cleanly
        unaligned_norm = torch.rand(1537, dtype=torch.bfloat16, device=device)[1:]
        self.assertNotEqual(unaligned_norm.data_ptr() % 16, 0)
        attn.indexer = None
        attn.q_norm = unaligned_norm
        x_small = torch.randn(4, 4096, dtype=torch.bfloat16, device=device)
        common_small = PrefillMeta(
            seqlen=4,
            seqlen_full=4,
            rd=64,
            device=device,
            cp_ctx=None,
            cp_on=False,
            freqs_cis=freqs[:4],
            topk_idxs=torch.empty(0, device=device),
            sp_int=0,
            any_cont=False,
            row_seqlens_full=torch.tensor([4], dtype=torch.long, device=device),
            use_varlen=True,
            workspace=ws,
        )
        qkv_unaligned = attn._prefill_compute_qkv(x_small, common_small)
        self.assertIsNotNone(qkv_unaligned.qr)
        self.assertIsNone(qkv_unaligned.qr_quant)


if __name__ == "__main__":
    unittest.main()
