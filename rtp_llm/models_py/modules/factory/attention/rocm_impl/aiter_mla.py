"""Torch-native MLA attention implementation for ROCm.

ROCm has no native MLA kernel registered in the python attention factory, so
models like tbstars_tse (DeepSeek-style MLA decoder) fail with
"can not find mla type". This module provides a correctness-first torch
implementation that mirrors the reference math in
``modules/hybrid/test/mla_attention_ref.py`` and the engine's flashinfer MLA
path (``cuda_mla_impl/flashinfer_mla_wrapper.py``).

The prefill impl uses MLA **absorb**: it writes current tokens' latent KV to
the paged cache, projects q into the compressed latent space (W_UK absorb), and
attends over ``[reused prefix latent ++ current latent]`` read directly from the
cache as compressed KV — it NEVER decompresses per-head K/V over the prefix.
This mirrors the CUDA flashinfer absorb decode op
(``cuda_mla_impl/flashinfer_mla.py``: ``bmm(q_nope, kc)`` → attend in latent →
``bmm(out, vc)``) and is what makes same-round ``enable_reuse_cache_in_batch``
both correct AND fast on ROCm: the shared prefix is a cheap latent gather, not a
redundant per-stream decompress GEMM. With no reused prefix it reduces to
self-contained causal attention in latent space. The decode impl reads the paged
latent cache. RoPE follows the engine's neox-style cos_sin_cache layout built in
``models/deepseek_v2.py`` (``[max_seq, rope_dim]`` with the first half cosines,
second half sines).
"""

from collections import defaultdict
from typing import Dict, List, Optional

import aiter
import torch

from rtp_llm.models_py.modules.factory.attention.fmha_impl_base import MlaImplBase
from rtp_llm.ops import AttentionConfigs, FMHAConfig, ParallelismConfig
from rtp_llm.ops.compute_ops import LayerKVCache, PyAttentionInputs
from rtp_llm.utils.model_weight import W


def _apply_neox_rope(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    """Apply neox-style (rotate_half) RoPE.

    Args:
        x: [..., rope_dim] tensor to rotate.
        cos: [tokens, rope_dim // 2] cosine values per token.
        sin: [tokens, rope_dim // 2] sine values per token.
    x is broadcast on token dim; cos/sin are reshaped to broadcast over heads.
    """
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    # cos/sin shape [T, half]; insert head broadcast dims as needed
    while cos.dim() < x.dim():
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
    out1 = x1 * cos - x2 * sin
    out2 = x2 * cos + x1 * sin
    return torch.cat((out1, out2), dim=-1)


class _RocmMlaTorchBase(MlaImplBase):
    """Shared setup for the torch-native ROCm MLA impls."""

    def __init__(
        self,
        attn_configs: AttentionConfigs,
        attn_inputs: PyAttentionInputs,
        weights: List[Dict[str, torch.Tensor]],
        cos_sin_cache: torch.Tensor,
        fmha_config: Optional[FMHAConfig] = None,
        use_trt_fmha: bool = False,
        quant_config: Optional[object] = None,
        max_seq_len: int = 0,
        is_cuda_graph: bool = False,
        parallelism_config: Optional[ParallelismConfig] = None,
    ) -> None:
        super().__init__(
            attn_configs,
            attn_inputs,
            weights,
            cos_sin_cache,
            fmha_config,
            use_trt_fmha=use_trt_fmha,
            quant_config=quant_config,
            max_seq_len=max_seq_len,
            is_cuda_graph=is_cuda_graph,
            parallelism_config=parallelism_config,
        )
        self.num_heads = attn_configs.head_num
        self.qk_nope_head_dim = attn_configs.nope_head_dim
        self.qk_rope_head_dim = attn_configs.rope_head_dim
        self.q_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        self.kv_lora_rank = attn_configs.kv_lora_rank
        self.v_head_dim = attn_configs.v_head_dim
        self.softmax_scale = self.q_head_dim ** (-0.5)
        if cos_sin_cache is None:
            raise Exception("RocmMla needs cos_sin_cache but got none")
        # Per-request cache for the layer-invariant prefill plan (positions,
        # cos/sin, cache write/gather indices, ragged assembly map, cu_seqlens).
        # One ``MlaImplBase`` instance is created per request and reused across
        # all decoder layers, so this is request-scoped. Keyed on the identity
        # of ``kv_cache_kernel_block_id_host`` (+ dtype) so the hybrid per-layer
        # block-group remap (``select_block_map_for_layer``) is handled
        # correctly: each distinct block-id table gets its own plan, while the
        # common single-group case computes the plan exactly once.
        self._plan_cache: Dict[object, Dict[str, object]] = {}

    def _rope_cos_sin(self, positions: torch.Tensor) -> tuple:
        """Gather cos/sin for the given token positions.

        cos_sin_cache layout: [max_seq, rope_dim] where [:, :rope_dim//2] are
        cosines and [:, rope_dim//2:] are sines (see models/deepseek_v2.py).
        """
        cache = self.cos_sin_cache
        half = cache.shape[-1] // 2
        rows = cache[positions]  # [T, rope_dim]
        cos = rows[:, :half]
        sin = rows[:, half:]
        return cos, sin

    def _build_positions(
        self, input_lengths: List[int], prefix_lengths: List[int]
    ) -> torch.Tensor:
        """Per-token position ids derived from input/prefix lengths."""
        positions: List[int] = []
        for L, p in zip(input_lengths, prefix_lengths):
            positions.extend(range(p, p + L))
        return torch.tensor(positions, dtype=torch.long)

    def _decompress(self, compressed_kv: torch.Tensor, layer_id: int) -> tuple:
        """Decompress latent KV to per-head ``k_nope`` and ``value``.

        ``mla_kv_b_w`` has shape ``[kv_lora_rank, H * (nope + v)]`` and satisfies
        ``kv = compressed_kv @ kv_b_w``. One GEMM over all input tokens (current
        or gathered prefix) decompresses them in a single batched op rather than
        per-stream; reshaping ``[N, H * (nope + v)]`` to ``[N, H, nope + v]``
        splits into ``k_nope`` ``[N, H, nope]`` and ``value`` ``[N, H, v]``.
        """
        kv_b = self.weights[layer_id][W.mla_kv_b_w].float()  # [r, H*(nope+v)]
        kv = compressed_kv.float() @ kv_b  # [N, H*(nope+v)]
        kv = kv.view(-1, self.num_heads, self.qk_nope_head_dim + self.v_head_dim)
        k_nope = kv[:, :, : self.qk_nope_head_dim]  # [N, H, nope]
        value = kv[:, :, self.qk_nope_head_dim :]  # [N, H, v]
        return k_nope, value

    def _prefix_lengths_list(self, num_streams: int) -> List[int]:
        prefix = self.attn_inputs.prefix_lengths
        if prefix is not None and prefix.numel() > 0:
            return prefix.cpu().tolist()
        return [0] * num_streams

    def _get_plan(
        self, device: torch.device, dtype: torch.dtype, kv_cache: Optional[LayerKVCache]
    ) -> Dict[str, object]:
        """Return the layer-invariant prefill plan, computing it once per request.

        Everything here depends only on the request's sequence lengths and the
        paged-cache block-id table — never on the per-layer q/kv tensors or
        ``weights[layer_id]`` — so it is hoisted out of the 48-layer decoder
        loop and memoized. The cache key includes the block-id table identity so
        the hybrid per-layer block remap is still correct.
        """
        block_ids_t = self.attn_inputs.kv_cache_kernel_block_id_host
        key = (id(block_ids_t), str(dtype))
        plan = self._plan_cache.get(key)
        if plan is None:
            plan = self._compute_plan(device, dtype, kv_cache, block_ids_t)
            self._plan_cache[key] = plan
        return plan

    def _compute_plan(
        self,
        device: torch.device,
        dtype: torch.dtype,
        kv_cache: Optional[LayerKVCache],
        block_ids_t: Optional[torch.Tensor],
    ) -> Dict[str, object]:
        """Build the layer-invariant prefill plan (see ``_get_plan``).

        Produces, in one shot for the whole request:
          - ``cos``/``sin`` gathered for every token position and cast to dtype;
          - ``write_blk``/``write_off``: flat scatter indices for persisting the
            current tokens' latent KV into the paged cache;
          - ``groups``: per shared-prefix group, the flat gather indices to read
            that group's latent prefix from the cache ONCE (the dedup);
          - ``kv_index``: a single gather map that assembles the ragged
            ``[prefix ++ current]`` key/value buffers from the concatenation of
            current-token KV and the per-group decompressed prefixes, replacing
            the old per-stream Python loop + multi-``cat``;
          - ``cu_seqlens_q``/``cu_seqlens_k`` + ``max_seqlen_q``/``max_seqlen_k``.
        """
        input_lengths = self.attn_inputs.input_lengths.cpu().tolist()
        n = len(input_lengths)
        prefix_lengths = self._prefix_lengths_list(n)

        positions = self._build_positions(input_lengths, prefix_lengths).to(device)
        cos, sin = self._rope_cos_sin(positions)
        cos = cos.to(dtype)
        sin = sin.to(dtype)

        cur_off: List[int] = []
        acc = 0
        for L in input_lengths:
            cur_off.append(acc)
            acc += L
        total_q = acc

        has_cache = kv_cache is not None and kv_cache.kv_cache_base is not None
        write_blk: Optional[torch.Tensor] = None
        write_off: Optional[torch.Tensor] = None
        block_ids: Optional[torch.Tensor] = None
        ssb = 0
        base_device = device
        if has_cache and block_ids_t is not None and block_ids_t.numel() > 0:
            base_device = kv_cache.kv_cache_base.device
            ssb = kv_cache.seq_size_per_block
            block_ids = block_ids_t.to(torch.long).to(base_device)
            blk_all: List[torch.Tensor] = []
            off_all: List[torch.Tensor] = []
            for i, L in enumerate(input_lengths):
                p = prefix_lengths[i]
                row = block_ids[i]
                pos = torch.arange(p, p + L, device=base_device)
                blk_all.append(row[pos // ssb])
                off_all.append(pos % ssb)
            if blk_all:
                write_blk = torch.cat(blk_all)
                write_off = torch.cat(off_all)

        # Shared-prefix grouping + dedup: reusers that share a first physical
        # block share the whole prefix chain, so one ``max_p``-token gather +
        # decompress per group covers all members.
        groups_plan: List[Dict[str, torch.Tensor]] = []
        grp_off_of: Dict[int, tuple] = {}  # stream -> (row offset in k_pfx_all, p)
        prefix_rows = 0
        any_reuse = False
        reuser_idx = [i for i, p in enumerate(prefix_lengths) if p > 0]
        if reuser_idx and block_ids is not None:
            first_block = block_ids[:, 0].tolist()
            groups: "defaultdict[int, List[int]]" = defaultdict(list)
            for i in reuser_idx:
                groups[first_block[i]].append(i)
            for members in groups.values():
                rep = max(members, key=lambda i: prefix_lengths[i])
                max_p = prefix_lengths[rep]
                row = block_ids[rep]
                pos = torch.arange(0, max_p, device=base_device)
                groups_plan.append(
                    {"gather_blk": row[pos // ssb], "gather_off": pos % ssb}
                )
                for i in members:
                    grp_off_of[i] = (prefix_rows, prefix_lengths[i])
                prefix_rows += max_p
            any_reuse = True

        # Ragged key/value assembly map into k_src = [current (total_q rows) ++
        # per-group decompressed prefixes (prefix_rows rows)], plus seqlens_k.
        seqlens_k: List[int] = []
        kv_rows: List[torch.Tensor] = []
        for i, L in enumerate(input_lengths):
            eff_p = 0
            if any_reuse and i in grp_off_of:
                grp_off, p = grp_off_of[i]
                kv_rows.append(
                    torch.arange(
                        total_q + grp_off, total_q + grp_off + p, dtype=torch.long
                    )
                )
                eff_p = p
            kv_rows.append(torch.arange(cur_off[i], cur_off[i] + L, dtype=torch.long))
            seqlens_k.append(eff_p + L)
        kv_index = torch.cat(kv_rows).to(device) if any_reuse else None

        cu_seqlens_q = torch.zeros(n + 1, dtype=torch.int32, device=device)
        cu_seqlens_k = torch.zeros(n + 1, dtype=torch.int32, device=device)
        cu_seqlens_q[1:] = torch.tensor(
            input_lengths, dtype=torch.int32, device=device
        ).cumsum(0)
        cu_seqlens_k[1:] = torch.tensor(
            seqlens_k, dtype=torch.int32, device=device
        ).cumsum(0)

        return {
            "cos": cos,
            "sin": sin,
            "write_blk": write_blk,
            "write_off": write_off,
            "any_reuse": any_reuse,
            "groups": groups_plan,
            "kv_index": kv_index,
            "cu_seqlens_q": cu_seqlens_q,
            "cu_seqlens_k": cu_seqlens_k,
            "max_seqlen_q": max(input_lengths),
            "max_seqlen_k": max(seqlens_k),
        }


class RocmMlaPrefillImpl(_RocmMlaTorchBase):
    """Prefill MLA via batched decompress + fused varlen flash attention.

    Mirrors the CUDA flashinfer ragged-prefill path
    (``cuda_mla_impl/flashinfer_mla.py``: decompress latent KV, then fused
    ragged attention with ``head_dim_qk=192``/``head_dim_vo=128``), and is what
    makes ``enable_reuse_cache_in_batch`` a net win on ROCm:

      1. Current tokens' latent KV (``compressed_kv ++ roped k_pe``) is written
         to the paged cache, then decompressed to per-head ``k_nope``/``value``
         in ONE GEMM for all tokens (``kv = compressed_kv @ kv_b_w``).
      2. Reusers' shared prefix is gathered from the cache and decompressed ONCE
         per group (concatenated -> a single gather + single GEMM), rather than
         re-decompressing the shared prefix per reuser per layer.
      3. Attention runs as a single fused ``aiter.flash_attn_varlen_func`` over
         ragged ``[prefix ++ current]`` per stream — the same fused kernel the
         encoder uses — instead of a chunked FP32 padded einsum. Query/key carry
         ``[nope ++ rope]`` (192) so the QK dot gives ``q_nope.k_nope +
         q_pe.k_pe`` (``k_pe`` broadcast across heads); ``causal=True`` uses FA2
         bottom-right alignment so current query ``j`` attends ``prefix[0:p]`` +
         ``current[0:j]``.

    Decompress (score over ``nope``) is used rather than absorb (score over the
    larger ``kv_lora_rank``) because for these prefill query lengths absorb's
    score/output einsums are ~``r/nope``x more FLOPs; this mirrors the CUDA path,
    which only switches to absorb for short-query decode with reuse cache.
    """

    # Class-level cache for whether aiter accepts asymmetric qk/v head dims
    # (192 vs 128). None=unknown, True=asymmetric ok, False=must pad v to qk dim.
    _ASYMMETRIC_HEAD_DIM_OK: Optional[bool] = None

    def _varlen_attn(
        self,
        q_full: torch.Tensor,
        k_full: torch.Tensor,
        v_full: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
        max_seqlen_q: int,
        max_seqlen_k: int,
    ) -> torch.Tensor:
        """Fused ragged varlen attention. q/k carry head_dim 192, v carries 128.

        aiter's default softmax_scale is ``q_head_dim**-0.5`` == ``self.softmax_scale``
        (== ``192**-0.5``), so it is left implicit to match the proven encoder call.
        If aiter rejects asymmetric qk/v head dims, value is zero-padded to the qk
        dim and the output is sliced back to ``v_head_dim``.
        """

        def _call(qq: torch.Tensor, kk: torch.Tensor, vv: torch.Tensor) -> torch.Tensor:
            return aiter.flash_attn_varlen_func(
                qq,
                kk,
                vv,
                cu_seqlens_q,
                cu_seqlens_k,
                max_seqlen_q,
                max_seqlen_k,
                dropout_p=0.0,
                causal=True,
            )

        cls = type(self)
        if cls._ASYMMETRIC_HEAD_DIM_OK is None:
            try:
                out = _call(q_full, k_full, v_full)
                cls._ASYMMETRIC_HEAD_DIM_OK = True
                return out
            except Exception:
                cls._ASYMMETRIC_HEAD_DIM_OK = False

        if cls._ASYMMETRIC_HEAD_DIM_OK:
            return _call(q_full, k_full, v_full)

        v_dim = v_full.shape[-1]
        qk_dim = q_full.shape[-1]
        v_pad = torch.zeros(
            (v_full.shape[0], v_full.shape[1], qk_dim),
            dtype=v_full.dtype,
            device=v_full.device,
        )
        v_pad[..., :v_dim] = v_full
        out = _call(q_full, k_full, v_pad)
        return out[..., :v_dim]

    @classmethod
    def support(
        cls, attn_configs: AttentionConfigs, attn_inputs: PyAttentionInputs
    ) -> bool:
        return (
            attn_configs.use_mla
            and attn_inputs.is_prefill
            and not attn_configs.is_sparse
        )

    def forward(
        self,
        q: torch.Tensor,
        compressed_kv: torch.Tensor,
        k_pe: torch.Tensor,
        kv_cache: Optional[LayerKVCache],
        layer_id: int,
        topk_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # q: [T, H, q_head_dim]; compressed_kv: [T, kv_lora_rank]; k_pe: [T, rope_dim]
        device = q.device
        H = self.num_heads
        nope = self.qk_nope_head_dim
        v_dim = self.v_head_dim
        q_dtype = q.dtype

        # All layer-invariant work (positions, cos/sin, cache write/gather
        # indices, ragged assembly map, cu_seqlens) is computed once per request
        # and memoized; only the q/kv tensor math below runs per layer.
        plan = self._get_plan(device, q_dtype, kv_cache)
        cos = plan["cos"]
        sin = plan["sin"]

        q_nope = q[:, :, :nope]
        q_pe = _apply_neox_rope(q[:, :, nope:], cos, sin).float()  # [T, H, rope]
        k_pe = _apply_neox_rope(k_pe, cos, sin)  # [T, rope] (shared across heads)

        # Persist current latent KV (compressed_kv ++ roped k_pe) to the paged
        # cache for all tokens before any prefix read below, so same-round
        # in-batch reusers see the prefix KV produced by the writer stream.
        write_blk = plan["write_blk"]
        if write_blk is not None:
            base = kv_cache.kv_cache_base
            latent = torch.cat([compressed_kv, k_pe], dim=-1).to(base.dtype)
            base[write_blk, plan["write_off"], :] = latent

        # Decompress all current tokens' latent KV in one GEMM, then build the
        # per-head query / key with [nope ++ rope] dims so the QK dot reproduces
        # q_nope.k_nope + q_pe.k_pe (k_pe broadcast across heads).
        k_nope_cur, v_cur = self._decompress(
            compressed_kv, layer_id
        )  # [T,H,nope],[T,H,v]
        k_pe_f = k_pe.float()
        q_full = torch.cat([q_nope.float(), q_pe], dim=-1)  # [T, H, dqk]
        k_cur_full = torch.cat(
            [k_nope_cur, k_pe_f[:, None, :].expand(-1, H, -1)], dim=-1
        )  # [T, H, dqk]

        if not plan["any_reuse"]:
            # No reused prefix: current-only KV is already in stream order, so
            # it IS the ragged key/value — no gather/cat needed.
            k_ragged = k_cur_full
            v_ragged = v_cur
        else:
            # Gather + decompress each *shared* prefix once (the dedup) using the
            # precomputed per-group gather indices, then assemble the ragged
            # [prefix ++ current] buffers via a SINGLE gather over
            # k_src = [current ++ per-group prefixes] (the precomputed kv_index),
            # replacing the old per-stream Python loop + multi-cat.
            base = kv_cache.kv_cache_base
            k_pfx_parts: List[torch.Tensor] = []
            v_pfx_parts: List[torch.Tensor] = []
            for g in plan["groups"]:
                latent = base[g["gather_blk"], g["gather_off"], :]  # [max_p, r+rope]
                ck = latent[:, : self.kv_lora_rank]
                kpe = latent[:, self.kv_lora_rank :]
                k_nope_p, v_p = self._decompress(ck, layer_id)  # [max_p,H,nope/v]
                k_pfx_parts.append(
                    torch.cat(
                        [k_nope_p, kpe.float()[:, None, :].expand(-1, H, -1)], dim=-1
                    )
                )
                v_pfx_parts.append(v_p)
            k_src = torch.cat([k_cur_full] + k_pfx_parts, dim=0)
            v_src = torch.cat([v_cur] + v_pfx_parts, dim=0)
            kv_index = plan["kv_index"]
            k_ragged = k_src[kv_index]
            v_ragged = v_src[kv_index]

        out = self._varlen_attn(
            q_full.to(q_dtype),
            k_ragged.to(q_dtype),
            v_ragged.to(q_dtype),
            plan["cu_seqlens_q"],
            plan["cu_seqlens_k"],
            plan["max_seqlen_q"],
            plan["max_seqlen_k"],
        )
        return out.reshape(-1, H * v_dim).to(q_dtype)


class RocmMlaDecodeImpl(_RocmMlaTorchBase):
    """Decode MLA via absorb + ``aiter.mla.mla_decode_fwd`` fused kernel.

    Single query token per sequence.  Uses the absorb trick (score in latent
    ``kv_lora_rank`` space) then delegates the batched paged attention to the
    aiter ASM kernel — no Python per-sequence loop.
    """

    @classmethod
    def support(
        cls, attn_configs: AttentionConfigs, attn_inputs: PyAttentionInputs
    ) -> bool:
        return (
            attn_configs.use_mla
            and not attn_inputs.is_prefill
            and not attn_configs.is_sparse
        )

    def _get_decode_plan(
        self, device: torch.device, dtype: torch.dtype, kv_cache: Optional[LayerKVCache]
    ) -> Dict[str, object]:
        block_ids_t = self.attn_inputs.kv_cache_kernel_block_id_host
        key = ("decode", id(block_ids_t), str(dtype))
        plan = self._plan_cache.get(key)
        if plan is None:
            plan = self._compute_decode_plan(device, dtype, kv_cache, block_ids_t)
            self._plan_cache[key] = plan
        return plan

    def _compute_decode_plan(
        self,
        device: torch.device,
        dtype: torch.dtype,
        kv_cache: Optional[LayerKVCache],
        block_ids_t: Optional[torch.Tensor],
    ) -> Dict[str, object]:
        seq_lengths = self.attn_inputs.sequence_lengths.cpu().tolist()
        B = len(seq_lengths)
        ssb = kv_cache.seq_size_per_block

        positions = torch.tensor([s - 1 for s in seq_lengths], dtype=torch.long).to(
            device
        )
        cos, sin = self._rope_cos_sin(positions)
        cos = cos.to(dtype)
        sin = sin.to(dtype)

        # Vectorized cache-write indices (no Python loop).
        pos_t = torch.tensor(seq_lengths, dtype=torch.long, device=device) - 1
        block_ids = block_ids_t.to(torch.long).to(device)
        write_blk = block_ids[torch.arange(B, device=device), pos_t // ssb]
        write_off = pos_t % ssb

        # Page-table metadata for aiter.mla.mla_decode_fwd.
        qo_indptr = torch.arange(0, B + 1, dtype=torch.int32, device=device)

        num_pages_list: List[int] = []
        kv_indices_list: List[torch.Tensor] = []
        kv_last_page_lens_list: List[int] = []
        for i in range(B):
            s = seq_lengths[i]
            n_pages = (s + ssb - 1) // ssb
            num_pages_list.append(n_pages)
            kv_indices_list.append(block_ids[i, :n_pages])
            last_len = s % ssb
            kv_last_page_lens_list.append(last_len if last_len > 0 else ssb)

        kv_indptr = torch.zeros(B + 1, dtype=torch.int32, device=device)
        kv_indptr[1:] = torch.tensor(
            num_pages_list, dtype=torch.int32, device=device
        ).cumsum(0)
        kv_indices = torch.cat(kv_indices_list).to(torch.int32)
        kv_last_page_lens = torch.tensor(
            kv_last_page_lens_list, dtype=torch.int32, device=device
        )

        return {
            "cos": cos,
            "sin": sin,
            "write_blk": write_blk,
            "write_off": write_off,
            "qo_indptr": qo_indptr,
            "kv_indptr": kv_indptr,
            "kv_indices": kv_indices,
            "kv_last_page_lens": kv_last_page_lens,
        }

    def forward(
        self,
        q: torch.Tensor,
        compressed_kv: torch.Tensor,
        k_pe: torch.Tensor,
        kv_cache: Optional[LayerKVCache],
        layer_id: int,
        topk_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        device = q.device
        H = self.num_heads
        nope = self.qk_nope_head_dim
        r = self.kv_lora_rank
        v_dim = self.v_head_dim
        q_dtype = q.dtype
        B = q.shape[0]
        ssb = kv_cache.seq_size_per_block

        plan = self._get_decode_plan(device, q_dtype, kv_cache)

        q_nope = q[:, :, :nope]
        q_pe = _apply_neox_rope(q[:, :, nope:], plan["cos"], plan["sin"])
        k_pe_cur = _apply_neox_rope(k_pe, plan["cos"], plan["sin"])

        # Vectorized cache write.
        base = kv_cache.kv_cache_base
        latent_cur = torch.cat([compressed_kv, k_pe_cur], dim=-1).to(base.dtype)
        base[plan["write_blk"], plan["write_off"], :] = latent_cur

        # Absorb: project q_nope into latent space.
        kc = self.weights[layer_id].get(W.mla_kc, None)
        vc = self.weights[layer_id].get(W.mla_vc, None)
        q_absorbed = torch.bmm(q_nope.to(kc.dtype).transpose(0, 1), kc).transpose(0, 1)

        # Concatenate absorbed q and roped q_pe for the fused kernel.
        q_fused = torch.cat([q_absorbed, q_pe.to(q_absorbed.dtype)], dim=-1)
        o = torch.empty(B, H, r, dtype=q_fused.dtype, device=device)

        # Reshape kv_cache_base to [num_blocks, page_size, 1, head_dim] for aiter.
        kv_head_dim = r + self.qk_rope_head_dim
        kv_buffer = base.view(-1, ssb, 1, kv_head_dim)

        aiter.mla.mla_decode_fwd(
            q_fused,
            kv_buffer,
            o,
            plan["qo_indptr"],
            plan["kv_indptr"],
            plan["kv_indices"],
            plan["kv_last_page_lens"],
            max_seqlen_q=1,
            page_size=ssb,
            nhead_kv=1,
            sm_scale=self.softmax_scale,
        )

        # Project from latent space to value space.
        attn_bmm_output = torch.bmm(o.transpose(0, 1), vc).transpose(0, 1)

        return attn_bmm_output.reshape(B, H * v_dim).to(q_dtype)
