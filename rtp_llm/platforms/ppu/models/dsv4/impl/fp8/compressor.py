"""PPU overrides; unchanged behavior is inherited from the shared DSV4 implementation."""

from __future__ import annotations
from rtp_llm.models_py.modules.dsv4.fp8.compressor import (
    _CUBLAS_GEMM_BF16_BF16_FP32,
    _SHARED_COS_SIN_CACHE,
    _CPStateReadSelection,
    _select_cp_state_read_tail_blocks,
    _select_cp_state_read_blocks,
    _fill_cp_state_read_cache,
    _CP_STATE_READ_SELECTION_CACHE,
    _build_cp_full_state_read_cache,
    _cp_sliced_state_read_needed,
    CompressorMeta,
    _cache_cp_state_read_selection,
    _CompressorPending,
    _CompressorNorm,
    _build_prefill_positions,
    build_prefill_metadata,
    build_prepare_metadata_args,
    build_decode_metadata,
)
import weakref
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple
import torch
import torch.nn as nn
from rtp_llm.models_py.distributed.collective_torch import Group, all_gather
from rtp_llm.models_py.modules.dsv4._profiler import record_function_range
from rtp_llm.ops.compute_ops import rtp_llm_ops
from rtp_llm.models_py.modules.dsv4.cp import (
    _CP_ROLE_INDEXER,
    _CP_ROLE_MAIN,
    CPContext,
    cp_all_gather_full_async,
    cp_should_gather,
    cp_wait_gather_full,
)
from rtp_llm.models_py.modules.dsv4.fp8._compressor_consts import (
    INDEXER_ENTRY_BYTES,
    INDEXER_HEAD_DIM,
    KV_ENTRY_BYTES,
    KV_HEAD_DIM,
)
from rtp_llm.models_py.modules.dsv4.fp8._compressor_vllm_triton import (
    build_cos_sin_cache,
    run_fused_compress_kv_write,
    run_save_partial_states,
)
from rtp_llm.models_py.modules.dsv4.fp8._kv_cache_utils import PoolBackedModule
from rtp_llm.models_py.modules.dsv4.fp8.compressor import CompressorFP8 as _BaseCompressorFP8

def _default_bf16_fp32_linear(activation, linear_weight):
    assert (
        _CUBLAS_GEMM_BF16_BF16_FP32 is not None
    ), "cublas_gemm_bf16_bf16_fp32 op is not built"
    return _CUBLAS_GEMM_BF16_BF16_FP32(activation, linear_weight)

def _linear_bf16_bf16_fp32(
    x: torch.Tensor, weight: torch.Tensor, *, linear_op=_default_bf16_fp32_linear
) -> torch.Tensor:
    """F.linear(x, weight) with BF16 operands and FP32 accumulation/output."""
    assert x.dtype == torch.bfloat16, f"expected BF16 input, got {x.dtype}"
    assert weight.dtype == torch.bfloat16, f"expected BF16 weight, got {weight.dtype}"
    assert x.is_contiguous(), "expected contiguous input"
    assert weight.is_contiguous(), "expected contiguous weight"
    leading_shape = x.shape[:-1]
    x_2d = x.reshape(-1, x.shape[-1])
    out_2d = linear_op(x_2d, weight)
    expected_shape = (int(x_2d.shape[0]), int(weight.shape[0]))
    if out_2d.dtype != torch.float32 or tuple(out_2d.shape) != expected_shape:
        raise RuntimeError(
            "BF16-to-FP32 linear returned an invalid output contract: "
            f"dtype={out_2d.dtype}, shape={tuple(out_2d.shape)}, "
            f"expected=torch.float32/{expected_shape}"
        )
    return out_2d.reshape(*leading_shape, weight.shape[0])

class CompressorFP8(_BaseCompressorFP8):
    def __init__(
        self,
        dim: int,
        head_dim: int,
        rope_head_dim: int,
        compress_ratio: int,
        max_batch_size: int,
        *,
        cp_role: str,
        norm_eps: float = 1e-6,
        rotate: bool = False,
        compressor_weights: Optional[Dict[str, torch.Tensor]] = None,
        platform_provider=None,
    ):
        """``compressor_weights`` is a 4-key dict ``{"ape", "wkv", "wgate",
        "norm"}`` extracted by the caller from ``layer_weights[W.v4_*compressor_*]``."""
        super(_BaseCompressorFP8, self).__init__()
        assert head_dim in (KV_HEAD_DIM, INDEXER_HEAD_DIM), (
            f"CompressorFP8 supports head_dim in {{{KV_HEAD_DIM}, "
            f"{INDEXER_HEAD_DIM}}}; got {head_dim}"
        )
        assert compressor_weights is not None, (
            "CompressorFP8 requires compressor_weights — meta-tensor / "
            "stand-alone construction is not supported (use the BF16 path)."
        )
        self._platform_provider = platform_provider
        from rtp_llm.platforms.ppu.models.dsv4.impl.platform_provider import build_dsv4_bf16_fp32_linear

        self._bf16_fp32_linear = build_dsv4_bf16_fp32_linear(
            _default_bf16_fp32_linear, platform_provider=platform_provider
        )
        self.dim = dim
        self.head_dim = head_dim
        # Which per-forward workspace CP buffer pair this compressor's gather
        # draws from — passed EXPLICITLY by the caller (main CSA/HCA vs nested
        # indexer) rather than inferred from ``head_dim``, so the semantics are
        # not hidden in a dimension constant. The two roles need distinct buffers
        # so both can be in flight within one layer under the overlap
        # orchestrator. Validated, never silently defaulted.
        assert cp_role in (_CP_ROLE_MAIN, _CP_ROLE_INDEXER), (
            f"CompressorFP8 cp_role must be one of "
            f"{{{_CP_ROLE_MAIN!r}, {_CP_ROLE_INDEXER!r}}}; got {cp_role!r}"
        )
        self._cp_role = cp_role
        self.rope_head_dim = rope_head_dim
        self.compress_ratio = compress_ratio
        self.overlap = compress_ratio == 4
        self.rotate = rotate
        self.norm_eps = norm_eps
        coff = 1 + self.overlap
        self.coff = coff
        # Per-slot bytes of the FP8 KV pool entry (selects layout/kernel).
        self._pool_entry_bytes = (
            KV_ENTRY_BYTES if head_dim == KV_HEAD_DIM else INDEXER_ENTRY_BYTES
        )

        # ape stays FP32. wkv/wgate are stored as BF16, while the fused
        # projection below uses BF16 operands with FP32 accumulation/output.
        # Register ape as a non-trainable Parameter so .to(device) follows the
        # module.
        self.ape = nn.Parameter(
            compressor_weights["ape"].float().contiguous(), requires_grad=False
        )
        self.wkv = nn.Linear(dim, coff * head_dim, bias=False)
        self.wgate = nn.Linear(dim, coff * head_dim, bias=False)
        with torch.no_grad():
            self.wkv.weight = nn.Parameter(
                compressor_weights["wkv"].to(torch.bfloat16), requires_grad=False
            )
            self.wgate.weight = nn.Parameter(
                compressor_weights["wgate"].to(torch.bfloat16), requires_grad=False
            )
        self.norm = _CompressorNorm(head_dim)
        self.norm.weight = nn.Parameter(
            compressor_weights["norm"].to(torch.bfloat16),
            requires_grad=False,
        )

        # Fuse wkv + wgate into one bf16 weight matrix; saves one
        # GEMM launch per compressor decode call (~92/step at bs16).
        self._wkv_wgate_fused: Optional[torch.Tensor] = None
        self._fuse_wkv_wgate(coff)

        # Legacy attribute kept for attention.py's cmp_T fallback (line 1583).
        self._kv_cache_t: int = 0

        # Built from freqs_cis during model loading. Runtime only reads it.
        self.freqs_cis: Optional[torch.Tensor] = None
        self._cos_sin_cache: Optional[torch.Tensor] = None
        self._state_tokens_per_block: int = 0
        self._cp_ctx: Optional[CPContext] = None
        self._cp_gather_stream: Optional[Any] = None
        self._kv_cache_sharded: bool = False
        self._profile_label: Optional[str] = None
        # MOEDBG: caller (Attention / IndexerFP8) sets this to a name
        # prefix like ``"L02_attn_cmp"`` before forward and clears after;
        # _forward_prefill_body uses it as the rec name root. None / empty
        # string suppresses recording.
        self._dbg_prefix: Optional[str] = None

    def _launch(
        self,
        kv_flat: torch.Tensor,  # [N, coff*head_dim] fp32
        score_flat: torch.Tensor,  # [N, coff*head_dim] fp32
        meta: CompressorMeta,
        seq_start: Optional[int] = None,
        is_decode: bool = False,
    ) -> None:
        """Launch the two vLLM kernels (state write + boundary KV write).

        ``seq_start`` is the absolute position of ``kv_flat[0]`` for
        sequentially-laid-out batches (prefill: ``sp_int``). When provided
        the fused kernel reads any overlap-window position with
        ``flat_idx = pos - seq_start in [0, N)`` directly from
        ``kv_flat / score_flat`` instead of reading back through the state
        pool, where current-launch writes are still in flight.

        Pass ``None`` to disable the raw path (decode: ``kv_flat`` is
        indexed by ``req_idx``, not by absolute position offset).

        All slot-mapping math is consumed from ``meta`` — this method only
        does kernel launches. Designed to stay branch-light so it composes
        cleanly with CUDA graph capture.
        """
        if (
            self._state_pool_3d is None
            or self._kv_pool_view is None
            or self._state_block_table is None
            or self._kv_block_table is None
        ):
            # Warmup / unbound: nothing to write.
            return

        N = int(meta.positions.shape[0])
        if N == 0:
            return

        cos_sin_cache = self._cos_sin_cache
        assert cos_sin_cache is not None

        with record_function_range("dsv4.fp8.compressor.launch.save_partial_states"):
            run_save_partial_states(
                kv_flat,
                score_flat,
                self.ape,
                meta.positions,
                self._state_pool_3d,
                meta.state_slots,
                compress_ratio=self.compress_ratio,
            )

        # Decode path passes seq_start=None: disable the raw branch so the
        # kernel only reads state_cache. seq_start value is then irrelevant.
        # Phase-3a part 4c: when meta carries per-request raw windows, the
        # kernel uses native varlen raw path (one address per request) and
        # the scalar ``seq_start`` is unused.
        use_varlen_raw = (
            meta.is_batched
            and meta.seq_start_per_req is not None
            and meta.cu_seq_per_req is not None
        )
        raw_disabled = (seq_start is None) and not use_varlen_raw
        state_cache_for_read = self._state_pool_3d
        state_block_table_for_read = self._state_block_table
        if _cp_sliced_state_read_needed(meta, self._cp_ctx, raw_disabled):
            with record_function_range(
                "dsv4.fp8.compressor.launch.cp_gather_state_read_cache"
            ):
                state_cache_for_read, state_block_table_for_read = (
                    _build_cp_full_state_read_cache(
                        self._state_pool_3d,
                        self._state_block_table,
                        int(self._cp_ctx.cp_size),
                        seq_start_per_req=(
                            meta.seq_start_per_req if use_varlen_raw else None
                        ),
                        token_count=(1 + int(self.overlap)) * self.compress_ratio,
                        state_tokens_per_block=self._state_tokens_per_block,
                        # All layers of the same attention-type share this
                        # meta object per forward, so layer 1 builds the
                        # selection and layers 2..N reuse it.
                        selection_cache_key=meta,
                    )
                )

        with record_function_range("dsv4.fp8.compressor.launch.compress_kv_write"):
            provider_launch = getattr(
                getattr(self, "_platform_provider", None), "run_compress_kv_write", None
            )
            decode_options = {"is_decode": is_decode} if provider_launch is not None else {}
            (provider_launch or run_fused_compress_kv_write)(
                state_cache_for_read,
                meta.token_to_req,
                meta.positions,
                meta.state_slots,
                state_block_table_for_read,
                self.norm.weight,
                self.norm_eps,
                cos_sin_cache,
                self._kv_pool_view,
                meta.kv_slots,
                kv_flat,
                score_flat,
                self.ape,
                0 if (raw_disabled or use_varlen_raw) else seq_start,
                disable_raw_path=raw_disabled,
                head_dim=self.head_dim,
                rope_head_dim=self.rope_head_dim,
                compress_ratio=self.compress_ratio,
                overlap=self.overlap,
                seq_start_per_req=meta.seq_start_per_req if use_varlen_raw else None,
                cu_seq_per_req=meta.cu_seq_per_req if use_varlen_raw else None,
                state_tokens_per_block=self._state_tokens_per_block,
                **decode_options,
            )

    def start_prefill(
        self,
        x: torch.Tensor,
        start_pos,
        *,
        meta: Optional[CompressorMeta] = None,
        cp_gather_stream: Optional[Any] = None,
        profile_label: Optional[str] = None,
        # REQUIRED, never Optional — by design. ``workspace`` is the single
        # authoritative per-forward prefill scratch (allocated once in
        # ``prefill/forward.py::forward_layers`` and threaded through the meta
        # broadcast). It is dereferenced only on the CP gather branch and is
        # unused on the non-CP path, but it is ALWAYS passed so there is exactly
        # one buffer-lifecycle contract and no hidden ``None``/fallback-allocate
        # branch a future edit could trip over. Do NOT relax this to Optional.
        workspace: "PrefillWorkspace",
    ) -> Optional[_CompressorPending]:
        """Begin a prefill compressor launch without waiting on the CP gather.

        Steps performed eagerly on the **default** stream:
          * shape / sp resolve (mirrors ``forward``);
          * warmup early return → ``None`` (no pool bound by framework);
          * fused KV/gate projection (``_linear_bf16_bf16_fp32``).

        Steps deferred to :meth:`finish_prefill`:
          * waiting the CP all-gather (queued here on
            ``cp_gather_stream`` when CP is active);
          * splitting fused → kv_flat + score_flat;
          * the ``_launch`` writer kernel.
        """
        if x.dim() == 2:
            bsz = 1
            seqlen = int(x.size(0))
        else:
            bsz, seqlen, _ = x.size()
        sp = (
            int(start_pos.item())
            if isinstance(start_pos, torch.Tensor)
            else int(start_pos)
        )
        if (
            self._state_pool_3d is None
            or self._kv_pool_view is None
            or self._kv_eb <= 0
        ):
            return None

        out_dim = (1 + self.overlap) * self.head_dim
        with record_function_range("dsv4.fp8.compressor.prefill.fused_linear"):
            fused_out = _linear_bf16_bf16_fp32(
                x, self._wkv_wgate_fused, linear_op=self._bf16_fp32_linear
            )
            N = bsz * seqlen
            fused_flat = fused_out.reshape(N, -1)

        cp_ctx = self._cp_ctx
        cp_gather = cp_should_gather(cp_ctx, start_pos)
        fused_gather_handle = None
        if cp_gather:
            assert cp_ctx is not None
            assert meta is not None, (
                "CompressorFP8 CP start_prefill requires hoisted "
                "CompressorMeta; non-CP path may pass meta=None"
            )
            N_full = int(cp_ctx.seq_len_full)
            assert int(meta.positions.numel()) == N_full, (
                f"CP compressor meta/token length mismatch: "
                f"meta={meta.positions.numel()} seq_len_full={N_full}"
            )
            # The non-prefix restore destination is the per-forward workspace's
            # role buffer (``cp_restore_{main,idx}``), selected in the gather
            # impl by ``cp_role`` — no fresh per-layer alloc here (that churn is
            # exactly what the workspace buffers eliminate under overlap).
            gather_stream = cp_gather_stream
            if gather_stream is None and fused_flat.is_cuda:
                gather_stream = torch.cuda.Stream(device=fused_flat.device)
            profile_name = self._cp_profile_name(profile_label)
            gather_range = "dsv4.fp8.compressor.prefill.cp_gather_kv_score"
            if profile_label:
                gather_range = (
                    f"dsv4.fp8.compressor.prefill.{profile_label}.cp_gather_kv_score"
                )
            with record_function_range(gather_range):
                fused_gather_handle = cp_all_gather_full_async(
                    fused_flat,
                    cp_ctx,
                    stream=gather_stream,
                    profile_name=profile_name,
                    workspace=workspace,
                    cp_role=self._cp_role,
                )

        return _CompressorPending(
            fused_flat=fused_flat,
            fused_gather_handle=fused_gather_handle,
            sp=sp,
            bsz=bsz,
            seqlen=seqlen,
            meta=meta,
            out_dim=out_dim,
            profile_label=profile_label,
            restored_buf=None,
        )

    def forward(
        self,
        x: torch.Tensor,
        start_pos,
        sequence_lengths: Optional[torch.Tensor] = None,
        meta: Optional[CompressorMeta] = None,
        *,
        # REQUIRED, never Optional — see ``start_prefill``: one authoritative
        # per-forward buffer, always passed (unused on non-CP), no fallback.
        workspace: "PrefillWorkspace",
    ) -> Optional[torch.Tensor]:
        """Prefill entry. ``bsz==1`` (FIFO scheduler).

        Returns ``None`` — downstream readers gather compressed K from the
        FP8 KV pool directly.

        ``meta`` lets the caller (typically ``attention.py``) hoist the
        slot-mapping compute out of the per-layer hot path and amortize it
        across the host compressor and any nested indexer compressor that
        share the same positions/b_idx. When ``None`` the compressor falls
        back to the in-body compute path (warmup / standalone / UT).
        """
        del sequence_lengths  # not needed: positions derived from start_pos+arange
        # Phase-3a: accept either flat ``[T_total, dim]`` (vLLM-native /
        # batched prefill) or legacy ``[B, S, dim]``. The compressor kernels
        # consume token-major 2D tensors, so the prefill path flattens once
        # immediately after the fused projection and never reintroduces B.
        if x.dim() == 2:
            bsz = 1
            seqlen = int(x.size(0))
        else:
            bsz, seqlen, _ = x.size()
        sp = (
            int(start_pos.item())
            if isinstance(start_pos, torch.Tensor)
            else int(start_pos)
        )
        # Warmup forward (no pool bound by framework): no-op.
        if (
            self._state_pool_3d is None
            or self._kv_pool_view is None
            or self._kv_eb <= 0
        ):
            return None

        device = x.device
        out_dim = (1 + self.overlap) * self.head_dim
        with record_function_range("dsv4.fp8.compressor.prefill.fused_linear"):
            fused_out = _linear_bf16_bf16_fp32(
                x, self._wkv_wgate_fused, linear_op=self._bf16_fp32_linear
            )
            N = bsz * seqlen
            fused_flat = fused_out.reshape(N, -1)

        cp_ctx = self._cp_ctx
        cp_gather = cp_should_gather(cp_ctx, start_pos)
        fused_gather_handle = None
        if cp_gather:
            assert cp_ctx is not None
            gather_stream = (
                torch.cuda.Stream(device=fused_flat.device)
                if fused_flat.is_cuda
                else None
            )
            # Gather the fused ``[kv | score]`` projection once. This removes
            # one NCCL launch and keeps the CP path token-major 2D; the split
            # below is a strided view and the Triton wrappers pass row strides
            # explicitly.
            with record_function_range(
                "dsv4.fp8.compressor.prefill.cp_gather_kv_score"
            ):
                fused_gather_handle = cp_all_gather_full_async(
                    fused_flat,
                    cp_ctx,
                    stream=gather_stream,
                    profile_name=self._cp_profile_name(None),
                    workspace=workspace,
                    cp_role=self._cp_role,
                )
            assert meta is not None, (
                "CompressorFP8 CP prefill requires full-sequence metadata from "
                "rtp_llm.models_py.modules.dsv4.fp8.prefill_meta; rebuilding it "
                "inside compressor.forward is intentionally disabled."
            )
            N = int(cp_ctx.seq_len_full)
            assert int(meta.positions.numel()) == N, (
                f"CP compressor meta/token length mismatch: meta={meta.positions.numel()} "
                f"seq_len_full={N}"
            )
            assert fused_gather_handle is not None
            with record_function_range("dsv4.fp8.compressor.prefill.cp_wait_kv_score"):
                fused_flat = cp_wait_gather_full(fused_gather_handle)

        with record_function_range("dsv4.fp8.compressor.prefill.split_kv_score"):
            assert fused_flat.dim() == 2, (
                f"CompressorFP8 prefill expects flat fused projection, got "
                f"{tuple(fused_flat.shape)}"
            )
            assert fused_flat.size(1) == 2 * out_dim, (
                f"CompressorFP8 fused hidden mismatch: got {fused_flat.size(1)}, "
                f"expected {2 * out_dim}"
            )
            kv_flat = fused_flat[:, :out_dim]
            score_flat = fused_flat[:, out_dim:]
        if meta is None:
            with record_function_range("dsv4.fp8.compressor.prefill.build_meta"):
                positions, b_idx = _build_prefill_positions(sp, bsz, seqlen, device)
                meta = self.prepare_metadata(
                    positions,
                    b_idx,
                    has_prefix=sp > 0,
                    seq_start_per_req=torch.tensor(
                        [sp], dtype=torch.long, device=device
                    ),
                    cu_seq_per_req=torch.tensor(
                        [0, seqlen], dtype=torch.long, device=device
                    ),
                )
        # Varlen prefill carries per-request raw arrays, so ``_launch`` routes
        # through the kernel's BATCHED branch even when CP has only one
        # request. Legacy scalar metadata keeps the old ``seq_start`` path.
        seq_start = None if meta.is_batched else sp
        with record_function_range("dsv4.fp8.compressor.prefill.launch"):
            self._launch(kv_flat, score_flat, meta, seq_start=seq_start)
        return None

    def forward_decode_vectorized(
        self,
        x: torch.Tensor,
        start_pos: torch.Tensor,
        meta: Optional[CompressorMeta] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        """Batched decode entry. ``position_ids`` enables q_len > 1 verify."""
        bsz, q_len = int(x.size(0)), int(x.size(1))
        T = bsz * q_len
        if position_ids is None:
            assert q_len == 1, "decode q_len > 1 requires flat position_ids"
        if (
            self._state_pool_3d is None
            or self._kv_pool_view is None
            or self._kv_eb <= 0
        ):
            return None

        device = x.device
        out_dim = (1 + self.overlap) * self.head_dim
        fused_out = _linear_bf16_bf16_fp32(
            x, self._wkv_wgate_fused, linear_op=self._bf16_fp32_linear
        )
        kv, score = fused_out[..., :out_dim], fused_out[..., out_dim:]

        # ``kv`` and ``score`` are last-dim slices of ``fused_out``. Keep
        # them as strided row views instead of materialising two BF16 copy
        # kernels; the Triton writer consumes row stride explicitly.
        kv_flat = kv.view(T, -1)
        score_flat = score.view(T, -1)
        if meta is None:
            if position_ids is None:
                positions = start_pos.to(device=device, dtype=torch.long).reshape(bsz)
                b_idx = torch.arange(bsz, device=device, dtype=torch.long)
                cu_seq_per_req = torch.arange(
                    0, bsz + 1, device=device, dtype=torch.int64
                )
                meta = self.prepare_metadata(
                    positions,
                    b_idx,
                    has_prefix=True,
                    seq_start_per_req=positions,
                    cu_seq_per_req=cu_seq_per_req,
                )
            else:
                positions = (
                    position_ids.to(device=device, dtype=torch.long)
                    .reshape(T)
                    .contiguous()
                )
                b_idx = torch.arange(bsz, device=device, dtype=torch.long)
                b_idx = b_idx.repeat_interleave(q_len).contiguous()
                position_ids_2d = positions.view(bsz, q_len)
                cu_seq_per_req = torch.arange(
                    0,
                    (bsz + 1) * q_len,
                    q_len,
                    device=device,
                    dtype=torch.long,
                )
                meta = self.prepare_metadata(
                    positions,
                    b_idx,
                    has_prefix=True,
                    is_batched=q_len > 1,
                    seq_start_per_req=position_ids_2d[:, 0].to(torch.long).contiguous(),
                    cu_seq_per_req=cu_seq_per_req,
                )
        self._launch(kv_flat, score_flat, meta, is_decode=True)
        return None
