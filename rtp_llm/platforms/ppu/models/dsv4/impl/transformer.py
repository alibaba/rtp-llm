"""PPU overrides; unchanged behavior is inherited from the shared DSV4 implementation."""

from __future__ import annotations
from rtp_llm.models_py.modules.dsv4.transformer import (
    _TRUE_ENV_VALUES,
    _FALSE_ENV_VALUES,
    _optional_env_bool,
    V4Args,
    _block_kwargs,
)
import logging
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence
import torch
import torch.nn as nn
import torch.nn.functional as F
from rtp_llm.models_py.modules import RMSNorm
from rtp_llm.models_py.modules.base.common.embedding import EmbeddingTorch
from rtp_llm.models_py.modules.dsv4 import _profiler
from rtp_llm.models_py.modules.dsv4 import _record_tensor as _rt
from rtp_llm.platforms.ppu.models.dsv4.impl.block import Block
from rtp_llm.models_py.modules.dsv4.cp import CPContext, build_cp_context
from rtp_llm.platforms.ppu.models.dsv4.impl.hc.factory import build_hc_head
from rtp_llm.platforms.ppu.models.dsv4.impl.platform_provider import (
    Dsv4PlatformProvider,
    Dsv4ProviderCapability,
    resolve_dsv4_platform_provider,
)
from rtp_llm.platforms.ppu.models.dsv4.impl.tp_norm import tp_gather_hidden, tp_rms_norm
from rtp_llm.models_py.modules.factory.fused_moe.utils.fp8_fp4.chunked_layer import (
    synchronized_moe_chunk_plan,
)
from rtp_llm.models_py.modules.dsv4.transformer import V4Transformer as _BaseV4Transformer

def _build_block(
    layer_id: int,
    args: V4Args,
    layer_weights: Optional[Dict[str, torch.Tensor]] = None,
    platform_provider: Optional[Dsv4PlatformProvider] = None,
    commit_only: bool = False,
) -> Block:
    if platform_provider is None:
        provider = resolve_dsv4_platform_provider({Dsv4ProviderCapability.BLOCK})
    else:
        provider = platform_provider
    return provider.build_block(
        Block,
        **_block_kwargs(layer_id, args, layer_weights, commit_only=commit_only),
        platform_provider=provider,
    )

class V4Transformer(_BaseV4Transformer):
    def __init__(
        self,
        args: V4Args,
        mw,
        platform_provider: Optional[Dsv4PlatformProvider] = None,
    ):
        """``mw`` is the framework's ``ModelWeights`` (with ``.global_weights``
        ``Dict[str, Tensor]`` keyed by ``W.*`` enum and ``.weights[layer_id]``
        per-layer dicts).  Required — every dsv4 sub-module reads its
        weights from ``mw`` at construction; there is no unit-test path
        that constructs the transformer with empty weights."""
        super(_BaseV4Transformer, self).__init__()
        if platform_provider is None:
            self.platform_provider = resolve_dsv4_platform_provider(
                {Dsv4ProviderCapability.BLOCK}
            )
        else:
            self.platform_provider = platform_provider
        self.args = args
        self.max_seq_len = args.max_seq_len
        self.hc_mult = args.hc_mult
        self.commit_only = bool(getattr(args, "commit_only", False))
        # Surface ``fp8_kv_cache`` as a top-level attr so
        # ``prefill/forward.py`` and ``DeepSeekV4Model.prepare_fmha_impl``
        # can dispatch via ``v4.fp8_kv_cache`` without reading args.
        self.fp8_kv_cache = args.fp8_kv_cache

        from rtp_llm.utils.model_weight import W

        gw = mw.global_weights
        self._mega_csa_runtime = None
        self._mega_decode_enabled = False
        embedding_weight = None
        if not self.commit_only:
            embedding_weight = gw[W.embedding]
            # Process-level deployment override, intentionally outside V4Args.
            # Set it consistently in each worker's environment.
            mega_request = _optional_env_bool("DSV4_MEGA")
            # SWA-only MTP/DSpark drafts share the process switch, but have no
            # Mega sublayers to initialize or require from the runtime.
            if mega_request is not False and any(args.compress_ratios[: args.n_layers]):
                from rtp_llm.models_py.modules.dsv4.fp8.decode.mega_support import (
                    mega_decode_unavailable_reason,
                )

                unavailable_reason = mega_decode_unavailable_reason(
                    args,
                    embedding_weight.device,
                )
                if unavailable_reason is not None and mega_request is True:
                    raise RuntimeError(
                        f"DSV4_MEGA=1 is unsupported: {unavailable_reason}"
                    )
                if unavailable_reason is not None:
                    logging.info(
                        "DSV4 Mega decode is unavailable; using the ordinary path: %s",
                        unavailable_reason,
                    )
                else:
                    self._mega_decode_enabled = True
                    logging.info(
                        "DSV4 Mega attention decode enabled on %s (TP1, dim=%d); "
                        "set DSV4_MEGA=0 to disable",
                        embedding_weight.device,
                        args.dim,
                    )

        self.layers = nn.ModuleList(
            [
                _build_block(
                    i,
                    args,
                    layer_weights=mw.weights[i],
                    platform_provider=self.platform_provider,
                    commit_only=self.commit_only,
                )
                for i in range(args.n_layers)
            ]
        )

        # MTP draft is a separate model (``DeepSeekV4MtpModel``) that
        # holds its own V4Transformer — no MTP layers live on the main
        # model's transformer.  ``args.n_mtp_layers`` exists only to
        # surface that contract via an assert.
        assert args.n_mtp_layers == 0, (
            "V4Transformer does not host MTP layers; the draft is a "
            "separate DeepSeekV4MtpModel instance."
        )

        if self.commit_only:
            # The commit path neither embeds tokens nor runs the final norm,
            # LM head, or mHC head. None-valued members make accidental use of
            # the ordinary transformer path fail at its call site.
            self.embed = None
            self.norm = None
            self.head_weight = None
            self.head_hc = None
        else:
            # ``EmbeddingTorch`` keeps ``self.weight`` as a plain attribute (no
            # ``nn.Parameter``); the framework dict supplies the real tensor.
            assert embedding_weight is not None
            self.embed = EmbeddingTorch(embedding_weight)
            self.norm = RMSNorm(gw[W.final_ln_gamma], args.norm_eps)
            if self._mega_decode_enabled:
                from rtp_llm.models_py.modules.dsv4.fp8.decode.mega_csa_runtime import (
                    MegaCSARuntime,
                )

                self._mega_csa_runtime = MegaCSARuntime()
                for layer_id, layer in enumerate(self.layers):
                    layer.enable_mega_csa(self._mega_csa_runtime, mw.weights[layer_id])
                    layer.enable_mega_hca(self._mega_csa_runtime, mw.weights[layer_id])

            # LM head — plain weight matrix [vocab_size, dim].  Accept either
            # BF16 (ckpt-native) or FP32 (legacy path).
            self.head_weight = gw[W.lm_head]
            if self.head_weight.dtype not in (torch.float32, torch.bfloat16):
                raise TypeError(
                    f"DSV4 lm_head must be FP32 or BF16, got {self.head_weight.dtype}"
                )
            self.head_hc = build_hc_head(
                gw[W.v4_hc_head_fn],
                gw[W.v4_hc_head_base],
                gw[W.v4_hc_head_scale],
                dim=args.dim,
                hc_mult=args.hc_mult,
                norm_eps=args.norm_eps,
                hc_eps=args.hc_eps,
                tp_size=args.tp_size,
                tp_rank=args.tp_rank,
                platform_provider=self.platform_provider,
            )

        self._dbg_step = 0
        self.register_buffer("_mtp_hidden_buffer", None, persistent=False)
        self.register_buffer("_mtp_last_hidden_buffer", None, persistent=False)
        self._mtp_hidden_valid_tokens = 0
        self._mtp_last_hidden_valid_tokens = 0
        # Max-sized per-forward prefill workspace dims (plain ints; not buffers).
        # Filled by ``_bind_prefill_workspace_dims`` at init; read by
        # ``forward_layers`` to build a per-forward ``PrefillWorkspace``.
        self._prefill_ws_q_rows = 0
        self._prefill_ws_q_dim = 0
        self._prefill_ws_full_rows = 0
        # Separate CP gather widths for the concurrent compressor gathers — each
        # backs a dedicated workspace buffer pair:
        #   main / indexer — back-to-back on the compressor side stream.
        self._prefill_ws_main_w = 0
        self._prefill_ws_idx_w = 0

        # DSpARK target-feature export. The target model config fills this
        # with 0-based layer ids (V4-Flash uses 40/41/42); the empty tuple is
        # the no-auxiliary-tensor default for ordinary inference and MTP.
        self.capture_aux_hidden_layer_ids: tuple[int, ...] = ()

    def _embed(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed(input_ids)

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x)

    @torch.inference_mode()
    def forward_decode(
        self,
        input_ids: torch.Tensor,  # [T_total] int (== [B] for q_len=1)
        attn_metadata: "DSv4DecodeAttnMetadata",  # type: ignore[name-defined]
        kv_cache=None,
    ) -> torch.Tensor:
        """Decode-only forward.

        Returns ``hidden [T_total, dim]`` (caller applies lm_head).
        Prefill ``forward`` is untouched — PD-disagg later splits cleanly.
        """
        B = attn_metadata.batch_size
        q_len = attn_metadata.q_len_per_req
        if input_ids.dim() == 1:
            input_ids_2d = input_ids.view(B, q_len)
        else:
            input_ids_2d = input_ids
        h = self._embed(input_ids_2d)  # [B, q_len, dim]
        h = h.unsqueeze(2).repeat(1, 1, self.hc_mult, 1)  # [B, q_len, hc, dim]
        begin_decode = getattr(self, "begin_decode", None)
        if begin_decode is not None:
            begin_decode(attn_metadata)
        layer_forward_range = _profiler.make_layer_forward_range()
        for layer_idx, layer in enumerate(self.layers):
            with layer_forward_range(layer_idx):
                h = layer.forward_decode(
                    h, attn_metadata, input_ids_2d, kv_cache=kv_cache
                )
        h = self._hc_head_reduce(h)  # [B, q_len, dim]
        # Framework RMSNorm wants 2D — flatten to [T_total, dim] and
        # return that directly (the next reshape would no-op anyway).
        return self._norm(h.reshape(B * q_len, h.shape[-1]))

    @torch.inference_mode()
    def forward(
        self,
        input_ids: torch.Tensor,
        start_pos=0,
        apply_lm_head: bool = True,
        kv_cache=None,
        block_tables_by_type=None,
        sequence_lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Standalone forward.

        Returns:
          if apply_lm_head: logits of LAST token [B, vocab_size] (official behavior)
          else:            pre-lm-head hidden state of ALL tokens [B, S, d] — for
                           framework wrapper which applies lm_head externally

        Empty-batch contract: when ``input_ids`` has zero tokens (S=0),
        every downstream module — ``apply_rotary_emb`` in particular,
        which does ``view_as_complex`` on an S=0 unflatten — would
        crash.  Under DP/EP the framework routinely issues forwards to
        ranks that have no local work to do; short-circuit with a
        correctly-shaped empty output so those ranks still participate
        in any collective that other ranks trigger (DeepEP dispatch,
        all_reduce, etc.).
        """
        B = input_ids.size(0)
        S = input_ids.size(1) if input_ids.dim() > 1 else input_ids.size(0)
        # S=0 short-circuit only applies for TP=EP=1.  With ep_size>1,
        # DeepEP dispatch/combine are collectives that ALL ranks must
        # enter together, even the ones with zero local tokens — else
        # the missing rank's absence deadlocks the buffer with "CPU
        # recv timeout".  The caller is responsible for clamping /
        # padding ``start_pos`` so downstream indexing stays in range
        # (see ``DeepSeekV4Model.forward``); individual layers are
        # empty-batch-safe (``apply_rotary_emb`` and the sparse-attn
        # path both short-circuit on zero-element inputs).
        if S == 0 and self.args.ep_size <= 1:
            # No nn.Parameter anymore — pull device from a known-bound tensor.
            device = self.embed.weight.device
            if apply_lm_head:
                return torch.zeros(
                    (B, self.head_weight.size(0)), dtype=torch.float32, device=device
                )
            return torch.zeros(
                (B, 0, self.args.dim), dtype=torch.bfloat16, device=device
            )

        # Build + propagate CP context once per forward.  When CP is
        # active (set_cp_info previously called with a real cp_info +
        # cp_size>1 + this is a prefill step), every layer's attn /
        # compressor / indexer gets a fresh CPContext with the derived
        # global_positions + unpad_restore tensors.  Otherwise we clear
        # any stale context with None so the modules fall through to
        # the single-rank path unchanged.
        _cp_info = getattr(self, "_cp_info", None)
        _cp_size = getattr(self, "_cp_size", 1)
        _cp_rank = getattr(self, "_cp_rank", 0)
        cp_ctx: Optional[CPContext] = None
        if _cp_info is not None and _cp_size > 1 and start_pos == 0 and S > 0:
            device = input_ids.device
            cp_ctx = build_cp_context(
                _cp_info,
                _cp_size,
                _cp_rank,
                S,
                device,
                kv_cache_sharded=bool(getattr(self, "_kv_cache_sharded", False)),
            )
        self._propagate_cp_ctx(cp_ctx)

        # Master switch: when MOEDBG=0 every record/begin/dump call site is
        # skipped at the call site (no f-string formatting, no function-call
        # overhead, no per-layer name allocation).  Read once per forward.
        _rt_on = _rt.ENABLED
        if _rt_on:
            _rt.begin(seqlen=int(input_ids.size(-1)))
            # begin() may suppress this forward (MOEDBG_MAX_SEQ); honour it.
            if _rt._get_buf() is None:
                _rt_on = False
        h = self._embed(input_ids)  # [B, S, d]
        if _rt_on:
            _rt.record("embed_out", h)
        h = h.unsqueeze(2).repeat(1, 1, self.hc_mult, 1)  # [B, S, hc, d]
        if _rt_on:
            _rt.record("embed_hc_expanded", h)

        # Bridge to flat Block.forward(x:[T,hc,d], input_ids:[T], positions:[T],
        # cu_seqlens:[B+1]). Standalone tests run with B==1; we flatten before
        # the loop and restore at the end so the hc_head_reduce / norm / rt
        # record paths below see the original [B, S, hc, d] shape.
        assert B == 1, (
            f"V4Transformer.forward standalone path expects B==1; got B={B}. "
            "Production inference goes through prefill/forward.py::forward_layers, "
            "which already handles multi-request via cu_seqlens."
        )
        h_flat = h.squeeze(0)  # [S, hc, d]
        input_ids_flat = input_ids.squeeze(0)  # [S]
        _start_pos_int = (
            int(start_pos)
            if isinstance(start_pos, int)
            else int(start_pos.view(-1)[0].item())
        )
        positions = _start_pos_int + torch.arange(
            S, device=input_ids.device, dtype=torch.int64
        )  # [S]
        cu_seqlens = torch.tensor(
            [0, S], dtype=torch.int64, device=input_ids.device
        )  # [2]
        layer_forward_range = _profiler.make_layer_forward_range()
        with synchronized_moe_chunk_plan(self.layers, S, input_ids.device):
            for li, layer in enumerate(self.layers):
                with layer_forward_range(li):
                    h_flat = layer(
                        h_flat,
                        input_ids_flat,
                        positions,
                        cu_seqlens,
                        kv_cache=kv_cache,
                        block_tables_by_type=block_tables_by_type,
                    )
                    if _rt_on:
                        _rt.record(f"layer{li:02d}_out", h_flat)
        h = h_flat.unsqueeze(0)  # [1, S, hc, d]
        h = self._hc_head_reduce(h)  # [B, S, d]
        if _rt_on:
            _rt.record("hc_reduced", h)
        # Framework RMSNorm wants 2D; collapse [B, S, d] → [B*S, d] and view back.
        bsz, seq, dim_ = h.shape
        h = self._norm(h.reshape(bsz * seq, dim_)).view(bsz, seq, -1)
        if _rt_on:
            _rt.record("final_norm", h)

        if _rt_on:
            extra: dict = {
                "apply_lm_head": bool(apply_lm_head),
                "input_ids_shape": tuple(input_ids.shape),
                "input_ids": input_ids.detach().cpu(),
                "start_pos": int(start_pos) if isinstance(start_pos, int) else -1,
            }
            if cp_ctx is not None:
                extra.update(
                    {
                        "cp_size": cp_ctx.cp_size,
                        "cp_rank": cp_ctx.cp_rank,
                        "chunk_length": cp_ctx.chunk_length,
                        "padded_seq_len": cp_ctx.padded_seq_len,
                        "seq_len_full": cp_ctx.seq_len_full,
                        "global_positions": cp_ctx.global_positions.detach().cpu(),
                        "unpad_restore": cp_ctx.unpad_restore.detach().cpu(),
                        "local_is_real": cp_ctx.local_is_real.detach().cpu(),
                    }
                )
            else:
                extra.update(
                    {
                        "cp_size": 1,
                        "cp_rank": 0,
                        "seq_len_full": int(S),
                    }
                )
            _rt.dump(step=self._dbg_step, extra=extra)
            self._dbg_step += 1

        if apply_lm_head:
            # Match h dtype to head_weight so F.linear works regardless of
            # load-time dtype (bf16 when enable_fp32_lm_head=False, f32
            # otherwise).  Final cast to f32 keeps logits dtype stable for
            # callers that feed into f32 sampling / softmax.
            return F.linear(
                h[:, -1].to(self.head_weight.dtype), self.head_weight
            ).float()
        return h
