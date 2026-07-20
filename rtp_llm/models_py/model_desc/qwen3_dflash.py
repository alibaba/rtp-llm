"""Qwen3 DFlash block-diffusion draft model (models_py) — base for DSpark.

Executable counterpart of the dense reference oracle
(rtp_llm/models_py/model_desc/test/dspark_reference.py) — stage names match:

    A. combine_hidden_states: fused = fc(concat(aux_hidden))          [T, H]
    B. inject_context_kv:     hidden_norm -> per-layer K/V -> k_norm
                              -> RoPE(K) -> paged-cache write          (no attention)
    C. block_forward:         [anchor + k masks] non-causal backbone  [B*(1+k), H]
    D. markov_correct:        DFlash = greedy argmax (no Markov head)  [B, k]

DSpark extends this base with a low-rank Markov transition-bias head that
overrides stage D (see qwen3_dspark.py), mirroring upstream vLLM's layering
(qwen3_dflash.py backbone + qwen3_dspark.py Markov head).  Stages A/B/C and the
propose/forward skeleton are shared; only markov_correct differs, so it is the
one virtual seam the subclass overrides.

The query block is the speculators "bonus anchor" layout: 1+k wide, anchor
(last verified token) at position committed_len, mask_j at committed_len+j;
predictions are read at the k mask positions.

Stage B is a write-only pass over the paged KV cache (design:
docs/dspark-phase1-design-2026-07-14.md, "特征注入 = 独立投影 pass"): it reuses
MhaRotaryEmbeddingOp + flashinfer append_paged_kv_cache with self-built page
indices, eager-only, never captured by CUDA graph.  Stage C is a regular
chunked-prefill decoder forward whose non-causal visibility comes entirely
from attn_config.is_causal=False (the model registration sets it).
"""

import math
from dataclasses import dataclass
from typing import Any, List, Optional

import torch
import torch.nn.functional as F
from torch import nn

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.model_loader.model_weight_info import ModelWeights
from rtp_llm.models_py.model_desc.block_map import select_block_map_for_layer
from rtp_llm.models_py.model_desc.module_base import GptModelBase
from rtp_llm.models_py.model_desc.qwen3 import Qwen3DecoderLayer
from rtp_llm.models_py.modules import Embedding, LinearFactory, RMSNorm
from rtp_llm.ops import ParallelismConfig
from rtp_llm.ops.compute_ops import PyAttentionInputs, PyModelInputs, PyModelOutputs
from rtp_llm.utils.model_weight import W


@dataclass
class DSparkDraftParams:
    """Draft-side extras parsed from the converted ckpt config.json.

    Shared by DFlash and DSpark (the "block-diffusion draft" family, matching
    the C++ SP_TYPE_DSPARK / dspark_config naming): markov_rank == 0 selects
    the DFlash argmax path, > 0 selects the DSpark Markov head.
    """

    aux_hidden_state_layer_ids: List[int]
    mask_token_id: int
    speculative_tokens: int  # k; query block is 1 + k wide
    block_size: int  # training block width; k + 1 <= block_size must hold
    markov_rank: int = 0  # 0 => DFlash (no Markov head)
    proposal_type: str = "greedy"

    @classmethod
    def from_ckpt_config(cls, cfg: dict) -> "DSparkDraftParams":
        params = cls(
            aux_hidden_state_layer_ids=list(cfg["aux_hidden_state_layer_ids"]),
            mask_token_id=cfg["mask_token_id"],
            speculative_tokens=cfg["speculative_tokens"],
            block_size=cfg["block_size"],
            markov_rank=cfg.get("markov_rank", 0) or 0,
            proposal_type=cfg.get("proposal_type") or "greedy",
        )
        if params.speculative_tokens + 1 > params.block_size:
            raise ValueError(
                f"speculative_tokens + 1 ({params.speculative_tokens + 1}) exceeds "
                f"ckpt block_size ({params.block_size}); the query block is "
                "anchor + k masks and must fit the training block width"
            )
        return params

    @property
    def block_width(self) -> int:
        return self.speculative_tokens + 1


@dataclass
class DSparkProposal:
    """Output contract of one draft block forward (rectangular, static k)."""

    draft_tokens: torch.Tensor  # [B, k] int64
    corrected_logits: torch.Tensor  # [B, k, V] fp32, Markov-corrected (= base for DFlash)
    base_logits: torch.Tensor  # [B, k, V] pre-correction lm_head logits
    head_hidden: torch.Tensor  # [B*(1+k), H] final block hidden states
    confidence_logits: Optional[torch.Tensor] = None  # phase-2 slot, never set here


class _PosIdsParams:
    """Duck-typed rope params: MhaRotaryEmbeddingOp._apply_rope only reads positions_d."""

    def __init__(self, positions_d: torch.Tensor):
        self.positions_d = positions_d


class Qwen3DFlashModel(GptModelBase):
    def __init__(
        self,
        config: ModelConfig,
        parallelism_config: ParallelismConfig,
        weights: ModelWeights,
        max_generate_batch_size: int,
        quant_config: Optional[object] = None,
        fmha_config=None,
        py_hw_kernel_config=None,
        device_resource_config=None,
    ):
        super().__init__(
            config,
            parallelism_config,
            weights,
            max_generate_batch_size=max_generate_batch_size,
            fmha_config=fmha_config,
            py_hw_kernel_config=py_hw_kernel_config,
            device_resource_config=device_resource_config,
        )
        assert config.dspark_config is not None, "dspark_config missing on ModelConfig"
        self.dspark_params: DSparkDraftParams = config.dspark_config
        if quant_config is not None:
            raise NotImplementedError("dspark draft does not support quantization yet")

        attn_configs = config.getAttentionConfigs(parallelism_config.get_attn_tp_size())
        assert (
            not attn_configs.is_causal
        ), "dspark block forward is non-causal; model registration must clear is_causal"
        self.attn_configs = attn_configs

        self.embed_tokens = Embedding(
            config, parallelism_config, weights.get_global_weight(W.embedding)
        )
        self.layers = nn.ModuleList(
            [
                Qwen3DecoderLayer(
                    config,
                    parallelism_config,
                    idx,
                    weights.weights[idx],
                    quant_config,
                    py_hw_kernel_config,
                )
                for idx in range(self.layer_num)
            ]
        )
        self.norm = RMSNorm(
            weights.get_global_weight(W.final_ln_gamma), eps=config.layernorm_eps
        )

        # Stage A: feature combine over layer-concatenated target aux hiddens.
        self.fc = LinearFactory.create_linear_from_weights(
            weights.global_weights, W.dspark_fc_w
        )
        # Stage B: hidden_norm applies only to the feature-KV injection pass.
        self.hidden_norm = RMSNorm(
            weights.get_global_weight(W.dspark_hidden_norm_gamma),
            eps=config.layernorm_eps,
        )
        # Per-layer K/V slices of the fused qkv weight (merge_qkv_hf layout:
        # [hidden, (nq + 2*nkv) * head_dim], per-rank after TP split).
        nq = attn_configs.head_num
        nkv = attn_configs.kv_head_num
        hd = attn_configs.size_per_head
        q_cols, kv_cols = nq * hd, nkv * hd
        self._ctx_k_w: List[torch.Tensor] = []
        self._ctx_v_w: List[torch.Tensor] = []
        self.ctx_k_norms = nn.ModuleList()
        for idx in range(self.layer_num):
            layer_weights = weights.weights[idx]
            qkv_w = layer_weights[W.attn_qkv_w]
            assert qkv_w.shape[-1] == q_cols + 2 * kv_cols, (
                f"unexpected qkv weight shape {tuple(qkv_w.shape)} for "
                f"nq={nq} nkv={nkv} hd={hd}"
            )
            self._ctx_k_w.append(qkv_w[:, q_cols : q_cols + kv_cols])
            self._ctx_v_w.append(qkv_w[:, q_cols + kv_cols :])
            self.ctx_k_norms.append(
                RMSNorm(layer_weights[W.k_ln_gamma], eps=config.layernorm_eps)
            )

        # Stage C output head (Stage D's Markov head is DSpark-only; see
        # qwen3_dspark.py).
        self.lm_head_weight = weights.get_global_weight(W.lm_head)  # [V, H]

        # Stage B rope: same op/cache/interleave flags as the block-forward path.
        from rtp_llm.models_py.modules.factory.attention.cuda_impl.flashinfer_rotary_emb import (
            MhaRotaryEmbeddingOp,
        )

        self._ctx_rope = MhaRotaryEmbeddingOp(attn_configs)

    # ---- Stage A ------------------------------------------------------

    def combine_hidden_states(self, aux_concat: torch.Tensor) -> torch.Tensor:
        """fc over layer-concatenated aux hiddens [T, n_aux*H] -> [T, H]."""
        return self.fc(aux_concat)

    # ---- Stage B ------------------------------------------------------

    def project_context_kv(
        self, fused: torch.Tensor, positions: torch.Tensor, layer_idx: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """One layer's feature K/V: hidden_norm -> K/V proj -> k_norm -> RoPE(K).

        fused: [T, H] fused features; positions: [T] int32 absolute positions.
        Returns (k, v) each [T, nkv, hd]; V is NOT rotated.
        """
        nkv, hd = self.attn_configs.kv_head_num, self.attn_configs.size_per_head
        normed = self.hidden_norm(fused)
        k = (normed @ self._ctx_k_w[layer_idx]).view(-1, nkv, hd)
        v = (normed @ self._ctx_v_w[layer_idx]).view(-1, nkv, hd)
        k = self.ctx_k_norms[layer_idx](k.reshape(-1, hd)).view(-1, nkv, hd)
        k = k.contiguous()
        # RoPE on K only, at the tokens' original positions (V not rotated).
        # _apply_rope rotates q and k in-place; feed a 1-head dummy q.
        dummy_q = k.new_zeros((k.shape[0], 1, hd))
        self._ctx_rope._apply_rope(dummy_q, k, _PosIdsParams(positions))
        return k, v

    def _build_ctx_write_indices(
        self,
        attn_inputs: PyAttentionInputs,
        ctx_lengths: torch.Tensor,  # [B] int, tokens to inject per request
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Self-built page indices for the write-only pass (eager, host-side).

        The injection window of request i is [prefix_i - ctx_i, prefix_i): the
        newly committed tokens, deliberately overwriting last round's
        rejected-position KV (no-rollback overwrite semantics).
        """
        page_size = self.attn_configs.kernel_tokens_per_block
        prefix = attn_inputs.prefix_lengths.tolist()
        ctx = ctx_lengths.tolist()
        block_table = attn_inputs.kv_cache_block_id_host
        if block_table.dim() == 3:
            # Engine layout is [group, batch, blocks]; the dspark draft is a
            # single FULL cache group (hybrid/linear targets are out of
            # phase-1 scope), so its rows live in group 0.
            block_table = block_table[0]

        batch_indices: List[int] = []
        positions: List[int] = []
        page_indices: List[int] = []
        page_indptr: List[int] = [0]
        last_page_len: List[int] = []
        for i, (p, c) in enumerate(zip(prefix, ctx)):
            assert 0 < c <= p, f"invalid ctx window: ctx={c}, prefix={p}"
            n_pages = math.ceil(p / page_size)
            page_indices.extend(block_table[i, :n_pages].tolist())
            page_indptr.append(page_indptr[-1] + n_pages)
            last_page_len.append(p - (n_pages - 1) * page_size)
            batch_indices.extend([i] * c)
            positions.extend(range(p - c, p))

        dev = torch.device("cuda")

        def to_dev(x: List[int]) -> torch.Tensor:
            return torch.tensor(x, dtype=torch.int32, device=dev)

        return (
            to_dev(batch_indices),
            to_dev(positions),
            to_dev(page_indices),
            to_dev(page_indptr),
            to_dev(last_page_len),
        )

    def _inject_context_kv_dense(
        self,
        fused: torch.Tensor,  # [B*width, H] fused features, request-major
        attn_inputs: PyAttentionInputs,
        ctx_starts: torch.Tensor,  # [B] int32, window base per request
    ) -> None:
        """Device fast path: fixed-width window [start, start + width).

        Pure-device index math (no .tolist()/.item()/D2H) and a direct
        advanced-indexing scatter into the paged cache — flashinfer's append
        needs a host-built ragged page_indptr, which would reintroduce the
        sync this path exists to remove.  Shapes are fixed given (B, width),
        so the whole pass is CUDA-graph capturable later.

        Rows past a request's accept_len carry garbage (rejected-trajectory)
        features by design: they land inside the block forward's own query
        window, whose attention path overwrites them (write-before-read)
        before anything reads them.  See the executor-side comment in
        MtpBatchStreamProcessor::updateDecodePostDSparkDraftModelInput.
        """
        num_rows = fused.shape[0]
        batch = ctx_starts.numel()
        assert batch > 0 and num_rows % batch == 0, (
            f"dense inject: rows ({num_rows}) not a multiple of batch ({batch})"
        )
        width = num_rows // batch
        dev = fused.device
        starts = ctx_starts.to(device=dev, dtype=torch.int32, non_blocking=True)
        offsets = torch.arange(width, dtype=torch.int32, device=dev)
        positions = (starts.unsqueeze(1) + offsets).reshape(-1)  # [B*width]

        page_size = self.attn_configs.kernel_tokens_per_block
        block_table = attn_inputs.kv_cache_block_id_device
        if block_table is None or block_table.numel() == 0:
            # The standard engine path only publishes the pinned host table
            # (kv_cache_block_id_device is a trtllm_gen/headwise extra); the
            # async H2D upload keeps this path free of D2H syncs.
            block_table = attn_inputs.kv_cache_block_id_host.to(
                device=dev, non_blocking=True
            )
        if block_table.dim() == 3:
            # [group, batch, blocks]; single FULL group in phase-1 => group 0.
            block_table = block_table[0]
        batch_idx = torch.repeat_interleave(
            torch.arange(batch, dtype=torch.int64, device=dev), width
        )
        pos_long = positions.long()
        page_ids = block_table[batch_idx, pos_long // page_size].long()  # [B*width]
        slots = pos_long % page_size

        assert self.kv_cache is not None, "kv_cache required for feature injection"
        for layer_idx in range(self.layer_num):
            k, v = self.project_context_kv(fused, positions, layer_idx)
            base = self.kv_cache.get_layer_cache(layer_idx).kv_cache_base
            # base is [P, 2, nkv, page, hd] (HND); advanced indices at dims
            # 0/1/3 broadcast to the front => LHS shape [B*width, nkv, hd].
            base[page_ids, 0, :, slots, :] = k.to(base.dtype)
            base[page_ids, 1, :, slots, :] = v.to(base.dtype)

    def inject_context_kv(
        self,
        fused: torch.Tensor,  # [T_ctx, H] fused features, request-major order
        attn_inputs: PyAttentionInputs,
        ctx_lengths: Optional[torch.Tensor] = None,  # [B]; None => whole prefix
        ctx_starts: Optional[torch.Tensor] = None,  # [B]; window base override
    ) -> None:
        """Write feature KV into the paged cache at the ctx positions.

        Write-only (no attention): per layer, project + k_norm + RoPE, then
        flashinfer append_paged_kv_cache with self-built indices.  The query
        block's own K/V are NOT written here — the block forward writes them
        at the future positions as part of its regular attention path.

        With ctx_starts given, the window is [starts[i], starts[i] + width)
        and the device fast path runs instead (dense decode-tail seeding);
        the ragged host path below stays for prefill seeding (naturally
        variable-length prompt suffixes, outside any graph-capture scope).
        """
        import flashinfer.page as page

        if ctx_starts is not None:
            self._inject_context_kv_dense(fused, attn_inputs, ctx_starts)
            return

        if ctx_lengths is None:
            ctx_lengths = attn_inputs.prefix_lengths
        assert fused.shape[0] == int(ctx_lengths.sum().item()), (
            f"input_hiddens rows ({fused.shape[0]}) != sum(ctx_lengths) "
            f"({int(ctx_lengths.sum().item())})"
        )
        (
            batch_indices,
            positions,
            page_indices,
            page_indptr,
            last_page_len,
        ) = self._build_ctx_write_indices(attn_inputs, ctx_lengths)

        assert self.kv_cache is not None, "kv_cache required for feature injection"
        for layer_idx in range(self.layer_num):
            k, v = self.project_context_kv(fused, positions, layer_idx)
            layer_cache = self.kv_cache.get_layer_cache(layer_idx)
            kv_base = layer_cache.kv_cache_base
            page.append_paged_kv_cache(
                k.to(kv_base.dtype),
                v.to(kv_base.dtype),
                batch_indices,
                positions,
                (kv_base[:, 0], kv_base[:, 1]),
                page_indices,
                page_indptr,
                last_page_len,
                "HND",
            )

    # ---- Stage C ------------------------------------------------------

    def block_forward(
        self, input_ids: torch.Tensor, inputs: PyModelInputs, fmha_impl: Any
    ) -> torch.Tensor:
        """Backbone over the [anchor + k masks] block.  Returns [B*(1+k), H].

        Non-causal visibility (feature prefix fully visible + intra-block
        bidirectional) comes from attn_configs.is_causal=False via the fmha
        impl; the block's own K/V are written at the future positions by the
        regular attention path.
        """
        hidden_states = self.embed_tokens(input_ids)
        attention_inputs = inputs.attention_inputs
        for i, decoder_layer in enumerate(self.layers[: self.layer_num]):
            select_block_map_for_layer(attention_inputs, i)
            hidden_states = decoder_layer(
                hidden_states,
                fmha_impl,
                kv_cache=self.kv_cache.get_layer_cache(i) if self.kv_cache else None,
            )
        return self.norm(hidden_states)

    def compute_base_logits(self, head_hidden: torch.Tensor) -> torch.Tensor:
        """lm_head over the k mask positions (bonus-anchor layout).

        head_hidden: [B*(1+k), H].  Returns [B, k, V] (weights' dtype).
        """
        width = self.dspark_params.block_width
        hidden = head_hidden.view(-1, width, head_hidden.shape[-1])[:, 1:, :]
        # The engine loader may keep lm_head in fp32 (enable_fp32_lm_head);
        # follow the weight dtype — logits go through .float() downstream.
        return F.linear(hidden.to(self.lm_head_weight.dtype), self.lm_head_weight)

    # ---- Stage D (virtual seam) -----------------------------------------

    def markov_correct(
        self, base_logits: torch.Tensor, anchor_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """DFlash stage D: greedy argmax, no transition bias.

        base_logits: [B, k, V]; anchor_ids: [B] int64 (unused by DFlash, kept
        for the uniform signature the DSpark subclass overrides).
        Returns (tokens [B, k] int64, corrected_logits [B, k, V] fp32).
        """
        corrected = base_logits.float()
        return corrected.argmax(dim=-1), corrected

    # ---- Full pipeline --------------------------------------------------

    def propose(
        self,
        inputs: PyModelInputs,
        fmha_impl: Any = None,
        ctx_lengths: Optional[torch.Tensor] = None,
        ctx_starts: Optional[torch.Tensor] = None,
    ) -> DSparkProposal:
        """One draft round: inject features, block forward, correct, decide.

        Contract (executor side):
          - inputs.input_ids: [B*(1+k)] block tokens, request-major; row 0 of
            each block is the anchor, rows 1..k are mask_token_id;
          - inputs.input_hiddens: [T_ctx, n_aux*H] target aux hiddens
            (concat by aux_hidden_state_layer_ids) of the tokens to inject,
            request-major; ctx_lengths[i] rows per request, ending at
            prefix_lengths[i].  ctx_lengths=None means the whole prefix
            (prefill seeding);
          - ctx_starts (or inputs.dspark_ctx_starts): dense decode-tail mode —
            1+k rows per request injected at [starts[i], starts[i] + 1 + k),
            device fast path, ctx_lengths ignored;
          - inputs.attention_inputs: chunked-prefill metadata with
            prefix_lengths = committed_len, input_lengths = 1+k each.
        """
        width = self.dspark_params.block_width
        attention_inputs = inputs.attention_inputs
        input_lengths = attention_inputs.input_lengths
        if not input_lengths.is_cuda:
            # Host-side sanity only on CPU inputs (UT paths): a CUDA read here
            # would D2H-sync every decode round.  The executor guarantees
            # uniform k+1 blocks on the engine path.
            assert bool((input_lengths == width).all()), (
                f"dspark block forward expects uniform input_lengths == {width}, "
                f"got {input_lengths.tolist()}"
            )
        if fmha_impl is None:
            fmha_impl = self.prepare_fmha_impl(inputs)

        # Executor channel for the dense decode-tail window base; when set,
        # injection takes the device fast path with a fixed [B, width] window.
        if ctx_starts is None:
            ctx_starts = getattr(inputs, "dspark_ctx_starts", None)
            if ctx_starts is not None and ctx_starts.numel() == 0:
                ctx_starts = None

        if ctx_lengths is None and ctx_starts is None:
            # Executor channel for the incremental (decode-tail) injection
            # window; unset means "whole prefix" (prefill seeding).
            ctx_from_inputs = getattr(inputs, "dspark_ctx_lengths", None)
            if ctx_from_inputs is not None and ctx_from_inputs.numel() > 0:
                ctx_lengths = ctx_from_inputs.cpu()

        aux = inputs.input_hiddens
        if aux is not None and aux.numel() > 0:
            fused = self.combine_hidden_states(aux)
            self.inject_context_kv(
                fused, attention_inputs, ctx_lengths, ctx_starts=ctx_starts
            )

        head_hidden = self.block_forward(inputs.input_ids, inputs, fmha_impl)
        base_logits = self.compute_base_logits(head_hidden)
        anchor_ids = inputs.input_ids.view(-1, width)[:, 0]
        draft_tokens, corrected = self.markov_correct(base_logits, anchor_ids)
        return DSparkProposal(
            draft_tokens=draft_tokens,
            corrected_logits=corrected,
            base_logits=base_logits,
            head_hidden=head_hidden,
        )

    def forward(self, inputs: PyModelInputs, fmha_impl: Any = None) -> PyModelOutputs:
        """Engine entry: full pipeline, returns hidden states + proposal.

        G3 contract: draft sampling lives in the model, so the outputs carry
        draft_tokens [B, k] and draft_probs [B, k, V] (softmax of the corrected
        logits — Markov-corrected for DSpark, raw base logits for DFlash)
        alongside the block hidden states.  Request temperature plumbing for
        the probabilistic q arrives with the executor splice; until then probs
        are softmax at T=1 (greedy verification is exact token match and does
        not read them).
        """
        if fmha_impl is None:
            fmha_impl = self.prepare_fmha_impl(inputs)
        proposal = self.propose(inputs, fmha_impl)
        outputs = PyModelOutputs(proposal.head_hidden, fmha_impl.fmha_params)
        outputs.draft_tokens = proposal.draft_tokens
        outputs.draft_probs = torch.softmax(proposal.corrected_logits, dim=-1)
        return outputs


__all__ = [
    "DSparkDraftParams",
    "DSparkProposal",
    "Qwen3DFlashModel",
]
