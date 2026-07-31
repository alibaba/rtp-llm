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

Stage B is a write-only pass over the paged KV cache (an independent
projection pass by design, not part of any decoder forward): both the ragged
(prefill-seeding) and dense (decode-tail) windows build (batch, position)
indices with pure device tensor math and share one advanced-indexing scatter
into the paged cache — no host-built ragged metadata, no D2H sync (layout
equivalence with flashinfer append_paged_kv_cache is pinned by test).  Stage C
is a regular chunked-prefill decoder forward whose non-causal visibility comes
entirely from attn_config.is_causal=False (the model registration sets it).
"""

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
        # qwen3_dspark.py).  Under TP the loader vocab-shards lm_head
        # (sp_0_pad8): each rank holds [V_pad/tp, H] and compute_base_logits
        # all-gathers the shard logits back to the full vocab.
        self.tp_size = parallelism_config.tp_size
        self.lm_head_weight = weights.get_global_weight(W.lm_head)  # [V(/tp), H]

        # Reduced draft vocab (speculators vocab pruning): lm_head/markov_w2
        # emit draft-vocab logits and d2t maps each argmax pick back to the
        # target vocab (ABSOLUTE ids — the ckpt converter normalizes the
        # speculators offset convention).  markov_w1 / embed_tokens stay
        # target-vocab, so anchors and chained prev tokens index them directly.
        self.d2t = weights.get_global_weight_or_none(W.multi_tokens_predict_d2t_map)
        if self.d2t is not None:
            self.d2t = self.d2t.long()
            assert self.tp_size == 1, (
                "reduced draft vocab (d2t) is not wired for tp > 1 yet: the "
                "vocab-shard merge slices by the target vocab size"
            )
            assert self.d2t.numel() == self.lm_head_weight.shape[0], (
                f"d2t rows ({self.d2t.numel()}) != draft lm_head rows "
                f"({self.lm_head_weight.shape[0]})"
            )
        elif self.tp_size == 1:
            assert self.lm_head_weight.shape[0] >= self.vocab_size, (
                f"draft lm_head rows ({self.lm_head_weight.shape[0]}) < target "
                f"vocab ({self.vocab_size}) but no d2t map was loaded — a "
                "trimmed draft vocab requires the ckpt's d2t tensor"
            )

        # Stage B rope: same op/cache/interleave flags as the block-forward path.
        from rtp_llm.models_py.modules.factory.attention.cuda_impl.flashinfer_rotary_emb import (
            MhaRotaryEmbeddingOp,
        )

        self._ctx_rope = MhaRotaryEmbeddingOp(attn_configs)

    # ---- CUDA graph capture hook --------------------------------------

    def cuda_graph_input_hidden_dim(self) -> int:
        """Width of input_hiddens the captured forward consumes: n_aux * H.

        Unlike a normal draft (input_hiddens = a single [T, H] hidden), the
        DSpark/DFlash draft consumes the target's layer-concatenated aux hiddens
        [T, n_aux*H] (stage A's fc input).  The CUDA graph runner sizes its
        static input_hiddens buffer to this instead of the model hidden H.
        H is the full model hidden on every rank: aux features come off the
        replicated residual stream, and stage A's fc weight is sp_id
        (replicated), so TP does not shard this dimension.
        """
        return len(self.dspark_params.aux_hidden_state_layer_ids) * self.config.hidden_size

    # ---- Stage A ------------------------------------------------------

    def combine_hidden_states(self, aux_concat: torch.Tensor) -> torch.Tensor:
        """fc over layer-concatenated aux hiddens [T, n_aux*H] -> [T, H]."""
        return self.fc(aux_concat)

    # ---- Stage B ------------------------------------------------------

    def project_context_kv(
        self,
        normed: torch.Tensor,
        positions: torch.Tensor,
        layer_idx: int,
        dummy_q: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """One layer's feature K/V over pre-normed features: K/V proj -> k_norm -> RoPE(K).

        normed: [T, H] hidden_norm(fused) — layer-invariant, so callers compute
        it once outside their layer loop; positions: [T] int32 absolute
        positions; dummy_q: [T, 1, hd] zero q shared across layers (_apply_rope
        rotates q and k in-place, and a rotated zero stays zero).
        Returns (k, v) each [T, nkv, hd]; V is NOT rotated.
        """
        nkv, hd = self.attn_configs.kv_head_num, self.attn_configs.size_per_head
        k = (normed @ self._ctx_k_w[layer_idx]).view(-1, nkv, hd)
        v = (normed @ self._ctx_v_w[layer_idx]).view(-1, nkv, hd)
        k = self.ctx_k_norms[layer_idx](k.reshape(-1, hd)).view(-1, nkv, hd)
        k = k.contiguous()
        # RoPE on K only, at the tokens' original positions (V not rotated).
        self._ctx_rope._apply_rope(dummy_q, k, _PosIdsParams(positions))
        return k, v

    def _prenorm_ctx_features(
        self, fused: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """hidden_norm(fused) plus the shared dummy q for project_context_kv."""
        hd = self.attn_configs.size_per_head
        normed = self.hidden_norm(fused)
        return normed, normed.new_zeros((normed.shape[0], 1, hd))

    def _ctx_block_table(
        self, attn_inputs: PyAttentionInputs, dev: torch.device
    ) -> torch.Tensor:
        """Device [B, pages] draft block table."""
        block_table = attn_inputs.kv_cache_block_id_device
        if block_table is None or block_table.numel() == 0:
            # The standard engine path only publishes the pinned host table
            # (kv_cache_block_id_device is a trtllm_gen/headwise extra); the
            # async H2D upload keeps this path free of D2H syncs.
            block_table = attn_inputs.kv_cache_block_id_host.to(
                device=dev, non_blocking=True
            )
        if block_table.dim() == 3:
            # Engine layout is [group, batch, blocks]; the dspark draft is a
            # single FULL cache group (hybrid/linear targets are out of
            # phase-1 scope), so its rows live in group 0.
            block_table = block_table[0]
        return block_table

    def _scatter_ctx_kv(
        self,
        fused: torch.Tensor,  # [T, H] fused features, request-major
        attn_inputs: PyAttentionInputs,
        batch_idx: torch.Tensor,  # [T] int64, owning request per row
        positions: torch.Tensor,  # [T] int32, absolute position per row
    ) -> None:
        """Project + scatter the rows' feature KV into the paged cache.

        Direct advanced-indexing scatter, layout-equivalent to flashinfer's
        append_paged_kv_cache (pinned by test) but with no host-built ragged
        page metadata: callers enqueue behind a busy stream without a D2H
        sync (prefill seeding overlaps the target's drain), and fixed-shape
        callers stay CUDA-graph capturable.
        """
        dev = fused.device
        page_size = self.attn_configs.kernel_tokens_per_block
        block_table = self._ctx_block_table(attn_inputs, dev)
        pos_long = positions.long()
        page_ids = block_table[batch_idx, pos_long // page_size].long()  # [T]
        slots = pos_long % page_size

        assert self.kv_cache is not None, "kv_cache required for feature injection"
        normed, dummy_q = self._prenorm_ctx_features(fused)
        for layer_idx in range(self.layer_num):
            k, v = self.project_context_kv(normed, positions, layer_idx, dummy_q)
            base = self.kv_cache.get_layer_cache(layer_idx).kv_cache_base
            # base is [P, 2, nkv, page, hd] (HND); advanced indices at dims
            # 0/1/3 broadcast to the front => LHS shape [T, nkv, hd].
            base[page_ids, 0, :, slots, :] = k.to(base.dtype)
            base[page_ids, 1, :, slots, :] = v.to(base.dtype)

    def _inject_context_kv_dense(
        self,
        fused: torch.Tensor,  # [B*width, H] fused features, request-major
        attn_inputs: PyAttentionInputs,
        ctx_starts: torch.Tensor,  # [B] int32, window base per request
    ) -> None:
        """Fixed-width window [start, start + width) per request (decode tail).

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
        batch_idx = torch.repeat_interleave(
            torch.arange(batch, dtype=torch.int64, device=dev), width
        )
        self._scatter_ctx_kv(fused, attn_inputs, batch_idx, positions)

    def inject_context_kv(
        self,
        fused: torch.Tensor,  # [T_ctx, H] fused features, request-major order
        attn_inputs: PyAttentionInputs,
        ctx_lengths: Optional[torch.Tensor] = None,  # [B]; None => whole prefix
        ctx_starts: Optional[torch.Tensor] = None,  # [B]; window base override
    ) -> None:
        """Write feature KV into the paged cache at the ctx positions.

        Write-only (no attention): per layer, project + k_norm + RoPE, then
        scatter into the paged cache.  The query block's own K/V are NOT
        written here — the block forward writes them at the future positions
        as part of its regular attention path.

        With ctx_starts given, the window is [starts[i], starts[i] + width)
        (dense decode-tail seeding); otherwise the ragged window of request i
        is [prefix_i - ctx_i, prefix_i) — the newly committed tokens,
        deliberately overwriting last round's rejected-position KV
        (no-rollback overwrite semantics).  Ragged row math needs the total
        row count, which comes from fused.shape[0] instead of a device
        reduction, keeping this path free of D2H syncs too.
        """
        if ctx_starts is not None:
            self._inject_context_kv_dense(fused, attn_inputs, ctx_starts)
            return

        if ctx_lengths is None:
            ctx_lengths = attn_inputs.prefix_lengths
        total = fused.shape[0]
        if not ctx_lengths.is_cuda:
            # Host-side sanity only on CPU metadata (UT paths): a CUDA read
            # here would stall enqueue behind the whole stream queue.  ctx == 0
            # would mean a fully prefix-cached prompt; the engine always
            # recomputes at least the last prompt token (input_lengths >= 1),
            # so an empty window is a broken caller, not a valid state.
            assert total == int(ctx_lengths.sum()), (
                f"input_hiddens rows ({total}) != sum(ctx_lengths) "
                f"({int(ctx_lengths.sum())})"
            )
            assert bool((ctx_lengths > 0).all()), (
                f"invalid ctx window: {ctx_lengths.tolist()}"
            )
        dev = torch.device("cuda")
        ctx = ctx_lengths.to(device=dev, dtype=torch.int64, non_blocking=True)
        prefix = attn_inputs.prefix_lengths.to(
            device=dev, dtype=torch.int64, non_blocking=True
        )
        batch = ctx.numel()
        batch_idx = torch.repeat_interleave(
            torch.arange(batch, dtype=torch.int64, device=dev),
            ctx,
            output_size=total,
        )
        # position of global row j in request i: window_start[i] + (j - row_base[i])
        row_base = torch.cumsum(ctx, 0) - ctx  # exclusive prefix sum
        offsets = torch.arange(total, dtype=torch.int64, device=dev)
        positions = ((prefix - ctx - row_base)[batch_idx] + offsets).to(torch.int32)
        self._scatter_ctx_kv(fused, attn_inputs, batch_idx, positions)

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

        TP: head_hidden is the replicated residual stream, so each rank's
        F.linear yields the [B, k, V_pad/tp] vocab shard; all-gather + reorder
        rebuilds the full vocab on every rank (same NCCL collective the engine
        uses for target logits in tpSyncEmbeddingOrLogits), then the sp_0_pad8
        padding rows are sliced off so argmax can never pick a pad id.  Under
        tp>1 this runs outside the CUDA graph (eager draft_tail); at tp==1 the
        full-tail graph captures it.
        """
        width = self.dspark_params.block_width
        hidden = head_hidden.view(-1, width, head_hidden.shape[-1])[:, 1:, :]
        # The engine loader may keep lm_head in fp32 (enable_fp32_lm_head);
        # follow the weight dtype — logits go through .float() downstream.
        logits = F.linear(hidden.to(self.lm_head_weight.dtype), self.lm_head_weight)
        if self.tp_size > 1:
            logits = self._gather_vocab_shards(logits)
        return logits

    def _gather_vocab_shards(self, shard_logits: torch.Tensor) -> torch.Tensor:
        """All-gather [B, k, V_pad/tp] shard logits into full-vocab [B, k, V]."""
        from rtp_llm.models_py.distributed.collective_torch import Group, all_gather

        rows = shard_logits.shape[0] * shard_logits.shape[1]
        gathered = all_gather(
            shard_logits.reshape(rows, -1).contiguous(), group=Group.TP
        )  # [tp * rows, V_pad/tp], rank-major
        return self._merge_vocab_shards(gathered, shard_logits.shape)

    def _merge_vocab_shards(
        self, gathered: torch.Tensor, shard_shape: torch.Size
    ) -> torch.Tensor:
        """Reorder rank-major gathered rows to [B, k, V] and strip pad rows.

        Pure tensor math (no collective) so tests can drive it with a
        hand-built gathered tensor.
        """
        batch, k, shard_vocab = shard_shape
        rows = batch * k
        full = (
            gathered.reshape(self.tp_size, rows, shard_vocab)
            .permute(1, 0, 2)
            .reshape(batch, k, self.tp_size * shard_vocab)
        )
        # sp_0_pad8 pads the vocab to a tp*8 multiple with zero rows on the
        # last rank; a zero logit can out-rank real negative logits, so the
        # pad columns must never reach argmax/softmax.
        return full[..., : self.vocab_size].contiguous()

    # ---- Stage D (virtual seam) -----------------------------------------

    def map_draft_to_target(self, draft_ids: torch.Tensor) -> torch.Tensor:
        """Draft-vocab argmax picks -> target-vocab token ids (identity for
        full-vocab drafts)."""
        if self.d2t is None:
            return draft_ids
        return self.d2t[draft_ids]

    def markov_correct(
        self, base_logits: torch.Tensor, anchor_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """DFlash stage D: greedy argmax, no transition bias.

        base_logits: [B, k, V_draft]; anchor_ids: [B] int64 (unused by DFlash,
        kept for the uniform signature the DSpark subclass overrides).
        Returns (tokens [B, k] int64 TARGET-vocab ids, corrected_logits
        [B, k, V_draft] fp32).
        """
        corrected = base_logits.float()
        return self.map_draft_to_target(corrected.argmax(dim=-1)), corrected

    # ---- Full pipeline --------------------------------------------------

    def _propose_backbone(
        self,
        inputs: PyModelInputs,
        fmha_impl: Any,
        ctx_lengths: Optional[torch.Tensor] = None,
        ctx_starts: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Stages A+B+C: feature combine, inject, block forward -> head_hidden.

        This region ends at head_hidden [B*(1+k), H], excluding the lm_head +
        Markov + softmax tail (stage D).  Two graph boundaries exist: the
        default full-tail graph captures forward (backbone + tail, vLLM
        FULL-graph boundary); the backbone-only graph (tp>1 or
        prefill seeding) captures forward_backbone and runs draft_tail
        eagerly after replay.
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

        # Executor channel for the dense decode-tail window base; when set,
        # injection takes the device fast path with a fixed [B, width] window.
        if ctx_starts is None:
            # pybind maps an undefined C++ tensor field to None.
            ctx_starts = inputs.dspark_ctx_starts
            if ctx_starts is not None and ctx_starts.numel() == 0:
                ctx_starts = None

        if ctx_lengths is None and ctx_starts is None:
            # Executor channel for the incremental (decode-tail) injection
            # window; unset means "whole prefix" (prefill seeding).  Stays on
            # its native device: the inject index math is device-side.
            ctx_from_inputs = inputs.dspark_ctx_lengths
            if ctx_from_inputs is not None and ctx_from_inputs.numel() > 0:
                ctx_lengths = ctx_from_inputs

        aux = inputs.input_hiddens
        if aux is not None and aux.numel() > 0:
            fused = self.combine_hidden_states(aux)
            self.inject_context_kv(
                fused, attention_inputs, ctx_lengths, ctx_starts=ctx_starts
            )

        return self.block_forward(inputs.input_ids, inputs, fmha_impl)

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
        if fmha_impl is None:
            fmha_impl = self.prepare_fmha_impl(inputs)
        width = self.dspark_params.block_width
        head_hidden = self._propose_backbone(inputs, fmha_impl, ctx_lengths, ctx_starts)
        base_logits = self.compute_base_logits(head_hidden)
        anchor_ids = inputs.input_ids.view(-1, width)[:, 0]
        draft_tokens, corrected = self.markov_correct(base_logits, anchor_ids)
        return DSparkProposal(
            draft_tokens=draft_tokens,
            corrected_logits=corrected,
            base_logits=base_logits,
            head_hidden=head_hidden,
        )

    # ---- CUDA-graph split: full forward (default) or backbone-only -------

    def forward_backbone(
        self, inputs: PyModelInputs, fmha_impl: Any = None
    ) -> PyModelOutputs:
        """Backbone-only graph entry: stages A+B+C, output = head_hidden.

        The CUDA graph runner binds this when the tail is not captured (tp>1
        graphs); the engine then runs draft_tail eagerly after
        replay to produce draft_tokens/draft_probs.  The default full-tail
        graph binds forward instead.
        """
        if fmha_impl is None:
            fmha_impl = self.prepare_fmha_impl(inputs)
        head_hidden = self._propose_backbone(inputs, fmha_impl)
        return PyModelOutputs(head_hidden, fmha_impl.fmha_params)

    def draft_tail(
        self, outputs: PyModelOutputs, inputs: PyModelInputs
    ) -> PyModelOutputs:
        """Stage D on head_hidden: lm_head + Markov + softmax.

        On the backbone-only graph boundary this runs eagerly after replay,
        reading the block hidden states the captured forward_backbone left in
        outputs.hidden_states and the anchor ids from inputs.input_ids (the
        static graph input buffer, refreshed pre-replay, valid until the next
        replay's refresh).  On the default full-tail boundary it runs inside
        the captured forward and the runner persists draft_tokens/draft_probs
        into static output buffers.
        """
        width = self.dspark_params.block_width
        head_hidden = outputs.hidden_states
        base_logits = self.compute_base_logits(head_hidden)
        anchor_ids = inputs.input_ids.view(-1, width)[:, 0]
        draft_tokens, corrected = self.markov_correct(base_logits, anchor_ids)
        outputs.draft_tokens = draft_tokens
        outputs.draft_probs = torch.softmax(corrected, dim=-1)
        return outputs

    def forward(self, inputs: PyModelInputs, fmha_impl: Any = None) -> PyModelOutputs:
        """Eager engine entry: full pipeline (backbone + tail).

        Used on the non-graph path (canRun false, e.g. prefill seeding).  The
        default full-tail CUDA graph captures this method whole; the
        backbone-only boundary (prefill graphs) captures
        forward_backbone and runs draft_tail eagerly after replay.  All paths
        keep the same output contract.

        G3 contract: draft sampling lives in the model, so the outputs carry
        draft_tokens [B, k] and draft_probs [B, k, V] (softmax of the corrected
        logits — Markov-corrected for DSpark, raw base logits for DFlash)
        alongside the block hidden states.  Request temperature plumbing for
        the probabilistic q arrives with the executor splice; until then probs
        are softmax at T=1 (greedy verification is exact token match and does
        not read them).
        """
        outputs = self.forward_backbone(inputs, fmha_impl)
        return self.draft_tail(outputs, inputs)


__all__ = [
    "DSparkDraftParams",
    "DSparkProposal",
    "Qwen3DFlashModel",
]
