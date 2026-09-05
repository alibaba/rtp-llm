"""DeepSeek-V4 DSpark draft model.

The checkpoint stores DSpark as three ordinary V4 SWA blocks under
``mtp.{0,1,2}``, plus a small target-feature projection and a Markov head.
Unlike autoregressive MTP, a DSpark step evaluates one runtime-fixed query block
in parallel::

    [anchor, noise, ..., noise]

Every query attends to the latest committed sliding-window context *and* the
whole query block (including future noise positions).  The backbone therefore
produces ``GEN_NUM_PER_CIRCLE`` base-logit rows in one pass.  A cheap Markov bias is then applied
left-to-right, beginning with the anchor, to recover intra-block dependency.

The target model supplies mean-pooled hidden states from its configured layers
through ``PyModelInputs.input_hiddens``.  Only newly committed rows are sent on
each step.  This model projects those rows once and inserts a layer-specific KV
projection into each draft layer's paged SWA cache before evaluating the next
query block.

The width is fixed for the lifetime of a service and comes only from
``GEN_NUM_PER_CIRCLE``. Decode proposal and commit calls use separate
batch-keyed decode CUDA-graph contracts (``gamma`` versus ``gamma + 1`` rows
per request); prompt seeding calls the same commit role through the standard
prefill/CP path and does not match its target-verify graph contract.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any, Optional, Tuple

import torch

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.model_loader.model_weight_info import ModelWeights
from rtp_llm.models_py.model_desc.deepseek_v4_model import DeepSeekV4Model
from rtp_llm.models_py.modules import RMSNorm
from rtp_llm.models_py.modules.dsv4._fused_rmsnorm_rope_triton import fused_rmsnorm_rope
from rtp_llm.models_py.modules.dsv4.cp import build_cp_context_for_forward
from rtp_llm.models_py.modules.dsv4.fp8._kv_cache_utils import (
    require_pool_tokens_per_block,
)
from rtp_llm.models_py.modules.dsv4.fp8._swa_ops_triton import (
    compute_swa_slot_mapping_from_positions,
)
from rtp_llm.models_py.modules.dsv4.fp8.attention import BIND_KEEP, bind_attn_cache
from rtp_llm.models_py.modules.dsv4.fp8.decode.compute_qkv import decode_compute_qkv
from rtp_llm.models_py.modules.dsv4.fp8.decode.decode_attn_metadata import (
    get_or_build_sched_meta,
)
from rtp_llm.models_py.modules.dsv4.fp8.decode.output_proj import decode_output_proj
from rtp_llm.models_py.modules.dsv4.fp8.decode.write_swa import decode_write_swa_fp8
from rtp_llm.models_py.modules.dsv4.kv_cache_utils import (
    SWA_KV,
    as_attention_inputs_by_tag,
    build_block_tables_for_tags,
    primary_attention_inputs,
)
from rtp_llm.models_py.modules.dsv4.utils import _v4_fp8_linear
from rtp_llm.models_py.modules.factory.attention.attn_factory import (
    CudaGraphSelectionMode,
    PrefillCudaGraphUnsupportedBackend,
)
from rtp_llm.models_py.modules.factory.attention.common import (
    create_write_cache_store_impl,
)
from rtp_llm.models_py.speculative.dspark_proposer_mixin import (
    DSparkProposerMixin,
    optional_tensor,
)
from rtp_llm.ops.compute_ops import PyModelInputs, PyModelOutputs
from rtp_llm.utils.model_weight import W


class DeepSeekV4DSparkModel(DSparkProposerMixin, DeepSeekV4Model):
    """DeepSeek-V4 implementation of the shared DSpark proposer.

    The engine input contract and query-block geometry are inherited from
    :class:`DSparkProposerMixin`; this class implements the
    model-specific hooks with DeepSeek-V4 specifics: mHC blocks, FP8 SWA
    paged-cache injection, FlashMLA non-causal top-k indices and the
    ``mtp.*`` checkpoint weights.

    Output is normalized ``[B * gamma, hidden_dim]``. The regular C++ model
    wrapper applies lm_head; the speculative executor then applies the Markov
    tail and the request's sampling parameters.
    """

    # Draft side: carries the capture ids for the shared-buffer row-width
    # derivation but consumes the features — only the target captures.
    _captures_aux_hidden = False

    def __init__(
        self,
        model_config: ModelConfig,
        parallelism_config,
        weights: ModelWeights,
        moe_config,
        max_generate_batch_size: int,
        fmha_config=None,
        py_hw_kernel_config=None,
        device_resource_config=None,
    ) -> None:
        super().__init__(
            model_config,
            parallelism_config,
            weights,
            moe_config,
            max_generate_batch_size=max_generate_batch_size,
            fmha_config=fmha_config,
            py_hw_kernel_config=py_hw_kernel_config,
            device_resource_config=device_resource_config,
        )

        noise_token_id = getattr(model_config, "dspark_noise_token_id", None)
        target_layer_ids = getattr(model_config, "dspark_target_layer_ids", None)
        markov_rank = getattr(model_config, "dspark_markov_rank", None)
        if noise_token_id is None or int(noise_token_id) < 0:
            raise ValueError(
                "DeepSeekV4DSparkModel requires a non-negative " "dspark_noise_token_id"
            )
        if not target_layer_ids:
            raise ValueError("DeepSeekV4DSparkModel requires dspark_target_layer_ids")
        if not markov_rank or int(markov_rank) <= 0:
            raise ValueError(
                "DeepSeekV4DSparkModel requires a positive dspark_markov_rank"
            )

        self._dspark_target_layer_ids = tuple(int(v) for v in target_layer_ids)
        self._dspark_markov_rank = int(markov_rank)
        self.tp_size = int(getattr(parallelism_config, "tp_size", 1) or 1)
        self.tp_rank = int(getattr(parallelism_config, "tp_rank", 0) or 0)
        # Commit-side CP context. Target features remain rank-local after the
        # target model's zigzag prefill; the commit forward goes through the
        # standard CP handleInputs, so its row map is derived per call from
        # the attached context_parallel_info (see map_commit_rows).
        self._dspark_commit_cp_enabled = False
        self._dspark_kv_cache_sharded = False
        _cp_cfg = getattr(parallelism_config, "prefill_cp_config", None)
        if _cp_cfg is not None and bool(_cp_cfg.is_enabled()) and self.tp_size > 1:
            self._dspark_commit_cp_enabled = True
            self._dspark_kv_cache_sharded = bool(
                getattr(_cp_cfg, "kv_cache_sharded", False)
            )

        if self._gen_num_per_cycle <= 0:
            raise ValueError(
                "DeepSeekV4DSparkModel requires a positive "
                f"gen_num_per_cycle, got {self._gen_num_per_cycle}"
            )
        if self._v4_args.n_layers != 3:
            raise ValueError(
                "DeepSeek-V4 DSpark checkpoint requires exactly three draft "
                f"layers, got {self._v4_args.n_layers}"
            )
        if any(int(ratio) != 0 for ratio in self._v4_args.compress_ratios):
            raise ValueError(
                "DeepSeek-V4 DSpark draft layers must all use SWA "
                f"(compress_ratio=0), got {self._v4_args.compress_ratios}"
            )
        if int(self._v4_args.window_size) <= 0:
            raise ValueError("DeepSeek-V4 DSpark requires a sliding window")

        self.init_dspark_proposer(
            width=int(self._gen_num_per_cycle),
            noise_token_id=int(noise_token_id),
            aux_feature_dim=len(self._dspark_target_layer_ids) * int(self._v4_args.dim),
            hidden_dim=int(self._v4_args.dim),
        )
        # Model-level weights are attached by ``_load_extra_weights`` after
        # the inherited V4Transformer has consumed the per-layer dictionaries.
        self.main_norm: Optional[RMSNorm] = None
        self.main_proj = None

        logging.info(
            "[DeepSeekV4DSparkModel] fixed gamma=%d noise=%d target_layers=%s "
            "markov_rank=%d window=%d",
            self._gen_num_per_cycle,
            self._dspark_noise_token_id,
            self._dspark_target_layer_ids,
            self._dspark_markov_rank,
            int(self._v4_args.window_size),
        )

    # ------------------------------------------------------------------
    # Initialization / graph policy
    # ------------------------------------------------------------------

    def cuda_graph_input_hidden_size(self) -> int:
        """Return the target-feature row width consumed by commit graphs."""

        return int(self._dspark_aux_feature_dim)

    def prepare_fmha_impl(
        self,
        inputs: PyModelInputs,
        is_cuda_graph: bool = False,
        cuda_graph_selection_mode: Optional[str] = None,
    ) -> Any:
        """Build the per-graph DSpARK metadata owner.

        DSpARK does not use the regular ``DSv4DecodeFmhaImpl`` metadata: its
        attention indices include the whole non-causal query block.  It only
        needs a persistent owner for FlashMLA's scheduler tensors.  The first
        call made while the stream is capturing recreates the scheduler inside
        the graph (via ``get_or_build_sched_meta``), so the schedule-build
        kernels replay with the current device-side ``topk_length`` values.
        """
        if cuda_graph_selection_mode == CudaGraphSelectionMode.PREFILL_GRAPH:
            raise PrefillCudaGraphUnsupportedBackend(
                "DeepSeek-V4 DSpARK attention does not support prefill CUDA Graph"
            )
        if not is_cuda_graph:
            return None
        return SimpleNamespace(
            sched_meta_cache={},
            prepare_cuda_graph=lambda _attention_inputs: None,
            support_cuda_graph=lambda: True,
        )

    def _load_extra_weights(self, weights: ModelWeights) -> None:
        gw = weights.global_weights
        self.main_norm = RMSNorm(
            gw[W.v4_dspark_main_norm], float(self._v4_args.norm_eps)
        )
        self.main_proj = _v4_fp8_linear(
            gw[W.v4_dspark_main_proj_w], gw[W.v4_dspark_main_proj_s]
        )

    # ------------------------------------------------------------------
    # DSparkProposerMixin hooks
    # ------------------------------------------------------------------

    def combine_hidden_states(self, features: torch.Tensor) -> torch.Tensor:
        assert self.main_norm is not None and self.main_proj is not None
        return self.main_norm(self.main_proj(features))

    # ------------------------------------------------------------------
    # Framework/paged-cache metadata helpers
    # ------------------------------------------------------------------

    def _swa_block_table(self, attention_inputs: Any, batch_size: int) -> torch.Tensor:
        """SWA_KV kernel block table for this forward, sliced to ``batch_size``.

        ``attention_inputs`` is ``PyModelInputs.attention_inputs``: a
        ``{tag: PyAttentionInputs}`` mapping for the multi-group DSV4 cache, or
        a single ``PyAttentionInputs`` when the cache collapses to one group.
        """
        block_tables = build_block_tables_for_tags(
            self.kv_cache, attention_inputs, (SWA_KV,)
        )
        table = (block_tables or {}).get(SWA_KV)
        if table is None:
            raise RuntimeError(
                "DSpark could not find the %s KV block table; available tags=%r"
                % (SWA_KV, list(as_attention_inputs_by_tag(attention_inputs)))
            )
        if int(table.shape[0]) < batch_size:
            raise RuntimeError(
                "DSpark SWA block table has fewer rows than the batch: "
                f"rows={table.shape[0]}, batch={batch_size}"
            )
        return table[:batch_size]

    @staticmethod
    def _global_pool_slots(
        block_table: torch.Tensor,
        req_ids: torch.Tensor,
        absolute_positions: torch.Tensor,
        entries_per_block: int,
        tokens_per_block: int,
    ) -> torch.Tensor:
        """Translate request-local positions to global paged-pool slots.

        This is the same mapping as ``compute_kv_pool_slot_mapping``, but
        accepts arbitrary request ids (needed for packed target-feature rows)
        and an arbitrary trailing position shape (needed for non-causal top-k).
        Negative requests/positions and unallocated block ids become ``-1``.
        """
        if absolute_positions.numel() == 0:
            return torch.empty_like(absolute_positions, dtype=torch.long)

        entries_per_block = int(entries_per_block)
        tokens_per_block = int(tokens_per_block)
        if entries_per_block <= 0 or tokens_per_block <= 0:
            raise ValueError(
                "DSpark paged-pool geometry must be positive, got "
                f"entries={entries_per_block}, tokens={tokens_per_block}"
            )

        positions = absolute_positions.to(torch.long)
        req = req_ids.to(device=positions.device, dtype=torch.long)
        while req.dim() < positions.dim():
            req = req.unsqueeze(-1)
        req = req.expand_as(positions)

        invalid = (positions < 0) | (req < 0)
        safe_req = req.clamp(0, max(int(block_table.shape[0]) - 1, 0))
        safe_pos = positions.clamp_min(0)
        block_in_request = safe_pos // tokens_per_block
        invalid = invalid | (block_in_request >= int(block_table.shape[1]))
        safe_block = block_in_request.clamp(0, max(int(block_table.shape[1]) - 1, 0))

        block_ids = block_table.to(torch.long)[safe_req, safe_block]
        invalid = invalid | (block_ids <= 0)
        slots = block_ids * entries_per_block + (safe_pos % entries_per_block)
        return torch.where(invalid, torch.full_like(slots, -1), slots)

    def _build_noncausal_indices(
        self,
        prefix_lengths: torch.Tensor,
        active_requests: torch.Tensor,
        block_table: torch.Tensor,
        entries_per_block: int,
        tokens_per_block: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Build latest-window + whole-query global slot ids.

        The physical index width is rounded to 128 for FlashMLA.  Valid
        entries are packed at the left, allowing ``topk_length`` to prevent
        the kernel from scanning padding.
        """
        batch_size = int(prefix_lengths.numel())
        gamma = self._gen_num_per_cycle
        window = int(self._v4_args.window_size)
        topk = ((window + gamma + 127) // 128) * 128
        device = prefix_lengths.device

        committed = torch.minimum(
            prefix_lengths.to(torch.long),
            torch.full_like(prefix_lengths, window, dtype=torch.long),
        )
        committed_start = prefix_lengths.to(torch.long) - committed
        columns = torch.arange(topk, device=device, dtype=torch.long).view(1, topk)
        in_context = columns < committed.view(batch_size, 1)
        in_query = columns < (committed + gamma).view(batch_size, 1)
        local_positions = torch.where(
            in_context,
            committed_start.view(batch_size, 1) + columns,
            torch.where(
                in_query,
                prefix_lengths.to(torch.long).view(batch_size, 1)
                + columns
                - committed.view(batch_size, 1),
                torch.full((batch_size, topk), -1, dtype=torch.long, device=device),
            ),
        )
        local_positions = torch.where(
            active_requests.to(torch.bool).view(batch_size, 1),
            local_positions,
            torch.full_like(local_positions, -1),
        )

        # Each query row sees the identical context+query set.  This is the
        # deliberate non-causal part of DSpark.
        local_positions = (
            local_positions.unsqueeze(1)
            .expand(batch_size, gamma, topk)
            .reshape(batch_size * gamma, topk)
        )
        req_ids = torch.arange(batch_size, device=device, dtype=torch.long)
        req_ids = req_ids.repeat_interleave(gamma)
        global_slots = self._global_pool_slots(
            block_table,
            req_ids,
            local_positions,
            entries_per_block,
            tokens_per_block,
        )
        topk_length = torch.where(
            active_requests.to(torch.bool),
            (committed + gamma).to(torch.int32),
            torch.zeros_like(committed, dtype=torch.int32),
        )
        return global_slots.to(torch.int32), topk_length

    # ------------------------------------------------------------------
    # DSpark attention / block forward
    # ------------------------------------------------------------------

    @contextmanager
    def _swa_cache_bound(
        self,
        layer_idx: int,
        block_table: torch.Tensor,
        cp_ctx: Any = None,
    ):
        """Temporarily point one draft layer's attention at this model's
        paged SWA cache and yield ``(attn, entries_per_block, pool)``.

        ``entries_per_block`` is the full ring-entry domain of one block
        (both CP slices together under a sharded prefill); ``pool`` is the
        packed 3-D FP8 view used by the regular writer. Byte-sliced commits
        also require it for layout validation, then write through the raw
        per-rank pool selected by ``attn._swa_cp_byte_sliced()``."""
        attn = self.v4.layers[layer_idx].attn
        if int(attn.compress_ratio) != 0:
            raise RuntimeError(
                f"DSpark layer {layer_idx} is not SWA-only: "
                f"compress_ratio={attn.compress_ratio}"
            )
        with bind_attn_cache(
            attn,
            self.kv_cache,
            {SWA_KV: block_table},
            cp_ctx=cp_ctx if cp_ctx is not None else BIND_KEEP,
        ):
            attn._ensure_freqs_cis_bound()
            entries_per_block = int(attn._swa_entries_per_block())
            pool = attn._pool_view_3d_fp8(SWA_KV)
            if entries_per_block <= 0 or pool is None:
                raise RuntimeError(
                    f"DSpark layer {layer_idx} has no FP8 SWA paged pool"
                )
            yield attn, entries_per_block, pool

    def _commit_layer_features(
        self,
        layer_idx: int,
        main_x: torch.Tensor,
        context_req_ids: torch.Tensor,
        context_positions: torch.Tensor,
        committed_ends: torch.Tensor,
        block_table: torch.Tensor,
        tokens_per_block: int,
        batch_size: int,
        gathered_req_ids: Optional[torch.Tensor] = None,
        gathered_positions: Optional[torch.Tensor] = None,
        cp_ctx: Any = None,
    ) -> None:
        """Insert committed target features through one layer's own
        wkv/kv_norm/RoPE pipeline. Rows superseded by newer SWA ring
        generations carry a -1 slot and are ignored by the FP8 writer.

        Under CP prefill ``main_x``/``context_positions`` are the rank-local
        zigzag rows; the projected KV is all-gathered across the CP group
        (proj-then-gather: the per-row wkv/RoPE projection commutes with the
        row split, and the projected rows are much narrower than the raw
        feature rows). Replicated caches write the full gathered domain on
        every rank; byte-sharded caches write only the local slice.
        """
        with self._swa_cache_bound(layer_idx, block_table, cp_ctx=cp_ctx) as (
            attn,
            entries_per_block,
            pool,
        ):
            safe_context_pos = context_positions.to(torch.long).clamp_min(0)
            context_freqs = attn.freqs_cis.index_select(
                0, safe_context_pos
            ).contiguous()
            context_kv = fused_rmsnorm_rope(
                attn._lin(attn.wkv, main_x).contiguous(),
                attn.kv_norm,
                context_freqs,
                int(attn.rope_head_dim),
                eps=float(attn.eps),
            )
            if gathered_positions is not None:
                from rtp_llm.models_py.distributed.collective_torch import (
                    Group,
                    all_gather,
                )

                context_kv = all_gather(context_kv.contiguous(), group=Group.TP)
                context_req_ids = gathered_req_ids
                context_positions = gathered_positions
            context_rows = int(context_positions.numel())
            context_slots = compute_swa_slot_mapping_from_positions(
                block_table=block_table[:batch_size]
                .to(device=main_x.device, dtype=torch.int32)
                .contiguous(),
                req_id_per_token=context_req_ids,
                positions=context_positions,
                seq_lens=committed_ends,
                num_tokens=context_rows,
                pool_entries_per_block=entries_per_block,
                tokens_per_block_for_block_table=tokens_per_block,
                ring_entries=entries_per_block,
            )
            if attn._swa_cp_byte_sliced():
                # Sharded prefill: same full-domain slots, but each rank owns
                # only its byte slice of every block — route through the
                # target pool's CP-sliced FP8 writer (used by the target's
                # own SWA layers) instead of the plain 3-D writer.
                from rtp_llm.models_py.modules.dsv4.fp8 import (
                    _swa_kv_insert_triton as insert_ops,
                )

                compaction = attn._build_swa_cp_byte_compaction(
                    context_slots,
                    full_entries_per_block=entries_per_block,
                    validation_site=f"dspark.commit.layer_{layer_idx}",
                    negative_mode="skip_minus_one",
                )
                if compaction is None:
                    raise RuntimeError(
                        f"DSpark layer {layer_idx}: CP-sliced slot compaction "
                        "is unavailable"
                    )
                insert_ops.quantize_and_insert_k_cache_cp_byte_sliced(
                    context_kv.reshape(-1, int(attn.head_dim)).to(torch.bfloat16),
                    attn._pool_raw_u8(SWA_KV),
                    context_slots,
                    full_entries_per_block=entries_per_block,
                    cp_rank=int(attn._cp_ctx.cp_rank),
                    cp_size=int(attn._cp_ctx.cp_size),
                    compaction=compaction,
                )
            else:
                decode_write_swa_fp8(
                    kv=context_kv,
                    slot_mapping=context_slots,
                    swa_pool_3d=pool,
                    bsz=context_rows,
                    q_len=1,
                    head_dim=int(attn.head_dim),
                )

    def map_commit_rows(
        self,
        starts: torch.Tensor,
        lengths: torch.Tensor,
        committed_ends: torch.Tensor,
        row_count: int,
        inputs: PyModelInputs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """CP prefill hands the commit the target's rank-local zigzag rows;
        this forward went through the same standard CP handleInputs, so the
        row -> (request, position) map is derived from the CP metadata it
        attached — the packed ``starts``/``lengths`` layout only describes
        the non-CP row order."""
        attention_inputs = primary_attention_inputs(
            inputs.attention_inputs, self.kv_cache
        )
        cp_info = getattr(attention_inputs, "context_parallel_info", None)
        if not self._dspark_commit_cp_enabled or cp_info is None:
            return super().map_commit_rows(
                starts, lengths, committed_ends, row_count, inputs
            )
        padding_mask = optional_tensor(
            getattr(cp_info, "prefill_qkv_padding_mask", None)
        )
        if padding_mask is None or int(padding_mask.numel()) == 0:
            # No CP-split prefill stream in this forward (e.g. a dense
            # decode-phase commit): the packed layout applies unchanged.
            return super().map_commit_rows(
                starts, lengths, committed_ends, row_count, inputs
            )
        device = committed_ends.device
        cp_ctx = build_cp_context_for_forward(
            cp_info,
            self.tp_size,
            self.tp_rank,
            row_count,
            device,
            prefix_lengths=optional_tensor(
                getattr(attention_inputs, "prefix_lengths", None)
            ),
            kv_cache_sharded=self._dspark_kv_cache_sharded,
        )
        positions = cp_ctx.global_positions
        if int(positions.numel()) != int(row_count):
            raise RuntimeError(
                "DSpark CP commit row mismatch: buffer supplied "
                f"{row_count} rank-local rows, CPContext maps "
                f"{int(positions.numel())}"
            )
        valid = cp_ctx.local_is_real.to(positions.device)
        req = cp_ctx.req_id_per_token
        if req is None:
            req = torch.zeros_like(valid, dtype=torch.int32)
        req32 = req.to(device=positions.device, dtype=torch.int32)
        pos32 = positions.to(torch.int32)
        # Padding slots must never reach the ring: -1 rows are skipped by the
        # slot mapping / FP8 writer on every rank after the gather.
        req32 = torch.where(valid, req32, torch.full_like(req32, -1))
        pos32 = torch.where(valid, pos32, torch.full_like(pos32, -1))
        return req32.contiguous(), pos32.contiguous(), cp_ctx

    def commit_feature_rows(
        self,
        main_x: torch.Tensor,
        context_req_ids: torch.Tensor,
        context_positions: torch.Tensor,
        committed_ends: torch.Tensor,
        inputs: PyModelInputs,
        commit_ctx: Any = None,
    ) -> None:
        batch_size = int(committed_ends.numel())
        if batch_size == 0 or int(main_x.shape[0]) == 0:
            return
        attention_inputs = inputs.attention_inputs
        block_table = self._swa_block_table(attention_inputs, batch_size)
        tokens_per_block = int(require_pool_tokens_per_block(self.kv_cache, tag=SWA_KV))
        write_cache_store_impl = create_write_cache_store_impl(
            primary_attention_inputs(attention_inputs, self.kv_cache), self.kv_cache
        )
        gathered_req_ids: Optional[torch.Tensor] = None
        gathered_positions: Optional[torch.Tensor] = None
        if commit_ctx is not None:
            # Proj-then-gather: the row map is gathered once here; each
            # layer's projected KV is gathered inside _commit_layer_features.
            from rtp_llm.models_py.distributed.collective_torch import Group, all_gather

            gathered_req_ids = all_gather(context_req_ids.contiguous(), group=Group.TP)
            gathered_positions = all_gather(
                context_positions.contiguous(), group=Group.TP
            )
        for layer_idx in range(len(self.v4.layers)):
            self._commit_layer_features(
                layer_idx,
                main_x,
                context_req_ids,
                context_positions,
                committed_ends,
                block_table,
                tokens_per_block,
                batch_size,
                gathered_req_ids=gathered_req_ids,
                gathered_positions=gathered_positions,
                cp_ctx=commit_ctx,
            )
            # PD-separated prefill publishes each committed draft-layer
            # cache from the commit call: the published range is exactly
            # the committed rows, so proposal rows never enter the store's
            # block plan.
            if write_cache_store_impl is not None:
                write_cache_store_impl(self.kv_cache.get_layer_cache_groups(layer_idx))

    def _forward_dspark_attention(
        self,
        layer_idx: int,
        x: torch.Tensor,
        query_positions: torch.Tensor,
        prefix_lengths: torch.Tensor,
        active_requests: torch.Tensor,
        block_table: Optional[torch.Tensor],
        tokens_per_block: int,
        graph_metadata: Any,
    ) -> torch.Tensor:
        batch_size, gamma, _ = x.shape
        if batch_size == 0:
            return torch.empty_like(x)
        if block_table is None:
            raise RuntimeError("DSpark attention requires an SWA block table")

        with self._swa_cache_bound(layer_idx, block_table) as (
            attn,
            entries_per_block,
            pool,
        ):
            # Project and insert all query KVs before attention so each query
            # can read every other query position, including future noise.
            qkv = decode_compute_qkv(attn, x, query_positions.reshape(-1))
            query_req_ids = torch.arange(
                batch_size, device=x.device, dtype=torch.long
            ).repeat_interleave(gamma)
            query_slots = self._global_pool_slots(
                block_table,
                query_req_ids,
                query_positions.reshape(-1),
                entries_per_block,
                tokens_per_block,
            )
            decode_write_swa_fp8(
                kv=qkv.kv,
                slot_mapping=query_slots,
                swa_pool_3d=pool,
                bsz=batch_size,
                q_len=gamma,
                head_dim=int(attn.head_dim),
            )

            global_indices, topk_length = self._build_noncausal_indices(
                prefix_lengths,
                active_requests,
                block_table,
                entries_per_block,
                tokens_per_block,
            )
            topk = int(global_indices.shape[-1])
            global_indices = global_indices.view(batch_size, gamma, topk).contiguous()

            sched_meta = get_or_build_sched_meta(
                graph_metadata,
                batch_size=batch_size,
                q_len=gamma,
                num_heads=int(attn.n_heads),
                topk=topk,
                extra_attn_type=None,
            )
            output = attn._get_fp8_decode_op().forward(
                q=qkv.q,
                kv_cache=pool,
                attn_sink=attn.attn_sink,
                topk_idxs=global_indices,
                sched_meta=sched_meta,
                topk_length=topk_length,
            )
            return decode_output_proj(attn, output, qkv.freqs_cis, batch_size, gamma)

    def _forward_layers(
        self,
        query_ids: torch.Tensor,
        query_positions: torch.Tensor,
        prefix_lengths: torch.Tensor,
        active_requests: torch.Tensor,
        block_table: Optional[torch.Tensor],
        tokens_per_block: int,
        graph_metadata: Any,
    ) -> torch.Tensor:
        hidden = self.v4.embed(query_ids)
        hidden = hidden.unsqueeze(2).repeat(1, 1, self.v4.hc_mult, 1)

        # The hyper-connection choreography lives in Block.forward_decode;
        # only the attention call is substituted with the non-causal
        # fixed-block variant.
        for layer_idx, layer in enumerate(self.v4.layers):
            hidden = layer.forward_decode(
                hidden,
                attn_metadata=None,
                input_ids=query_ids,
                attn_fn=lambda x_pre, layer_idx=layer_idx: self._forward_dspark_attention(
                    layer_idx,
                    x_pre,
                    query_positions,
                    prefix_lengths,
                    active_requests,
                    block_table,
                    tokens_per_block,
                    graph_metadata,
                ),
            )

        return hidden

    # ------------------------------------------------------------------
    # Head / framework forward
    # ------------------------------------------------------------------

    def forward_query_block(
        self,
        query_ids: torch.Tensor,
        query_positions: torch.Tensor,
        prefix_lengths: torch.Tensor,
        active_requests: torch.Tensor,
        inputs: PyModelInputs,
        fmha_impl: Any,
    ) -> torch.Tensor:
        batch_size = int(query_ids.shape[0])
        attention_inputs = inputs.attention_inputs
        block_table = (
            self._swa_block_table(attention_inputs, batch_size)
            if batch_size > 0
            else None
        )
        tokens_per_block = int(require_pool_tokens_per_block(self.kv_cache, tag=SWA_KV))

        # Eager forwards get a fresh owner each round, while CUDA graph capture
        # receives the persistent object created by ``prepare_fmha_impl``.
        graph_metadata = (
            fmha_impl
            if fmha_impl is not None and hasattr(fmha_impl, "sched_meta_cache")
            else SimpleNamespace(sched_meta_cache={})
        )
        return self._forward_layers(
            query_ids,
            query_positions,
            prefix_lengths,
            active_requests,
            block_table,
            tokens_per_block,
            graph_metadata,
        )

    def compute_draft_hidden_states(self, hidden: torch.Tensor) -> torch.Tensor:
        batch_size, gamma = int(hidden.shape[0]), int(hidden.shape[1])
        dim = int(self._v4_args.dim)

        head_hidden = self.v4._hc_head_reduce(hidden).reshape(batch_size * gamma, dim)
        return self.v4.norm(head_hidden)

    def _forward_device(self) -> torch.device:
        if self.v4 is None:
            raise RuntimeError("DeepSeekV4DSparkModel is not initialized")
        device = self.v4.embed.weight.device
        if self.kv_cache is not None and not bool(self.fp8_kv_cache):
            raise RuntimeError("DeepSeekV4DSparkModel currently requires FP8 KV cache")
        return device

    @torch.inference_mode()
    def forward_propose(
        self, inputs: PyModelInputs, fmha_impl: Any = None
    ) -> PyModelOutputs:
        device = self._forward_device()
        # PyWrappedModel warmup intentionally has no KVCache.  Produce stable
        # shapes without invoking any paged-cache or FlashMLA kernels.
        if self.kv_cache is None:
            gamma = self._gen_num_per_cycle
            input_tokens = int(inputs.input_ids.numel())
            batch_size = max((input_tokens + gamma - 1) // gamma, 1)
            logging.warning(
                "[DeepSeekV4DSparkModel] proposal forward with kv_cache=None; "
                "returning warmup placeholders for batch=%d",
                batch_size,
            )
            return self.dspark_empty_outputs(batch_size, device)
        return self.run_propose_step(inputs, fmha_impl, device)

    @torch.inference_mode()
    def forward_commit(
        self, inputs: PyModelInputs, fmha_impl: Any = None
    ) -> PyModelOutputs:
        device = self._forward_device()
        if self.kv_cache is None:
            return PyModelOutputs(
                torch.zeros(
                    (0, int(self._v4_args.dim)),
                    dtype=torch.bfloat16,
                    device=device,
                )
            )
        return self.run_commit_step(inputs, device)

    def forward(self, inputs: PyModelInputs, fmha_impl: Any = None) -> PyModelOutputs:
        raise RuntimeError(
            "DeepSeekV4DSparkModel requires a fixed forward_propose or "
            "forward_commit entrypoint"
        )


__all__ = ["DeepSeekV4DSparkModel"]
