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
``GEN_NUM_PER_CIRCLE``. Decode CUDA graph capture/replay keeps that query width,
target-feature width, and FlashMLA scheduler metadata stable for each captured
batch size.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import Any, Optional, Tuple

import torch
import torch.nn.functional as F

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.model_loader.model_weight_info import ModelWeights
from rtp_llm.models_py.model_desc.deepseek_v4_model import DeepSeekV4Model
from rtp_llm.models_py.modules import RMSNorm
from rtp_llm.models_py.modules.dsv4._fused_rmsnorm_rope_triton import (
    fused_rmsnorm_rope,
)
from rtp_llm.models_py.modules.dsv4.attn_type import SWA_KV
from rtp_llm.models_py.modules.dsv4.fp8._kv_cache_utils import (
    require_pool_tokens_per_block,
)
from rtp_llm.models_py.modules.dsv4.fp8.decode.compute_qkv import (
    decode_compute_qkv,
)
from rtp_llm.models_py.modules.dsv4.fp8.decode.decode_attn_metadata import (
    get_or_build_sched_meta,
)
from rtp_llm.models_py.modules.dsv4.fp8.decode.output_proj import (
    decode_output_proj,
)
from rtp_llm.models_py.modules.dsv4.fp8.decode.write_swa import (
    decode_write_swa_fp8,
)
from rtp_llm.models_py.modules.dsv4.utils import _v4_fp8_linear
from rtp_llm.models_py.modules.factory.attention.common import (
    create_write_cache_store_impl,
)
from rtp_llm.ops.compute_ops import PyModelInputs, PyModelOutputs
from rtp_llm.utils.model_weight import W


class DeepSeekV4DSparkModel(DeepSeekV4Model):
    """Runtime-fixed-width DeepSeek-V4 DSpark proposer.

    Input contract (all token rows are request-major):

    * ``input_ids``: ``[B * gamma]`` query block.  Column zero is the anchor;
      the remaining columns are forced to the configured noise token here.
    * ``input_hiddens``: target auxiliary features, flattenable to
      ``[rows, len(target_layers) * hidden_size]``.
    * ``dspark_ctx_lengths`` / ``dspark_ctx_starts``: number and source-row
      start of newly committed feature rows for each request.
    * ``attention_inputs.prefix_lengths``: committed sequence length *after*
      those context rows and immediately before the query block.

    Output ``draft_tokens`` and ``draft_probs`` are respectively
    ``[B, gamma]`` and ``[B, gamma, vocab]``.
    """

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
                "DeepSeekV4DSparkModel requires a non-negative "
                "dspark_noise_token_id"
            )
        if not target_layer_ids:
            raise ValueError(
                "DeepSeekV4DSparkModel requires dspark_target_layer_ids"
            )
        if not markov_rank or int(markov_rank) <= 0:
            raise ValueError(
                "DeepSeekV4DSparkModel requires a positive dspark_markov_rank"
            )

        self._dspark_noise_token_id = int(noise_token_id)
        self._dspark_target_layer_ids = tuple(int(v) for v in target_layer_ids)
        self._dspark_markov_rank = int(markov_rank)

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

        self._dspark_aux_dim = (
            len(self._dspark_target_layer_ids) * int(self._v4_args.dim)
        )
        # Model-level weights are attached by ``_load_extra_weights`` after
        # the inherited V4Transformer has consumed the per-layer dictionaries.
        self.main_norm: Optional[RMSNorm] = None
        self.main_proj = None
        self.markov_w1: Optional[torch.Tensor] = None
        self.markov_w2: Optional[torch.Tensor] = None

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

    def _should_capture_cuda_graph(self, attn: Any, is_target_verify: bool) -> bool:
        # DSpARK always evaluates a fixed-width query block.  Accepted context
        # lengths remain device data in the persistent CudaGraphRunner input
        # buffers, so they do not change the captured launch topology.
        return True

    def prepare_fmha_impl(
        self, inputs: PyModelInputs, is_cuda_graph: bool = False
    ) -> Any:
        """Build the per-graph DSpARK metadata owner.

        DSpARK does not use the regular ``DSv4DecodeFmhaImpl`` metadata: its
        attention indices include the whole non-causal query block.  It only
        needs a persistent owner for FlashMLA's scheduler tensors.  The first
        call made while the stream is capturing recreates the scheduler inside
        the graph (via ``get_or_build_sched_meta``), so the schedule-build
        kernels replay with the current device-side ``topk_length`` values.
        """
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
        self.markov_w1 = gw[W.v4_dspark_markov_w1]
        self.markov_w2 = gw[W.v4_dspark_markov_w2]

        if tuple(self.markov_w1.shape) != (
            int(self._v4_args.vocab_size),
            self._dspark_markov_rank,
        ):
            raise ValueError(
                "unexpected DSpark markov_w1 shape: "
                f"{tuple(self.markov_w1.shape)}"
            )
        if tuple(self.markov_w2.shape) != tuple(self.markov_w1.shape):
            raise ValueError(
                "DSpark markov_w2 shape must match markov_w1, got "
                f"{tuple(self.markov_w2.shape)} vs {tuple(self.markov_w1.shape)}"
            )

    # ------------------------------------------------------------------
    # Framework/paged-cache metadata helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _optional_tensor(value: Any) -> Optional[torch.Tensor]:
        if value is None or not isinstance(value, torch.Tensor):
            return None
        return value if value.numel() > 0 else None

    def _swa_block_table(
        self, attention_inputs: Any, batch_size: int
    ) -> torch.Tensor:
        by_group = getattr(
            attention_inputs, "kv_cache_kernel_block_id_device_by_group", None
        )
        regions = getattr(self.kv_cache, "group_region_names", None)
        if by_group is None or regions is None:
            raise RuntimeError(
                "DSpark requires per-group KV block tables and group region names"
            )
        for group_id, region in enumerate(regions):
            if int(region) != int(SWA_KV) or group_id >= len(by_group):
                continue
            table = by_group[group_id]
            if table is not None and table.numel() > 0:
                if int(table.shape[0]) < batch_size:
                    raise RuntimeError(
                        "DSpark SWA block table has fewer rows than the batch: "
                        f"rows={table.shape[0]}, batch={batch_size}"
                    )
                return table[:batch_size]
        raise RuntimeError("DSpark could not find the SWA KV block table")

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
        safe_block = block_in_request.clamp(
            0, max(int(block_table.shape[1]) - 1, 0)
        )

        block_ids = block_table.to(torch.long)[safe_req, safe_block]
        invalid = invalid | (block_ids <= 0)
        slots = block_ids * entries_per_block + (safe_pos % entries_per_block)
        return torch.where(invalid, torch.full_like(slots, -1), slots)

    @staticmethod
    def _map_context_rows(
        starts: torch.Tensor,
        lengths: torch.Tensor,
        prefix_lengths: torch.Tensor,
        row_count: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Map source feature rows to request ids and absolute positions.

        Rows outside every ``[start, start + length)`` interval are padding
        from a dense target-verify output and receive ``(-1, -1)``.  Keeping
        this transform independent from the feature projection also makes its
        packed/dense layout semantics directly unit-testable on CPU.
        """
        device = starts.device
        if row_count == 0:
            return (
                torch.empty(0, dtype=torch.int32, device=device),
                torch.empty(0, dtype=torch.int32, device=device),
            )
        batch_size = int(starts.numel())
        if batch_size == 0:
            raise ValueError("cannot map non-empty DSpark context without requests")

        rows = torch.arange(row_count, device=device, dtype=torch.long)
        req = torch.searchsorted(starts.contiguous(), rows, right=True) - 1
        safe_req = req.clamp(0, batch_size - 1)
        valid = (req >= 0) & (rows >= starts[safe_req])
        valid = valid & (rows < starts[safe_req] + lengths[safe_req])

        local_offset = rows - starts[safe_req]
        positions = prefix_lengths[safe_req] - lengths[safe_req] + local_offset
        req = torch.where(valid, req, torch.full_like(req, -1))
        positions = torch.where(
            valid, positions, torch.full_like(positions, -1)
        )
        return req.to(torch.int32), positions.to(torch.int32)

    def _project_target_features(
        self,
        inputs: PyModelInputs,
        prefix_lengths: torch.Tensor,
        batch_size: int,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``main_x, req_ids, positions`` for received target rows.

        Explicit ``starts`` may describe a dense target-verify buffer with
        holes after each request's accepted prefix.  ``searchsorted`` maps all
        source rows to requests without a device-to-host length read; hole rows
        retain a ``-1`` request/position and are skipped by the cache writer.
        """
        assert self.main_norm is not None and self.main_proj is not None
        hidden = self._optional_tensor(getattr(inputs, "input_hiddens", None))
        lengths = self._optional_tensor(
            getattr(inputs, "dspark_ctx_lengths", None)
        )
        if batch_size == 0:
            return (
                torch.empty(
                    (0, int(self._v4_args.dim)),
                    dtype=torch.bfloat16,
                    device=device,
                ),
                torch.empty(0, dtype=torch.int32, device=device),
                torch.empty(0, dtype=torch.int32, device=device),
            )
        if hidden is None:
            raise RuntimeError("DSpark requires target features in input_hiddens")
        if lengths is None or int(lengths.numel()) < batch_size:
            raise RuntimeError(
                "DSpark requires dspark_ctx_lengths with one value per request"
            )

        lengths = lengths[:batch_size].to(device=device, dtype=torch.long)
        starts_in = self._optional_tensor(getattr(inputs, "dspark_ctx_starts", None))
        if starts_in is None:
            starts = lengths.cumsum(0) - lengths
        else:
            if int(starts_in.numel()) < batch_size:
                raise RuntimeError(
                    "DSpark dspark_ctx_starts has fewer values than the batch"
                )
            starts = starts_in[:batch_size].to(device=device, dtype=torch.long)

        if hidden.numel() % self._dspark_aux_dim != 0:
            raise RuntimeError(
                "DSpark target feature tensor cannot be reshaped to the "
                f"configured width {self._dspark_aux_dim}: "
                f"shape={tuple(hidden.shape)}"
            )
        features = hidden.reshape(-1, self._dspark_aux_dim).to(device=device)
        row_count = int(features.shape[0])
        if row_count == 0:
            return (
                torch.empty(
                    (0, int(self._v4_args.dim)),
                    dtype=torch.bfloat16,
                    device=device,
                ),
                torch.empty(0, dtype=torch.int32, device=device),
                torch.empty(0, dtype=torch.int32, device=device),
            )

        main_x = self.main_norm(self.main_proj(features))
        req, positions = self._map_context_rows(
            starts,
            lengths,
            prefix_lengths,
            row_count,
        )
        return main_x, req, positions

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
                torch.full(
                    (batch_size, topk), -1, dtype=torch.long, device=device
                ),
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

    def _forward_dspark_attention(
        self,
        layer_idx: int,
        x: torch.Tensor,
        query_positions: torch.Tensor,
        main_x: torch.Tensor,
        context_req_ids: torch.Tensor,
        context_positions: torch.Tensor,
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

        layer = self.v4.layers[layer_idx]
        attn = layer.attn
        if int(attn.compress_ratio) != 0:
            raise RuntimeError(
                f"DSpark layer {layer_idx} is not SWA-only: "
                f"compress_ratio={attn.compress_ratio}"
            )

        previous_cache = attn._kv_cache
        previous_tables = attn._block_tables_by_type
        attn._kv_cache = self.kv_cache
        attn._block_tables_by_type = {int(SWA_KV): block_table}
        try:
            attn._ensure_freqs_cis_bound()
            entries_per_block = int(attn._pool_entries_per_block(SWA_KV))
            pool = attn._pool_view_3d_fp8(SWA_KV)
            if entries_per_block <= 0 or pool is None:
                raise RuntimeError(
                    f"DSpark layer {layer_idx} has no FP8 SWA paged pool"
                )

            # Insert newly committed target features through this layer's own
            # wkv/kv_norm/RoPE pipeline.  Hole rows carry a -1 slot and are
            # ignored by the FP8 writer.
            context_rows = int(main_x.shape[0])
            if context_rows > 0:
                safe_context_pos = context_positions.to(torch.long).clamp_min(0)
                context_freqs = (
                    attn.freqs_cis.index_select(0, safe_context_pos).contiguous()
                )
                context_kv = fused_rmsnorm_rope(
                    attn._lin(attn.wkv, main_x).contiguous(),
                    attn.kv_norm,
                    context_freqs,
                    int(attn.rope_head_dim),
                    eps=float(attn.eps),
                )
                context_slots = self._global_pool_slots(
                    block_table,
                    context_req_ids,
                    context_positions,
                    entries_per_block,
                    tokens_per_block,
                )
                decode_write_swa_fp8(
                    kv=context_kv,
                    slot_mapping=context_slots,
                    swa_pool_3d=pool,
                    bsz=context_rows,
                    q_len=1,
                    head_dim=int(attn.head_dim),
                )

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
            global_indices = global_indices.view(
                batch_size, gamma, topk
            ).contiguous()

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
            return decode_output_proj(
                attn, output, qkv.freqs_cis, batch_size, gamma
            )
        finally:
            attn._kv_cache = previous_cache
            attn._block_tables_by_type = previous_tables

    def _forward_layers(
        self,
        query_ids: torch.Tensor,
        query_positions: torch.Tensor,
        main_x: torch.Tensor,
        context_req_ids: torch.Tensor,
        context_positions: torch.Tensor,
        prefix_lengths: torch.Tensor,
        active_requests: torch.Tensor,
        block_table: Optional[torch.Tensor],
        tokens_per_block: int,
        graph_metadata: Any,
        write_cache_store_impl: Any = None,
    ) -> torch.Tensor:
        batch_size, gamma = query_ids.shape
        hidden = self.v4.embed(query_ids)
        hidden = hidden.unsqueeze(2).repeat(1, 1, self.v4.hc_mult, 1)

        for layer_idx, layer in enumerate(self.v4.layers):
            residual = hidden
            x_pre, post, comb = layer.attn_hc.pre(hidden)
            x_pre = layer.attn_norm(
                x_pre.reshape(batch_size * gamma, int(self._v4_args.dim))
            ).view(batch_size, gamma, int(self._v4_args.dim))
            attention_output = self._forward_dspark_attention(
                layer_idx,
                x_pre,
                query_positions,
                main_x,
                context_req_ids,
                context_positions,
                prefix_lengths,
                active_requests,
                block_table,
                tokens_per_block,
                graph_metadata,
            )
            hidden = layer.attn_hc.post(
                attention_output, residual, post, comb
            )

            residual = hidden
            x_pre, post, comb = layer.ffn_hc.pre(hidden)
            x_pre = layer.ffn_norm(
                x_pre.reshape(batch_size * gamma, int(self._v4_args.dim))
            ).view(batch_size, gamma, int(self._v4_args.dim))
            ffn_output = layer.ffn(x_pre, query_ids)
            hidden = layer.ffn_hc.post(ffn_output, residual, post, comb)

            # The custom non-causal attention path bypasses the ordinary V4
            # prefill loop, so it must explicitly publish each completed
            # draft-layer cache in PD-separated prefill.  The C++ input
            # overrides describe the newly injected context plus this fixed
            # query block; decode receives exactly those SWA cache entries.
            if write_cache_store_impl is not None:
                write_cache_store_impl(self.kv_cache.get_layer_caches(layer_idx))

        return hidden

    # ------------------------------------------------------------------
    # Head / framework forward
    # ------------------------------------------------------------------

    def _empty_outputs(self, batch_size: int, device: torch.device) -> PyModelOutputs:
        gamma = self._gen_num_per_cycle
        dim = int(self._v4_args.dim)
        vocab = int(self._v4_args.vocab_size)
        outputs = PyModelOutputs(
            torch.zeros(
                (batch_size * gamma, dim),
                dtype=torch.bfloat16,
                device=device,
            )
        )
        outputs.draft_tokens = torch.zeros(
            (batch_size, gamma), dtype=torch.int32, device=device
        )
        outputs.draft_probs = torch.zeros(
            (batch_size, gamma, vocab), dtype=torch.float32, device=device
        )
        return outputs

    def _forward_head(
        self, hidden: torch.Tensor, anchors: torch.Tensor
    ) -> PyModelOutputs:
        assert self.markov_w1 is not None and self.markov_w2 is not None
        batch_size, gamma = int(hidden.shape[0]), int(hidden.shape[1])
        dim = int(self._v4_args.dim)

        head_hidden = self.v4._hc_head_reduce(hidden).reshape(
            batch_size * gamma, dim
        )
        # PyModelOutputs.hidden_states convention is post-final-norm and
        # pre-lm-head.  Reuse it for the base DSpark logits and for the
        # framework's (unused) regular logits output.
        normalized = self.v4.norm(head_hidden)
        base_logits = torch.mm(
            normalized.to(self.v4.head_weight.dtype), self.v4.head_weight.t()
        ).float()
        base_logits = base_logits.view(batch_size, gamma, -1)

        previous = anchors
        tokens = []
        probabilities = []
        for step in range(gamma):
            markov_embed = F.embedding(previous, self.markov_w1)
            markov_bias = F.linear(markov_embed, self.markov_w2).float()
            logits = base_logits[:, step] + markov_bias
            probs = torch.softmax(logits, dim=-1, dtype=torch.float32)
            next_token = torch.argmax(probs, dim=-1)
            probabilities.append(probs)
            tokens.append(next_token.to(torch.int32))
            previous = next_token

        outputs = PyModelOutputs(normalized)
        outputs.draft_tokens = torch.stack(tokens, dim=1).contiguous()
        outputs.draft_probs = torch.stack(probabilities, dim=1).contiguous()
        return outputs

    @torch.inference_mode()
    def forward(
        self, inputs: PyModelInputs, fmha_impl: Any = None
    ) -> PyModelOutputs:
        if self.v4 is None:
            raise RuntimeError("DeepSeekV4DSparkModel is not initialized")
        device = self.v4.embed.weight.device
        gamma = self._gen_num_per_cycle

        # PyWrappedModel warmup intentionally has no KVCache.  Produce stable
        # shapes without invoking any paged-cache or FlashMLA kernels.
        if self.kv_cache is None:
            input_tokens = int(inputs.input_ids.numel())
            batch_size = max((input_tokens + gamma - 1) // gamma, 1)
            logging.warning(
                "[DeepSeekV4DSparkModel] forward with kv_cache=None; "
                "returning warmup placeholders for batch=%d",
                batch_size,
            )
            return self._empty_outputs(batch_size, device)

        if not bool(self.fp8_kv_cache):
            raise RuntimeError(
                "DeepSeekV4DSparkModel currently requires FP8 KV cache"
            )

        attention_inputs = inputs.attention_inputs
        input_lengths = self._optional_tensor(
            getattr(attention_inputs, "input_lengths", None)
        )
        batch_size = int(input_lengths.numel()) if input_lengths is not None else 0
        expected_tokens = batch_size * gamma
        if int(inputs.input_ids.numel()) != expected_tokens:
            raise RuntimeError(
                "DSpark input_ids must contain exactly B*gamma tokens: "
                f"numel={inputs.input_ids.numel()}, batch={batch_size}, "
                f"gamma={gamma}"
            )

        prefix = self._optional_tensor(
            getattr(attention_inputs, "prefix_lengths", None)
        )
        if batch_size > 0 and (prefix is None or int(prefix.numel()) < batch_size):
            raise RuntimeError(
                "DSpark requires prefix_lengths with one value per request"
            )
        prefix_lengths = (
            prefix[:batch_size].to(device=device, dtype=torch.int32)
            if prefix is not None
            else torch.empty(0, dtype=torch.int32, device=device)
        )

        raw_ids = inputs.input_ids.to(device=device, dtype=torch.int32).view(
            batch_size, gamma
        )
        anchors = raw_ids[:, 0].clone()
        query_ids = torch.full_like(raw_ids, self._dspark_noise_token_id)
        if batch_size > 0:
            query_ids[:, 0].copy_(anchors)
        query_positions = prefix_lengths.to(torch.long).view(batch_size, 1)
        query_positions = query_positions + torch.arange(
            gamma, device=device, dtype=torch.long
        ).view(1, gamma)

        main_x, context_req_ids, context_positions = self._project_target_features(
            inputs, prefix_lengths.to(torch.long), batch_size, device
        )
        ctx_lengths = self._optional_tensor(getattr(inputs, "dspark_ctx_lengths", None))
        active_requests = (
            ctx_lengths[:batch_size].to(device=device) > 0
            if ctx_lengths is not None
            else torch.zeros(batch_size, dtype=torch.bool, device=device)
        )
        block_table = (
            self._swa_block_table(attention_inputs, batch_size)
            if batch_size > 0
            else None
        )
        tokens_per_block = int(
            require_pool_tokens_per_block(self.kv_cache, region=int(SWA_KV))
        )
        write_cache_store_impl = create_write_cache_store_impl(
            attention_inputs, self.kv_cache
        )

        # Eager forwards get a fresh owner each round, while CUDA graph capture
        # receives the persistent object created by ``prepare_fmha_impl``.
        graph_metadata = (
            fmha_impl
            if fmha_impl is not None and hasattr(fmha_impl, "sched_meta_cache")
            else SimpleNamespace(sched_meta_cache={})
        )
        hidden = self._forward_layers(
            query_ids,
            query_positions,
            main_x,
            context_req_ids,
            context_positions,
            prefix_lengths,
            active_requests,
            block_table,
            tokens_per_block,
            graph_metadata,
            write_cache_store_impl,
        )

        # Empty DP ranks must still execute every MoE layer above so EP
        # collectives remain balanced; only the non-collective head is skipped.
        if batch_size == 0:
            return self._empty_outputs(0, device)
        return self._forward_head(hidden, anchors)


__all__ = ["DeepSeekV4DSparkModel"]
