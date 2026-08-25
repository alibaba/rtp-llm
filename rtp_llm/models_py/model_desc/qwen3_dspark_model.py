"""Qwen3 DSpark backbone on the shared DSpark proposer contract."""

from typing import Any

import torch
from torch import nn

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.model_loader.model_weight_info import ModelWeights
from rtp_llm.models_py.model_desc.block_map import (
    select_attention_inputs_for_layer,
)
from rtp_llm.models_py.model_desc.qwen3 import Qwen3Model
from rtp_llm.models_py.distributed.collective_torch import Group, all_gather
from rtp_llm.models_py.modules import LinearFactory, RMSNorm
from rtp_llm.models_py.modules.factory.attention.common import (
    create_write_cache_store_impl,
)
from rtp_llm.models_py.speculative.dspark_proposer_mixin import (
    DSparkProposerMixin,
    graph_captured_greedy_markov_decode,
)
from rtp_llm.ops import ParallelismConfig
from rtp_llm.ops.compute_ops import (
    DSparkCallPhase,
    PyModelInputs,
    PyModelOutputs,
)
from rtp_llm.utils.model_weight import W


class _RopePositions:
    def __init__(self, positions: torch.Tensor) -> None:
        self.positions_d = positions


class Qwen3DSparkModel(DSparkProposerMixin, Qwen3Model):
    def __init__(
        self,
        config: ModelConfig,
        parallelism_config: ParallelismConfig,
        weights: ModelWeights,
        max_generate_batch_size: int,
        quant_config=None,
        fmha_config=None,
        py_hw_kernel_config=None,
        device_resource_config=None,
    ) -> None:
        if quant_config is not None:
            raise NotImplementedError("Qwen3 DSpark quantization is not supported")
        super().__init__(
            config,
            parallelism_config,
            weights,
            max_generate_batch_size=max_generate_batch_size,
            quant_config=quant_config,
            fmha_config=fmha_config,
            py_hw_kernel_config=py_hw_kernel_config,
            device_resource_config=device_resource_config,
        )
        proposal_width = int(config.gen_num_per_cycle)
        query_width = proposal_width + int(not config.dspark_sample_from_anchor)
        self.init_dspark_proposer(
            width=proposal_width,
            query_width=query_width,
            noise_token_id=config.dspark_noise_token_id,
            aux_feature_dim=len(config.dspark_target_layer_ids)
            * config.hidden_size,
            hidden_dim=config.hidden_size,
        )

        self.attn_configs = config.getAttentionConfigs(
            parallelism_config.get_attn_tp_size()
        )
        if self.attn_configs.is_causal:
            raise ValueError("Qwen3 DSpark proposal attention must be non-causal")
        self.fc = LinearFactory.create_linear_from_weights(
            weights.global_weights, W.dspark_fc_w
        )
        self.hidden_norm = RMSNorm(
            weights.get_global_weight(W.dspark_hidden_norm_gamma),
            eps=config.layernorm_eps,
        )

        self._dspark_proposal_driver = parallelism_config.tp_rank == 0
        lm_head = weights.get_global_weight(W.lm_head).contiguous()
        if parallelism_config.tp_size > 1:
            # The serving target may be TP2 while the online fusion kernel
            # is vocab-local. Reconstruct and retain the small 20k draft
            # head once at initialization; steady-state proposal replay
            # then has no vocab all-gather or host synchronization.
            lm_head = all_gather(lm_head, group=Group.TP)
        self._dspark_lm_head = (
            lm_head[: int(config.vocab_size)].contiguous()
            if self._dspark_proposal_driver
            else None
        )
        self._dspark_markov_w1 = None
        self._dspark_markov_w2 = None
        self._dspark_d2t = None
        if self._dspark_proposal_driver:
            self._dspark_markov_w1 = weights.get_global_weight(
                W.dspark_markov_w1
            ).contiguous()
            raw_markov_w2 = weights.get_global_weight(
                W.dspark_markov_w2
            ).contiguous()
            draft_vocab_size = int(config.vocab_size)
            padded_vocab_size = (draft_vocab_size + 127) // 128 * 128
            if (
                self._dspark_markov_w1.dim() != 2
                or raw_markov_w2.dim() != 2
                or self._dspark_markov_w1.shape[1] != raw_markov_w2.shape[1]
            ):
                raise ValueError("DSpARK Markov weights must have the same rank")
            if not draft_vocab_size <= raw_markov_w2.shape[0] <= padded_vocab_size:
                raise ValueError(
                    "DSpARK Markov W2 rows must cover exactly the draft "
                    "vocabulary, with at most one 128-token padding tile"
                )
            if self._dspark_markov_w1.shape[0] < int(config.input_vocab_size):
                raise ValueError(
                    "DSpARK Markov W1 must cover the target/input vocabulary"
                )
            # The generic linear consumes the logical draft vocabulary. A
            # checkpoint may retain one alignment tile physically; discard it
            # once at load rather than carrying stride-specific serving logic.
            self._dspark_markov_w2 = raw_markov_w2[:draft_vocab_size].contiguous()

            d2t = weights.get_global_weight_or_none(
                W.multi_tokens_predict_d2t_map
            )
            if int(config.input_vocab_size) != draft_vocab_size and d2t is None:
                raise ValueError(
                    "reduced-vocabulary DSpARK requires an absolute d2t map"
                )
            self._dspark_d2t = (
                torch.arange(
                    draft_vocab_size,
                    dtype=torch.int64,
                    device=self._dspark_markov_w1.device,
                )
                if d2t is None
                else d2t.to(dtype=torch.int64).contiguous()
            )
            if self._dspark_d2t.shape != (draft_vocab_size,):
                raise ValueError("DSpARK d2t must cover the draft vocabulary")

        heads = self.attn_configs.head_num
        kv_heads = self.attn_configs.kv_head_num
        head_dim = self.attn_configs.size_per_head
        q_cols = heads * head_dim
        context_kv_weights = []
        self.context_k_norms = nn.ModuleList()
        for layer_weights in weights.weights[: self.layer_num]:
            qkv = layer_weights[W.attn_qkv_w]
            context_kv_weights.append(qkv[:, q_cols:])
            self.context_k_norms.append(
                RMSNorm(layer_weights[W.k_ln_gamma], eps=config.layernorm_eps)
            )
        # Context features are identical for every draft layer. Project every
        # layer's K/V in one GEMM, matching the DSpARK/DFlash runtime design
        # and avoiding five launch-bound small GEMMs for the current model.
        self.context_kv_projection = LinearFactory.create_linear(
            torch.cat(context_kv_weights, dim=1),
            None,
            None,
            None,
            py_hw_kernel_config,
        )

        from rtp_llm.models_py.modules.factory.attention.cuda_impl.flashinfer_rotary_emb import (
            MhaRotaryEmbeddingOp,
        )

        self.context_rope = MhaRotaryEmbeddingOp(self.attn_configs)

    def cuda_graph_input_hidden_size(self) -> int:
        """Width of the concatenated target auxiliary features per token."""
        return self._dspark_aux_feature_dim

    def combine_hidden_states(self, features: torch.Tensor) -> torch.Tensor:
        return self.fc(features)

    def build_proposal_outputs(
        self, hidden: torch.Tensor, query_ids: torch.Tensor
    ) -> PyModelOutputs:
        head_hidden = self.compute_draft_hidden_states(hidden)
        if not self._dspark_proposal_driver:
            # Target verification input is assembled and TP-synchronized by
            # the C++ driver rank.  Other TP ranks still execute the sharded
            # draft backbone/collectives, but must not duplicate the full
            # 20k-vocabulary LM head and sequential Markov kernel.
            if self.config.dspark_sample_from_anchor:
                placeholder = query_ids[:, : self._dspark_width]
            else:
                placeholder = query_ids[:, 1 : self._dspark_width + 1]
            outputs = PyModelOutputs(head_hidden)
            outputs.speculative_token_ids = placeholder
            return outputs

        batch_size = int(query_ids.shape[0])
        shaped_hidden = head_hidden.view(
            batch_size, self._dspark_query_width, self._dspark_hidden_dim
        )
        if self.config.dspark_sample_from_anchor:
            sample_hidden = shaped_hidden[:, : self._dspark_width]
        else:
            sample_hidden = shaped_hidden[:, 1 : self._dspark_width + 1]
        # The full-vocabulary LM head and dependent dense Markov chain are
        # captured together in the proposal CUDA graph. Replay therefore has
        # no Python scheduling or eager fallback while keeping one portable
        # implementation for every graph bucket and supported GPU.
        base_logits = torch.nn.functional.linear(
            sample_hidden.contiguous(), self._dspark_lm_head
        )
        proposal_ids = graph_captured_greedy_markov_decode(
            base_logits,
            query_ids[:, 0],
            self._dspark_markov_w1,
            self._dspark_markov_w2,
            self._dspark_d2t,
        )
        outputs = PyModelOutputs(head_hidden)
        outputs.speculative_token_ids = proposal_ids
        return outputs

    def dspark_attention_inputs(self, inputs: PyModelInputs):
        attention = select_attention_inputs_for_layer(inputs, self.kv_cache, 0)
        if isinstance(attention, list):
            if len(attention) != 1:
                raise RuntimeError(
                    "Qwen3 DSpark commit requires exactly one KV cache group "
                    f"per draft layer, got {len(attention)}"
                )
            attention = attention[0]
        return attention

    def _block_table(self, inputs: PyModelInputs) -> torch.Tensor:
        attention = self.dspark_attention_inputs(inputs)
        # The draft KV tensor is indexed by the attention kernel's local page
        # ids.  ``kv_cache_block_id_device`` is the physical/cache-store table
        # in main's split cache API and can contain ids outside this tensor.
        table = attention.kv_cache_kernel_block_id_device
        if table is None or table.numel() == 0:
            table = attention.kv_cache_kernel_block_id.to(
                device=self.embed_tokens.weight.device, non_blocking=True
            )
        return table[0] if table.dim() == 3 else table

    def commit_feature_rows(
        self,
        main_x: torch.Tensor,
        context_req_ids: torch.Tensor,
        context_positions: torch.Tensor,
        committed_ends: torch.Tensor,
        inputs: PyModelInputs,
        commit_ctx: Any = None,
    ) -> None:
        del committed_ends, commit_ctx
        table = self._block_table(inputs)
        page_size = self.attn_configs.kernel_tokens_per_block
        positions = context_positions.long()
        pages = table[context_req_ids.long(), positions // page_size].long()
        slots = positions % page_size
        hidden = self.hidden_norm(main_x)
        head_dim = self.attn_configs.size_per_head
        kv_heads = self.attn_configs.kv_head_num
        dummy_q = hidden.new_zeros((hidden.shape[0], 1, head_dim))

        all_kv = self.context_kv_projection(hidden).view(
            -1, self.layer_num, 2, kv_heads, head_dim
        )
        for layer_idx in range(self.layer_num):
            key, value = all_kv[:, layer_idx].unbind(1)
            key = self.context_k_norms[layer_idx](key.reshape(-1, head_dim)).view(
                -1, kv_heads, head_dim
            )
            self.context_rope._apply_rope(
                dummy_q, key, _RopePositions(context_positions)
            )
            cache = self.kv_cache.get_layer_cache(layer_idx).kv_cache_base
            cache[pages, 0, :, slots, :] = key.to(cache.dtype)
            cache[pages, 1, :, slots, :] = value.to(cache.dtype)

        writer = create_write_cache_store_impl(
            self.dspark_attention_inputs(inputs)
        )
        if writer is not None:
            for layer_idx in range(self.layer_num):
                layer_caches = self.kv_cache.get_layer_cache_groups(layer_idx)
                if len(layer_caches) != 1:
                    raise RuntimeError(
                        "Qwen3 DSpark commit requires exactly one KV cache "
                        f"group per draft layer, got {len(layer_caches)} for "
                        f"layer {layer_idx}"
                    )
                writer(layer_caches[0])

    def forward_query_block(
        self,
        query_ids: torch.Tensor,
        query_positions: torch.Tensor,
        prefix_lengths: torch.Tensor,
        active_requests: torch.Tensor,
        inputs: PyModelInputs,
        fmha_impl: Any,
    ) -> torch.Tensor:
        del query_ids, query_positions, prefix_lengths, active_requests
        return super().forward(inputs, fmha_impl).hidden_states

    @torch.inference_mode()
    def forward(self, inputs: PyModelInputs, fmha_impl: Any = None) -> PyModelOutputs:
        phase = getattr(inputs, "dspark_call_phase", DSparkCallPhase.NONE)
        if phase == DSparkCallPhase.NONE:
            raise RuntimeError("Qwen3 DSpark requires an explicit phase")
        device = self.embed_tokens.weight.device
        if self.kv_cache is None:
            if phase == DSparkCallPhase.COMMIT:
                return PyModelOutputs(
                    torch.empty(
                        (0, self.config.hidden_size),
                        dtype=torch.bfloat16,
                        device=device,
                    )
                )
            tokens = int(inputs.input_ids.numel())
            batch = max(
                (tokens + self._dspark_query_width - 1)
                // self._dspark_query_width,
                1,
            )
            return self.dspark_empty_outputs(batch, device)
        if phase == DSparkCallPhase.COMMIT:
            return self.run_commit_step(inputs, device)
        return self.run_propose_step(inputs, fmha_impl, device)


__all__ = ["Qwen3DSparkModel"]
