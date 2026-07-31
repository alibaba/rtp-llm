"""DeepSeek-V4 DSpark block-diffusion draft model.

This is the DSV4-specific sibling of the Qwen DFlash/DSpark implementation.
The checkpoint contract differs in two load-bearing ways:

* three ordinary V4 blocks are stored under ``mtp.{0,1,2}``;
* ``sample_from_anchor`` uses exactly ``k`` query rows (anchor + ``k-1``
  noise rows), and all rows predict a draft token.  The checkpoint's
  ``dspark_block_size`` is that prediction count, not anchor-plus-predictions.

The implementation is deliberately a thin composition over
:class:`DeepSeekV4Model`: target features are combined and written into each
draft layer's existing SWA paged cache, the inherited V4 layer loop executes
the query block, and a replicated low-rank Markov head corrects the logits.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional

import torch
import torch.nn.functional as F
from torch import nn

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.model_loader.model_weight_info import ModelWeights
from rtp_llm.models_py.model_desc.deepseek_v4_model import DeepSeekV4Model
from rtp_llm.models_py.modules import RMSNorm
from rtp_llm.models_py.modules.dsv4.attn_type import SWA_KV
from rtp_llm.models_py.modules.dsv4.fp8._kv_cache_utils import (
    require_pool_tokens_per_block,
)
from rtp_llm.models_py.modules.dsv4.kv_cache_utils import build_block_tables_batched
from rtp_llm.models_py.modules.dsv4.utils import _v4_fp8_linear_from_dict
from rtp_llm.ops.compute_ops import PyModelInputs, PyModelOutputs
from rtp_llm.utils.model_weight import W


@dataclass(frozen=True)
class DeepSeekV4DSparkParams:
    """DSpark fields embedded in the 0731 target checkpoint config."""

    target_layer_ids: List[int]
    mask_token_id: int
    speculative_tokens: int
    block_size: int
    markov_rank: int
    proposal_type: str = "greedy"
    sample_from_anchor: bool = True

    @classmethod
    def from_ckpt_config(cls, cfg: dict) -> "DeepSeekV4DSparkParams":
        block_size = int(cfg["dspark_block_size"])
        params = cls(
            target_layer_ids=[int(x) for x in cfg["dspark_target_layer_ids"]],
            mask_token_id=int(cfg["dspark_noise_token_id"]),
            speculative_tokens=block_size,
            block_size=block_size,
            markov_rank=int(cfg["dspark_markov_rank"]),
            proposal_type=str(cfg.get("dspark_proposal_type") or "greedy"),
            sample_from_anchor=bool(cfg.get("sample_from_anchor", True)),
        )
        if not params.target_layer_ids:
            raise ValueError("dspark_target_layer_ids must not be empty")
        if params.block_size < 2:
            raise ValueError(f"invalid dspark_block_size={params.block_size}")
        if params.markov_rank < 1:
            raise ValueError(f"invalid dspark_markov_rank={params.markov_rank}")
        if not params.sample_from_anchor:
            raise ValueError("DSV4 checkpoint requires sample_from_anchor=true")
        return params

    @property
    def aux_hidden_state_layer_ids(self) -> List[int]:
        """Generic engine-facing spelling used by ModelFactory."""
        return self.target_layer_ids

    @property
    def block_width(self) -> int:
        # Official DSV4 layout: anchor + (k-1) noise rows, all k predict;
        # dspark_block_size is k (vLLM rejects a smaller speculative length).
        return self.speculative_tokens


@dataclass
class DeepSeekV4DSparkProposal:
    draft_tokens: torch.Tensor
    corrected_logits: torch.Tensor
    base_logits: torch.Tensor
    head_hidden: torch.Tensor


class DSparkMarkovHead(nn.Module):
    """Replicated low-rank transition head kept in checkpoint BF16."""

    def __init__(self, markov_w1: torch.Tensor, markov_w2: torch.Tensor):
        super().__init__()
        self.markov_w1 = markov_w1
        self.markov_w2 = markov_w2

    def bias(self, previous_tokens: torch.Tensor) -> torch.Tensor:
        embedded = self.markov_w1[previous_tokens.long()]
        return F.linear(embedded, self.markov_w2).float()


class DeepSeekV4DSparkModel(DeepSeekV4Model):
    """Three-layer DSV4 DSpark draft with feature-KV injection."""

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
    ):
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
        assert model_config.dspark_config is not None
        self.dspark_params: DeepSeekV4DSparkParams = model_config.dspark_config
        assert self._v4_args.n_layers == len(self.dspark_params.target_layer_ids)
        self._v4_args.n_hash_layers = 0
        self._v4_args.compress_ratios = [0] * self._v4_args.n_layers
        self.tp_size = int(getattr(parallelism_config, "tp_size", 1) or 1)

        self.main_proj = None
        self.main_norm: Optional[RMSNorm] = None
        self.markov_head: Optional[DSparkMarkovHead] = None

    def _should_capture_cuda_graph(self, attn: Any, is_target_verify: bool) -> bool:
        # Like MTP, the multi-token draft arrives marked as prefill even though
        # it is a decode-tail operation.
        return True

    def _load_extra_weights(self, weights: ModelWeights) -> None:
        gw = weights.global_weights
        self.main_proj = _v4_fp8_linear_from_dict(
            gw, W.v4_dspark_main_proj_w, W.v4_dspark_main_proj_s
        )
        self.main_norm = RMSNorm(
            gw[W.v4_dspark_main_norm], float(self._v4_args.norm_eps)
        )
        self.markov_head = DSparkMarkovHead(
            gw[W.v4_dspark_markov_w1], gw[W.v4_dspark_markov_w2]
        )
        assert self.v4 is not None
        for layer in self.v4.layers:
            # Route A: the existing FlashMLA sparse kernel accepts arbitrary
            # K row indices. P3's metadata builder explicitly appends the
            # complete query block, yielding bidirectional intra-block
            # visibility without a new attention kernel.
            layer.attn.dspark_noncausal = True

    def cuda_graph_input_hidden_dim(self) -> int:
        return len(self.dspark_params.target_layer_ids) * int(self._v4_args.dim)

    @staticmethod
    def _apply_linear(layer: Any, value: torch.Tensor) -> torch.Tensor:
        shape = value.shape
        if value.dim() <= 2:
            return layer(value)
        return layer(value.reshape(-1, shape[-1])).view(*shape[:-1], -1)

    def combine_hidden_states(self, aux_hidden_states: torch.Tensor) -> torch.Tensor:
        """``main_norm(main_proj(concat(aux)))`` -> ``[T, H]``."""
        assert self.main_proj is not None and self.main_norm is not None
        if aux_hidden_states.dim() == 3:
            aux_hidden_states = aux_hidden_states.flatten(1)
        expected = self.cuda_graph_input_hidden_dim()
        if aux_hidden_states.dim() != 2 or aux_hidden_states.size(-1) != expected:
            raise ValueError(
                f"DSV4 DSpark expected aux [T,{expected}], got "
                f"{tuple(aux_hidden_states.shape)}"
            )
        return self.main_norm(self._apply_linear(self.main_proj, aux_hidden_states))

    @staticmethod
    def _context_rows(
        total: int,
        attn_inputs: Any,
        device: torch.device,
        ctx_lengths: Optional[torch.Tensor],
        ctx_starts: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if ctx_starts is not None and ctx_starts.numel() > 0:
            starts = ctx_starts.to(device=device, dtype=torch.long, non_blocking=True)
            batch = int(starts.numel())
            if total % batch:
                raise ValueError(
                    f"dense DSpark context rows {total} not divisible by batch {batch}"
                )
            width = total // batch
            batch_idx = torch.repeat_interleave(
                torch.arange(batch, device=device, dtype=torch.long), width
            )
            offsets = torch.arange(width, device=device, dtype=torch.long).repeat(batch)
            return batch_idx, starts[batch_idx] + offsets

        lengths = ctx_lengths
        if lengths is None or lengths.numel() == 0:
            lengths = attn_inputs.input_lengths
        lengths = lengths.to(device=device, dtype=torch.long, non_blocking=True)
        if int(lengths.numel()) == 0:
            raise ValueError("DSpark context injection requires per-request lengths")
        batch = int(lengths.numel())
        batch_idx = torch.repeat_interleave(
            torch.arange(batch, device=device, dtype=torch.long),
            lengths,
            output_size=total,
        )
        prefix = attn_inputs.prefix_lengths.to(
            device=device, dtype=torch.long, non_blocking=True
        )
        row_base = torch.cumsum(lengths, dim=0) - lengths
        offsets = torch.arange(total, device=device, dtype=torch.long)
        positions = (prefix - lengths - row_base)[batch_idx] + offsets
        return batch_idx, positions

    def inject_context_kv(
        self,
        main_x: torch.Tensor,
        attn_inputs: Any,
        ctx_lengths: Optional[torch.Tensor] = None,
        ctx_starts: Optional[torch.Tensor] = None,
    ) -> None:
        """Project the same target feature through each draft ``wkv`` and
        scatter RoPE'd FP8 K into that layer's ordinary SWA paged pool."""
        if not self.fp8_kv_cache:
            raise NotImplementedError("DSV4 DSpark requires the production FP8 KV cache")
        assert self.v4 is not None and self.kv_cache is not None
        block_tables = build_block_tables_batched(self.kv_cache, attn_inputs)
        if block_tables is None or int(SWA_KV) not in block_tables:
            raise RuntimeError("DSpark SWA block table is unavailable")
        block_table = block_tables[int(SWA_KV)]
        batch_idx, positions = self._context_rows(
            int(main_x.size(0)),
            attn_inputs,
            main_x.device,
            ctx_lengths,
            ctx_starts,
        )

        from rtp_llm.models_py.modules.dsv4.fp8 import (
            _swa_kv_insert_triton as insert_ops,
        )
        from rtp_llm.models_py.modules.dsv4.fp8.attention import fused_rmsnorm_rope

        tokens_per_block = require_pool_tokens_per_block(
            self.kv_cache, region=int(SWA_KV)
        )
        safe_positions = positions.clamp_min(0)
        block_in_seq = safe_positions // tokens_per_block
        block_ids = block_table.to(torch.long)[batch_idx, block_in_seq]

        for layer in self.v4.layers:
            attn = layer.attn
            attn._kv_cache = self.kv_cache
            attn._block_tables_by_type = block_tables
            if attn._swa_cp_byte_sliced():
                raise NotImplementedError("DSpark CP byte-sliced injection lands in P6")
            pool = attn._pool_view_3d_fp8(SWA_KV)
            if pool is None:
                raise RuntimeError(f"DSpark layer {layer.layer_id} SWA FP8 pool unavailable")
            entries_per_block = int(attn._swa_entries_per_block())
            slots = block_ids * entries_per_block + (
                safe_positions % entries_per_block
            )
            slots = torch.where(
                (positions >= 0) & (block_ids > 0),
                slots,
                torch.full_like(slots, -1),
            )
            kv = attn._lin(attn.wkv, main_x.unsqueeze(0))
            freqs = attn.freqs_cis.index_select(
                0, safe_positions.to(device=attn.freqs_cis.device, dtype=torch.long)
            )
            kv = fused_rmsnorm_rope(
                kv,
                attn.kv_norm,
                freqs,
                int(attn.rope_head_dim),
                eps=float(attn.eps),
            )
            kv_bf16 = kv.reshape(-1, int(attn.head_dim)).to(torch.bfloat16)
            insert_ops.quantize_and_insert_k_cache(kv_bf16, pool, slots)

    def _propose_backbone(
        self, inputs: PyModelInputs, fmha_impl: Any = None
    ) -> PyModelOutputs:
        width = self.dspark_params.block_width
        input_lengths = inputs.attention_inputs.input_lengths
        if not input_lengths.is_cuda and input_lengths.numel() > 0:
            assert bool((input_lengths == width).all()), (
                f"DSV4 DSpark expects {width} query rows per request, "
                f"got {input_lengths.tolist()}"
            )
        aux = inputs.input_hiddens
        if aux is not None and aux.numel() > 0:
            main_x = self.combine_hidden_states(aux)
            ctx_lengths = inputs.dspark_ctx_lengths
            ctx_starts = inputs.dspark_ctx_starts
            self.inject_context_kv(
                main_x,
                inputs.attention_inputs,
                ctx_lengths if ctx_lengths is not None else None,
                ctx_starts if ctx_starts is not None else None,
            )
        return super().forward(inputs, fmha_impl)

    def compute_base_logits(self, head_hidden: torch.Tensor) -> torch.Tensor:
        """All ``k`` anchor-layout rows predict: ``[B*k,H] -> [B,k,V]``."""
        assert self.v4 is not None
        width = self.dspark_params.block_width
        hidden = head_hidden.view(-1, width, head_hidden.size(-1))
        weight = self.v4.head_weight
        logits = F.linear(hidden.to(weight.dtype), weight)
        if self.tp_size > 1:
            from rtp_llm.models_py.distributed.collective_torch import Group, all_gather

            rows = int(logits.size(0) * logits.size(1))
            gathered = all_gather(logits.reshape(rows, -1).contiguous(), group=Group.TP)
            shard_vocab = int(logits.size(-1))
            logits = (
                gathered.reshape(self.tp_size, rows, shard_vocab)
                .permute(1, 0, 2)
                .reshape(logits.size(0), width, self.tp_size * shard_vocab)
            )
        return logits[..., : self.vocab_size].contiguous()

    def map_draft_to_target(self, draft_ids: torch.Tensor) -> torch.Tensor:
        return draft_ids

    def markov_correct(
        self, base_logits: torch.Tensor, anchor_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert self.markov_head is not None
        corrected = base_logits.float()
        if corrected is base_logits:
            corrected = corrected.clone()
        batch, width = corrected.shape[:2]
        tokens = torch.empty((batch, width), dtype=torch.long, device=corrected.device)
        previous = anchor_ids.long()
        for step in range(width):
            corrected[:, step] += self.markov_head.bias(previous)
            previous = corrected[:, step].argmax(dim=-1)
            tokens[:, step] = previous
        return tokens, corrected

    def propose(
        self, inputs: PyModelInputs, fmha_impl: Any = None
    ) -> DeepSeekV4DSparkProposal:
        outputs = self._propose_backbone(inputs, fmha_impl)
        base_logits = self.compute_base_logits(outputs.hidden_states)
        anchors = inputs.input_ids.view(-1, self.dspark_params.block_width)[:, 0]
        tokens, corrected = self.markov_correct(base_logits, anchors)
        return DeepSeekV4DSparkProposal(
            tokens, corrected, base_logits, outputs.hidden_states
        )

    def forward_backbone(
        self, inputs: PyModelInputs, fmha_impl: Any = None
    ) -> PyModelOutputs:
        if fmha_impl is None:
            fmha_impl = self.prepare_fmha_impl(inputs)
        return self._propose_backbone(inputs, fmha_impl)

    def draft_tail(self, outputs: PyModelOutputs, inputs: PyModelInputs) -> PyModelOutputs:
        base_logits = self.compute_base_logits(outputs.hidden_states)
        anchors = inputs.input_ids.view(-1, self.dspark_params.block_width)[:, 0]
        tokens, corrected = self.markov_correct(base_logits, anchors)
        outputs.draft_tokens = tokens
        outputs.draft_probs = torch.softmax(corrected, dim=-1)
        return outputs

    def forward(self, inputs: PyModelInputs, fmha_impl: Any = None) -> PyModelOutputs:
        return self.draft_tail(self.forward_backbone(inputs, fmha_impl), inputs)


__all__ = [
    "DeepSeekV4DSparkParams",
    "DeepSeekV4DSparkProposal",
    "DSparkMarkovHead",
    "DeepSeekV4DSparkModel",
]
