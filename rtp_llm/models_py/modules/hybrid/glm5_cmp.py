"""GLM5 CMP execution path using CUDA Graph, multistream, and PDL."""

from __future__ import annotations

import importlib
import logging
import os
from dataclasses import dataclass
from typing import Any

import torch

from rtp_llm.utils.model_weight import W

_ENABLE_ENV = "RTP_LLM_GLM5_CMP"
_SUPPORTED_MODEL_TYPES = frozenset(("glm_5", "glm_5_mtp"))
logger = logging.getLogger(__name__)


def _load_ops() -> Any:
    return importlib.import_module("rtp_kernel.glm5")


def _configure_deep_gemm_pdl(ops: Any) -> None:
    deep_gemm = importlib.import_module("deep_gemm")
    deep_gemm.set_pdl(ops.get_pdl())


def resolve_glm5_cmp_enabled() -> bool:
    value = os.environ.get(_ENABLE_ENV, "0").strip().lower()
    if value in ("1", "true", "yes", "on"):
        return True
    if value in ("0", "false", "no", "off", ""):
        return False
    raise ValueError(f"invalid {_ENABLE_ENV}={value!r}")


def _is_capturing() -> bool:
    try:
        return bool(torch.cuda.is_current_stream_capturing())
    except RuntimeError:
        return False


def get_weight_and_scale_from_linear(
    linear: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the FP8 weight pair required by RTP-kernel operators."""
    weight = getattr(linear, "weight", None)
    scale = getattr(linear, "weight_scales", None)
    if not isinstance(weight, torch.Tensor) or not isinstance(scale, torch.Tensor):
        raise TypeError("RTP-kernel requires FP8 weight and weight_scales")
    return weight, scale


def _page_block_table(attn_inputs: Any) -> torch.Tensor | None:
    """Return RTP's request-local table at the 64-token kernel granularity."""
    table = getattr(attn_inputs, "kv_cache_block_id_device", None)
    return table if isinstance(table, torch.Tensor) and table.numel() > 0 else None


def _tokens_per_request(
    rows: int,
    context_lens: Any,
    block_table: Any,
) -> int | None:
    if (
        rows <= 0
        or not isinstance(context_lens, torch.Tensor)
        or context_lens.dtype != torch.int32
        or context_lens.numel() != rows
        or not isinstance(block_table, torch.Tensor)
        or block_table.dim() != 2
        or block_table.size(0) <= 0
    ):
        return None
    request_bs = int(block_table.size(0))
    return rows // request_bs if rows % request_bs == 0 else None


def _unsupported_parallelism_reason(parallelism: Any) -> str | None:
    raw_tp = int(getattr(parallelism, "tp_size", 1) or 1)
    attn_tp = int(parallelism.get_attn_tp_size())
    # Decode carries the prefill CP width and sharded-cache metadata so that it
    # can read P->D KV cache correctly, but it does not run context parallelism.
    # Active prefill CP repurposes the raw TP group and therefore has raw_tp > 1.
    if raw_tp != 1 or attn_tp != 1:
        return "GLM5 CMP requires TP=1"
    return None


@dataclass(frozen=True)
class _StreamEvents:
    caller_to_main: torch.cuda.Event
    side_streams_complete: torch.cuda.Event
    norm_to_indexer_k: torch.cuda.Event
    qkv_to_indexer_q: torch.cuda.Event
    indexer_q_to_score: torch.cuda.Event
    q_path_complete: torch.cuda.Event
    indexer_complete: torch.cuda.Event


class Glm5Cmp:
    """Coordinate GLM5 CUDA Graph, multistream, and PDL (CMP) execution.

    CMP names the scheduling scheme, not a kernel provider. Its operators may
    come from RTP-kernel, DeepGEMM, or existing RTP-LLM implementations.
    """

    _streams_by_device: dict[int, tuple[Any, Any, Any]] = {}

    def __init__(
        self,
        *,
        layer_idx: int,
        config: Any,
        parallelism_config: Any,
        self_attn: Any,
        input_layernorm: Any,
        mlp: Any,
        post_attention_layernorm: Any,
    ) -> None:
        self.layer_idx = int(layer_idx)
        self.config = config
        self.parallelism_config = parallelism_config
        self.self_attn = self_attn
        self.input_layernorm = input_layernorm
        self.mlp = mlp
        self.post_attention_layernorm = post_attention_layernorm
        self._is_moe_layer = self.layer_idx in config.moe_layer_index
        self._router_weight = None
        self._events: _StreamEvents | None = None
        self._packed_head_gate_weight: torch.Tensor | None = None
        self._draft_prefill_clone = False
        self._disabled_reason = self._static_disabled_reason()
        self._moe_prepack_disabled_reason = None
        self._moe_prepack_abi_validated = False
        # Merely setting RTP_LLM_GLM5_CMP must not import RTP-kernel or change
        # DeepGEMM's process-wide PDL setting. These fields are initialized only
        # after the current model call has passed the CMP capability checks.
        self.ops = None
        self._qkv_projection = None
        self._q_b_proj = None
        self._output_projection = None
        self._indexer_k_projection = None
        self._indexer_q_projection = None

    @property
    def has_indexer(self) -> bool:
        return bool(getattr(self.self_attn, "has_indexer", False))

    @property
    def is_moe_layer(self) -> bool:
        return self._is_moe_layer

    def _static_disabled_reason(self) -> str | None:
        reason = _unsupported_parallelism_reason(self.parallelism_config)
        if reason is not None:
            return reason
        if str(getattr(self.config, "model_type", "")) not in _SUPPORTED_MODEL_TYPES:
            return "unsupported model type"
        attn = self.config.attn_config
        if not bool(getattr(attn, "use_mla", False)):
            return "attention is not MLA"
        if int(getattr(attn, "kernel_tokens_per_block", 0)) != 64:
            return "GLM5 CMP requires 64-token KV pages"
        if (
            self.has_indexer
            and os.environ.get("GLM5_INDEXER_TOPK_BACKEND", "dsv4_persistent")
            .strip()
            .lower()
            != "topk_v3"
        ):
            return "GLM5 CMP requires GLM5_INDEXER_TOPK_BACKEND=topk_v3"

        if not self.is_moe_layer:
            if not (
                bool(getattr(self.mlp, "accepts_fp8_input", False))
                and bool(
                    getattr(getattr(self.mlp, "up_proj", None), "scale_ue8m0", False)
                )
            ):
                return "Dense GLM5 CMP requires FP8 input with UE8M0 scales"
        return None

    def _static_moe_prepack_disabled_reason(self) -> str | None:
        # RTP-kernel writes router outputs directly into MegaMoE's stable input
        # buffers. Keep this optional optimization limited to the deployed ABI.
        config = self.config
        if (
            not self.is_moe_layer
            or self._router_weight is None
            or getattr(self.mlp, "correction_bias", None) is None
            or not callable(
                getattr(
                    getattr(self.mlp, "fused_moe", None),
                    "prepacked_input_views",
                    None,
                )
            )
            or not callable(
                getattr(
                    getattr(self.mlp, "fused_moe", None),
                    "forward_prepacked",
                    None,
                )
            )
            or getattr(self.mlp, "shared_expert", None) is None
            or getattr(self.mlp, "shared_expert_gate", None) is not None
            or int(getattr(self.mlp, "ffn_tp_size", 1)) != 1
            or (
                int(getattr(config, "expert_num", 0)),
                int(getattr(config, "moe_k", 0)),
                int(getattr(config, "moe_n_group", 0)),
                int(getattr(config, "moe_topk_group", 0)),
                bool(getattr(config, "has_moe_norm", False)),
                float(getattr(config, "routed_scaling_factor", 0.0)),
            )
            != (256, 8, 1, 1, True, 2.5)
        ):
            return "unsupported MegaMoE contract"
        return None

    def _initialize_for_cmp(self, ops: Any) -> None:
        """Cache operator inputs after an actual model call selects CMP."""
        if self.ops is not None:
            return

        if self.is_moe_layer:
            self._router_weight = self._prepare_router_weight(self.mlp)
            self._moe_prepack_disabled_reason = (
                self._static_moe_prepack_disabled_reason()
            )

        qkv_projection = get_weight_and_scale_from_linear(
            self.self_attn.fused_qkv_a_proj
        )
        q_b_projection = get_weight_and_scale_from_linear(self.self_attn.q_b_proj)
        output_projection = get_weight_and_scale_from_linear(self.self_attn.o_proj)
        if self.has_indexer:
            indexer_k_projection = get_weight_and_scale_from_linear(
                self.self_attn.indexer.wk
            )
            indexer_q_projection = get_weight_and_scale_from_linear(
                self.self_attn.indexer.wq_b
            )
        else:
            indexer_k_projection = None
            indexer_q_projection = None

        self._qkv_projection = qkv_projection
        self._q_b_proj = q_b_projection
        self._output_projection = output_projection
        self._indexer_k_projection = indexer_k_projection
        self._indexer_q_projection = indexer_q_projection
        self.ops = ops

    @staticmethod
    def _moe_prepack_views_disabled_reason(views: Any, rows: int) -> str | None:
        if not isinstance(views, (tuple, list)) or len(views) != 4:
            return "MegaMoE prepacked views must contain four tensors"
        activation, scale, topk_indices, topk_weights = views
        if not all(isinstance(tensor, torch.Tensor) for tensor in views):
            return "MegaMoE prepacked views must be tensors"

        expected = (
            (activation, torch.float8_e4m3fn, (rows, 6144), "activation"),
            (scale, torch.int32, (rows, 48), "activation scale"),
            (topk_indices, torch.int64, (rows, 8), "TopK indices"),
            (topk_weights, torch.float32, (rows, 8), "TopK weights"),
        )
        for tensor, dtype, shape, name in expected:
            if tensor.dtype != dtype or tuple(tensor.shape) != shape:
                return f"unsupported MegaMoE {name} dtype or shape"
            if not tensor.is_contiguous():
                return f"unsupported MegaMoE {name} layout"
            if tensor.device != activation.device:
                return "MegaMoE prepacked views must share a device"
        return None

    def moe_prepacked_input_views(self, rows: int) -> tuple[
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
    ]:
        """Return writable FP8 MegaMoE views, or four empty slots."""
        if not self.is_moe_layer or self._moe_prepack_disabled_reason is not None:
            return None, None, None, None
        try:
            views = self.mlp.fused_moe.prepacked_input_views(int(rows))
        except (AttributeError, TypeError, ValueError, RuntimeError):
            return None, None, None, None
        if not self._moe_prepack_abi_validated:
            reason = self._moe_prepack_views_disabled_reason(views, int(rows))
            if reason is not None:
                self._moe_prepack_disabled_reason = reason
                return None, None, None, None
            self._moe_prepack_abi_validated = True
        return tuple(views)

    @staticmethod
    def _prepare_router_weight(mlp: Any) -> torch.Tensor | None:
        weight = getattr(getattr(mlp, "gate", None), "weight", None)
        if (
            not isinstance(weight, torch.Tensor)
            or tuple(weight.shape) != (256, 6144)
            or weight.dtype != torch.bfloat16
        ):
            return None
        # CudaF16Linear keeps RTP's contiguous [K,N] storage as an [N,K]
        # transpose view for F.linear. Router projection consumes row-major
        # [N,K] through TMA,
        # so materialize that layout once and share it across graph clones.
        return weight if weight.is_contiguous() else weight.contiguous()

    def clone_for_cuda_graph(
        self,
        *,
        mlp: Any,
        draft_prefill: bool = False,
    ) -> Glm5Cmp:
        clone = object.__new__(type(self))
        clone.layer_idx = self.layer_idx
        clone.config = self.config
        clone.parallelism_config = self.parallelism_config
        clone.self_attn = self.self_attn
        clone.input_layernorm = self.input_layernorm
        clone.mlp = mlp
        clone.post_attention_layernorm = self.post_attention_layernorm
        clone._is_moe_layer = self._is_moe_layer
        clone._router_weight = self._router_weight
        clone._events = None
        clone._packed_head_gate_weight = self._packed_head_gate_weight
        clone._draft_prefill_clone = bool(draft_prefill)
        clone._disabled_reason = self._disabled_reason
        clone._moe_prepack_disabled_reason = self._moe_prepack_disabled_reason
        clone._moe_prepack_abi_validated = False
        clone.ops = self.ops
        clone._qkv_projection = self._qkv_projection
        clone._q_b_proj = self._q_b_proj
        clone._output_projection = self._output_projection
        clone._indexer_k_projection = self._indexer_k_projection
        clone._indexer_q_projection = self._indexer_q_projection
        return clone

    @staticmethod
    def _attention_impl(fmha_impl: Any) -> Any:
        if hasattr(fmha_impl, "weights") and hasattr(fmha_impl, "fmha_params"):
            return fmha_impl
        nested = getattr(fmha_impl, "fmha_impl", None)
        if nested is not None and hasattr(nested, "weights"):
            return nested
        raise TypeError("cannot locate SparseMlaImpl")

    def _unsupported_call_reason(
        self, hidden_states: torch.Tensor, fmha_impl: Any, kv_cache: Any
    ) -> str | None:
        """Validate the dynamic call once before the model enters GLM5 CMP."""
        implementation = self._attention_impl(fmha_impl)
        attn_inputs = implementation.attn_inputs
        params = implementation.fmha_params
        rows = int(hidden_states.size(0)) if hidden_states.dim() == 2 else 0
        block_table = _page_block_table(attn_inputs)
        context_lens = getattr(params, "expanded_seq_lens", None)
        tokens_per_request = _tokens_per_request(rows, context_lens, block_table)
        is_prefill = bool(getattr(attn_inputs, "is_prefill", False))
        is_target_verify = bool(getattr(attn_inputs, "is_target_verify", False))
        is_draft_prefill = (
            str(getattr(self.config, "model_type", "")) == "glm_5_mtp"
            and self._draft_prefill_clone
            and is_prefill
            and not is_target_verify
        )

        if kv_cache is None:
            return "KV cache is unavailable"
        if hidden_states.dim() != 2 or rows <= 0 or hidden_states.size(1) != 6144:
            return "unsupported hidden-state shape"
        if is_prefill and not is_target_verify and not is_draft_prefill:
            return "ordinary prefill is unsupported"
        if self.has_indexer and tokens_per_request is None:
            return "invalid Indexer request metadata"
        if is_draft_prefill:
            expected = int(getattr(self.config, "gen_num_per_cycle", 0) or 0) + 1
            if expected <= 1 or tokens_per_request != expected:
                return "draft-prefill width does not match MTP"
        return None

    @classmethod
    def _side_streams(cls, device: torch.device) -> tuple[Any, Any, Any]:
        device_index = (
            torch.cuda.current_device() if device.index is None else int(device.index)
        )
        streams = cls._streams_by_device.get(device_index)
        if streams is None:
            if _is_capturing():
                raise RuntimeError(
                    "GLM5 CMP side streams must be created before capture"
                )
            stream_device = torch.device("cuda", device_index)
            streams = (
                torch.cuda.Stream(device=stream_device, priority=-1),
                torch.cuda.Stream(device=stream_device),
                torch.cuda.Stream(device=stream_device),
            )
            cls._streams_by_device[device_index] = streams
        return streams

    @staticmethod
    def _new_events(device: torch.device) -> _StreamEvents:
        if _is_capturing():
            raise RuntimeError("GLM5 CMP events must be created before capture")
        events = _StreamEvents(
            *(torch.cuda.Event(enable_timing=False) for _ in range(7))
        )
        # torch.cuda.Event owns the lifetime; this eager record only creates
        # the cudaEvent_t handle needed by programmatic producer launches.
        for event in events.__dict__.values():
            event.record()
        return events

    def mla_prologue(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        fmha_impl: Any,
        kv_cache: Any,
        prev_topk_indices: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run MLA preparation Ops; RTP only supplies the stream schedule."""
        implementation = self._attention_impl(fmha_impl)
        ops = self.ops
        rows = int(hidden_states.size(0))
        params = implementation.fmha_params
        attention = self.self_attn
        qkv_weight, qkv_scale = self._qkv_projection
        q_b_weight, q_b_scale = self._q_b_proj
        mla_cache = kv_cache.kv_cache_base
        if mla_cache.dtype != torch.uint8:
            mla_cache = mla_cache.view(torch.uint8)
        mla_cache = mla_cache.view(-1, 64, 656)

        if not self.has_indexer:
            if prev_topk_indices is None:
                raise RuntimeError("main-only layer requires prior TopK indices")
            # Residual add + RMSNorm, then group-128 FP8 quantization for the
            # following attention projections.
            residual_out, _, hidden_fp8, hidden_scale = ops.add_norm_quant(
                hidden_states,
                residual,
                self.input_layernorm.weight.data,
                epsilon=float(self.input_layernorm.variance_epsilon),
            )
            # Fused Q/KV-A FP8 projection; the output packs Q-LoRA, latent KV,
            # and the positional-key slice in one BF16 tensor.
            projected = ops.qkv_a_proj(
                hidden_fp8,
                hidden_scale,
                qkv_weight,
                qkv_scale,
            )
            # Split the projection, RMSNorm and quantize Q/latent-KV, apply
            # RoPE to K-PE, and write the latent KV and K-PE to the MLA cache.
            q_fp8, q_scale, mla_cache = ops.qkv_rmsnorm_quant_rope_cached(
                projected,
                attention.q_a_layernorm.weight.data,
                attention.kv_a_layernorm.weight.data,
                implementation._cos_sin_cache,
                params.positions_d,
                params.slot_mapping,
                q_epsilon=float(attention.q_a_layernorm.variance_epsilon),
                kv_epsilon=float(attention.kv_a_layernorm.variance_epsilon),
                cache=mla_cache,
            )
            # Project Q-LoRA into per-head [NoPE | PE], keep NoPE for the
            # absorbed BMM, and RoPE the PE suffix of the SparseMLA query.
            q_nope, q_for_sparse_mla = ops.q_b_proj(
                q_fp8,
                q_scale,
                q_b_weight,
                q_b_scale,
                implementation._cos_sin_cache,
                params.positions_d,
            )
            # Apply the per-head absorbed Q-NoPE x W_KC BMM and fill the NoPE
            # prefix, completing the [absorbed NoPE | RoPE] SparseMLA query.
            ops.absorbed_q_nope_bmm(
                q_nope,
                implementation.weights[self.layer_idx][W.mla_kc],
                out=q_for_sparse_mla,
            )
            return residual_out, q_for_sparse_mla, prev_topk_indices

        block_table = _page_block_table(implementation.attn_inputs)
        assert block_table is not None
        if self._events is None:
            self._events = self._new_events(hidden_states.device)
        events = self._events
        main_stream, index_stream, indexer_q_stream = self._side_streams(
            hidden_states.device
        )
        indexer = attention.indexer
        if self._packed_head_gate_weight is None:
            if _is_capturing():
                raise RuntimeError("warm up head-gate weight packing before capture")
            # Pack the FP32 Head-Gate projection weight once for the CUDA
            # kernel layout; this performs no per-token computation.
            self._packed_head_gate_weight = ops.pack_head_gate_weight(
                indexer.weights_proj.weight.float().contiguous()
            )
        indexer_cache = kv_cache.kv_scale_base
        if indexer_cache.dtype != torch.uint8:
            indexer_cache = indexer_cache.view(torch.uint8)
        indexer_cache = indexer_cache.view(-1, 64, 1, 132)
        indexer_k_weight, indexer_k_scale = self._indexer_k_projection
        indexer_q_weight, indexer_q_scale = self._indexer_q_projection
        # At long KV, paged Indexer scoring is heavy enough that overlapping it
        # with both main-query projections slows the critical path.
        serialize_score_after_q_path = int(block_table.size(1)) * 64 >= (1 << 19)

        events.caller_to_main.record()
        with torch.cuda.stream(main_stream):
            events.caller_to_main.wait()
            gate_output = torch.empty(
                (rows, 32), device=hidden_states.device, dtype=torch.float32
            )
            # Residual add + RMSNorm, then emit both BF16 normalized hidden
            # states and group-128 FP8 input for the projection branches.
            residual_out, norm_out, hidden_fp8, hidden_scale = ops.add_norm_quant(
                hidden_states,
                residual,
                self.input_layernorm.weight.data,
                epsilon=float(self.input_layernorm.variance_epsilon),
                head_gate=gate_output,
                notify_event=events.norm_to_indexer_k,
            )

            # Run Q/KV-A projection and the independent Head-Gate linear
            # together; gate_output is raw Linear(norm), before Q-scale folding.
            projected = ops.qkv_a_proj(
                hidden_fp8,
                hidden_scale,
                qkv_weight,
                qkv_scale,
                head_gate_norm=norm_out,
                head_gate_weight=self._packed_head_gate_weight,
                gate_output=gate_output,
            )
            # Split Q/KV-A, RMSNorm and quantize Q/latent-KV, apply K-PE RoPE,
            # and write the latent KV plus K-PE into the paged MLA cache.
            q_fp8, q_scale, mla_cache = ops.qkv_rmsnorm_quant_rope_cached(
                projected,
                attention.q_a_layernorm.weight.data,
                attention.kv_a_layernorm.weight.data,
                implementation._cos_sin_cache,
                params.positions_d,
                params.slot_mapping,
                q_epsilon=float(attention.q_a_layernorm.variance_epsilon),
                kv_epsilon=float(attention.kv_a_layernorm.variance_epsilon),
                notify_event=events.qkv_to_indexer_q,
                cache=mla_cache,
            )

            # Project Q-LoRA into per-head [NoPE | PE], retain NoPE for the
            # absorbed BMM, and RoPE the PE suffix of the SparseMLA query.
            q_nope, q_for_sparse_mla = ops.q_b_proj(
                q_fp8,
                q_scale,
                q_b_weight,
                q_b_scale,
                implementation._cos_sin_cache,
                params.positions_d,
            )
            # Apply Q-NoPE x W_KC per head and fill the query prefix, producing
            # the complete [absorbed NoPE | RoPE] input consumed by SparseMLA.
            ops.absorbed_q_nope_bmm(
                q_nope,
                implementation.weights[self.layer_idx][W.mla_kc],
                out=q_for_sparse_mla,
            )
            if serialize_score_after_q_path:
                events.q_path_complete.record()

            with torch.cuda.stream(indexer_q_stream):
                events.qkv_to_indexer_q.wait()
                # FP8-project Q-LoRA to 32 x 128 Indexer-Q values. RoPE,
                # Hadamard, quantization, and Head-Gate folding follow below.
                indexer_q = ops.indexer_q_proj(
                    q_fp8,
                    q_scale,
                    indexer_q_weight,
                    indexer_q_scale,
                )
                # Apply RoPE + Hadamard + FP8 quantization and fold the per-head
                # Q scale into the early gate result: head_weight=gate_raw*q_scale/64.
                indexer_q_fp8, head_weights = ops.indexer_q_rope_quant(
                    indexer_q,
                    implementation._cos_sin_cache,
                    params.positions_d,
                    gate_output,
                    notify_event=events.indexer_q_to_score,
                )

            with torch.cuda.stream(index_stream):
                events.norm_to_indexer_k.wait()
                # Project Indexer-K, apply LayerNorm + RoPE + Hadamard + FP8
                # quantization, and write K plus its scale to the paged cache.
                ops.indexer_k_cache(
                    hidden_fp8,
                    hidden_scale,
                    indexer_k_weight,
                    indexer_k_scale,
                    indexer.k_norm.weight.data,
                    indexer.k_norm.beta.data,
                    implementation._cos_sin_cache,
                    params.positions_d,
                    params.slot_mapping,
                    indexer_cache,
                    epsilon=float(indexer.k_norm.variance_epsilon),
                )
                events.indexer_q_to_score.wait()
                if serialize_score_after_q_path:
                    events.q_path_complete.wait()
                # Compute paged Indexer logits with DeepGEMM, then immediately
                # select request-local TopK indices on the same stream.
                topk_indices = indexer.indexer_op._get_topk_paged(
                    indexer_q_fp8,
                    head_weights,
                    kv_cache,
                    params,
                    implementation.attn_inputs,
                )
                events.indexer_complete.record()

            events.indexer_complete.wait()
            events.side_streams_complete.record()
        events.side_streams_complete.wait()
        return residual_out, q_for_sparse_mla, topk_indices

    def sparse_mla(
        self,
        query: torch.Tensor,
        topk_indices: torch.Tensor,
        fmha_impl: Any,
        kv_cache: Any,
    ) -> torch.Tensor:
        implementation = self._attention_impl(fmha_impl)
        cache = kv_cache.kv_cache_base
        if not implementation.fmha_impl.expects_paged_kv:
            cache = cache.view(-1, 1, cache.size(-1))
        return implementation.fmha_impl.forward(
            query, cache, topk_indices, layer_id=self.layer_idx
        )

    def mla_post_moe_pre(
        self,
        mla_output: torch.Tensor,
        residual: torch.Tensor,
        fmha_impl: Any,
        moe_activation: torch.Tensor | None = None,
        moe_scale: torch.Tensor | None = None,
        routed_indices: torch.Tensor | None = None,
        routed_weights: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, ...]:
        """Project the MLA output and prepare an optional routed MoE input."""
        ops = self.ops
        implementation = self._attention_impl(fmha_impl)
        mla_weight = implementation.weights[self.layer_idx][W.mla_vc]
        # Apply the per-head absorbed attention-output x W_VC BMM and quantize
        # the expanded [M, 16384] activation for the output projection.
        quantized, quant_scale = ops.mla_absorbed_output_bmm_quant(
            mla_output, mla_weight
        )
        output_weight, output_scale = self._output_projection
        # FP8 output projection from the expanded 64-head activation to the
        # model hidden size, producing BF16 attention output [M, 6144].
        attention_output = ops.project_attention_output(
            quantized,
            quant_scale,
            output_weight,
            output_scale,
        )

        if self.is_moe_layer and moe_activation is not None:
            assert moe_scale is not None
            assert routed_indices is not None
            assert routed_weights is not None
            # Post-attention residual add + RMSNorm, while writing BF16 Router
            # input and group-32 FP8/UE8M0 activation directly to MegaMoE buffers.
            moe_residual = torch.empty_like(attention_output)
            moe_norm = torch.empty_like(attention_output)
            ops.add_rms_norm_mega_moe_quant(
                attention_output,
                residual,
                self.post_attention_layernorm.weight.data,
                out=(moe_residual, moe_norm, moe_activation, moe_scale),
                epsilon=float(self.post_attention_layernorm.variance_epsilon),
            )
            # Project normalized hidden states to FP32 logits for 256 experts.
            router_logits = ops.router_proj(
                moe_norm,
                self._router_weight,
            )
            # Apply sigmoid and correction bias, select corrected Top-8 experts,
            # normalize/scale their weights, and pack MegaMoE routing buffers.
            ops.router_topk(
                router_logits,
                self.mlp.correction_bias,
                topk_indices=routed_indices,
                topk_weights=routed_weights,
            )
            return moe_norm, moe_residual, routed_indices, routed_weights
        return attention_output, residual


def should_enable_glm5_cmp(
    layers: Any,
    layer_num: int,
    hidden_states: torch.Tensor,
    fmha_impl: Any,
    kv_cache: Any,
) -> bool:
    """Return whether the complete model forward should use GLM5 CMP."""
    if layer_num <= 0:
        return False

    cmps = []
    for layer in layers[:layer_num]:
        layer_cmp = layer.cmp
        if layer_cmp is None or layer_cmp._disabled_reason is not None:
            return False
        cmps.append(layer_cmp)
    if len(cmps) != layer_num:
        return False

    # Layer 0 must seed the request-local TopK indices reused by main-only layers.
    # It validates the dynamic attention contract shared by the complete model.
    cmp = cmps[0]
    first_cache = kv_cache.get_layer_cache(0) if kv_cache is not None else None
    if (
        not cmp.has_indexer
        or cmp._unsupported_call_reason(hidden_states, fmha_impl, first_cache)
        is not None
    ):
        return False

    # Initialize RTP-kernel and its DeepGEMM PDL contract only after this
    # concrete call has been selected for CMP. CUDA Graph performs eager warmup
    # forwards before capture, so graph clones are initialized outside capture.
    uninitialized = [layer_cmp for layer_cmp in cmps if layer_cmp.ops is None]
    if uninitialized:
        ops = next(
            (layer_cmp.ops for layer_cmp in cmps if layer_cmp.ops is not None),
            None,
        )
        if ops is None:
            ops = _load_ops()
        for layer_cmp in uninitialized:
            layer_cmp._initialize_for_cmp(ops)
        _configure_deep_gemm_pdl(ops)
        logger.info("GLM5 CMP activated for %d layers", layer_num)
    return True
