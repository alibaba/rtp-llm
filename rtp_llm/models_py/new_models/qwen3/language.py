from __future__ import annotations

import logging
import math
from numbers import Real
from typing import TYPE_CHECKING, Any, Optional

import torch
import torch.nn as nn

from rtp_llm.models_py.downstream_heads import ReplicatedHandlerHeads
from rtp_llm.models_py.layers.activation import silu_and_mul
from rtp_llm.models_py.layers.embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
    output_vocab_head_kwargs,
)
from rtp_llm.models_py.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from rtp_llm.models_py.layers.norm import RMSNorm, decoder_rms_norm
from rtp_llm.models_py.model_desc.block_map import get_primary_attention_inputs
from rtp_llm.models_py.model_desc.module_base import GptModelBase
from rtp_llm.models_py.module_base import CheckpointLoadTarget, RtpModule
from rtp_llm.models_py.new_models.model_base import (
    positive_int as _positive_int,
    required_config_value,
    resolve_head_dim as _resolve_head_dim,
    select_fmha_impl_for_layer,
)
from rtp_llm.models_py.quant_methods.base import QuantizationConfig
from rtp_llm.models_py.read_spec import unwrap_checkpoint_tensor
from rtp_llm.models_py.weight_mapper import WeightsMapper

if TYPE_CHECKING:
    from rtp_llm.ops.compute_ops import (
        LayerKVCache,
        PyModelInputs,
        PyModelOutputs,
    )


class Qwen3MLP(RtpModule):

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        tp_size: int = 1,
        tp_rank: int = 0,
        quant_config: Optional[QuantizationConfig] = None,
        params_dtype: torch.dtype = torch.float16,
        prefix: str = "mlp",
    ):
        super().__init__()
        self.tp_size = tp_size
        self.gate_up_proj = MergedColumnParallelLinear(
            input_size=hidden_size,
            output_size=2 * intermediate_size,
            tp_size=tp_size,
            tp_rank=tp_rank,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
            bias=False,
            shard_names=["gate_proj", "up_proj"],
            params_dtype=params_dtype,
        )
        self.down_proj = RowParallelLinear(
            input_size=intermediate_size,
            output_size=hidden_size,
            tp_size=tp_size,
            tp_rank=tp_rank,
            quant_config=quant_config,
            prefix=f"{prefix}.down_proj",
            bias=False,
            params_dtype=params_dtype,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up = self.gate_up_proj(x)
        act = silu_and_mul(gate_up)
        x = self.down_proj(act)
        return x


class Qwen3Attention(RtpModule):
    """Qwen3 dense attention.

    Differs from Qwen2 attention by:
      * No qkv bias (Qwen3 dense ckpts ship without it).
      * Adds per-head RMSNorm on Q and K (`q_norm`, `k_norm`) applied between
        qkv_proj and fmha_impl. Weight shape is [head_dim].
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        layer_idx: int = 0,
        tp_size: int = 1,
        tp_rank: int = 0,
        quant_config: Optional[QuantizationConfig] = None,
        params_dtype: torch.dtype = torch.float16,
        rms_norm_eps: float = 1e-6,
        prefix: str = "self_attn",
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.layer_idx = layer_idx
        self.tp_size = tp_size

        self.qkv_proj = QKVParallelLinear(
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            tp_size=tp_size,
            tp_rank=tp_rank,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
            bias=False,
            params_dtype=params_dtype,
        )
        self.num_heads_per_partition = self.qkv_proj.num_heads_per_partition
        self.num_kv_heads_per_partition = self.qkv_proj.num_kv_heads_per_partition
        self.q_size = self.qkv_proj.q_size
        self.kv_size = self.qkv_proj.kv_size
        self.q_norm = RMSNorm(head_dim, eps=rms_norm_eps, params_dtype=params_dtype)
        self.k_norm = RMSNorm(head_dim, eps=rms_norm_eps, params_dtype=params_dtype)
        self.o_proj = RowParallelLinear(
            input_size=num_heads * head_dim,
            output_size=hidden_size,
            tp_size=tp_size,
            tp_rank=tp_rank,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
            bias=False,
            params_dtype=params_dtype,
        )

    def _apply_qk_norm(self, qkv: torch.Tensor) -> torch.Tensor:
        if (
            qkv.is_cuda
            and qkv.dim() == 2
            and getattr(torch.version, "hip", None) is not None
        ):
            qkv = qkv.contiguous()
            m, n = qkv.shape
            from rtp_llm.ops.compute_ops import rtp_llm_ops

            rtp_llm_ops.fused_qk_rmsnorm_v2(
                qkv,
                self.q_norm.weight.data,
                self.k_norm.weight.data,
                self.q_norm.eps,
                self.num_heads_per_partition,
                self.num_kv_heads_per_partition,
                m,
                n,
                self.head_dim,
            )
            return qkv

        if qkv.is_cuda and qkv.dim() == 2:
            try:
                import flashinfer

                m, n = qkv.shape
                qkv_view = qkv.reshape(
                    m,
                    self.num_heads_per_partition + self.num_kv_heads_per_partition * 2,
                    self.head_dim,
                )
                q = qkv_view[:, : self.num_heads_per_partition, :]
                k = qkv_view[
                    :,
                    self.num_heads_per_partition : self.num_heads_per_partition
                    + self.num_kv_heads_per_partition,
                    :,
                ]
                flashinfer.norm.rmsnorm(
                    q, self.q_norm.weight.data, eps=self.q_norm.eps, out=q
                )
                flashinfer.norm.rmsnorm(
                    k, self.k_norm.weight.data, eps=self.k_norm.eps, out=k
                )
                return qkv_view.reshape(m, n)
            except ModuleNotFoundError:
                pass

        prefix_shape = qkv.shape[:-1]
        q = qkv[..., : self.q_size].reshape(
            *prefix_shape, self.num_heads_per_partition, self.head_dim
        )
        k = qkv[..., self.q_size : self.q_size + self.kv_size].reshape(
            *prefix_shape, self.num_kv_heads_per_partition, self.head_dim
        )
        v = qkv[..., self.q_size + self.kv_size :]
        q = self.q_norm(q).reshape(*prefix_shape, self.q_size)
        k = self.k_norm(k).reshape(*prefix_shape, self.kv_size)
        return torch.cat([q, k, v], dim=-1)

    def forward(
        self,
        hidden_states: torch.Tensor,
        fmha_impl: Any,
        kv_cache: Optional[LayerKVCache] = None,
    ) -> torch.Tensor:
        input_shape = hidden_states.shape[:-1]
        qkv = self.qkv_proj(hidden_states)
        qkv = self._apply_qk_norm(qkv)
        attn_output = fmha_impl.forward(qkv, kv_cache, self.layer_idx)
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        output = self.o_proj(attn_output)
        return output


class Qwen3DecoderLayer(RtpModule):

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        intermediate_size: int,
        head_dim: int,
        layer_idx: int = 0,
        attn_tp_size: int = 1,
        attn_tp_rank: int = 0,
        ffn_tp_size: int = 1,
        ffn_tp_rank: int = 0,
        quant_config: Optional[QuantizationConfig] = None,
        params_dtype: torch.dtype = torch.float16,
        rms_norm_eps: float = 1e-6,
        prefix: str = "layer",
    ):
        super().__init__()
        self.input_layernorm = decoder_rms_norm(
            hidden_size, eps=rms_norm_eps, params_dtype=params_dtype, quant_config=quant_config
        )
        self.self_attn = Qwen3Attention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            layer_idx=layer_idx,
            tp_size=attn_tp_size,
            tp_rank=attn_tp_rank,
            quant_config=quant_config,
            params_dtype=params_dtype,
            rms_norm_eps=rms_norm_eps,
            prefix=f"{prefix}.self_attn",
        )
        self.post_attention_layernorm = decoder_rms_norm(
            hidden_size, eps=rms_norm_eps, params_dtype=params_dtype, quant_config=quant_config
        )
        self.mlp = Qwen3MLP(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            tp_size=ffn_tp_size,
            tp_rank=ffn_tp_rank,
            quant_config=quant_config,
            params_dtype=params_dtype,
            prefix=f"{prefix}.mlp",
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        fmha_impl: Any,
        kv_cache: Optional[LayerKVCache] = None,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, fmha_impl, kv_cache)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


def _extract_config_values(
    model_config: Any,
    load_config: Any,
    *,
    validate_intermediate_size: bool = True,
):
    if isinstance(model_config, dict):
        hidden_size = required_config_value(model_config, "hidden_size")
        num_heads = required_config_value(model_config, "num_attention_heads")
        num_kv_heads = model_config.get("num_key_value_heads", num_heads)
        intermediate_size = required_config_value(model_config, "intermediate_size")
        num_layers = required_config_value(model_config, "num_hidden_layers")
        vocab_size = required_config_value(model_config, "vocab_size")
        head_dim = _resolve_head_dim(
            hidden_size, num_heads, model_config.get("head_dim")
        )
        rms_norm_eps = model_config.get("rms_norm_eps", 1e-6)
    else:
        hidden_size = required_config_value(model_config, "hidden_size")
        num_layers = required_config_value(
            model_config, "num_layers", "num_hidden_layers"
        )
        vocab_size = required_config_value(model_config, "vocab_size")
        intermediate_size = required_config_value(
            model_config, "inter_size", "intermediate_size"
        )
        attn_config = getattr(model_config, "attn_config", None)
        if attn_config is not None:
            num_heads = required_config_value(attn_config, "head_num")
            num_kv_heads = getattr(attn_config, "kv_head_num", num_heads)
            if num_kv_heads == -1:
                num_kv_heads = num_heads
            head_dim = _resolve_head_dim(
                hidden_size,
                num_heads,
                getattr(attn_config, "size_per_head", None),
            )
        else:
            num_heads = required_config_value(model_config, "num_attention_heads")
            num_kv_heads = getattr(model_config, "num_key_value_heads", num_heads)
            head_dim = _resolve_head_dim(
                hidden_size, num_heads, getattr(model_config, "head_dim", None)
            )
        rms_norm_eps = getattr(
            model_config,
            "layernorm_eps",
            getattr(model_config, "rms_norm_eps", 1e-6),
        )

    tp_size = required_config_value(load_config, "tp_size")
    tp_rank = required_config_value(load_config, "tp_rank")
    attn_tp_size = required_config_value(load_config, "attn_tp_size")
    attn_tp_rank = required_config_value(load_config, "attn_tp_rank")
    ffn_tp_size = required_config_value(load_config, "ffn_tp_size")
    ffn_tp_rank = required_config_value(load_config, "ffn_tp_rank")
    lm_head_tp_size = required_config_value(load_config, "lm_head_tp_size")
    lm_head_tp_rank = required_config_value(load_config, "lm_head_tp_rank")
    quant_config = getattr(load_config, "quant_config", None)
    params_dtype = getattr(load_config, "compute_dtype", torch.float16)
    enable_fp32_lm_head = (
        model_config.get("enable_fp32_lm_head", True)
        if isinstance(model_config, dict)
        else getattr(model_config, "enable_fp32_lm_head", True)
    )

    dimensions = {
        "hidden_size": hidden_size,
        "num_heads": num_heads,
        "num_kv_heads": num_kv_heads,
        "num_layers": num_layers,
        "vocab_size": vocab_size,
        "head_dim": head_dim,
    }
    if validate_intermediate_size:
        dimensions["intermediate_size"] = intermediate_size
    for name, value in dimensions.items():
        _positive_int(value, name)
    for name, size, rank in (
        ("TP", tp_size, tp_rank),
        ("attention TP", attn_tp_size, attn_tp_rank),
        ("FFN TP", ffn_tp_size, ffn_tp_rank),
        ("LM head TP", lm_head_tp_size, lm_head_tp_rank),
    ):
        _positive_int(size, f"{name} size")
        if isinstance(rank, bool) or not isinstance(rank, int) or not 0 <= rank < size:
            raise ValueError(f"Invalid {name} partition: rank={rank}, size={size}")
    if (lm_head_tp_size, lm_head_tp_rank) != (tp_size, tp_rank):
        raise ValueError(
            "A separate LM head topology is not supported by the Qwen dense "
            "newloader slice"
        )

    return dict(
        hidden_size=hidden_size,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        intermediate_size=intermediate_size,
        num_layers=num_layers,
        vocab_size=vocab_size,
        head_dim=head_dim,
        rms_norm_eps=rms_norm_eps,
        tp_size=tp_size,
        tp_rank=tp_rank,
        attn_tp_size=attn_tp_size,
        attn_tp_rank=attn_tp_rank,
        ffn_tp_size=ffn_tp_size,
        ffn_tp_rank=ffn_tp_rank,
        lm_head_tp_size=lm_head_tp_size,
        lm_head_tp_rank=lm_head_tp_rank,
        quant_config=quant_config,
        params_dtype=params_dtype,
        lm_head_params_dtype=torch.float32 if enable_fp32_lm_head else params_dtype,
    )


def _validate_supported_parallelism(parallelism_config: Any) -> None:
    ffn_disaggregate = getattr(parallelism_config, "ffn_disaggregate_config", None)
    if bool(getattr(ffn_disaggregate, "enable_ffn_disaggregate", False)):
        raise ValueError(
            "FFN disaggregation is not supported by the Qwen dense newloader slice"
        )


class Qwen3ForCausalLM(GptModelBase, RtpModule):
    """Direct parameter owner using the current runtime's PP/cache contract.

    The loading mixin is local to this model; legacy runtime models keep the
    unchanged GptModelBase. Production admission remains gated until the
    matching NewLoader core and native PP execution are validated together.
    """

    supports_checkpoint_weight_updates = True
    supports_pp_global_weight_aliases = True
    supports_custom_weight_mappings = True
    supports_handler_projection_layers = True
    supports_ft_handler_heads = True

    def load_ft_checkpoint(self, checkpoint):
        # Multimodal/draft subclasses have other owned parameters. They need
        # their own format contract, not the dense language-model one.
        if type(self) is not Qwen3ForCausalLM:
            raise ValueError(f"No Qwen3 dense FT adapter for {type(self).__name__}")
        from rtp_llm.models_py.new_models.qwen2.ft_checkpoint import (
            load_gqa_ffn_ft_checkpoint,
        )

        load_gqa_ffn_ft_checkpoint(self, checkpoint)

    _decoder_layer = Qwen3DecoderLayer
    _dense_config = staticmethod(_extract_config_values)

    # Plain CUDA dense inference reads its final Parameters directly. Record after
    # head normalization, logit scaling and (if requested) startup LoRA merge.
    resident_after_post_load = True

    def resident_constructor_state(self):
        from rtp_llm.models_py.layers.linear import LinearBase
        from rtp_llm.models_py.quant_methods.unquantized import UnquantizedLinearMethod

        # A derived MoE or draft model can add non-parameter execution state.
        # Only a concrete owner that declares this final-dense layout uses it.
        if not type(self).__dict__.get("resident_after_post_load", False) or any(
            not isinstance(m.quant_method, UnquantizedLinearMethod)
            for m in self.modules() if isinstance(m, LinearBase)
        ):
            raise ValueError("This resident layout describes only unquantized Qwen3 language-model owners")
        return {"handlers": self.handler_heads.resident_shapes(),
                "head_explicit": bool(self.lm_head and self.lm_head._explicit_checkpoint_loaded),
                "head_tied": bool(self.lm_head and self.lm_head._tied_checkpoint_loaded)}

    def prepare_resident_checkpoint_restore(self, state, values):
        self.handler_heads.bind_resident_parameters_from_record(state["handlers"], values)

    def finish_resident_checkpoint_restore(self):
        from rtp_llm.models_py.resident_checkpoint import restore_final_linears

        self._lm_head_postprocessed = True
        state = self._native_resident_state
        if self.lm_head is not None:
            self.lm_head._explicit_checkpoint_loaded = state["head_explicit"]
            self.lm_head._tied_checkpoint_loaded = state["head_tied"]
        restore_final_linears(self)

    def _decoder_kwargs(self, parallelism_config):
        return {}

    WEIGHTS_MAPPER = WeightsMapper(
        prefix_mapping={"model.": ""},
    )

    def _owns_mapped_layer(self, mapped_name):
        parts = mapped_name.split(".", 2)
        if len(parts) == 3 and parts[0] == "layers" and parts[2]:
            index = parts[1]
            if index.isascii() and index.isdigit():
                layer = int(index)
                limit = getattr(
                    self.config, "_checkpoint_num_layers_before_override", self.layer_num
                )
                if str(layer) == index and layer < limit:
                    return layer in self._owned_layer_ids
        # Leave malformed or genuinely out-of-range layers for strict dispatch.
        return True

    def owns_lora_checkpoint_target(self, mapped_name):
        # A full-depth adapter and its base checkpoint have the same layer
        # domain, even when the service builds only a prefix or one PP stage.
        return self._owns_mapped_layer(mapped_name)

    def checkpoint_weight_name_filter(self):
        checkpoint_layers = getattr(
            self.config, "_checkpoint_num_layers_before_override", self.layer_num
        )
        if (
            not self._pipeline_partitioned
            and self.lm_head is not None
            and checkpoint_layers == self.layer_num
        ):
            return None

        def owns(name):
            mapped = self.WEIGHTS_MAPPER.map_name(name)
            if mapped in self.handler_heads.sources:
                return True
            if mapped.startswith("layers."):
                return self._owns_mapped_layer(mapped)
            if mapped == "embed_tokens.weight":
                return self.embed_tokens is not None or (
                    self.lm_head is not None and self.tie_word_embeddings
                )
            if mapped == "lm_head.weight" or (
                self._pipeline_partitioned and mapped.startswith("lm_head.")
            ):
                return self.lm_head is not None
            if mapped.startswith("norm."):
                return self.norm is not None
            return True  # Unknown names remain subject to strict dispatch validation.

        return owns

    def resolve_checkpoint_load_plan(self, name, shape):
        if name in self.handler_heads.sources:
            # Both consumers must see the full incoming tensor. The language
            # projection still takes its own normal TP/vocabulary slice.
            return None, None
        if (
            name == "embed_tokens.weight"
            and self.embed_tokens is None
            and self.lm_head is not None
            and self.tie_word_embeddings
        ):
            return (
                CheckpointLoadTarget(self.lm_head, "weight", True),
                self.lm_head.tied_checkpoint_read_spec("weight", shape),
            )
        return super().resolve_checkpoint_load_plan(name, shape)

    def load_weights(self, weights):
        if isinstance(weights, dict):
            weights_iter = iter(weights.items())
        else:
            weights_iter = weights
        has_lm_head = False

        def _track(it):
            nonlocal has_lm_head
            for name, tensor in it:
                if name in self.handler_heads.sources:
                    tensor = self.handler_heads.capture(name, tensor)
                    if name not in self.handler_heads.vocabulary_sources:
                        continue
                    if name == "lm_head.weight" and self.lm_head is None:
                        continue
                    if (
                        name == "embed_tokens.weight"
                        and self.embed_tokens is None
                        and (self.lm_head is None or not self.tie_word_embeddings)
                    ):
                        continue
                if name == "embed_tokens.weight" and self.embed_tokens is None:
                    value, spec = unwrap_checkpoint_tensor(tensor)
                    self.lm_head.load_tied_checkpoint_weight(value, spec)
                    continue
                if name == "lm_head.weight":
                    has_lm_head = True
                yield name, tensor

        mapped_iter = self.WEIGHTS_MAPPER.apply(weights_iter)
        super().load_weights(_track(mapped_iter))

        # Qwen3 small variants (e.g. Qwen3-0.6B) tie lm_head to embed_tokens.
        # Mirror HF transformers: when no lm_head.weight is in the ckpt, copy
        # the embedding weights into lm_head so the projection isn't random.
        if (
            not has_lm_head
            and self.tie_word_embeddings
            and self.lm_head is not None
            and self.embed_tokens is not None
        ):
            logging.info(
                "[Qwen3ForCausalLM] lm_head.weight not found in ckpt; "
                "tying lm_head to embed_tokens (tie_word_embeddings)"
            )
            self.lm_head._copy_local_tied_weight(self.embed_tokens.weight.data)

    def process_weights_after_loading(self) -> None:
        if self._lm_head_postprocessed or self.lm_head is None:
            return
        processed = self.lm_head.weight
        if self.normalize_lm_head_weight:
            processed = torch.nn.functional.normalize(processed, dim=1)
        if self.logit_scale != 1.0:
            processed = processed * self.logit_scale
        if processed is not self.lm_head.weight:
            with torch.no_grad():
                self.lm_head.weight.copy_(processed)
        self._lm_head_postprocessed = True

    def _init_handler_heads(self, load_config, cfg):
        if getattr(getattr(load_config, "quant_config", None), "quant_type", None) == "pertensor_quant":
            from rtp_llm.models_py.quant_methods.per_tensor_int8 import StaticInt8Mapper

            self.WEIGHTS_MAPPER = StaticInt8Mapper(self.WEIGHTS_MAPPER)
        self.handler_heads = ReplicatedHandlerHeads(
            getattr(load_config, "custom_weight_mappings", ()),
            self.WEIGHTS_MAPPER,
            cfg["hidden_size"],
            cfg["params_dtype"],
            projection_layers=getattr(load_config, "custom_projection_layers", ()),
            protected_roots={
                "embed_tokens",
                "lm_head",
                "norm",
                "layers",
                "handler_heads",
            },
        )

    def __init__(self, model_config: Any, load_config: Any):
        parallelism_config = getattr(load_config, "parallelism_config", None)
        if parallelism_config is None:
            raise ValueError("Qwen3 newloader requires parallelism_config")
        fmha_config = getattr(load_config, "fmha_config", None)
        py_hw_kernel_config = getattr(load_config, "py_hw_kernel_config", None)

        super().__init__(
            config=model_config,
            parallelism_config=parallelism_config,
            weight=None,
            max_generate_batch_size=getattr(load_config, "max_generate_batch_size", 0),
            fmha_config=fmha_config,
            py_hw_kernel_config=py_hw_kernel_config,
        )

        self._native_resident_state = getattr(load_config, "resident_construction", None)
        cfg = self._dense_config(model_config, load_config)
        self._init_handler_heads(load_config, cfg)
        _validate_supported_parallelism(parallelism_config)
        pp_size = int(getattr(parallelism_config, "pp_size", 1))
        is_draft = bool(
            model_config.get("is_mtp", False)
            if isinstance(model_config, dict)
            else getattr(model_config, "is_mtp", False)
        )
        self._pipeline_partitioned = pp_size > 1 and not is_draft
        # Partition finalization and cross-rank agreement are owned by main's
        # engine configuration. Do not independently repartition at load time.
        self._owned_layer_ids = self.layer_ids()
        owns_embedding = 0 in self._owned_layer_ids
        owns_head = cfg["num_layers"] - 1 in self._owned_layer_ids
        self.tie_word_embeddings = (
            bool(model_config.get("tie_word_embeddings", False))
            if isinstance(model_config, dict)
            else bool(getattr(model_config, "tie_word_embeddings", False))
        )
        normalize_lm_head_weight = (
            model_config.get("normalize_lm_head_weight", False)
            if isinstance(model_config, dict)
            else getattr(model_config, "normalize_lm_head_weight", False)
        )
        if not isinstance(normalize_lm_head_weight, bool):
            raise TypeError("normalize_lm_head_weight must be a bool")
        raw_logit_scale = (
            model_config.get("logit_scale", 1.0)
            if isinstance(model_config, dict)
            else getattr(model_config, "logit_scale", 1.0)
        )
        if isinstance(raw_logit_scale, bool) or not isinstance(raw_logit_scale, Real):
            raise TypeError("logit_scale must be a finite real number")
        logit_scale = float(raw_logit_scale)
        if not math.isfinite(logit_scale):
            raise ValueError("logit_scale must be finite")
        self.normalize_lm_head_weight = normalize_lm_head_weight
        self.logit_scale = logit_scale
        self._lm_head_postprocessed = False

        self.embed_tokens = (
            VocabParallelEmbedding(
                vocab_size=cfg.get("input_vocab_size", cfg["vocab_size"]),
                embedding_dim=cfg["hidden_size"],
                allow_vocab_padding=True,
                tp_size=cfg["attn_tp_size"],
                tp_rank=cfg["attn_tp_rank"],
                params_dtype=cfg["params_dtype"],
            )
            if owns_embedding
            else None
        )
        owned_layers = {
            str(i): self._decoder_layer(
                hidden_size=cfg["hidden_size"],
                num_heads=cfg["num_heads"],
                num_kv_heads=cfg["num_kv_heads"],
                intermediate_size=cfg["intermediate_size"],
                head_dim=cfg["head_dim"],
                layer_idx=i,
                attn_tp_size=cfg["attn_tp_size"],
                attn_tp_rank=cfg["attn_tp_rank"],
                ffn_tp_size=cfg["ffn_tp_size"],
                ffn_tp_rank=cfg["ffn_tp_rank"],
                quant_config=cfg["quant_config"],
                params_dtype=cfg["params_dtype"],
                rms_norm_eps=cfg["rms_norm_eps"],
                prefix=f"layers.{i}",
                **self._decoder_kwargs(parallelism_config),
            )
            for i in self._owned_layer_ids
        }
        self.layers = (
            nn.ModuleDict(owned_layers)
            if self._pipeline_partitioned
            else nn.ModuleList(owned_layers.values())
        )
        self.norm = (
            RMSNorm(
                cfg["hidden_size"],
                eps=cfg["rms_norm_eps"],
                params_dtype=cfg["params_dtype"],
            )
            if owns_head
            else None
        )
        # Embedding/classification/reranking consumes normalized hidden states.
        # A handler may independently own the raw full LM projection.
        unspecified = object()
        task_type = (
            model_config.get("task_type", unspecified)
            if isinstance(model_config, dict)
            else getattr(model_config, "task_type", unspecified)
        )
        # A plain HF/draft config defaults to language modeling and does not
        # need native serving config bindings just to construct its parameters.
        needs_lm_head = owns_head
        if needs_lm_head and task_type is not unspecified:
            from rtp_llm.ops import TaskType

            needs_lm_head = task_type == TaskType.LANGUAGE_MODEL
        self.lm_head = (
            ParallelLMHead(
                vocab_size=cfg["vocab_size"],
                hidden_size=cfg["hidden_size"],
                tp_size=cfg["lm_head_tp_size"],
                tp_rank=cfg["lm_head_tp_rank"],
                params_dtype=cfg["lm_head_params_dtype"],
                **output_vocab_head_kwargs(model_config),
            )
            if needs_lm_head
            else None
        )
        if (
            self.tie_word_embeddings
            and self.embed_tokens is not None
            and self.lm_head is not None
        ):
            self.embed_tokens.tie_output_head(self.lm_head)
        self._init_capture_context(
            self._capture_canonical_layer,
            self._capture_canonical_final,
        )

    def _capture_canonical_layer(self, hidden_states, residual):
        return hidden_states

    def _capture_canonical_final(self, hidden_states, residual):
        return self.norm(hidden_states)

    def runtime_weight_view(self) -> dict[str, torch.Tensor]:
        result = self.handler_heads.exports()
        if self.embed_tokens is not None:
            result["embedding"] = self.embed_tokens.weight
        if self.norm is not None:
            result["final_layernorm.gamma"] = self.norm.weight
        if self.lm_head is not None:
            result["lm_head"] = self.lm_head.weight
        return result

    def _input_embeddings(self, inputs):
        return self.embed_tokens(inputs.input_ids)

    def make_empty_intermediate_tensors(self, hidden_template):
        tensors = super().make_empty_intermediate_tensors(hidden_template)
        if self._mtp_aux_capture_layer_ids and not self.config.is_mtp:
            previous_parts, _ = self._aux_hidden_capture_parts()
            if previous_parts:
                # A real upstream stage sends just its completed captures.
                tensors["mtp_aux_hidden_states"] = hidden_template.new_zeros(
                    (hidden_template.shape[0], previous_parts * hidden_template.shape[-1])
                )
        return tensors

    def validate_weights_loaded(self, loaded_tensor_ids=None):
        self.handler_heads.validate()
        super().validate_weights_loaded(loaded_tensor_ids)

    def forward(self, inputs: PyModelInputs, fmha_impl: Any = None) -> PyModelOutputs:
        hidden_states = (
            self._input_embeddings(inputs)
            if self.embed_tokens is not None
            else inputs.pp_intermediates["hidden_states"]
        )
        if fmha_impl is None:
            fmha_impl = self.prepare_fmha_impl(inputs)
        capture = self.capture_context(inputs.capture_hidden_states)
        capture_aux = bool(self._mtp_aux_capture_layer_ids) and not self.config.is_mtp
        if capture_aux:
            self.begin_aux_hidden_capture(
                hidden_states,
                get_primary_attention_inputs(inputs).is_target_verify,
                upstream_hidden=inputs.pp_intermediates.get("mtp_aux_hidden_states"),
            )
        local_layers = (
            self.layers.values()
            if isinstance(self.layers, nn.ModuleDict)
            else self.layers
        )
        for i, layer in enumerate(local_layers):
            layer_fmha_impl = select_fmha_impl_for_layer(fmha_impl, self.kv_cache, i)
            hidden_states = layer(
                hidden_states,
                layer_fmha_impl,
                kv_cache=self.kv_cache.get_layer_cache(i) if self.kv_cache else None,
            )
            global_id = self._owned_layer_ids[i]
            if capture_aux and global_id in self._mtp_aux_capture_layer_id_set:
                self.capture_aux_hidden(global_id, hidden_states)
            capture.capture_layer(global_id, hidden_states)
        if capture_aux:
            self.finish_aux_hidden_capture()
        if self.norm is None:
            from rtp_llm.ops.compute_ops import PyModelOutputs

            outputs = PyModelOutputs(hidden_states)
            intermediates = {"hidden_states": hidden_states}
            if capture_aux and self._mtp_aux_capture_buffer is not None:
                intermediates["mtp_aux_hidden_states"] = self._mtp_aux_capture_buffer
            outputs.pp_intermediates = intermediates
            return outputs
        outputs = capture.finalize(hidden_states)
        if capture_aux:
            outputs.mtp_target_hidden_states = self._mtp_target_hidden_states
        return outputs
