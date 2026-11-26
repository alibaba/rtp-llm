from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

import torch

from rtp_llm.models_py.layers.activation import silu_and_mul
from rtp_llm.models_py.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from rtp_llm.models_py.layers.norm import decoder_rms_norm
from rtp_llm.models_py.module_base import RtpModule
from rtp_llm.models_py.new_models.model_base import (
    positive_int as _positive_int,
    required_config_value,
    resolve_head_dim as _resolve_head_dim,
)
from rtp_llm.models_py.new_models.qwen2.trt_checkpoint import Qwen2CheckpointMapper
from rtp_llm.models_py.new_models.qwen3.language import Qwen3ForCausalLM
from rtp_llm.models_py.quant_methods.base import QuantizationConfig
if TYPE_CHECKING:
    from rtp_llm.ops.compute_ops import LayerKVCache


class Qwen2MLP(RtpModule):

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


class Qwen2Attention(RtpModule):
    """Dense GQA attention with configurable Q/K/V bias and no Q/K norm."""

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
        prefix: str = "self_attn",
        qkv_bias: bool = True,
        qkv_bias_as_matrix: bool = False,
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
            bias=qkv_bias,
            # Splitting the concatenated Q/K/V biases leaves a [1, N] row
            # on every attention TP rank. F.linear may select a different
            # CUDA bias epilogue for [N]; the two can round differently.
            bias_as_matrix=qkv_bias_as_matrix or tp_size > 1,
            params_dtype=params_dtype,
        )
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

    def forward(
        self,
        hidden_states: torch.Tensor,
        fmha_impl: Any,
        kv_cache: Optional[LayerKVCache] = None,
    ) -> torch.Tensor:
        input_shape = hidden_states.shape[:-1]
        qkv = self.qkv_proj(hidden_states)
        attn_output = fmha_impl.forward(qkv, kv_cache, self.layer_idx)
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        output = self.o_proj(attn_output)
        return output


class Qwen2DecoderLayer(RtpModule):

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
        qkv_bias: bool = True,
        qkv_bias_as_matrix: bool = False,
    ):
        super().__init__()
        self.input_layernorm = decoder_rms_norm(
            hidden_size, eps=rms_norm_eps, params_dtype=params_dtype, quant_config=quant_config
        )
        self.self_attn = Qwen2Attention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            layer_idx=layer_idx,
            tp_size=attn_tp_size,
            tp_rank=attn_tp_rank,
            quant_config=quant_config,
            params_dtype=params_dtype,
            prefix=f"{prefix}.self_attn",
            qkv_bias=qkv_bias,
            qkv_bias_as_matrix=qkv_bias_as_matrix,
        )
        self.post_attention_layernorm = decoder_rms_norm(
            hidden_size, eps=rms_norm_eps, params_dtype=params_dtype, quant_config=quant_config
        )
        self.mlp = Qwen2MLP(
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
    allow_context_parallel: bool = False,
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

    input_vocab_size = (
        model_config.get("input_vocab_size", 0)
        if isinstance(model_config, dict)
        else getattr(model_config, "input_vocab_size", 0)
    )
    if input_vocab_size == 0 and not isinstance(input_vocab_size, bool):
        input_vocab_size = vocab_size
    _positive_int(input_vocab_size, "input_vocab_size")

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
    context_parallel = _context_parallel_enabled(
        getattr(load_config, "parallelism_config", None)
    )
    if context_parallel and allow_context_parallel:
        # CP partitions tokens, not attention or dense FFN weights. The existing
        # CP FMHA implementation owns KV communication across the physical TP
        # group; a row-parallel reduction here would mix unrelated token shards.
        if (attn_tp_size, attn_tp_rank, ffn_tp_size, ffn_tp_rank) != (1, 0, 1, 0):
            raise ValueError(
                "Context parallelism requires replicated attention and dense FFN "
                "weights (attention TP=1/rank=0, FFN TP=1/rank=0)"
            )
    elif (attn_tp_size, attn_tp_rank) != (tp_size, tp_rank):
        raise ValueError(
            "Context parallelism is not supported by the dense GQA newloader path"
        )
    if (ffn_tp_size, ffn_tp_rank) != (attn_tp_size, attn_tp_rank):
        raise ValueError(
            "Independent FFN TP/sequence parallelism is not supported by the "
            "dense GQA newloader path"
        )
    if (lm_head_tp_size, lm_head_tp_rank) != (tp_size, tp_rank):
        raise ValueError(
            "A separate LM head topology is not supported by the dense GQA "
            "newloader path"
        )

    return dict(
        hidden_size=hidden_size,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        intermediate_size=intermediate_size,
        num_layers=num_layers,
        vocab_size=vocab_size,
        input_vocab_size=input_vocab_size,
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


def _context_parallel_enabled(
    parallelism_config: Any, *, include_prefill_peer: bool = False
) -> bool:
    prefill_cp = getattr(parallelism_config, "prefill_cp_config", None)
    if prefill_cp is not None:
        # PREFILL_CP denotes a decode peer of a CP prefill service; it retains
        # ordinary TP weight sharding. Match the C++ get_attn/ffn_tp_* getters,
        # which only return replicated partitions for is_enabled().
        checkers = (
            ("is_enabled", "is_prefill_enabled")
            if include_prefill_peer
            else ("is_enabled",)
        )
        for checker_name in checkers:
            checker = getattr(prefill_cp, checker_name, None)
            if callable(checker) and bool(checker()):
                return True
    return False


def _validate_supported_parallelism(
    parallelism_config: Any, *, allow_context_parallel: bool = False
) -> None:
    if (
        _context_parallel_enabled(parallelism_config, include_prefill_peer=True)
        and not allow_context_parallel
    ):
        raise ValueError(
            "Context parallelism is not supported by this dense GQA consumer"
        )
    ffn_disaggregate = getattr(parallelism_config, "ffn_disaggregate_config", None)
    if bool(getattr(ffn_disaggregate, "enable_ffn_disaggregate", False)):
        raise ValueError(
            "FFN disaggregation is not supported by the dense GQA newloader path"
        )


class Qwen2ForCausalLM(Qwen3ForCausalLM):
    """Qwen2 owns its biased GQA decoder and shares the main dense PP runtime."""

    supports_ft_handler_heads = True
    QKV_BIAS = True
    WEIGHTS_MAPPER = Qwen2CheckpointMapper()
    _decoder_layer = Qwen2DecoderLayer

    def __init__(self, model_config, load_config):
        if (
            type(self.WEIGHTS_MAPPER) is Qwen2CheckpointMapper
            and getattr(getattr(load_config, "quant_config", None), "quant_type", None)
            == "smooth_quant"
        ):
            from rtp_llm.models_py.new_models.qwen2.trt_checkpoint import (
                Qwen2SmoothTrtMapper,
            )

            self.WEIGHTS_MAPPER = Qwen2SmoothTrtMapper(
                exact_mapping=self.WEIGHTS_MAPPER.exact_mapping
            )
        super().__init__(model_config, load_config)

    @staticmethod
    def _dense_config(model_config, load_config):
        return _extract_config_values(
            model_config, load_config, allow_context_parallel=True
        )

    @classmethod
    def _decoder_kwargs(cls, parallelism_config):
        dp_size = getattr(parallelism_config, "dp_size", 1)
        if isinstance(dp_size, bool) or not isinstance(dp_size, int) or dp_size <= 0:
            raise ValueError(f"DP size must be a positive integer, got {dp_size!r}")
        # DP/EP also run the split when attention weights are replicated.
        ep_size = getattr(parallelism_config, "ep_size", 1)
        return dict(
            qkv_bias=cls.QKV_BIAS, qkv_bias_as_matrix=dp_size > 1 or ep_size > 1
        )

    def load_ft_checkpoint(self, checkpoint):
        from rtp_llm.models_py.new_models.qwen2.ft_checkpoint import (
            load_dense_ft_checkpoint,
        )

        load_dense_ft_checkpoint(self, checkpoint)


class Qwen2EmbeddingBackbone(Qwen2ForCausalLM):
    """Bidirectional Qwen2 blocks; the service handler owns sentence pooling."""


    def __init__(self, model_config: Any, load_config: Any):
        # Both the FMHA selector and the default pooling handler consult this
        # same config. A causal config can yield plausible but wrong vectors.
        if model_config.attn_config.is_causal:
            raise ValueError("qwen_2_embedding requires bidirectional attention")
        super().__init__(model_config, load_config)
