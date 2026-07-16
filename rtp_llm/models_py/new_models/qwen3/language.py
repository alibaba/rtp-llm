import logging
from typing import Any, Optional

import torch
import torch.nn as nn

from rtp_llm.models_py.layers.activation import silu_and_mul
from rtp_llm.models_py.layers.embedding import ParallelLMHead, VocabParallelEmbedding
from rtp_llm.models_py.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from rtp_llm.models_py.layers.norm import RMSNorm
from rtp_llm.models_py.module_base import RtpModule
from rtp_llm.models_py.new_models.model_base import (
    NewLoaderModelBase,
    required_config_value,
)
from rtp_llm.models_py.quant_methods.base import QuantizationConfig
from rtp_llm.models_py.weight_mapper import WeightsMapper
from rtp_llm.ops.compute_ops import (
    LayerKVCache,
    PyModelInputs,
    PyModelOutputs,
    rtp_llm_ops,
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
    ):
        super().__init__()
        self.tp_size = tp_size
        self.gate_up_proj = MergedColumnParallelLinear(
            input_size=hidden_size,
            output_size=2 * intermediate_size,
            tp_size=tp_size,
            tp_rank=tp_rank,
            quant_config=quant_config,
            prefix="gate_up_proj",
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
            prefix="down_proj",
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
            prefix="qkv_proj",
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
            prefix="o_proj",
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
        tp_size: int = 1,
        tp_rank: int = 0,
        quant_config: Optional[QuantizationConfig] = None,
        params_dtype: torch.dtype = torch.float16,
        rms_norm_eps: float = 1e-6,
    ):
        super().__init__()
        self.input_layernorm = RMSNorm(
            hidden_size, eps=rms_norm_eps, params_dtype=params_dtype
        )
        self.self_attn = Qwen3Attention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            layer_idx=layer_idx,
            tp_size=tp_size,
            tp_rank=tp_rank,
            quant_config=quant_config,
            params_dtype=params_dtype,
            rms_norm_eps=rms_norm_eps,
        )
        self.post_attention_layernorm = RMSNorm(
            hidden_size, eps=rms_norm_eps, params_dtype=params_dtype
        )
        self.mlp = Qwen3MLP(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            tp_size=tp_size,
            tp_rank=tp_rank,
            quant_config=quant_config,
            params_dtype=params_dtype,
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


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return value


def _resolve_head_dim(hidden_size: Any, num_heads: Any, configured: Any) -> int:
    hidden_size = _positive_int(hidden_size, "hidden_size")
    num_heads = _positive_int(num_heads, "num_heads")
    if configured is not None:
        return _positive_int(configured, "head_dim")
    if hidden_size % num_heads != 0:
        raise ValueError(
            f"hidden_size={hidden_size} must be divisible by num_heads={num_heads} "
            "when head_dim is not configured"
        )
    return hidden_size // num_heads


def _extract_config_values(model_config: Any, load_config: Any):
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
        "intermediate_size": intermediate_size,
        "num_layers": num_layers,
        "vocab_size": vocab_size,
        "head_dim": head_dim,
    }
    for name, value in dimensions.items():
        _positive_int(value, name)

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
        quant_config=quant_config,
        params_dtype=params_dtype,
        lm_head_params_dtype=torch.float32 if enable_fp32_lm_head else params_dtype,
    )


class Qwen3ForCausalLM(NewLoaderModelBase):

    WEIGHTS_MAPPER = WeightsMapper(
        prefix_mapping={"model.": ""},
    )

    def load_weights(self, weights):
        if isinstance(weights, dict):
            weights_iter = iter(weights.items())
        else:
            weights_iter = weights
        has_lm_head = False

        def _track(it):
            nonlocal has_lm_head
            for name, tensor in it:
                if name == "lm_head.weight":
                    has_lm_head = True
                yield name, tensor

        mapped_iter = self.WEIGHTS_MAPPER.apply(weights_iter)
        super().load_weights(_track(mapped_iter))

        # Qwen3 small variants (e.g. Qwen3-0.6B) tie lm_head to embed_tokens.
        # Mirror HF transformers: when no lm_head.weight is in the ckpt, copy
        # the embedding weights into lm_head so the projection isn't random.
        if not has_lm_head:
            logging.info(
                "[Qwen3ForCausalLM] lm_head.weight not found in ckpt; "
                "tying lm_head to embed_tokens (tie_word_embeddings)"
            )
            self.lm_head._copy_weight(self.embed_tokens.weight.data)

    def __init__(self, model_config: Any, load_config: Any):
        parallelism_config = getattr(load_config, "parallelism_config", None)
        if parallelism_config is None:
            raise ValueError("Qwen3 newloader requires parallelism_config")
        fmha_config = getattr(load_config, "fmha_config", None)
        device_resource_config = getattr(load_config, "device_resource_config", None)

        super().__init__(
            config=model_config,
            parallelism_config=parallelism_config,
            fmha_config=fmha_config,
            device_resource_config=device_resource_config,
        )

        cfg = _extract_config_values(model_config, load_config)

        self.embed_tokens = VocabParallelEmbedding(
            vocab_size=cfg["vocab_size"],
            embedding_dim=cfg["hidden_size"],
            tp_size=cfg["tp_size"],
            tp_rank=cfg["tp_rank"],
            params_dtype=cfg["params_dtype"],
        )
        self.layers = nn.ModuleList(
            [
                Qwen3DecoderLayer(
                    hidden_size=cfg["hidden_size"],
                    num_heads=cfg["num_heads"],
                    num_kv_heads=cfg["num_kv_heads"],
                    intermediate_size=cfg["intermediate_size"],
                    head_dim=cfg["head_dim"],
                    layer_idx=i,
                    tp_size=cfg["tp_size"],
                    tp_rank=cfg["tp_rank"],
                    quant_config=cfg["quant_config"],
                    params_dtype=cfg["params_dtype"],
                    rms_norm_eps=cfg["rms_norm_eps"],
                )
                for i in range(cfg["num_layers"])
            ]
        )
        self.norm = RMSNorm(
            cfg["hidden_size"],
            eps=cfg["rms_norm_eps"],
            params_dtype=cfg["params_dtype"],
        )
        self.lm_head = ParallelLMHead(
            vocab_size=cfg["vocab_size"],
            hidden_size=cfg["hidden_size"],
            tp_size=cfg["tp_size"],
            tp_rank=cfg["tp_rank"],
            params_dtype=cfg["lm_head_params_dtype"],
        )

    def runtime_weight_view(self) -> dict[str, torch.Tensor]:
        return {
            "embedding": self.embed_tokens.weight,
            "final_layernorm.gamma": self.norm.weight,
            "lm_head": self.lm_head.weight,
        }

    def forward(self, inputs: PyModelInputs, fmha_impl: Any = None) -> PyModelOutputs:
        input_ids = inputs.input_ids
        hidden_states = self.embed_tokens(input_ids)
        if fmha_impl is None:
            fmha_impl = self.prepare_fmha_impl(inputs)
        for i, layer in enumerate(self.layers):
            hidden_states = layer(
                hidden_states,
                fmha_impl,
                kv_cache=self.kv_cache.get_layer_cache(i) if self.kv_cache else None,
            )
        hidden_states = self.norm(hidden_states)
        return PyModelOutputs(hidden_states, fmha_impl.fmha_params)
