from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from rtp_llm.models_py.distributed.collective_torch import Group, all_reduce
from rtp_llm.models_py.layers.embedding import ParallelLMHead, VocabParallelEmbedding
from rtp_llm.models_py.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from rtp_llm.models_py.layers.norm import RMSNorm
from rtp_llm.models_py.model_desc.module_base import GptModelBase
from rtp_llm.models_py.module_base import rtp_module
from rtp_llm.models_py.quant_methods.base import QuantizationConfig
from rtp_llm.models_py.weight_mapper import WeightsMapper
from rtp_llm.ops.compute_ops import LayerKVCache, PyModelInputs, PyModelOutputs


@rtp_module
class Qwen2MLP(nn.Module):

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
        gate, up = gate_up.chunk(2, dim=-1)
        x = torch.nn.functional.silu(gate) * up
        x = self.down_proj(x)
        if self.tp_size > 1:
            x = all_reduce(x, group=Group.TP)
        return x


@rtp_module
class Qwen2Attention(nn.Module):

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
            bias=True,
            params_dtype=params_dtype,
        )
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
        if self.tp_size > 1:
            output = all_reduce(output, group=Group.TP)
        return output


@rtp_module
class Qwen2DecoderLayer(nn.Module):

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
    ):
        super().__init__()
        self.input_layernorm = RMSNorm(hidden_size, params_dtype=params_dtype)
        self.self_attn = Qwen2Attention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            layer_idx=layer_idx,
            tp_size=tp_size,
            tp_rank=tp_rank,
            quant_config=quant_config,
            params_dtype=params_dtype,
        )
        self.post_attention_layernorm = RMSNorm(hidden_size, params_dtype=params_dtype)
        self.mlp = Qwen2MLP(
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


def _extract_config_values(model_config: Any, load_config: Any):
    if isinstance(model_config, dict):
        hidden_size = model_config.get("hidden_size", 4096)
        num_heads = model_config.get("num_attention_heads", 32)
        num_kv_heads = model_config.get("num_key_value_heads", num_heads)
        intermediate_size = model_config.get("intermediate_size", 11008)
        num_layers = model_config.get("num_hidden_layers", 32)
        vocab_size = model_config.get("vocab_size", 152064)
        head_dim = model_config.get("head_dim", hidden_size // num_heads)
    else:
        hidden_size = getattr(model_config, "hidden_size", 4096)
        num_layers = getattr(
            model_config, "num_layers", getattr(model_config, "num_hidden_layers", 32)
        )
        vocab_size = getattr(model_config, "vocab_size", 152064)
        intermediate_size = getattr(
            model_config,
            "inter_size",
            getattr(model_config, "intermediate_size", 11008),
        )
        attn_config = getattr(model_config, "attn_config", None)
        if attn_config is not None:
            num_heads = getattr(attn_config, "head_num", 32)
            num_kv_heads = getattr(attn_config, "kv_head_num", num_heads)
            head_dim = getattr(attn_config, "size_per_head", hidden_size // num_heads)
        else:
            num_heads = getattr(model_config, "num_attention_heads", 32)
            num_kv_heads = getattr(model_config, "num_key_value_heads", num_heads)
            head_dim = getattr(model_config, "head_dim", hidden_size // num_heads)

    tp_size = getattr(load_config, "tp_size", 1)
    tp_rank = getattr(load_config, "tp_rank", 0)
    quant_config = getattr(load_config, "quant_config", None)
    params_dtype = getattr(load_config, "compute_dtype", torch.float16)

    return dict(
        hidden_size=hidden_size,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        intermediate_size=intermediate_size,
        num_layers=num_layers,
        vocab_size=vocab_size,
        head_dim=head_dim,
        tp_size=tp_size,
        tp_rank=tp_rank,
        quant_config=quant_config,
        params_dtype=params_dtype,
    )


@rtp_module
class Qwen2ForCausalLM(GptModelBase):

    WEIGHTS_MAPPER = WeightsMapper(
        prefix_mapping={"model.": ""},
    )

    def load_weights(self, weights):
        import logging

        if isinstance(weights, dict):
            weights_iter = iter(weights.items())
        else:
            weights_iter = weights
        has_lm_head = False

        def _track(it):
            nonlocal has_lm_head
            for name, tensor in it:
                if name == "lm_head.weight" or name.startswith("lm_head."):
                    has_lm_head = True
                yield name, tensor

        mapped_iter = self.WEIGHTS_MAPPER.apply(_track(weights_iter))
        from rtp_llm.models_py.module_base import _default_load_weights

        _default_load_weights(self, mapped_iter)

        # Handle tied embeddings: when ckpt has no lm_head.weight (e.g. Qwen2.5-0.5B
        # with tie_word_embeddings=true), HF transformers reuses embed_tokens.weight
        # at runtime. Mirror that here so lm_head doesn't stay uninitialized.
        if not has_lm_head:
            logging.info(
                "[Qwen2ForCausalLM] lm_head.weight not found in ckpt; "
                "tying lm_head to embed_tokens (tie_word_embeddings)"
            )
            self.lm_head.weight.data.copy_(self.embed_tokens.weight.data)

    def __init__(self, model_config: Any, load_config: Any):
        parallelism_config = getattr(load_config, "parallelism_config", None)
        fmha_config = getattr(load_config, "fmha_config", None)
        device_resource_config = getattr(load_config, "device_resource_config", None)

        super().__init__(
            config=model_config,
            parallelism_config=parallelism_config,
            weight=None,
            max_generate_batch_size=0,
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
                Qwen2DecoderLayer(
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
                )
                for i in range(cfg["num_layers"])
            ]
        )
        self.norm = RMSNorm(cfg["hidden_size"], params_dtype=cfg["params_dtype"])
        self.lm_head = ParallelLMHead(
            vocab_size=cfg["vocab_size"],
            hidden_size=cfg["hidden_size"],
            tp_size=cfg["tp_size"],
            tp_rank=cfg["tp_rank"],
            params_dtype=cfg["params_dtype"],
        )

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
