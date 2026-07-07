"""Old Qwen (1.0) dense model for the new (vLLM-style) weight loader.

Faithfully ports the legacy ``rtp_llm/models/qwen.py`` checkpoint layout
(Qwen/Qwen-7B-Chat, Qwen-1B8, Qwen-13B, etc.):

  transformer.wte.weight                          -> embed_tokens
  transformer.h.{i}.ln_1.weight                   -> input_layernorm (RMSNorm)
  transformer.h.{i}.attn.c_attn.{weight,bias}     -> FUSED qkv (+bias), split q/k/v
  transformer.h.{i}.attn.c_proj.weight            -> o_proj (no bias)
  transformer.h.{i}.ln_2.weight                   -> post_attention_layernorm (RMSNorm)
  transformer.h.{i}.mlp.w1.weight                 -> gate_proj (SiLU gate)
  transformer.h.{i}.mlp.w2.weight                 -> up_proj
  transformer.h.{i}.mlp.c_proj.weight             -> down_proj
  transformer.ln_f.weight                         -> norm
  lm_head.weight                                 -> lm_head (NOT tied by default)

Old Qwen specifics vs Qwen2/Qwen3:
  * HF ckpt uses Conv1D format ([in, out]) for ALL linear weights -> transposed to
    nn.Linear ([out, in]) in ``_rewrite``.
  * Fused ``c_attn`` (single tensor for q/k/v) + bias -> split into q_proj/k_proj/
    v_proj shard names so ``QKVParallelLinear`` can do TP slicing per-shard.
  * MLP naming: w1=gate, w2=up, c_proj=down (not gate_proj/up_proj/down_proj).
  * ``intermediate_size`` in HF config is total (w1+w2) -> /2 for per-projection.
  * RMSNorm eps = 1e-5 (``layer_norm_epsilon``), not 1e-6.
  * No q_norm/k_norm (unlike Qwen3).
  * Prefix: ``transformer.`` (not ``model.``).  Layer naming: ``h.{i}`` (not ``layers.{i}``).
  * QKV has bias=True; o_proj / MLP projections have no bias.
"""

from typing import Any, Iterator, Optional, Tuple

import torch
import torch.nn as nn

from rtp_llm.models_py.distributed.collective_torch import Group, all_reduce
from rtp_llm.models_py.layers.activation import silu_and_mul
from rtp_llm.models_py.layers.embedding import ParallelLMHead, VocabParallelEmbedding
from rtp_llm.models_py.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from rtp_llm.models_py.layers.norm import RMSNorm
from rtp_llm.models_py.model_desc.module_base import GptModelBase
from rtp_llm.models_py.module_base import RtpModule
from rtp_llm.models_py.quant_methods.base import QuantizationConfig
from rtp_llm.ops.compute_ops import LayerKVCache, PyModelInputs, PyModelOutputs


class QwenMLP(RtpModule):

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
        if self.tp_size > 1:
            x = all_reduce(x, group=Group.TP)
        return x


class QwenAttention(RtpModule):
    """Old Qwen (1.0) attention.

    Differs from Qwen2 attention only by having the same default eps;
    Qwen2 also uses bias=True. Differs from Qwen3 by:
      * Has qkv bias (Qwen3 dense does not).
      * No per-head q_norm / k_norm.
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
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.layer_idx = layer_idx
        self.tp_size = tp_size

        self.num_heads_per_partition = num_heads // tp_size
        self.num_kv_heads_per_partition = max(1, num_kv_heads // tp_size)
        self.q_size = self.num_heads_per_partition * head_dim
        self.kv_size = self.num_kv_heads_per_partition * head_dim

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


class QwenDecoderLayer(RtpModule):

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
        rms_norm_eps: float = 1e-5,
    ):
        super().__init__()
        self.input_layernorm = RMSNorm(
            hidden_size, eps=rms_norm_eps, params_dtype=params_dtype
        )
        self.self_attn = QwenAttention(
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
        self.post_attention_layernorm = RMSNorm(
            hidden_size, eps=rms_norm_eps, params_dtype=params_dtype
        )
        self.mlp = QwenMLP(
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
        num_heads = model_config.get(
            "n_head", model_config.get("num_attention_heads", 32)
        )
        num_kv_heads = num_heads  # Old Qwen (1.0) has no GQA
        # HF Qwen's intermediate_size is the total (w1+w2); each projection
        # uses intermediate_size // 2.
        raw_inter = model_config.get(
            "intermediate_size", model_config.get("ffn_hidden_size", 22016)
        )
        intermediate_size = raw_inter // 2
        num_layers = model_config.get(
            "num_hidden_layers", model_config.get("n_layer", 32)
        )
        vocab_size = model_config.get(
            "vocab_size", model_config.get("padded_vocab_size", 152064)
        )
        head_dim = model_config.get("kv_channels", hidden_size // num_heads)
        rms_norm_eps = model_config.get("layer_norm_epsilon", 1e-5)
    else:
        hidden_size = getattr(model_config, "hidden_size", 4096)
        num_layers = getattr(
            model_config, "num_layers", getattr(model_config, "num_hidden_layers", 32)
        )
        vocab_size = getattr(model_config, "vocab_size", 152064)
        # inter_size is already the per-projection size (divided by 2 in the
        # legacy _from_hf).
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
        rms_norm_eps = getattr(
            model_config,
            "layernorm_eps",
            getattr(model_config, "rms_norm_eps", 1e-5),
        )

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
        rms_norm_eps=rms_norm_eps,
        tp_size=tp_size,
        tp_rank=tp_rank,
        quant_config=quant_config,
        params_dtype=params_dtype,
    )


class QwenForCausalLM(GptModelBase):

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

        # Stored for _rewrite: QKV fusion splitting needs head counts.
        self._num_heads = cfg["num_heads"]
        self._num_kv_heads = cfg["num_kv_heads"]
        self._head_dim = cfg["head_dim"]

        self.embed_tokens = VocabParallelEmbedding(
            vocab_size=cfg["vocab_size"],
            embedding_dim=cfg["hidden_size"],
            tp_size=cfg["tp_size"],
            tp_rank=cfg["tp_rank"],
            params_dtype=cfg["params_dtype"],
        )
        self.layers = nn.ModuleList(
            [
                QwenDecoderLayer(
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
            params_dtype=cfg["params_dtype"],
        )

    # ------------------------------------------------------------------ #
    #  Weight loading: rename Qwen ckpt names + transpose Conv1D +
    #                    split fused qkv
    # ------------------------------------------------------------------ #
    def _rewrite(
        self, name: str, tensor: torch.Tensor
    ) -> Iterator[Tuple[str, torch.Tensor]]:
        # Global tensors.
        if name == "transformer.wte.weight":
            yield "embed_tokens.weight", tensor
            return
        if name == "transformer.ln_f.weight":
            yield "norm.weight", tensor
            return
        if name == "lm_head.weight":
            yield "lm_head.weight", tensor
            return

        # Per-layer tensors.
        prefix = "transformer.h."
        if not name.startswith(prefix):
            # Unknown global (e.g. rotary buffers) -- drop.
            return
        rest = name[len(prefix) :]
        layer_idx, sub = rest.split(".", 1)
        base = f"layers.{layer_idx}"

        if sub == "ln_1.weight":
            yield f"{base}.input_layernorm.weight", tensor
        elif sub == "ln_2.weight":
            yield f"{base}.post_attention_layernorm.weight", tensor
        elif sub in ("attn.c_attn.weight", "attn.c_attn.bias"):
            # Fused qkv: Conv1D stores weight as [in, out]; transpose to
            # [out, in] then split rows [q | k | v] into separate shard names
            # so the attention module's redirect maps them onto qkv_proj (same
            # contract as qwen3's separate q_proj/k_proj/v_proj ckpt tensors).
            q = self._num_heads * self._head_dim
            kv = self._num_kv_heads * self._head_dim
            if sub.endswith(".weight"):
                t = tensor.t().contiguous()
                yield f"{base}.self_attn.q_proj.weight", t[0:q]
                yield f"{base}.self_attn.k_proj.weight", t[q : q + kv]
                yield f"{base}.self_attn.v_proj.weight", t[q + kv : q + 2 * kv]
            else:  # bias (1-D, no transpose)
                yield f"{base}.self_attn.q_proj.bias", tensor[0:q]
                yield f"{base}.self_attn.k_proj.bias", tensor[q : q + kv]
                yield f"{base}.self_attn.v_proj.bias", tensor[q + kv : q + 2 * kv]
        elif sub == "attn.c_proj.weight":
            # Conv1D [in, out] -> nn.Linear [out, in]
            yield f"{base}.self_attn.o_proj.weight", tensor.t().contiguous()
        elif sub == "mlp.w1.weight":
            # w1 = gate; Conv1D [in, out] -> nn.Linear [out, in]
            yield f"{base}.mlp.gate_proj.weight", tensor.t().contiguous()
        elif sub == "mlp.w2.weight":
            # w2 = up; Conv1D [in, out] -> nn.Linear [out, in]
            yield f"{base}.mlp.up_proj.weight", tensor.t().contiguous()
        elif sub == "mlp.c_proj.weight":
            # c_proj = down; Conv1D [in, out] -> nn.Linear [out, in]
            yield f"{base}.mlp.down_proj.weight", tensor.t().contiguous()
        # else: drop unknown per-layer weights

    def load_weights(self, weights):
        import logging

        if isinstance(weights, dict):
            weights = iter(weights.items())

        has_lm_head = False

        def _stream():
            nonlocal has_lm_head
            for name, tensor in weights:
                if name == "lm_head.weight" or name.startswith("lm_head."):
                    has_lm_head = True
                yield from self._rewrite(name, tensor)

        super().load_weights(_stream())

        # Old Qwen defaults to tie_word_embeddings=False, but some ckpts
        # (e.g. Qwen-1B8 variants) may tie. Mirror HF: when no lm_head.weight
        # is in the ckpt, copy embed_tokens into lm_head.
        if not has_lm_head:
            logging.info(
                "[QwenForCausalLM] lm_head.weight not found in ckpt; "
                "tying lm_head to embed_tokens (tie_word_embeddings)"
            )
            self.lm_head.weight.data.copy_(self.embed_tokens.weight.data)

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
