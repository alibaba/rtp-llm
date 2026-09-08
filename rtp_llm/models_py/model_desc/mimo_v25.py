"""MiMo V2.5 forward implementation.

What sets this apart from the qwen3 / generic_moe templates is that the two attention
kinds disagree on four things -- KV head count, window size, RoPE base and whether a
per-head sink bias applies -- so one FMHA implementation is built per kind and layers
pick theirs by cache group tag.
"""

from typing import Any, Dict, List, Optional

import torch
from torch import nn

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.model_loader.model_weight_info import ModelWeights
from rtp_llm.models_py.model_desc.block_map import (
    get_attention_inputs_value,
    select_attention_inputs_for_tag,
)
from rtp_llm.models_py.model_desc.generic_moe import GenericMoeLayer
from rtp_llm.models_py.model_desc.module_base import GptModelBase
from rtp_llm.models_py.modules import (
    CausalAttention,
    DenseMLP,
    Embedding,
    FMHAImplBase,
    RMSNorm,
)
from rtp_llm.models_py.modules.factory.attention.attn_factory import get_fmha_impl
from rtp_llm.models_py.modules.factory.attention.cuda_impl.py_flashinfer_mha import (
    PyFlashinferPagedPrefillImpl,
    PyFlashinferPrefillImpl,
)
from rtp_llm.ops import (
    HWKernelConfig,
    HybridAttentionType,
    MoeConfig,
    ParallelismConfig,
)
from rtp_llm.ops.compute_ops import LayerKVCache, PyModelInputs, PyModelOutputs
from rtp_llm.utils.model_weight import W

# Consumer side of the cache-group tags ``rtp_llm/models/mimo_v25.py`` puts into
# ``ModelConfig.kv_cache_spec_descs`` and that CacheConfig/CacheTopology then publishes
# as ``KVCache.group_tags`` and as the keys of ``PyModelInputs.attention_inputs``.
# Duplicated rather than imported because model_desc modules sit below rtp_llm.models in
# the dependency order (see the dsv4 precedent in modules/dsv4/kv_cache_utils.py). Keep
# both in sync.
GA_KV_TAG = "ga_kv"
SWA_KV_TAG = "swa_kv"


class MiMoV25DecoderLayer(nn.Module):
    def __init__(
        self,
        config: ModelConfig,
        parallelism_config: ParallelismConfig,
        layer_idx: int,
        weights: Dict[str, torch.Tensor],
        moe_config: MoeConfig,
        max_generate_batch_size: int = 0,
        enable_cuda_graph: bool = False,
        quant_config: Optional[object] = None,
        hw_kernel_config: Optional["HWKernelConfig"] = None,
        is_ga: bool = False,
    ):
        super().__init__()
        self.is_ga = is_ga
        self.attn_configs = build_layer_attn_configs(config, parallelism_config, is_ga)

        self.self_attn = CausalAttention(
            self.attn_configs,
            parallelism_config,
            weights,
            config.layernorm_eps,
            quant_config,
            hw_kernel_config,
            layer_idx,
        )
        # Sliding-window layers carry a per-head sink bias; global-attention layers do not.
        self.sink_bias: Optional[torch.Tensor] = weights.get(W.attn_sink_bias, None)
        # Layer 0 is a dense FFN (inter=16384); the rest are 256-expert MoE layers
        # (noaux_tc routing, with the correction_bias logic handled inside GenericMoeLayer)
        if layer_idx in config.moe_layer_index:
            self.mlp = GenericMoeLayer(
                config,
                parallelism_config,
                weights,
                moe_config,
                max_generate_batch_size,
                enable_cuda_graph=enable_cuda_graph,
                hw_kernel_config=hw_kernel_config,
            )
        else:
            self.mlp = DenseMLP(
                config.activation_type,
                parallelism_config,
                weights,
                quant_config,
                hw_kernel_config,
            )
        self.input_layernorm = RMSNorm(
            weights[W.pre_ln_gamma], eps=config.layernorm_eps
        )
        self.post_attention_layernorm = RMSNorm(
            weights[W.post_ln_gamma], eps=config.layernorm_eps
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        fmha_impl: FMHAImplBase,
        kv_cache: Optional[LayerKVCache] = None,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states, fmha_impl=fmha_impl, kv_cache=kv_cache
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states


def build_layer_attn_configs(
    config: ModelConfig, parallelism_config: ParallelismConfig, is_ga: bool
):
    """Attention configs for one layer kind.

    ``getAttentionConfigs`` returns the model-wide values; the four fields the two kinds
    disagree on are overridden here. ``rope_config.dim`` (64, partial RoPE) is shared, so
    the value from config parsing carries through unchanged.
    """
    swa_config = config.hybrid_attention_config.swa_attention_config
    tp_size = parallelism_config.get_attn_tp_size()
    attn_configs = config.getAttentionConfigs(tp_size)
    if is_ga:
        attn_configs.kv_head_num = swa_config.ga_kv_head_num // tp_size  # 4 / tp
        attn_configs.sliding_window = 0  # not windowed
        attn_configs.rope_config.base = int(config.attn_config.rope_config.base)  # 1e7
        attn_configs.add_sink_bias = False
    else:
        attn_configs.kv_head_num = swa_config.swa_kv_head_num // tp_size  # 8 / tp
        attn_configs.sliding_window = swa_config.window_size  # 128
        attn_configs.rope_config.base = int(swa_config.swa_rope_theta)  # 1e4
        attn_configs.add_sink_bias = swa_config.add_sink_bias  # True
    return attn_configs


class MiMoV25Model(GptModelBase):
    def __init__(
        self,
        config: ModelConfig,
        parallelism_config: ParallelismConfig,
        weights: ModelWeights,
        moe_config: MoeConfig,
        max_generate_batch_size: int,
        fmha_config=None,
        py_hw_kernel_config=None,
        device_resource_config=None,
    ):
        super().__init__(
            config,
            parallelism_config,
            weights,
            max_generate_batch_size=max_generate_batch_size,
            fmha_config=fmha_config,
            py_hw_kernel_config=py_hw_kernel_config,
            device_resource_config=device_resource_config,
        )
        types = config.hybrid_attention_config.hybrid_attention_types
        self.is_ga_layer: List[bool] = [
            t != HybridAttentionType.SLIDING_WINDOW for t in types
        ]

        # The fused-qkv FP8 ckpt is natively sliced into 4 slabs, so MiMoPerBlockFp8Weight
        # only accepts tp_size == 4 (see models/mimo_v25_weight.py). Both KV head counts
        # divide evenly at that size, which is what lets build_layer_attn_configs use plain
        # floor division. Assert the precondition rather than carrying a non-divisible
        # fallback that the weight loader would reject anyway.
        swa_config = config.hybrid_attention_config.swa_attention_config
        tp_size = parallelism_config.get_attn_tp_size()
        for name, kv_head_num in (
            ("ga_kv_head_num", swa_config.ga_kv_head_num),
            ("swa_kv_head_num", swa_config.swa_kv_head_num),
        ):
            assert (
                kv_head_num % tp_size == 0
            ), f"{name}={kv_head_num} must be divisible by attn tp_size={tp_size}"

        enable_cuda_graph = (
            py_hw_kernel_config.enable_cuda_graph
            if py_hw_kernel_config is not None
            else False
        )

        self.embed_tokens = Embedding(
            config, parallelism_config, weights.get_global_weight(W.embedding)
        )
        self.layers = nn.ModuleList(
            [
                MiMoV25DecoderLayer(
                    config,
                    parallelism_config,
                    idx,
                    weights.weights[idx],
                    moe_config,
                    max_generate_batch_size,
                    enable_cuda_graph=enable_cuda_graph,
                    quant_config=config.quant_config,
                    hw_kernel_config=py_hw_kernel_config,
                    is_ga=self.is_ga_layer[idx],
                )
                for idx in range(self.layer_num)
            ]
        )
        self.norm = RMSNorm(
            weights.get_global_weight(W.final_ln_gamma), eps=config.layernorm_eps
        )

    def prepare_fmha_impl(
        self, inputs: PyModelInputs, is_cuda_graph: bool = False
    ) -> dict[str, Any]:
        """Build one FMHA implementation per attention kind.

        The base implementation derives a single ``AttentionConfigs`` from the model
        config, which cannot express two KV head counts, two RoPE bases and two window
        sizes. Call the factory directly with the per-kind configs instead.
        """
        attention_inputs = get_attention_inputs_value(inputs)
        impls: dict[str, Any] = {}
        for tag, is_ga in ((GA_KV_TAG, True), (SWA_KV_TAG, False)):
            group_inputs = select_attention_inputs_for_tag(attention_inputs, tag)
            impl = get_fmha_impl(
                build_layer_attn_configs(self.config, self.parallelism_config, is_ga),
                self.weight,
                group_inputs,
                self.fmha_config,
                self.config.quant_config,
                is_cuda_graph,
                self.config.max_seq_len,
                self.parallelism_config,
            )
            if group_inputs.is_prefill:
                if not is_ga and not isinstance(
                    impl, (PyFlashinferPrefillImpl, PyFlashinferPagedPrefillImpl)
                ):
                    raise RuntimeError(
                        "MiMo sliding-window prefill requires ragged or SWA-paged KV attention, "
                        f"got {type(impl).__name__}"
                    )
                backend = getattr(getattr(impl, "fmha_impl", None), "backend", None)
                if backend != "fa2":
                    raise RuntimeError(
                        f"MiMo prefill requires the FlashInfer FA2 backend for tag {tag}, "
                        f"got {type(impl).__name__} with backend={backend!r}"
                    )
            impls[tag] = impl

        # Sliding-window layers need an implementation that both honours window_left and
        # accepts a per-head sink bias. Probe the capability here rather than having the
        # shared implementations opt out on MiMo's config fields, which would also change
        # dispatch for other models that set them: set_sink_bias is implemented only by
        # the PyFlashinfer family, which is also the only family that forwards
        # sliding_window to the kernel. Global-attention layers are checked too because
        # forward() clears their bias through the same method.
        for tag, impl in impls.items():
            assert hasattr(impl, "set_sink_bias"), (
                f"MiMo needs a window- and sink-capable fmha impl for tag {tag}, "
                f"got {type(impl).__name__}"
            )
        return impls

    def forward(self, inputs: PyModelInputs, fmha_impl: Any = None) -> PyModelOutputs:
        input_ids: torch.Tensor = inputs.input_ids
        hidden_states = self.embed_tokens(input_ids)
        if fmha_impl is None:
            fmha_impl = self.prepare_fmha_impl(inputs)

        for i, decoder_layer in enumerate(self.layers[: self.layer_num]):
            # Resolved from the layer kind rather than through select_fmha_impl_for_layer:
            # that helper reads the tag off the KV cache, which does not exist during
            # warmup, and this model already knows which kind each layer is.
            layer_impl = fmha_impl[GA_KV_TAG if self.is_ga_layer[i] else SWA_KV_TAG]
            # Global-attention layers must be handed None so the previous layer's bias
            # cannot leak forward.
            layer_impl.set_sink_bias(decoder_layer.sink_bias)
            hidden_states = decoder_layer(
                hidden_states,
                layer_impl,
                kv_cache=self.kv_cache.get_layer_cache(i) if self.kv_cache else None,
            )
        hidden_states = self.norm(hidden_states)
        return PyModelOutputs(hidden_states)


__all__ = [
    "MiMoV25Model",
]
