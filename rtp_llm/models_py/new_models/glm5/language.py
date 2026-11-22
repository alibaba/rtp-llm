"""Direct GLM-5 score/draft owners. Checkpoint storage and graph state are separate."""

import copy
from types import SimpleNamespace

import torch
from torch import nn

from rtp_llm.config.mega_moe import (
    FP4_MEGA_MOE_STRATEGIES,
    FUSED_SHARED_STRATEGIES,
    MEGA_MOE_STRATEGIES,
    MegaMoeExecutionPolicy,
)
from rtp_llm.models_py.layers.moe_experts import BaseMoEExperts
from rtp_llm.models_py.model_desc.generic_moe_mtp import GenericMoeMTPModel
from rtp_llm.models_py.model_desc.glm5 import (
    DecodeLayerOutput,
    Glm5MoeDecoderLayer,
    Glm5MoeLayer,
    Glm5MoeModel,
)
from rtp_llm.models_py.model_desc.module_base import GptModelBase
from rtp_llm.models_py.modules.hybrid.glm5_cmp import Glm5Cmp, resolve_glm5_cmp_enabled
from rtp_llm.models_py.new_models.deepseek_v3.attention import _fp8_scale_parameter
from rtp_llm.models_py.new_models.deepseek_v3.language import (
    DeepSeekV32ForCausalLM,
    checkpoint_path,
    nonnegative_int,
    positive_int,
)
from rtp_llm.models_py.new_models.deepseek_v3.model import DeepSeekV32DecoderLayer
from rtp_llm.models_py.new_models.deepseek_v3.moe import DeepSeekV32MoEBlock
from rtp_llm.models_py.new_models.deepseek_v3_mtp.language import (
    DeepSeekV32MTPForCausalLM,
)
from rtp_llm.models_py.new_models.glm5.packed_experts import Glm5PackedExpertMethod
from rtp_llm.models_py.read_spec import unwrap_checkpoint_tensor
from rtp_llm.models_py.utils.fuse_config import fuse_kernels_enabled
from rtp_llm.ops import RoleType
from rtp_llm.utils.model_weight import W


def _graph_shell(module):
    """New module registry, same owned tensors; replace only graph-local executors."""
    clone = copy.copy(module)
    clone._parameters = module._parameters.copy()
    clone._buffers = module._buffers.copy()
    clone._modules = module._modules.copy()
    return clone


class _Glm5Experts(BaseMoEExperts):
    def _configure_factory_adapter(self, adapter):
        super()._configure_factory_adapter(adapter)
        if adapter.moe_strategy in MEGA_MOE_STRATEGIES:
            limit = self._model_config.swiglu_limit
            adapter.mega_moe_policy = MegaMoeExecutionPolicy(
                activation_clamp=float(limit) if limit > 0 else None
            )
            adapter.layer_id = self.layer_idx
            adapter.max_generate_batch_size = self.max_generate_batch_size

    def _maybe_build_fused_moe(self):
        if (
            getattr(self._moe_config, "moe_strategy", None)
            not in FUSED_SHARED_STRATEGIES
        ):
            super()._maybe_build_fused_moe()
        # A staged expert leaf can finish long before its sibling shared MLP.
        # Bind the fused executor only after both owners have finished loading.

    def bind_shared_executor(self, shared):
        if (
            self.fused_moe is not None
            or not self._post_load_done
            or getattr(self._moe_config, "moe_strategy", None)
            not in FUSED_SHARED_STRATEGIES
        ):
            return
        self._shared_kernel_weights = {}
        if shared is not None:
            for linear, weight, scale in (
                (shared.gate_up_proj, W.ffn_w13, W.ffn_s13),
                (shared.down_proj, W.ffn_w2, W.ffn_s2),
            ):
                self._shared_kernel_weights[weight] = linear.weight
                self._shared_kernel_weights[scale] = _fp8_scale_parameter(linear)
        try:
            super()._maybe_build_fused_moe()
        finally:
            del self._shared_kernel_weights

    def _build_weights_dict(self):
        weights = super()._build_weights_dict()
        weights.update(getattr(self, "_shared_kernel_weights", {}))
        return weights


class _Glm5MoE(DeepSeekV32MoEBlock):
    experts_cls = _Glm5Experts
    _gate_logits = Glm5MoeLayer._gate_logits

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        pc = kwargs["parallelism_config"]
        self.gate_chunk_rows = (
            8192
            if kwargs["model_config"].model_type == "glm_5"
            and getattr(kwargs["moe_config"], "moe_strategy", None)
            in MEGA_MOE_STRATEGIES
            and pc.role_type != RoleType.DECODE
            and pc.prefill_cp_config.is_enabled()
            else 0
        )

    @property
    def fused_moe(self):
        return self.experts.fused_moe

    def _merge_experts(self, hidden_states, topk_weights, topk_ids):
        if getattr(self.fused_moe, "includes_shared_expert", False):
            return self.experts(hidden_states, topk_weights, topk_ids)
        return super()._merge_experts(hidden_states, topk_weights, topk_ids)

    def forward_prepacked(self, hidden_states, topk_ids, topk_weights):
        if self.fake_balance_expert is not None:
            self.fake_balance_expert(topk_ids, topk_weights)
        routed = self.fused_moe.forward_prepacked(hidden_states)
        if self.fused_moe.includes_shared_expert or self.shared_experts is None:
            return routed
        return routed + self.shared_experts(hidden_states)

    def clone_for_cuda_graph(self, *, share_mega_buf):
        clone = _graph_shell(self)
        clone.experts = _graph_shell(self.experts)
        if getattr(self.fused_moe, "supports_cuda_graph_clone", False):
            clone.experts.fused_moe = self.fused_moe.clone_for_cuda_graph(
                share_mega_buf=share_mega_buf
            )
        elif share_mega_buf:
            raise RuntimeError("GLM-5 draft executor cannot share MegaMoE buffers")
        return clone


class _CmpLinear:
    """Kernel view of an already-loaded FP8 projection; no checkpoint reads/copies."""

    def __init__(self, linear):
        self.linear = linear
        self.weight = getattr(linear, "weight", None)
        self.weight_scales = getattr(linear, "weight_scale", None)

    def __call__(self, x):
        return self.linear(x)


class _Glm5Decoder(DeepSeekV32DecoderLayer):
    moe_cls = _Glm5MoE

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.self_attn.has_indexer = self.indexer is not None
        self.cmp = None

    def bind_cmp(self, model):
        self.cmp = None
        if not resolve_glm5_cmp_enabled(model.py_hw_kernel_config):
            return
        attn = self.self_attn
        idx = attn.indexer
        indexer = (
            None
            if idx is None
            else SimpleNamespace(
                wk=_CmpLinear(idx.wk),
                wq_b=_CmpLinear(idx.wq_b),
                k_norm=SimpleNamespace(
                    weight=idx.k_norm.weight,
                    beta=idx.k_norm.bias,
                    variance_epsilon=idx.k_norm.eps,
                ),
                weights_proj=idx.weights_proj,
                indexer_op=idx.indexer_op,
            )
        )
        attention = SimpleNamespace(
            fused_qkv_a_proj=_CmpLinear(attn._fused_qkv_a_runtime),
            q_b_proj=_CmpLinear(attn.q_b_proj),
            o_proj=_CmpLinear(attn.o_proj),
            q_a_layernorm=attn.q_a_layernorm,
            kv_a_layernorm=attn.kv_a_layernorm,
            has_indexer=idx is not None,
            indexer=indexer,
            fuse_kernels_enabled=fuse_kernels_enabled(model.py_hw_kernel_config),
        )
        mlp = self.mlp
        if isinstance(mlp, _Glm5MoE):
            # Prepacking writes physical expert ids. Never bypass EPLB.
            cmp_mlp = SimpleNamespace(
                gate=mlp.gate,
                correction_bias=mlp.gate.e_score_correction_bias,
                fused_moe=(
                    None if mlp.experts.requires_expert_id_mapping else mlp.fused_moe
                ),
                shared_expert=mlp.shared_experts,
                ffn_tp_size=mlp.ffn_tp_size,
            )
        else:
            scale = _fp8_scale_parameter(mlp.gate_up_proj)
            cmp_mlp = SimpleNamespace(
                accepts_fp8_input=mlp.gate_up_proj.weight.dtype == torch.float8_e4m3fn,
                up_proj=SimpleNamespace(
                    scale_ue8m0=scale is not None and scale.dtype == torch.int32
                ),
            )
        self.cmp = Glm5Cmp(
            layer_idx=self.layer_idx,
            config=model.config,
            parallelism_config=model.parallelism_config,
            self_attn=attention,
            mlp=cmp_mlp,
            input_layernorm=SimpleNamespace(
                weight=self.input_layernorm.weight,
                variance_epsilon=self.input_layernorm.eps,
            ),
            post_attention_layernorm=SimpleNamespace(
                weight=self.post_attention_layernorm.weight,
                variance_epsilon=self.post_attention_layernorm.eps,
            ),
        )
        if getattr(self, "_draft_prefill_clone", False):
            self.cmp = self.cmp.clone_for_cuda_graph(
                mlp=self.cmp.mlp, draft_prefill=True
            )

    _forward_cmp = Glm5MoeDecoderLayer._forward_cmp

    def _fwd_mlp_or_moe(self, hidden_states, residual):
        normalized, residual = self.post_attention_layernorm(hidden_states, residual)
        return self.mlp(normalized), residual

    def forward(
        self,
        hidden_states,
        residual,
        fmha_impl,
        kv_cache=None,
        prev_topk_indices=None,
        enable_cmp=False,
        force_reuse_topk_indices=False,
    ):
        if enable_cmp:
            return self._forward_cmp(
                hidden_states,
                residual,
                fmha_impl,
                kv_cache,
                prev_topk_indices,
                force_reuse_topk_indices,
            )
        normalized, residual = self.input_layernorm(hidden_states, residual)
        attention, topk = self.self_attn(
            normalized,
            fmha_impl,
            kv_cache,
            previous_topk=prev_topk_indices,
            return_topk=True,
            force_reuse_topk_indices=force_reuse_topk_indices,
        )
        output, residual = self._fwd_mlp_or_moe(attention, residual)
        return DecodeLayerOutput(output, residual, topk)

    def clone_for_cuda_graph(self, *, draft_prefill=False, share_mega_buf=False):
        clone = _graph_shell(self)
        clone.mlp = self.mlp.clone_for_cuda_graph(share_mega_buf=share_mega_buf)
        clone._draft_prefill_clone = draft_prefill
        clone.cmp = None
        return clone


class _OwnedGlm5:
    decoder_layer_cls = _Glm5Decoder

    def __init__(self, model_config, load_config):
        # Glm5._create_config already normalized quantization_config.expert_dtype.
        # Other model-wide formats (notably ModelOpt NVFP4) use a different
        # FP4 encoding. The GLM mixed recipe has block-FP8 model quantization.
        model_quant = model_config.quant_config
        dtype = getattr(model_config, "expert_weight_dtype", "")
        mixed_block_fp8 = (
            model_quant is None or model_quant.get_method() == "FP8_PER_BLOCK"
        )
        packed = mixed_block_fp8 and dtype == "fp4"
        if (
            mixed_block_fp8
            and not dtype
            and getattr(load_config.moe_config, "moe_strategy", None)
            in FP4_MEGA_MOE_STRATEGIES
        ):
            packed = _legacy_packed_checkpoint(checkpoint_path(model_config))
        if packed:
            self.decoder_layer_cls = _PackedGlm5Decoder
        super().__init__(model_config, load_config)
        self.WEIGHTS_MAPPER = _Glm5Mapper(self.WEIGHTS_MAPPER, packed)
        self.moe_config = load_config.moe_config
        self.max_generate_batch_size = load_config.max_generate_batch_size
        for layer in self.layers:
            if isinstance(layer.mlp, _Glm5MoE):
                layer.mlp.experts.max_generate_batch_size = self.max_generate_batch_size

    def _cp_handoff_reference(self):
        return self.embed_tokens.weight

    def _apply(self, fn, recurse=True):
        result = super()._apply(fn, recurse)
        for layer in self.layers:
            if layer.cmp is not None:
                # CMP views and event handles are not torch submodules.
                layer.bind_cmp(self)
        return result

    def runtime_weight_view(self):
        for layer in self.layers:
            if isinstance(layer.mlp, _Glm5MoE):
                layer.mlp.experts.bind_shared_executor(layer.mlp.shared_experts)
        return super().runtime_weight_view()

    def initialize(self, init_resource):
        ok = super().initialize(init_resource)
        if ok:
            for layer in self.layers:
                if isinstance(layer.mlp, _Glm5MoE):
                    layer.mlp.experts.bind_shared_executor(layer.mlp.shared_experts)
                layer.bind_cmp(self)
        return ok


class _PackedGlm5Experts(_Glm5Experts):
    def validate_runtime_device(self, device):
        super().validate_runtime_device(device)
        if device.type != "cpu" and (
            device.type != "cuda"
            or torch.version.hip is not None
            or getattr(self._moe_config, "moe_strategy", None)
            not in FP4_MEGA_MOE_STRATEGIES
        ):
            raise ValueError(
                "GLM-5 group-32 FP4 experts require a CUDA FP4 MegaMoE strategy"
            )

    def _init_buffers(self, params_dtype):
        self.quant_method = Glm5PackedExpertMethod()
        super()._init_buffers(params_dtype)


class _PackedGlm5MoE(_Glm5MoE):
    experts_cls = _PackedGlm5Experts


class _PackedGlm5Decoder(_Glm5Decoder):
    moe_cls = _PackedGlm5MoE


class _Glm5Mapper:
    def __init__(self, mapper, packed):
        self.mapper = mapper
        self.packed = packed

    def map_name(self, name):
        dest = self.mapper.map_name(name)
        if self.packed and ".mlp.experts." in dest:
            return (
                dest.removesuffix(".weight_scale") + ".scale"
                if dest.endswith(".weight_scale")
                else dest
            )
        if dest.endswith(".scale"):
            return dest.removesuffix(".scale") + ".weight_scale_inv"
        return dest

    def apply(self, weights):
        for name, value in weights:
            dest = self.map_name(name)
            # Direct FP8 linears take numerical FP32 block scales, not raw
            # exponent bytes. Packed routed scale storage stays byte-for-byte.
            if dest.endswith(".weight_scale_inv"):
                tensor, spec = unwrap_checkpoint_tensor(value)
                if tensor.dtype == getattr(torch, "float8_e8m0fnu", None):
                    converted = tensor.float()
                    # The reader may already have applied a TP slice. Retain
                    # its proof so the destination does not shard the scale twice.
                    value = spec.wrap_sharded_tensor(converted) if spec else converted
            yield dest, value


def _legacy_packed_checkpoint(path):
    """Older mixed exports identify group-32 FP4 by expert .weight_scale."""
    from safetensors import safe_open

    from rtp_llm.models_py.weight_mapper import (
        discover_ckpt_files,
        safetensor_index_entries,
    )

    def marker(name):
        return ".mlp.experts." in name and name.endswith(
            (
                ".gate_proj.weight_scale",
                ".up_proj.weight_scale",
                ".down_proj.weight_scale",
            )
        )

    entries = safetensor_index_entries(path)
    if entries is not None:
        return any(marker(name) for name, _ in entries)
    for filename in discover_ckpt_files(path):
        if filename.endswith(".safetensors"):
            with safe_open(filename, framework="pt", device="cpu") as reader:
                if any(marker(name) for name in reader.keys()):
                    return True
    return False


class Glm5ForCausalLM(_OwnedGlm5, DeepSeekV32ForCausalLM, Glm5MoeModel):
    """Load HF score parameters directly; execute the GLM score/CP/CMP loop."""


class Glm5MTPForCausalLM(_OwnedGlm5, DeepSeekV32MTPForCausalLM, GenericMoeMTPModel):
    """Load one draft layer; run GLM masking, seed/reuse and CP capture."""

    reverse_mtp_concat = False

    @staticmethod
    def draft_checkpoint_layer(config, ckpt_path):
        count = nonnegative_int(
            config.get("num_nextn_predict_layers", 1), "num_nextn_predict_layers"
        )
        if count > 1:
            raise ValueError("GLM-5 MTP requires a single draft layer")
        depth = positive_int(config["num_hidden_layers"], "num_hidden_layers")
        return depth if depth > 1 else 0

    def _vocab_mapping(self, prefix):
        if self._checkpoint_layer:
            return {
                "model.embed_tokens.weight": "embed_tokens.weight",
                "lm_head.weight": "lm_head.weight",
            }
        return super()._vocab_mapping(prefix)

    def checkpoint_weight_name_filter(self):
        keep = super().checkpoint_weight_name_filter()
        if not self._checkpoint_layer:
            return keep
        # Combined GLM uses the score vocabulary even if an extraction tool
        # also wrote layer-local copies. Never depend on shard traversal order.
        unused = {
            self._checkpoint_prefix + k
            for k in ("embed_tokens.weight", "shared_head.head.weight")
        }
        return lambda name: name not in unused and keep(name)

    def __init__(self, model_config, load_config):
        super().__init__(model_config, load_config)
        self._init_mtp_indexer_state(
            self.max_generate_batch_size, self.embed_tokens.weight.device
        )

    @property
    def pre_fc_norm_embedding(self):
        return self.mtp_block.e_norm

    @property
    def pre_fc_norm_hidden(self):
        return self.mtp_block.h_norm

    @property
    def fc(self):
        return self.mtp_block.fc

    def clone_for_cuda_graph(self, *, share_mega_buf=False):
        clone = object.__new__(type(self))
        GptModelBase.__init__(
            clone,
            self.config,
            self.parallelism_config,
            self.weight,
            self.max_generate_batch_size,
            self.fmha_config,
            self.py_hw_kernel_config,
        )
        clone.max_generate_batch_size = self.max_generate_batch_size
        clone.moe_config = self.moe_config
        clone._mla_kernel_layout = self._mla_kernel_layout
        clone._keep_mla_checkpoint_weights = self._keep_mla_checkpoint_weights
        clone.register_buffer("cos_sin_cache", self.cos_sin_cache, persistent=False)
        clone.embed_tokens, clone.mtp_block = self.embed_tokens, self.mtp_block
        clone.norm, clone.lm_head = self.norm, self.lm_head
        clone.layers = nn.ModuleList(
            layer.clone_for_cuda_graph(
                draft_prefill=True, share_mega_buf=share_mega_buf
            )
            for layer in self.layers
        )
        clone._mtp_indexer_share_enabled = self._mtp_indexer_share_enabled
        clone.set_mtp_indexer_role(0)
        clone.register_buffer(
            "_mtp_shared_topk_indices", self._mtp_shared_topk_indices, persistent=False
        )
        return clone
