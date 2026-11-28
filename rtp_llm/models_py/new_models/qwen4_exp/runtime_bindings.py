"""Typed direct-owner adapters to current-main execution classes.

No internal W.* weight dictionary is synthesized. Existing main forward,
QSA transaction, page/cache and graph implementations remain authoritative.
Native imports are delayed until execution binding, not checkpoint allocation.
"""
import torch
from torch import nn
from rtp_llm.models_py.new_models.qwen4_exp.dense_weights import bind_projection


def bind_attention(owner, parallelism_config, *, layer_idx, is_mtp_draft,
                   rope_config, hw_kernel_config):
    from rtp_llm.models_py.model_desc.qwen4_exp import Qwen4ExpAttention
    from rtp_llm.models_py.modules.base.common.zero_centered_rms_norm import ZeroCenteredFusedQKRMSNorm
    from rtp_llm.models_py.modules.qwen4_exp.weight_views import Qwen4LinearFactory

    class DirectAttention(Qwen4ExpAttention):
        def __init__(self):
            # The legacy constructor consumes W.* dictionaries. All runtime
            # fields are initialized here from typed parameter owners instead.
            nn.Module.__init__(self)
            self.linear_factory = Qwen4LinearFactory
            self.layer_idx = layer_idx
            self.parallelism_config = parallelism_config
            self.tp_size = owner.tp_size
            self.head_num = owner.local_heads
            self.head_dim = owner.head_dim
            self.num_key_value_groups = owner.local_heads // owner.local_kv_heads
            self.q_size = owner.q_size
            for name in ("qkv_proj", "o_proj", "gate"):
                module = bind_projection(getattr(owner, name), hw_kernel_config=hw_kernel_config,
                    role=getattr(parallelism_config, "role_type", "PDFUSION"))
                setattr(self, name, module)
            self.cache_scale_len = 1024
            self.o_proj.maybe_cache_quant_scale(self.cache_scale_len)
            self.qk_fuse_norm = ZeroCenteredFusedQKRMSNorm(owner.q_norm.weight,
                owner.k_norm.weight, owner.local_heads, owner.local_kv_heads,
                owner.head_dim, owner.norm_eps)
            self.is_mtp_draft = is_mtp_draft
            self.qsa_rope_config = rope_config
            self.qsa_indexer = None if owner.qsa_indexer is None else owner.qsa_indexer.runtime

    # Main's trusted QSA path deliberately checks the exact attention class.
    # Only construction differs for direct parameters; retain the real runtime
    # type as well as its unmodified methods, rather than silently disabling it.
    runtime = Qwen4ExpAttention.__new__(Qwen4ExpAttention)
    DirectAttention.__init__(runtime)
    return runtime


def bind_gdn(owner, linear_config, parallelism_config, *, norm_activation="silu",
             role="PDFUSION", hw_kernel_config=None):
    """Reuse main GDN cache converters and prefill/decode/target-verification."""
    from rtp_llm.models_py.model_desc.qwen3_next import (
        Qwen3NextGatedDeltaNet,
        Qwen3NextGatedDeltaNetDecode,
        Qwen3NextGdnStateParameters,
    )
    from rtp_llm.models_py.modules.qwen4_exp.gdn_prefill_adapter import Qwen4Sm120GdnPrefill
    from rtp_llm.models_py.modules.qwen4_exp.gdn_prefill_norm import Qwen4PrefillGatedNorm
    from rtp_llm.models_py.modules.qwen4_exp.weight_views import Qwen4LinearFactory
    from rtp_llm.models_py.triton_kernels.common.layernorm_gated import RmsNormGated

    class DirectGDN(Qwen3NextGatedDeltaNet):
        def __init__(self):
            nn.Module.__init__(self)
            self.linear_factory = Qwen4LinearFactory
            self.linear_attn_config = linear_config
            self.parallelism_config = parallelism_config
            self.quant_config = None
            for name in ("head_k_dim","head_v_dim","local_num_k_heads","local_num_v_heads","num_key_value_heads"):
                setattr(self,name,getattr(owner,name))
            qkvz = owner.in_proj_qkvz.weight.t()
            ba = owner.in_proj_ba_w
            # Preserve main's CUDA BF16 QKVZ+BA single-GEMM layout and transfer
            # the views back to the parameter owner, with no stale source copy.
            self._qkvz_ba_fused = qkvz.dtype == ba.dtype == torch.bfloat16
            if self._qkvz_ba_fused:
                fused = torch.cat((qkvz,ba),dim=1).contiguous()
                self._qkvz_size, self._ba_size = qkvz.shape[1], ba.shape[1]
                owner.in_proj_qkvz.weight.data = fused[:,:self._qkvz_size].t()
                owner.in_proj_ba_w.data = fused[:,self._qkvz_size:]
                self.in_proj_fused = Qwen4LinearFactory.create_linear(fused,None,None,None,hw_kernel_config=hw_kernel_config)
                self.in_proj_qkvz = self.in_proj_ba = None
            else:
                self.in_proj_fused = None
                self.in_proj_qkvz = bind_projection(owner.in_proj_qkvz, hw_kernel_config=hw_kernel_config, role=role)
                if owner.in_proj_ba_storage_scale is None:
                    self.in_proj_ba = Qwen4LinearFactory.create_linear(ba,None,None,None,hw_kernel_config=hw_kernel_config)
                else:
                    from rtp_llm.models_py.modules.base.common.fp8_linear import Fp8Linear
                    weight = ba.t().contiguous()
                    owner.in_proj_ba_w.data = weight.t()
                    self.in_proj_ba = Fp8Linear(weight, owner.in_proj_ba_storage_scale, role=role)
            state = Qwen3NextGdnStateParameters(owner.conv1d_w, owner.dt_bias, owner.a_log)
            self.prefill_gdn = Qwen4Sm120GdnPrefill(linear_config, parallelism_config, state)
            self.decode_gdn = Qwen3NextGatedDeltaNetDecode(linear_config, parallelism_config, state)
            self.norm = Qwen4PrefillGatedNorm(RmsNormGated(owner.norm_w,
                eps=owner.rms_norm_eps,group_size=owner.head_v_dim,activation=norm_activation),role=role)
            self.out_proj = bind_projection(owner.out_proj, hw_kernel_config=hw_kernel_config, role=role)

    runtime = DirectGDN()
    for parameter in owner.parameters():
        parameter._rtp_requires_restart_for_update = True
    return runtime


def bind_moe(owner, config, parallelism_config, *, moe_config=None, hw_kernel_config=None):
    """Main routing/shared-FFN execution around direct EP-local Marlin experts."""
    from rtp_llm.models_py.modules.qwen4_exp.moe.shared_overlap import Qwen4MoeLayer
    from rtp_llm.models_py.modules.qwen4_exp.dense_mlp import Qwen4DenseMLP
    from rtp_llm.models_py.modules import SelectTopk, SigmoidGateScaleAdd, FakeBalanceExpert
    from rtp_llm.models_py.modules.base import FusedSiluAndMul
    from rtp_llm.models_py.modules.qwen4_exp.moe.topk import Qwen4ExpMoeTopk
    from rtp_llm.models_py.modules.qwen4_exp.weight_views import Qwen4LinearFactory
    from rtp_llm.models_py.distributed.collective_torch import Group, all_reduce
    role = getattr(parallelism_config, "role_type", "PDFUSION")

    class DirectShared(Qwen4DenseMLP):
        def __init__(self):
            nn.Module.__init__(self)
            self.linear_factory = Qwen4LinearFactory
            self.activation_type = config.activation_type
            self.parallelism_config = parallelism_config
            self.act_fn = FusedSiluAndMul()
            self.is_gated = True
            shared = owner.shared_expert
            fused = torch.cat((shared.gate_proj.weight, shared.up_proj.weight),dim=0)
            width = shared.gate_proj.weight.shape[0]
            shared.gate_proj.weight.data = fused[:width]
            shared.up_proj.weight.data = fused[width:]
            if shared.up_storage_scale is None:
                self.up_proj = Qwen4LinearFactory.create_linear(fused.t(),None,None,None,hw_kernel_config=hw_kernel_config)
            else:
                from rtp_llm.models_py.modules.base.common.fp8_linear import Fp8Linear
                self.up_proj = Fp8Linear(fused, shared.up_storage_scale, role=role)
            self.down_proj = bind_projection(shared.down_proj, hw_kernel_config=hw_kernel_config, role=role)

    class DirectExperts(nn.Module):
        @property
        def topk_ids_dtype(self):
            if native_experts is None:
                return torch.int32
            return native_experts.topk_ids_dtype

        @property
        def router(self):
            return getattr(native_experts, "router", None)

        def forward(self, hidden_states, topk_weights, topk_ids, activation, skip_tp_allreduce=False):
            if activation != "SiGLU":
                raise ValueError("Qwen4 experts require SiGLU")
            if hasattr(owner.experts, "runtime"):
                return owner.experts(
                    hidden_states, topk_weights, topk_ids,
                    skip_tp_allreduce=skip_tp_allreduce,
                )
            output = owner.experts(hidden_states,topk_ids,topk_weights)
            if not skip_tp_allreduce and parallelism_config.tp_size > 1:
                output = all_reduce(output,group=Group.TP)
            return output

    native_experts = getattr(owner.experts, "runtime", None)

    class DirectMoe(Qwen4MoeLayer):
        def __init__(self):
            nn.Module.__init__(self)
            self.linear_factory = Qwen4LinearFactory
            self.config = config
            self.parallelism_config = parallelism_config
            self.ffn_tp_size = parallelism_config.get_ffn_tp_size()
            self.ep_size = owner.experts.ep_size
            self.hidden_dim, self.ffn_dim = config.hidden_size, config.inter_size
            self.num_experts, self.num_local_experts = owner.experts.num_experts, owner.experts.local
            self.top_k = config.moe_k
            self.gate = bind_projection(owner.gate, hw_kernel_config=hw_kernel_config, role=role)
            self.select_topk = Qwen4ExpMoeTopk(SelectTopk(config=config))
            self.fake_balance_expert = None
            if moe_config is not None and moe_config.fake_balance_expert:
                self.fake_balance_expert = FakeBalanceExpert(expert_num=config.expert_num,
                    moe_k=config.moe_k,dp_rank=parallelism_config.dp_rank,
                    dp_size=parallelism_config.dp_size,ep_size=parallelism_config.ep_size)
            # The naked executor consumes physical ids; the gate returns logical ids.
            self.fused_moe = (
                native_experts
                if native_experts is not None and not owner.experts.requires_expert_id_mapping
                else DirectExperts()
            )
            self.correction_bias = None
            self.shared_expert = None
            if owner.shared_expert is not None:
                shared = Qwen4DenseMLP.__new__(Qwen4DenseMLP)
                DirectShared.__init__(shared)
                self.shared_expert = shared
            self.add_shared_expert = self.shared_expert is not None
            self.shared_expert_gate = None if owner.shared_expert_gate is None else bind_projection(owner.shared_expert_gate, hw_kernel_config=hw_kernel_config, role=role)
            self.sigmoid_gate_scale_add = None if self.shared_expert_gate is None else SigmoidGateScaleAdd()
            self.use_ep_shared_allreduce = self.shared_expert is not None and self.ffn_tp_size > 1 and self.ep_size > 1
            router = getattr(self.fused_moe, "router", None)
            router_tp_size = getattr(router, "tp_collective_size", None)
            self.use_unified_tp_allreduce = (self.shared_expert is not None
                and self.ffn_tp_size > 1 and self.ep_size == 1
                and self.ffn_tp_size == router_tp_size
                and bool(getattr(router, "supports_skip_tp_allreduce", False)))

    runtime = Qwen4MoeLayer.__new__(Qwen4MoeLayer)
    DirectMoe.__init__(runtime)
    runtime.prepare_shared_expert_overlap()
    return runtime


def bind_decoder(owner, config, load_config, *, is_mtp_draft=False):
    """Public draft/target decoder binding; the original main forward is shared."""
    from rtp_llm.models_py.model_desc.qwen4_exp import Qwen4ExpDecoderLayer
    from rtp_llm.models_py.modules.qwen4_exp.tp_reduce import install_target_graph_all_reduce
    from rtp_llm.ops import HybridAttentionType
    pc = load_config.parallelism_config
    hw = load_config.py_hw_kernel_config

    class DirectDecoder(Qwen4ExpDecoderLayer):
        def __init__(self):
            nn.Module.__init__(self)
            self.layer_idx = owner.layer_idx
            self.layer_type = HybridAttentionType.LINEAR if owner.layer_type == "linear_attention" else HybridAttentionType.NONE
            self._execution_role = getattr(pc,"role_type","PDFUSION")
            if owner.layer_type == "linear_attention":
                self.self_attn = bind_gdn(owner.linear_attn,config.linear_attention_config,pc,
                    norm_activation=getattr(config,"linear_attn_norm_activation","silu"),role=self._execution_role,hw_kernel_config=hw)
            else:
                self.self_attn = owner.self_attn.bind_runtime(pc,layer_idx=owner.layer_idx,
                    is_mtp_draft=is_mtp_draft,rope_config=config.attn_config.rope_config,hw_kernel_config=hw)
            self.attn_hyper_connection = owner.attn_hyper_connection.runtime
            self.mlp_hyper_connection = owner.mlp_hyper_connection.runtime
            self.mlp = bind_moe(owner.mlp,config,pc,moe_config=load_config.moe_config,hw_kernel_config=hw)
            install_target_graph_all_reduce(self.self_attn,self.mlp)
    return DirectDecoder()


def bind_target(owner, config, load_config):
    from rtp_llm.models_py.model_desc.qwen4_exp import Qwen4ExpModel
    from rtp_llm.models_py.model_desc.module_base import GptModelBase
    from rtp_llm.models_py.modules import Embedding, MultimodalEmbeddingInjector
    from rtp_llm.models_py.modules.qwen4_exp.cpu_target_status import prepare_cpu_target_status
    from rtp_llm.models_py.modules.qwen4_exp.tp_reduce import prepare_target_reduce, install_target_graph_all_reduce
    pc = load_config.parallelism_config
    if pc is None:
        raise ValueError("Qwen4 runtime requires resolved parallelism_config")

    class DirectTarget(Qwen4ExpModel):
        def __init__(self):
            GptModelBase.__init__(self,config,pc,None,load_config.max_generate_batch_size,
                load_config.fmha_config,load_config.py_hw_kernel_config)
            if owner.embed_tokens.storage_scale is None:
                self.embed_tokens = Embedding(config,pc,owner.embed_tokens.weight)
            else:
                from rtp_llm.models_py.modules.base.common.fp8_embedding import Fp8VocabEmbedding
                self.embed_tokens = Fp8VocabEmbedding(owner.embed_tokens.weight,
                    pc.get_attn_tp_size(), owner.embed_tokens.weight.device,
                    weight_scale=owner.embed_tokens.storage_scale)
                self.embed_tokens.prepare_byte_gather(role=getattr(pc,"role_type","PDFUSION"))
            self.layers = nn.ModuleList([bind_decoder(layer,config,load_config) for layer in owner.layers])
            self.hc_mult = config.hc_mult
            self.hyper_connection_mixer = owner.hyper_connection_mixer.runtime
            self.multimodal_embedding_injector = MultimodalEmbeddingInjector()
            self.ple_layers = nn.ModuleDict({str(i):layer.ple.runtime for i,layer in enumerate(owner.layers) if layer.ple is not None})
            self._ple_target_transaction = None
            self._ple_graph_target_transactions = {}
            self._capture_mtp_target_hidden = bool(getattr(config,"is_mtp",False))
            self._cpu_target_status = prepare_cpu_target_status(config,pc)
            self._target_reduce = prepare_target_reduce(config,pc)
            if self._target_reduce is not None:
                for layer in self.layers:
                    install_target_graph_all_reduce(layer.self_attn,layer.mlp,self._target_reduce)
    # Main uses exact target identity to distinguish target PLE prefetch and
    # FP8 HC-embedding semantics from draft subclasses. Preserve that identity;
    # the initializer above replaces only the legacy weight-dictionary ctor.
    runtime = Qwen4ExpModel.__new__(Qwen4ExpModel)
    DirectTarget.__init__(runtime)
    return runtime
