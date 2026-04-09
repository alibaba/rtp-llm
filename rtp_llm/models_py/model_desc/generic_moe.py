import logging
import os
from typing import Any, Dict, Optional

import torch
from torch import nn

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.model_loader.model_weight_info import ModelWeights
from rtp_llm.models_py.model_desc.block_map import select_block_map_for_layer
from rtp_llm.models_py.model_desc.module_base import GptModelBase
from rtp_llm.models_py.modules import (
    CausalAttention,
    DenseMLP,
    Embedding,
    FakeBalanceExpert,
    FMHAImplBase,
    FusedMoeFactory,
    GroupTopK,
    LinearFactory,
    MlaAttention,
    RMSNorm,
    RMSResNorm,
    SelectTopk,
    SigmoidGateScaleAdd,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.config_adapter import (
    MoEConfigAdapter,
)
from rtp_llm.ops import HWKernelConfig, MoeConfig, ParallelismConfig
from rtp_llm.ops.compute_ops import LayerKVCache, PyModelInputs, PyModelOutputs
from rtp_llm.utils.model_weight import W

# ── Determinism checker ──────────────────────────────────────────────
# SAVE_DETERMINISM_REF=1  → save tensors to /tmp/determinism_ref/
# CHECK_DETERMINISM=1     → compare against saved tensors, report diffs
_DET_SAVE = os.environ.get("SAVE_DETERMINISM_REF", "0") == "1"
_DET_CHECK = os.environ.get("CHECK_DETERMINISM", "0") == "1"
_DET_DIR = os.environ.get("DETERMINISM_REF_DIR", "/tmp/determinism_ref")
_det_call_count = 0  # track forward call index (prefill vs decode steps)

if _DET_SAVE:
    os.makedirs(_DET_DIR, exist_ok=True)
    print(f"[DETERMINISM] Saving reference tensors to {_DET_DIR}", flush=True)
if _DET_CHECK:
    print(f"[DETERMINISM] Checking against reference tensors in {_DET_DIR}", flush=True)


def _det_checkpoint(tag: str, tensor: torch.Tensor):
    """Save or compare a tensor at a named checkpoint."""
    global _det_call_count
    path = os.path.join(_DET_DIR, f"call{_det_call_count}_{tag}.pt")

    if _DET_SAVE:
        torch.save(tensor.detach().cpu(), path)
    elif _DET_CHECK:
        if not os.path.exists(path):
            print(f"[DETERMINISM] SKIP {tag} — no reference file", flush=True)
            return
        ref = torch.load(path, map_location="cpu", weights_only=True)
        cur = tensor.detach().cpu()
        if ref.shape != cur.shape:
            print(
                f"[DETERMINISM] SHAPE MISMATCH {tag}: ref={list(ref.shape)} cur={list(cur.shape)}",
                flush=True,
            )
            return
        diff = (ref.float() - cur.float()).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
        exact = torch.equal(ref, cur)
        status = "EXACT" if exact else f"DIFF max={max_diff:.8f} mean={mean_diff:.8f}"
        print(f"[DETERMINISM] {tag}: {status}", flush=True)
        if not exact:
            # Show where the biggest diffs are
            n_nonzero = (diff > 0).sum().item()
            n_total = diff.numel()
            print(
                f"[DETERMINISM]   non-zero diffs: {n_nonzero}/{n_total} ({100*n_nonzero/n_total:.2f}%)",
                flush=True,
            )


def _det_bump_call():
    """Increment the forward call counter (call after each full model forward)."""
    global _det_call_count
    _det_call_count += 1


class GenericMoeLayer(nn.Module):
    """Generic MoE layer supporting both Qwen3 and internal model."""

    def __init__(
        self,
        config: ModelConfig,
        parallelism_config: ParallelismConfig,
        weights: Dict[str, torch.Tensor],
        moe_config: MoeConfig,
        max_generate_batch_size: int = 0,
        enable_cuda_graph: bool = False,
        hw_kernel_config: Optional["HWKernelConfig"] = None,
    ):
        super().__init__()
        self.config = config
        self.parallelism_config = parallelism_config

        self.hidden_dim = config.hidden_size
        self.ffn_dim = config.inter_size
        self.num_experts = config.eplb_config.phy_exp_num(config.expert_num)
        self.top_k = config.moe_k

        # Get quant_config from model_config
        quant_config = config.quant_config
        self.gate = LinearFactory.create_linear_from_weights(
            weights, W.moe_gate, None, None, quant_config, hw_kernel_config
        )
        self.select_topk = SelectTopk(config=config)
        if moe_config.fake_balance_expert:
            self.fake_balance_expert = FakeBalanceExpert(
                expert_num=config.expert_num,
                moe_k=config.moe_k,
                dp_rank=parallelism_config.dp_rank,
                dp_size=parallelism_config.dp_size,
                ep_size=parallelism_config.ep_size,
            )
        else:
            self.fake_balance_expert = None
        config_adapter = MoEConfigAdapter(
            model_config=config,
            parallelism_config=parallelism_config,
            moe_config=moe_config,
            quant_config=quant_config,
            enable_cuda_graph=enable_cuda_graph,
        )
        self.fused_moe = FusedMoeFactory().create_fused_moe(config_adapter, weights)

        self.w1 = weights.get(W.moe_w1, None)
        self.w2 = weights.get(W.moe_w2, None)
        assert (
            self.w1 is not None and self.w2 is not None
        ), "Weights w1 and w2 must be provided"
        self.num_local_experts = self.w1.shape[0]
        self.add_shared_expert = config.moe_style == 2
        if self.add_shared_expert:
            self.shared_expert = DenseMLP(
                config.activation_type, parallelism_config, weights, quant_config
            )
        else:
            self.shared_expert = None
        if weights.get(W.shared_expert_gate, None) is not None:
            self.shared_expert_gate = LinearFactory.create_linear_from_weights(
                weights, W.shared_expert_gate, None, None, config
            )
            self.sigmoid_gate_scale_add = SigmoidGateScaleAdd()
        else:
            self.shared_expert_gate = None
            self.sigmoid_gate_scale_add = None

        # for group topk
        self.correction_bias = weights.get(W.e_score_correction_b, None)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        num_tokens, _ = hidden_states.shape
        router_logits = self.gate(hidden_states)
        router_logits_fp32 = router_logits.float()

        topk_weights = torch.empty(
            (num_tokens, self.top_k),
            dtype=torch.float32,
            device=hidden_states.device,
        )
        # different executor may need different topk_ids dtype
        topk_ids_dtype = self.fused_moe.topk_ids_dtype
        topk_ids = torch.empty(
            (num_tokens, self.top_k),
            dtype=topk_ids_dtype,
            device=hidden_states.device,
        )

        if self.correction_bias is not None:
            self.group_topk = GroupTopK()
            self.renormalize = self.config.has_moe_norm
            self.num_expert_group = self.config.moe_n_group

            self.topk_group = self.config.moe_topk_group
            self.n_routed_experts = self.config.expert_num  # config.n_routed_experts
            self.routed_scaling_factor = self.config.routed_scaling_factor
            self.group_topk(
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                scores=router_logits_fp32,
                correction_bias=self.correction_bias,
                n_group=self.num_expert_group,
                topk_group=self.topk_group,
                topk=self.top_k,
                renormalize=self.renormalize,
                routed_scaling_factor=self.routed_scaling_factor,
            )
        else:
            # Top-K selection using C++ SelectTopkOp
            self.select_topk(router_logits_fp32, topk_ids, topk_weights)

        if self.fake_balance_expert is not None:
            self.fake_balance_expert(topk_ids, topk_weights)

        experts_output = self.fused_moe(
            hidden_states=hidden_states,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            activation="SiGLU",
        )

        if self.shared_expert is not None:
            shared_expert_output = self.shared_expert(hidden_states)
            if self.shared_expert_gate is not None:
                gate_output = self.shared_expert_gate(hidden_states)  # [T, 1]
                # Fused: experts_output += sigmoid(gate_output) * shared_expert_output
                self.sigmoid_gate_scale_add(
                    gate_output, shared_expert_output, experts_output
                )
            else:
                experts_output = experts_output + shared_expert_output
        return experts_output


class DecodeLayerOutput:
    def __init__(self, hidden_states: torch.Tensor, residual: torch.Tensor):
        self.hidden_states = hidden_states
        self.residual = residual


class GenericMoeDecoderLayer(nn.Module):
    """Generic MoE decoder layer supporting Dense/MoE hybrid and shared experts."""

    def __init__(
        self,
        config: ModelConfig,
        parallelism_config: ParallelismConfig,
        weights: Dict[str, torch.Tensor],
        global_weights: Dict[str, torch.Tensor],
        layer_idx: int,
        moe_config: MoeConfig,
        max_generate_batch_size: int = 0,
        enable_cuda_graph: bool = False,
        hw_kernel_config: Optional["HWKernelConfig"] = None,
    ):
        super().__init__()
        self.layer_idx = layer_idx

        # Get quant_config from model_config
        quant_config = config.quant_config
        if config.attn_config.use_mla:
            self.self_attn = MlaAttention(
                config.attn_config,
                parallelism_config,
                weights,
                layer_idx,
                config.layernorm_eps,
                quant_config,
                hw_kernel_config,
                global_weights=global_weights,
            )
        else:
            attn_configs = config.getAttentionConfigs(
                parallelism_config.get_attn_tp_size()
            )
            self.self_attn = CausalAttention(
                attn_configs,
                parallelism_config,
                weights,
                config.layernorm_eps,
                quant_config,
                hw_kernel_config,
                layer_idx,
            )

        # Determine if this is a Dense layer (before first MoE layer or dense only)
        if layer_idx not in config.moe_layer_index:
            self.mlp = DenseMLP(
                config.activation_type, parallelism_config, weights, quant_config
            )
        else:
            self.mlp = GenericMoeLayer(
                config,
                parallelism_config,
                weights,
                moe_config,
                max_generate_batch_size,
                enable_cuda_graph=enable_cuda_graph,
                hw_kernel_config=hw_kernel_config,
            )

        # 使用 RMSResNorm 来 fuse residual add 和 layernorm
        self.input_layernorm = RMSResNorm(
            weights[W.pre_ln_gamma], eps=config.layernorm_eps
        )
        self.post_attention_layernorm = RMSResNorm(
            weights[W.post_ln_gamma], eps=config.layernorm_eps
        )

    def _debug_dump(self, tag: str, layer_idx: int, tensor: torch.Tensor):
        """Dump tensor stats for debugging. Enable with DEBUG_LAYER_DUMP=1."""
        import os

        if not os.environ.get("DEBUG_LAYER_DUMP"):
            return
        t = tensor.float()
        print(
            f"[DEBUG] layer={layer_idx} {tag}: shape={list(tensor.shape)} "
            f"dtype={tensor.dtype} "
            f"mean={t.mean().item():.6f} std={t.std().item():.6f} "
            f"min={t.min().item():.6f} max={t.max().item():.6f} "
            f"abs_mean={t.abs().mean().item():.6f} "
            f"nan={torch.isnan(t).sum().item()} inf={torch.isinf(t).sum().item()}",
            flush=True,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        fmha_impl: FMHAImplBase,
        kv_cache: Optional[LayerKVCache] = None,
    ) -> DecodeLayerOutput:
        layer_idx = self.layer_idx

        self._debug_dump("input_hidden", layer_idx, hidden_states)
        self._debug_dump("input_residual", layer_idx, residual)

        if _DET_SAVE or _DET_CHECK:
            _det_checkpoint(f"L{layer_idx}_input_hidden", hidden_states)
            _det_checkpoint(f"L{layer_idx}_input_residual", residual)

        hidden_states = self.input_layernorm(hidden_states, residual)
        self._debug_dump("after_input_ln", layer_idx, hidden_states)

        if _DET_SAVE or _DET_CHECK:
            _det_checkpoint(f"L{layer_idx}_after_input_ln", hidden_states)

        hidden_states = self.self_attn(
            hidden_states=hidden_states, fmha_impl=fmha_impl, kv_cache=kv_cache
        )
        self._debug_dump("after_attn", layer_idx, hidden_states)

        if _DET_SAVE or _DET_CHECK:
            _det_checkpoint(f"L{layer_idx}_after_attn", hidden_states)

        hidden_states = self.post_attention_layernorm(hidden_states, residual)
        self._debug_dump("after_post_ln", layer_idx, hidden_states)

        if _DET_SAVE or _DET_CHECK:
            _det_checkpoint(f"L{layer_idx}_after_post_ln", hidden_states)

        hidden_states = self.mlp(hidden_states)
        self._debug_dump("after_mlp", layer_idx, hidden_states)

        if _DET_SAVE or _DET_CHECK:
            _det_checkpoint(f"L{layer_idx}_after_mlp", hidden_states)

        return DecodeLayerOutput(hidden_states, residual)


class GenericMoeModel(GptModelBase):
    """Generic MoE model supporting Qwen3-MoE, internal model, and other MoE architectures."""

    def __init__(
        self,
        model_config: ModelConfig,
        parallelism_config: ParallelismConfig,
        weights: ModelWeights,
        moe_config: MoeConfig,
        max_generate_batch_size: int,
        fmha_config=None,
        py_hw_kernel_config=None,
        device_resource_config=None,
    ):
        super().__init__(
            model_config,
            parallelism_config,
            weights,
            max_generate_batch_size=max_generate_batch_size,
            fmha_config=fmha_config,
            py_hw_kernel_config=py_hw_kernel_config,
            device_resource_config=device_resource_config,
        )
        # Determine attention_type from model_config.attn_config.use_mla
        self.embed_tokens = Embedding(
            model_config, parallelism_config, weights.get_global_weight(W.embedding)
        )
        # Get enable_cuda_graph from py_hw_kernel_config
        enable_cuda_graph = (
            py_hw_kernel_config.enable_cuda_graph
            if py_hw_kernel_config is not None
            else False
        )
        self.layers = nn.ModuleList(
            [
                GenericMoeDecoderLayer(
                    model_config,
                    parallelism_config,
                    weights.weights[idx],
                    weights.global_weights,
                    idx,
                    moe_config,
                    max_generate_batch_size,
                    enable_cuda_graph=enable_cuda_graph,
                    hw_kernel_config=py_hw_kernel_config,
                )
                for idx in range(self.layer_num)
            ]
        )
        self.norm = RMSResNorm(
            weights.get_global_weight(W.final_ln_gamma), eps=model_config.layernorm_eps
        )

    def forward(self, inputs: PyModelInputs, fmha_impl: Any = None) -> PyModelOutputs:
        input_ids: torch.Tensor = inputs.input_ids
        hidden_states = self.embed_tokens(input_ids)
        if fmha_impl is None:
            fmha_impl = self.prepare_fmha_impl(
                inputs
            )  # pyright: ignore[reportUnreachable]

        _debug = os.environ.get("DEBUG_LAYER_DUMP")
        if _debug:
            t = hidden_states.float()
            print(
                f"[DEBUG] embedding: shape={list(hidden_states.shape)} "
                f"mean={t.mean().item():.6f} std={t.std().item():.6f} "
                f"min={t.min().item():.6f} max={t.max().item():.6f}",
                flush=True,
            )

        if _DET_SAVE or _DET_CHECK:
            _det_checkpoint("embedding", hidden_states)
            _det_checkpoint("input_ids", input_ids)

        residual = torch.zeros_like(hidden_states)
        for i, decoder_layer in enumerate(self.layers[: self.layer_num]):
            select_block_map_for_layer(inputs.attention_inputs, i)
            output = decoder_layer(
                hidden_states,
                residual,
                fmha_impl,
                kv_cache=self.kv_cache.get_layer_cache(i) if self.kv_cache else None,
            )
            hidden_states = output.hidden_states
            residual = output.residual

        hidden_states = self.norm(hidden_states, residual)

        if _debug:
            t = hidden_states.float()
            print(
                f"[DEBUG] final_norm: shape={list(hidden_states.shape)} "
                f"mean={t.mean().item():.6f} std={t.std().item():.6f} "
                f"min={t.min().item():.6f} max={t.max().item():.6f}",
                flush=True,
            )

        if _DET_SAVE or _DET_CHECK:
            _det_checkpoint("final_norm", hidden_states)
            _det_bump_call()

        return PyModelOutputs(hidden_states, fmha_impl.fmha_params)


__all__ = [
    "GenericMoeLayer",
    "GenericMoeDecoderLayer",
    "GenericMoeModel",
]
