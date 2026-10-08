"""K3 latent and shared branches composed with RTP's fused-MoE factory."""

import inspect

import torch
from torch import nn

from rtp_llm.models.kimi_k3.kimi_k3_weight import (
    KimiK3WeightNames as K3W,
    shared_expert_weight_shard_enabled,
)
from rtp_llm.models_py.distributed.collective_torch import Group, all_gather
from rtp_llm.models_py.modules.base import RMSNorm
from .norm import KimiK3LatentRMSNorm
from rtp_llm.models_py.modules.factory.fused_moe import FusedMoeFactory
from rtp_llm.models_py.modules.factory.fused_moe.defs.config_adapter import (
    MoEConfigAdapter,
)
from rtp_llm.utils.model_weight import W
from rtp_llm.ops import MoeConfig, RoleType
from .attention import linear, profile_scope
from .router import KimiK3RouterProjection
from .routing import grouped_topk
from .moe_backend import get_k3_moe_backend
from .linear import (
    KimiK3Bf16Linear,
    KimiK3LatentDownLinear,
    bf16_linear,
    bf16_linear_add_inplace,
)
from rtp_llm.models_py.triton_kernels.common.activation import situ_and_mul


_FUSED_SHARED_BUFFERS = {}
_FUSED_SHARED_OUTPUTS = {}


class _FusedSharedExperts(nn.Module):
    """Use DeepGEMM's BF16 shared expert with the routed MXFP4 kernel."""

    def __init__(self, cfg, packed, shared_gate_up, shared_down, shared_hidden):
        super().__init__()
        import deep_gemm
        import torch.distributed as dist

        from rtp_llm.models_py.kernels.cuda.quant_layouts import (
            FP4_BLOCK,
            prepare_fp4_weight_scale_for_deepgemm,
        )
        from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.mega_moe import (
            _get_validated_world_ep_group,
        )
        from rtp_llm.models_py.modules.factory.fused_moe.utils.mega_moe.input_packer import (
            get_mega_moe_input_packer,
        )

        params = inspect.signature(deep_gemm.fp8_fp4_mega_moe).parameters
        required = {"shared_x", "shared_y", "shared_l1_weights", "shared_l2_weights",
                    "activation_beta", "activation_linear_beta"}
        if not required.issubset(params):
            raise RuntimeError("DeepGEMM lacks BF16 fused shared-expert SiTU support")
        if "shared_intermediate_hidden" not in inspect.signature(
            deep_gemm.get_symm_buffer_for_mega_moe
        ).parameters:
            raise RuntimeError("DeepGEMM lacks BF16 shared-expert workspace support")
        if cfg.world_size != cfg.ep_size:
            raise RuntimeError("fused shared expert requires EP group equal WORLD")

        experts = cfg.n_local_experts
        latent = cfg.dim
        intermediate = cfg.moe_inter_dim
        w13 = packed.pop(W.moe_w1)
        device = w13.device
        s13 = prepare_fp4_weight_scale_for_deepgemm(
            packed.pop(W.moe_s1), 2 * intermediate, latent, experts, backend=deep_gemm
        )
        w2 = packed.pop(W.moe_w2)
        s2 = prepare_fp4_weight_scale_for_deepgemm(
            packed.pop(W.moe_s2), latent, intermediate, experts, backend=deep_gemm
        )
        self.routed_l1, self.routed_l2 = deep_gemm.transform_weights_for_mega_moe(
            (w13, s13), (w2, s2), activation="situ"
        )
        del w13, s13, w2, s2

        shared_intermediate = shared_gate_up.size(0) // 2
        if (shared_gate_up.dtype != torch.bfloat16
                or shared_down.dtype != torch.bfloat16
                or tuple(shared_gate_up.shape) != (2 * shared_intermediate, shared_hidden)
                or tuple(shared_down.shape) != (shared_intermediate, shared_hidden)):
            raise ValueError("fused shared-expert BF16 weights have unexpected layout")
        self.shared_l1, self.shared_l2 = deep_gemm.transform_weights_for_mega_moe(
            shared_gate_up.contiguous(), shared_down.T.contiguous(), activation="situ"
        )
        group = _get_validated_world_ep_group(cfg, dist)
        capacity = max(int(cfg.max_tokens_per_rank), 1)
        key = (id(group), cfg.n_routed_experts, capacity, cfg.n_activated_experts,
               latent, intermediate, shared_intermediate)
        buf = _FUSED_SHARED_BUFFERS.get(key)
        if buf is None:
            buf = deep_gemm.get_symm_buffer_for_mega_moe(
                group=group, num_experts=cfg.n_routed_experts,
                num_max_tokens_per_rank=capacity, num_topk=cfg.n_activated_experts,
                hidden=latent, intermediate_hidden=intermediate,
                mma_type="fp8xfp4", activation="situ",
                shared_intermediate_hidden=shared_intermediate,
            )
            expected = (int(buf.num_max_tokens_per_rank), shared_intermediate)
            if tuple(buf.shared_l2_acts.shape) != expected:
                raise RuntimeError("DeepGEMM returned the wrong shared workspace layout")
            _FUSED_SHARED_BUFFERS[key] = buf
        self.buf = buf
        output_key = (device, int(buf.num_max_tokens_per_rank), latent, shared_hidden)
        outputs = _FUSED_SHARED_OUTPUTS.get(output_key)
        if outputs is None:
            outputs = (
                torch.empty((output_key[1], latent), device=device, dtype=torch.bfloat16),
                torch.empty((output_key[1], shared_hidden), device=device, dtype=torch.bfloat16),
                torch.empty((output_key[1], shared_hidden), device=device, dtype=torch.bfloat16),
            )
            _FUSED_SHARED_OUTPUTS[output_key] = outputs
        self.routed_y, self.shared_x, self.shared_y = outputs
        self.packer = get_mega_moe_input_packer()
        self.beta = float(cfg.activation_beta)
        self.linear_beta = cfg.activation_linear_beta
        self.fp4_block = FP4_BLOCK

    def forward(self, hidden, routed_input, routing, ids, valid_mask=None):
        import deep_gemm

        tokens = int(routed_input.size(0))
        if tokens > self.buf.num_max_tokens_per_rank:
            raise ValueError("fused shared-expert input exceeds token capacity")
        if self.packer.name == "fused":
            from rtp_llm.models_py.triton_kernels.moe.mega_moe_input_pack import (
                fused_pack_mega_moe_inputs,
            )

            fused_pack_mega_moe_inputs(
                routed_input, routing, ids, self.buf.x[:tokens],
                self.buf.x_sf[:tokens], self.buf.topk_idx[:tokens],
                self.buf.topk_weights[:tokens], shared_input=hidden,
                shared_out=self.shared_x[:tokens], valid_mask=valid_mask,
            )
        else:
            if valid_mask is not None:
                ids = torch.where(valid_mask[:, None], ids, 0)
                routing = torch.where(valid_mask[:, None], routing, 0)
            self.packer.pack(routed_input, routing, ids, self.buf, tokens)
            self.shared_x[:tokens].copy_(hidden)
            if valid_mask is not None:
                self.shared_x[:tokens].masked_fill_(~valid_mask[:, None], 0)
        deep_gemm.fp8_fp4_mega_moe(
            self.routed_y[:tokens], self.routed_l1, self.routed_l2, self.buf,
            shared_x=self.shared_x, shared_y=self.shared_y,
            shared_l1_weights=self.shared_l1, shared_l2_weights=self.shared_l2,
            recipe=(1, 1, self.fp4_block), activation="situ",
            activation_clamp=None, activation_beta=self.beta,
            activation_linear_beta=self.linear_beta, fast_math=True,
        )
        return self.routed_y[:tokens], self.shared_y[:tokens]


def situ(gate, up, beta, linear_beta, *, inplace=False):
    if gate.is_cuda:
        from rtp_llm.models_py.triton_kernels.common.situ import situ as fused_situ

        return fused_situ(
            gate, up, beta, linear_beta, inplace=inplace, native_k3=True
        )
    g, u = gate.float(), up.float()
    g = beta * torch.tanh(g / beta) * torch.sigmoid(g)
    if linear_beta is not None:
        u = linear_beta * torch.tanh(u / linear_beta)
    result = (g * u).to(gate.dtype)
    return gate.copy_(result) if inplace else result


class KimiK3LatentMoE(nn.Module):
    def __init__(
        self,
        config,
        parallelism,
        weights,
        layer_idx,
        moe_config,
        hardware,
        moe_capacity,
    ):
        super().__init__()
        runtime = config.k3_runtime_config
        self.beta, self.linear_beta = (
            runtime.activation_situ_beta,
            runtime.activation_situ_linear_beta,
        )
        self.top_k, self.groups, self.top_groups = (
            config.moe_k,
            config.moe_n_group,
            config.moe_topk_group,
        )
        self.renormalize, self.route_scale = (
            config.has_moe_norm,
            config.routed_scaling_factor,
        )
        self.router = KimiK3RouterProjection(weights[K3W.MOE_GATE])
        # The router prepares a second layout for GEMM. Keep the prepared
        # storage in ModelWeights so the original layout can be released.
        weights[K3W.MOE_GATE] = self.router.weight
        self.correction = weights[K3W.MOE_CORRECTION_BIAS].float()
        down_weight = weights[K3W.MOE_ROUTED_DOWN]
        self.down = (
            KimiK3LatentDownLinear(down_weight)
            if down_weight.is_cuda and down_weight.dtype == torch.bfloat16
            else linear(weights, K3W.MOE_ROUTED_DOWN, hardware)
        )
        if isinstance(self.down, KimiK3LatentDownLinear):
            # The small-batch plan may make a contiguous copy of the down
            # weight. Preserve its original [in, out] view without retaining
            # the old allocation for every layer.
            weights[K3W.MOE_ROUTED_DOWN] = self.down.weight.T
        self.up = linear(weights, K3W.MOE_ROUTED_UP, hardware)
        self.shared_gate_up = weights[K3W.MOE_SHARED_GATE_UP]
        self.shared_expert_weight_shard = shared_expert_weight_shard_enabled(
            parallelism.role_type
        )
        self.shared_weight_tp_size = int(parallelism.tp_size)
        if self.shared_expert_weight_shard:
            if self.shared_weight_tp_size <= 0 or self.shared_weight_tp_size % 2:
                raise ValueError("shared expert sharding requires an even TP size")
            if config.inter_size % self.shared_weight_tp_size:
                raise ValueError("shared expert intermediate size must divide TP")
            expected_gate_up = (2 * config.inter_size // self.shared_weight_tp_size, config.hidden_size)
            expected_down = (config.inter_size // self.shared_weight_tp_size, config.hidden_size)
            self.shared_down_weight = weights[K3W.MOE_SHARED_DOWN]
            self.shared_down = None
        else:
            expected_gate_up = (2 * config.inter_size, config.hidden_size)
            expected_down = (config.inter_size, config.hidden_size)
            self.shared_down_weight = None
            self.shared_down = linear(weights, K3W.MOE_SHARED_DOWN, hardware)
        if self.shared_gate_up.shape != expected_gate_up or weights[K3W.MOE_SHARED_DOWN].shape != expected_down:
            raise ValueError(
                "K3 shared projection layout does not match the FFN TP placement: "
                f"gate/up={tuple(self.shared_gate_up.shape)} expected={expected_gate_up}, "
                f"down={tuple(weights[K3W.MOE_SHARED_DOWN].shape)} expected={expected_down}"
            )
        cfg = MoEConfigAdapter(
            config,
            parallelism,
            moe_config or MoeConfig(),
            enable_cuda_graph=bool(hardware and hardware.enable_cuda_graph),
            expert_activation="situ",
            activation_beta=self.beta,
            activation_linear_beta=self.linear_beta,
            mega_moe_backend=get_k3_moe_backend(),
        )
        if cfg.moe_strategy not in ("auto", "mega_moe", "mega_moe_se"):
            raise ValueError("K3 native MXFP4 requires the MegaMoE executor")
        fused_shared = cfg.moe_strategy == "mega_moe_se"
        self.fuses_layer_residual = fused_shared
        if fused_shared and parallelism.role_type != RoleType.DECODE:
            raise ValueError("BF16 fused shared expert requires Decode's full weights")
        norm_cls = RMSNorm if fused_shared else KimiK3LatentRMSNorm
        self.norm = (
            norm_cls(weights[K3W.MOE_ROUTED_NORM], config.layernorm_eps)
            if runtime.latent_moe_use_norm else None
        )
        cfg.moe_strategy = "mega_moe"
        cfg.max_tokens_per_rank = max(int(moe_capacity), 1)
        cfg.moe_quant_method = "FP8_FP4"
        cfg.dim = cfg.hidden_size = runtime.routed_expert_hidden_size
        cfg.layer_id = layer_idx
        cfg.moe_w1_layout = "gate_up"
        packed = {
            W.moe_w1: torch.cat(
                (weights.pop(K3W.MOE_W1_PACKED), weights.pop(K3W.MOE_W3_PACKED)), dim=1
            ).view(torch.int8),
            W.moe_s1: torch.cat(
                (weights.pop(K3W.MOE_W1_SCALE), weights.pop(K3W.MOE_W3_SCALE)), dim=1
            ).view(torch.float8_e8m0fnu),
            W.moe_w2: weights.pop(K3W.MOE_W2_PACKED).view(torch.int8),
            W.moe_s2: weights.pop(K3W.MOE_W2_SCALE).view(torch.float8_e8m0fnu),
        }
        if fused_shared:
            self.experts = _FusedSharedExperts(
                cfg, packed, self.shared_gate_up, weights[K3W.MOE_SHARED_DOWN],
                config.hidden_size,
            )
            # The transformed weights own the only live copies on Decode.
            weights.pop(K3W.MOE_SHARED_GATE_UP)
            weights.pop(K3W.MOE_SHARED_DOWN)
            self.shared_gate_up = None
            self.shared_down = None
            self.expert_step = self._fused_expert_step
            self.mask_routes = self._keep_routes
        else:
            self.experts = FusedMoeFactory().create_fused_moe(cfg, packed)
            self.expert_step = self._separate_expert_step
            self.mask_routes = self._mask_routes

    @staticmethod
    def _keep_routes(routing, ids, valid_mask):
        return routing, ids

    @staticmethod
    def _mask_routes(routing, ids, valid_mask):
        if valid_mask is None:
            return routing, ids
        return (
            torch.where(valid_mask[:, None], routing, 0),
            torch.where(valid_mask[:, None], ids, 0),
        )

    def _fused_expert_step(self, hidden, routed_input, routing, ids, valid_mask):
        with profile_scope("RTP::moe.routed_and_shared_experts"):
            return self.experts(hidden, routed_input, routing, ids, valid_mask)

    def _separate_expert_step(self, hidden, routed_input, routing, ids, valid_mask):
        with profile_scope("RTP::moe.routed_experts"):
            routed = self.experts(routed_input, routing, ids, activation="situ")
        with profile_scope("RTP::moe.shared_gate_up_proj"):
            gate_up_weight = self.shared_gate_up
            if self.shared_expert_weight_shard:
                gate_up_weight = all_gather(gate_up_weight.contiguous(), group=Group.TP)
            gate_up = bf16_linear(hidden, gate_up_weight)
            if self.shared_expert_weight_shard:
                del gate_up_weight
            gate, up = gate_up.chunk(2, dim=-1)
        with profile_scope("RTP::moe.shared_activation"):
            shared_input = situ_and_mul(gate, up, self.beta, self.linear_beta)
            del gate, up, gate_up
        with profile_scope("RTP::moe.shared_down_proj"):
            if self.shared_expert_weight_shard:
                down_weight = all_gather(self.shared_down_weight.contiguous(), group=Group.TP)
                shared = torch.matmul(shared_input, down_weight)
                del down_weight
            else:
                shared = self.shared_down(shared_input)
            del shared_input
        return routed, shared

    def forward(self, hidden, valid_mask=None, residual=None):
        with profile_scope("RTP::moe.router"):
            routing, ids = grouped_topk(
                self.router(hidden),
                self.correction,
                top_k=self.top_k,
                groups=self.groups,
                top_groups=self.top_groups,
                renormalize=self.renormalize,
                scale=self.route_scale,
            )
        routing, ids = self.mask_routes(routing, ids, valid_mask)
        with profile_scope("RTP::moe.routed_down_proj"):
            routed_input = self.down(hidden)
        routed, shared = self.expert_step(
            hidden, routed_input, routing, ids, valid_mask
        )
        if self.norm is not None:
            with profile_scope("RTP::moe.routed_norm"):
                routed = self.norm(routed.contiguous())
        if residual is not None:
            if not self.fuses_layer_residual or not isinstance(self.up, KimiK3Bf16Linear):
                raise ValueError("MoE residual fusion requires the BF16 Decode shared expert")
            from rtp_llm.models_py.triton_kernels.moe.output_add import add_moe_output

            with profile_scope("RTP::moe.routed_up_proj"):
                routed_up = self.up(routed)
            with profile_scope("RTP::moe.output_add"):
                return add_moe_output(routed_up, shared, residual)
        if isinstance(self.up, KimiK3Bf16Linear):
            # Match native K3: combine the routed projection and shared
            # output in a single addmm GEMM call.
            with profile_scope("RTP::moe.routed_up_proj_add_shared"):
                if (
                    routed.is_cuda
                    and routed.ndim == 2
                    and routed.shape[0] >= 4096
                    and routed.dtype == self.up.weight.dtype == shared.dtype == torch.bfloat16
                    and shared.is_contiguous()
                    and self.up.bias is None
                    and not torch.cuda.is_current_stream_capturing()
                ):
                    return bf16_linear_add_inplace(routed, self.up.weight, shared)
                return self.up(routed, residual=shared)
        with profile_scope("RTP::moe.routed_up_proj"):
            routed_up = self.up(routed)
        return routed_up + shared
