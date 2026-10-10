"""K3 latent and shared branches composed with RTP's fused-MoE factory."""

import inspect

import torch
from torch import nn
from torch.nn import functional as F

from rtp_llm.models.kimi_k3.kimi_k3_weight import (
    KimiK3WeightNames as K3W,
    shared_expert_weight_shard_enabled,
)
from rtp_llm.models_py.distributed.collective_torch import Group, all_gather
from rtp_llm.models_py.modules.base import RMSNorm
from rtp_llm.models_py.modules.factory.fused_moe import FusedMoeFactory
from rtp_llm.models_py.modules.factory.fused_moe.defs.config_adapter import (
    MoEConfigAdapter,
)
from rtp_llm.utils.model_weight import W
from rtp_llm.ops import MoeConfig, RoleType
from .router import KimiK3Router
from rtp_llm.models_py.modules.factory.linear import LinearFactory
from rtp_llm.models_py.modules.factory.linear.impl.cuda.f16_linear import (
    CudaF16Linear,
    linear_add_inplace,
)
from rtp_llm.models_py.triton_kernels.moe.activation import situ_and_mul


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
        self.router = KimiK3Router(
            weights[K3W.MOE_GATE],
            weights[K3W.MOE_CORRECTION_BIAS],
            top_k=config.moe_k,
            groups=config.moe_n_group,
            top_groups=config.moe_topk_group,
            renormalize=config.has_moe_norm,
            scale=config.routed_scaling_factor,
        )
        # The router prepares a second layout for GEMM. Keep the prepared
        # storage in ModelWeights so the original layout can be released.
        weights[K3W.MOE_GATE] = self.router.weight
        if any(
            weights[key].dtype == torch.float8_e4m3fn
            for key in (K3W.MOE_ROUTED_DOWN, K3W.MOE_ROUTED_UP, K3W.MOE_SHARED_DOWN)
        ):
            raise ValueError("K3 latent MoE projections require unquantized weights")
        self.down = LinearFactory.create_linear_from_weights(
            weights, K3W.MOE_ROUTED_DOWN, hw_kernel_config=hardware
        )
        self.up = LinearFactory.create_linear_from_weights(
            weights, K3W.MOE_ROUTED_UP, hw_kernel_config=hardware
        )
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
            self.shared_down = LinearFactory.create_linear_from_weights(
                weights, K3W.MOE_SHARED_DOWN, hw_kernel_config=hardware
            )
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
        )
        if cfg.moe_strategy not in ("auto", "mega_moe", "mega_moe_se"):
            raise ValueError("K3 native MXFP4 requires the MegaMoE executor")
        fused_shared = cfg.moe_strategy == "mega_moe_se"
        self.fuses_layer_residual = fused_shared
        if fused_shared and parallelism.role_type != RoleType.DECODE:
            raise ValueError("BF16 fused shared expert requires Decode's full weights")
        self.norm = (
            RMSNorm(weights[K3W.MOE_ROUTED_NORM], config.layernorm_eps)
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
        return self.experts(hidden, routed_input, routing, ids, valid_mask)

    def _separate_expert_step(self, hidden, routed_input, routing, ids, valid_mask):
        routed = self.experts(routed_input, routing, ids, activation="situ")
        gate_up_weight = self.shared_gate_up
        if self.shared_expert_weight_shard:
            gate_up_weight = all_gather(gate_up_weight.contiguous(), group=Group.TP)
        gate_up = F.linear(hidden, gate_up_weight)
        if self.shared_expert_weight_shard:
            del gate_up_weight
        gate, up = gate_up.chunk(2, dim=-1)
        shared_input = situ_and_mul(gate, up, self.beta, self.linear_beta)
        del gate, up, gate_up
        if self.shared_expert_weight_shard:
            down_weight = all_gather(self.shared_down_weight.contiguous(), group=Group.TP)
            shared = torch.matmul(shared_input, down_weight)
            del down_weight
        else:
            shared = self.shared_down(shared_input)
        del shared_input
        return routed, shared

    def forward(self, hidden, valid_mask=None, residual=None):
        routing, ids = self.router(hidden)
        routing, ids = self.mask_routes(routing, ids, valid_mask)
        routed_input = self.down(hidden)
        routed, shared = self.expert_step(
            hidden, routed_input, routing, ids, valid_mask
        )
        if self.norm is not None:
            routed = self.norm(routed.contiguous())
        bf16_up = (
            isinstance(self.up, CudaF16Linear)
            and self.up.weight.is_cuda
            and self.up.weight.dtype == torch.bfloat16
        )
        if residual is not None:
            if not self.fuses_layer_residual or not bf16_up:
                raise ValueError("MoE residual fusion requires the BF16 Decode shared expert")
            from rtp_llm.models_py.triton_kernels.moe.output_add import add_moe_output

            routed_up = self.up(routed)
            return add_moe_output(routed_up, shared, residual)
        if bf16_up:
            # Match native K3: combine the routed projection and shared
            # output in a single addmm GEMM call.
            if (
                routed.is_cuda
                and routed.ndim == 2
                and routed.shape[0] >= 4096
                and routed.dtype == self.up.weight.dtype == shared.dtype == torch.bfloat16
                and shared.is_contiguous()
                and self.up.bias is None
                and not torch.cuda.is_current_stream_capturing()
            ):
                return linear_add_inplace(routed, self.up.weight, shared)
            return self.up(routed, residual=shared)
        routed_up = self.up(routed)
        return routed_up + shared
