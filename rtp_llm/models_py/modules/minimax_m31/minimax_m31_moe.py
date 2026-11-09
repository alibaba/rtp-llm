"""MiniMax-M3.1 prepacked NVFP4 routed experts on main's FusedMoe interface."""

from __future__ import annotations

import math
from contextvars import ContextVar
from dataclasses import dataclass

import torch
import torch.distributed as dist
from torch import nn

from rtp_llm.models_py.model_desc.generic_moe import GenericMoeLayer
from rtp_llm.models_py.modules.factory.fused_moe.defs.fused_moe import (
    CombineForwardPayload,
    FusedMoe,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.quant_config import (
    FusedMoEQuantConfig,
)
from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.fp8_fp4_base import (
    Fp8Fp4ExecutorBase,
)
from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.mega_moe import (
    _activate_mega_moe_rank_nvcc_tmpdir,
    _get_validated_world_ep_group,
    _mega_output_capacity,
    _restore_tmpdir,
)
from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.routers.fp8_fp4_router import (
    MegaMoeRouter,
)
from rtp_llm.models_py.modules.factory.fused_moe.utils.mega_moe.buffer import (
    _get_or_create_mega_output,
)
from rtp_llm.models_py.modules.minimax_m31.kernels.minimax_m31_moe_pack import (
    pack_m31_nvfp4_inputs,
)
from rtp_llm.models_py.modules.minimax_m31.kernels.minimax_m31_prefill_router import (
    expand_fp32_router_weight,
    minimax_m31_prefill_router_logits,
)
from rtp_llm.models_py.modules.minimax_m31.minimax_m31_mlp import M31DenseMLP
from rtp_llm.ops import RoleType
from rtp_llm.utils.model_weight import W
from rtp_llm.utils.warmup import model_warm_up_enabled

_NVFP4_BUFFERS = {}
_NVFP4_WARMED_KEYS = set()


@dataclass(frozen=True)
class M31PrefillChunkPlan:
    capacity: int
    chunks: int


_PREFILL_CHUNK_PLAN: ContextVar[M31PrefillChunkPlan | None] = ContextVar(
    "m31_prefill_chunk_plan", default=None
)


def prepare_m31_prefill_chunks(layers, local_rows: int, device):
    """Agree on routed-MoE calls across EP ranks before either forward phase."""
    executors = [
        layer.mlp.fused_moe.fused_experts
        for layer in layers
        if isinstance(getattr(layer, "mlp", None), M31MoeLayer)
    ]
    if not executors:
        return None
    capacity = int(executors[0].cfg.max_tokens_per_rank)
    if (
        capacity < 1
        or capacity > 16384
        or any(
            int(executor.cfg.max_tokens_per_rank) != capacity for executor in executors
        )
    ):
        raise ValueError("M3.1 Prefill requires equal MoE chunk capacity <=16384")
    group = _get_validated_world_ep_group(executors[0].cfg, dist)
    local_chunks = max(1, (local_rows + capacity - 1) // capacity)
    agreed = torch.tensor(
        [local_chunks, capacity, -capacity], dtype=torch.int64, device=device
    )
    dist.all_reduce(agreed, op=dist.ReduceOp.MAX, group=group)
    chunks, max_capacity, neg_min_capacity = agreed.tolist()
    if max_capacity != -neg_min_capacity:
        raise ValueError("M3.1 Prefill MoE chunk capacity differs across EP ranks")
    return M31PrefillChunkPlan(capacity, chunks)


def run_m31_prefill_chunks(hidden, weights, ids, forward, plan):
    rows = hidden.shape[0]
    if rows > plan.capacity * plan.chunks:
        raise ValueError("M3.1 Prefill rows exceed agreed MoE chunk plan")
    if plan.chunks == 1 and rows:
        return forward(hidden, weights, ids)
    output = torch.empty_like(hidden)
    dummy = None
    for chunk in range(plan.chunks):
        start = chunk * plan.capacity
        end = min(start + plan.capacity, rows)
        if start < rows:
            output[start:end].copy_(
                forward(hidden[start:end], weights[start:end], ids[start:end])
            )
        else:
            if dummy is None:
                dummy = (
                    hidden.new_zeros((1, hidden.shape[1])),
                    weights.new_zeros((1, weights.shape[1])),
                    torch.arange(
                        weights.shape[1], dtype=ids.dtype, device=ids.device
                    ).unsqueeze(0),
                )
            forward(*dummy)
    return output


def _nvfp4_buffer(config):
    import deep_gemm

    group = _get_validated_world_ep_group(config, dist)
    key = (
        id(group),
        config.n_routed_experts,
        config.max_tokens_per_rank,
        config.n_activated_experts,
        config.dim,
        config.moe_inter_dim,
    )
    buffer = _NVFP4_BUFFERS.get(key)
    if buffer is None:
        buffer = deep_gemm.get_symm_buffer_for_mega_moe_nvfp4(
            group=group,
            num_experts=config.n_routed_experts,
            num_max_tokens_per_rank=max(config.max_tokens_per_rank, 1),
            num_topk=config.n_activated_experts,
            hidden=config.dim,
            intermediate_hidden=config.moe_inter_dim,
            activation="swiglu_oai",
        )
        _NVFP4_BUFFERS[key] = buffer
    return buffer


def _nvfp4_warmup_token_counts(config, num_sms):
    """Choose one token count per DeepGEMM 2.8.2 NVFP4 JIT tile/wave bucket."""
    import deep_gemm

    counts = []
    previous = None
    local_experts = config.n_local_experts
    for tokens in range(1, config.max_tokens_per_rank + 1):
        block_m = deep_gemm.get_block_m_for_mega_moe_nvfp4(
            config.ep_size,
            config.n_routed_experts,
            config.max_tokens_per_rank,
            tokens,
            config.n_activated_experts,
        )
        expected = tokens * config.n_activated_experts / local_experts
        if expected < 1:
            wave = local_experts
        else:
            m_blocks = math.ceil(math.ceil(expected) / block_m)
            n_blocks = 2 * config.moe_inter_dim // 128
            imbalance = 3 if block_m == 192 else 6
            wave = min(
                local_experts, math.ceil(imbalance * num_sms / (m_blocks * n_blocks))
            )
            if block_m == 192:
                wave = max(wave, min(4, local_experts))
            while wave < local_experts and local_experts % wave:
                wave += 1
        signature = (block_m, wave)
        if signature != previous:
            counts.append(tokens)
            previous = signature
    return counts


def _pack_weight_scales(scale: torch.Tensor, experts: int, rows: int, cols: int):
    import deep_gemm

    expected = (experts, rows, cols // 16)
    if scale.dtype != torch.float8_e4m3fn or tuple(scale.shape) != expected:
        raise ValueError(f"M3.1 NVFP4 scale requires E4M3 {expected}")
    packed = scale.contiguous().view(torch.int32)
    return deep_gemm.transform_sf_into_required_layout(
        packed, rows, cols, (1, 16), experts
    )


def reorder_m31_nvfp4_checkpoint(w13, s13, g13, g2):
    """Convert RTP [up|gate] stacks and inverse scales to DeepGEMM order."""
    up_w, gate_w = w13.chunk(2, dim=1)
    up_s, gate_s = s13.chunk(2, dim=1)
    gate_up_w = torch.cat((gate_w, up_w), dim=1).contiguous()
    gate_up_s = torch.cat((gate_s, up_s), dim=1).contiguous()
    inverse_gsf = torch.stack((g13[:, 1], g13[:, 0]), dim=1)
    if not bool(torch.isfinite(inverse_gsf).all()) or bool((inverse_gsf <= 0).any()):
        raise ValueError("M3.1 L1 inverse global scales must be positive finite")
    if not bool(torch.isfinite(g2).all()) or bool((g2 <= 0).any()):
        raise ValueError("M3.1 L2 inverse global scales must be positive finite")
    return (
        gate_up_w,
        gate_up_s,
        inverse_gsf.reciprocal().contiguous(),
        g2.reciprocal().contiguous(),
    )


class M31NVFP4MegaExecutor(Fp8Fp4ExecutorBase):
    """DeepGEMM NVFP4 MegaMoE with OAI activation and checkpoint scales."""

    execute_empty_inputs = True

    def setup_weights(self, weights):
        import deep_gemm

        cfg = self.cfg
        if cfg.max_tokens_per_rank > 16384:
            raise ValueError(
                "M3.1 NVFP4 MegaMoE requires model-level Prefill chunking "
                "to at most 16384 tokens per rank"
            )
        if cfg.ep_size <= 1 or cfg.world_size != cfg.ep_size:
            raise ValueError("M3.1 NVFP4 MegaMoE requires EP=WORLD>1")
        if cfg.dim != 6144 or cfg.n_activated_experts not in (4, 8):
            raise ValueError("M3.1 NVFP4 MegaMoE requires hidden=6144, TopK=4 or 8")
        e, d, inter = cfg.n_local_experts, cfg.dim, cfg.moe_inter_dim
        w13, s13, g13 = (weights[W.moe_w1], weights[W.moe_s1], weights[W.moe_w1_s2])
        w2, s2, g2 = (weights[W.moe_w2], weights[W.moe_s2], weights[W.moe_w2_s2])
        for name, tensor, dtype, shape in (
            ("w13", w13, torch.int8, (e, 2 * inter, d // 2)),
            ("s13", s13, torch.float8_e4m3fn, (e, 2 * inter, d // 16)),
            ("g13", g13, torch.float32, (e, 2)),
            ("w2", w2, torch.int8, (e, d, inter // 2)),
            ("s2", s2, torch.float8_e4m3fn, (e, d, inter // 16)),
            ("g2", g2, torch.float32, (e,)),
        ):
            if tensor.dtype != dtype or tuple(tensor.shape) != shape:
                raise ValueError(f"M3.1 NVFP4 {name} requires {dtype} {shape}")
        if any(t.device != w13.device for t in (s13, g13, w2, s2, g2)):
            raise ValueError("M3.1 NVFP4 routed weights must share a device")
        gate_up_w, gate_up_s, l1_raw_gsf, l2_raw_gsf = reorder_m31_nvfp4_checkpoint(
            w13, s13, g13, g2
        )
        with torch.cuda.device(w13.device):
            packed_s13 = _pack_weight_scales(gate_up_s, e, 2 * inter, d)
            packed_s2 = _pack_weight_scales(s2, e, d, inter)
            (self.l1_w, self.l1_sf, self.l1_gsf), (
                self.l2_w,
                self.l2_sf,
                self.l2_gsf,
            ) = deep_gemm.transform_weights_for_mega_moe_nvfp4(
                (gate_up_w, packed_s13, l1_raw_gsf),
                (w2.contiguous(), packed_s2, l2_raw_gsf),
            )
        weight_device = w13.device
        for key in (W.moe_w1, W.moe_s1, W.moe_w1_s2, W.moe_w2, W.moe_s2, W.moe_w2_s2):
            weights.pop(key)
        del w13, s13, g13, w2, s2, g2
        del gate_up_w, gate_up_s, l1_raw_gsf, l2_raw_gsf, packed_s13, packed_s2
        torch.cuda.empty_cache()
        self.buffer = _nvfp4_buffer(cfg)
        self.output = _get_or_create_mega_output(
            _mega_output_capacity(self.buffer, cfg.max_tokens_per_rank),
            d,
            torch.bfloat16,
            weight_device,
        )
        self._maybe_warmup_jit_once()

    @torch.inference_mode()
    def _maybe_warmup_jit_once(self):
        import deep_gemm

        cfg = self.cfg
        device = self.l1_w.device
        with torch.cuda.device(device):
            enabled = model_warm_up_enabled()
            if enabled and torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "M3.1 NVFP4 MegaMoE warmup cannot run during graph capture"
                )
            num_sms = int(deep_gemm.get_num_sms()) if enabled else None
            key = (
                id(self.buffer),
                num_sms,
                cfg.dim,
                cfg.moe_inter_dim,
                cfg.n_activated_experts,
                cfg.max_tokens_per_rank,
                cfg.model_config.swiglu_alpha,
                cfg.swiglu_limit,
            )
            needed = enabled and key not in _NVFP4_WARMED_KEYS
            counts = _nvfp4_warmup_token_counts(cfg, num_sms) if needed else None
            group = _get_validated_world_ep_group(cfg, dist)
            agreement = [None] * dist.get_world_size(group)
            signature = (
                enabled,
                needed,
                num_sms,
                cfg.n_routed_experts,
                cfg.n_local_experts,
                cfg.dim,
                cfg.moe_inter_dim,
                cfg.n_activated_experts,
                cfg.max_tokens_per_rank,
                cfg.model_config.swiglu_alpha,
                cfg.swiglu_limit,
                counts,
            )
            dist.all_gather_object(agreement, signature, group=group)
            if any(peer != signature for peer in agreement):
                raise ValueError("M3.1 NVFP4 MegaMoE warmup differs across EP ranks")
            if not needed:
                return
            rows = max(counts)
            x = torch.zeros((rows, cfg.dim), dtype=torch.bfloat16, device=device)
            weights = torch.full(
                (rows, cfg.n_activated_experts),
                1.0 / cfg.n_activated_experts,
                dtype=torch.float32,
                device=device,
            )
            ids = (
                (
                    cfg.local_expert_start
                    + torch.arange(
                        cfg.n_activated_experts, dtype=torch.int64, device=device
                    )
                    % cfg.n_local_experts
                )
                .expand(rows, -1)
                .contiguous()
            )
            rank = dist.get_rank(group)
            _, previous_tmpdir = _activate_mega_moe_rank_nvcc_tmpdir(rank)
            try:
                for count in counts:
                    dist.barrier(group=group)
                    pack_m31_nvfp4_inputs(
                        x[:count], weights[:count], ids[:count], self.buffer
                    )
                    deep_gemm.nvfp4_nvfp4_mega_moe(
                        self.output[:count],
                        (self.l1_w, self.l1_sf, self.l1_gsf),
                        (self.l2_w, self.l2_sf, self.l2_gsf),
                        self.buffer,
                        recipe=(1, 1, 16),
                        activation="swiglu_oai",
                        activation_clamp=cfg.swiglu_limit,
                        activation_alpha=cfg.model_config.swiglu_alpha,
                        fast_math=True,
                        assume_all_topk_valid=True,
                    )
                    torch.cuda.synchronize(device)
            finally:
                _restore_tmpdir(previous_tmpdir)
            _NVFP4_WARMED_KEYS.add(key)
            del x, weights, ids
            torch.cuda.empty_cache()

    def execute(
        self,
        payload,
        activation,
        expert_map,
        a2_scale,
        apply_router_weight_on_input,
        extra_expert_args,
    ) -> CombineForwardPayload:
        import deep_gemm

        if activation.lower() != "swiglu_oai":
            raise ValueError("M3.1 NVFP4 experts require swiglu_oai")
        if (
            expert_map is not None
            or a2_scale is not None
            or apply_router_weight_on_input
        ):
            raise ValueError(
                "M3.1 NVFP4 MegaMoE does not accept external expert transforms"
            )
        args = extra_expert_args or {}
        alpha = float(args.get("swiglu_alpha", 0.0))
        limit = float(args.get("swiglu_limit", 0.0))
        if alpha <= 0 or limit <= 0:
            raise ValueError("M3.1 NVFP4 MegaMoE requires positive OAI alpha and limit")
        x = payload.expert_x
        topk_weights = payload.expert_topk_weights
        topk_ids = payload.expert_topk_ids
        if topk_weights is None or topk_ids is None:
            raise ValueError("M3.1 NVFP4 MegaMoE requires routed TopK")
        if topk_ids.dtype != torch.int64:
            raise ValueError("M3.1 NVFP4 MegaMoE requires int64 expert IDs")
        pack_m31_nvfp4_inputs(x, topk_weights, topk_ids, self.buffer)
        y = self.output[: x.shape[0]]
        with torch.cuda.device(x.device):
            deep_gemm.nvfp4_nvfp4_mega_moe(
                y,
                (self.l1_w, self.l1_sf, self.l1_gsf),
                (self.l2_w, self.l2_sf, self.l2_gsf),
                self.buffer,
                recipe=(1, 1, 16),
                activation="swiglu_oai",
                activation_clamp=limit,
                activation_alpha=alpha,
                fast_math=True,
                assume_all_topk_valid=True,
            )
        return CombineForwardPayload(fused_expert_output=y)


class _M31Fp32Router(nn.Module):
    def __init__(self, linear, prepare_prefill=False):
        super().__init__()
        self.linear = linear
        if prepare_prefill:
            for name, part in zip(
                ("_prefill_high", "_prefill_middle", "_prefill_low"),
                expand_fp32_router_weight(linear.weight),
            ):
                self.register_buffer(name, part, persistent=False)

    def forward(self, hidden_states):
        if _PREFILL_CHUNK_PLAN.get() is not None and hasattr(self, "_prefill_high"):
            return minimax_m31_prefill_router_logits(
                hidden_states.contiguous(),
                (self._prefill_high, self._prefill_middle, self._prefill_low),
            )
        return self.linear(hidden_states.float())


class M31MoeLayer(GenericMoeLayer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        weight = self.gate.weight
        prepare_prefill = (
            self.parallelism_config.role_type in (RoleType.PREFILL, RoleType.PDFUSION)
            and weight.is_cuda
            and weight.dtype == torch.float32
            and torch.cuda.get_device_capability(weight.device) == (10, 3)
        )
        self.gate = _M31Fp32Router(self.gate, prepare_prefill)

    def _run_fused_moe(
        self,
        hidden_states,
        topk_weights,
        topk_ids,
        activation,
        extra_expert_args,
        skip_tp_allreduce,
    ):
        plan = _PREFILL_CHUNK_PLAN.get()
        if plan is None:
            return super()._run_fused_moe(
                hidden_states,
                topk_weights,
                topk_ids,
                activation,
                extra_expert_args,
                skip_tp_allreduce,
            )

        def forward(x, weights, ids):
            return super(M31MoeLayer, self)._run_fused_moe(
                x,
                weights,
                ids,
                activation,
                extra_expert_args,
                skip_tp_allreduce,
            )

        return run_m31_prefill_chunks(
            hidden_states, topk_weights, topk_ids, forward, plan
        )

    def _create_shared_expert(
        self,
        activation_type,
        parallelism_config,
        weights,
        quant_config,
        hw_kernel_config,
    ):
        return M31DenseMLP(
            activation_type,
            parallelism_config,
            weights,
            quant_config,
            hw_kernel_config=hw_kernel_config,
            alpha=self.config.swiglu_alpha,
            limit=self.config.swiglu_limit,
        )

    def _create_fused_moe(self, config_adapter, weights):
        quant = FusedMoEQuantConfig(quant_dtype="nvfp4", block_shape=[1, 16])
        router = MegaMoeRouter(config_adapter, quant)
        executor = M31NVFP4MegaExecutor(config_adapter, quant, weights)
        return FusedMoe(router, executor, config_adapter.expert_num, "m31_nvfp4_mega")

    def _moe_activation_args(self):
        return "swiglu_oai", {
            "swiglu_alpha": self.config.swiglu_alpha,
            "swiglu_limit": self.config.swiglu_limit,
        }
