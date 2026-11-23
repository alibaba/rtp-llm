"""Native ModelOpt NVFP4 routed experts with a BF16 shared expert."""

import logging
from dataclasses import replace

import torch
import torch.distributed as dist

from rtp_llm.models.glm53_prefill_parallel import shared_expert_local_enabled
from rtp_llm.ops import RoleType
from rtp_llm.utils.model_weight import W

from .jit_warmup import mega_moe_jit_warmup_enabled
from .mega_moe import GLM5MegaMoE, _activation_clamp, _sync_cuda_graph_warmup_ranks
from .mega_moe_wrapper import MegaMoeWrapper

logger = logging.getLogger(__name__)

_ACTIVATION = "swiglu_upper"
_RUNTIME_CACHE: dict[tuple, tuple] = {}
_JIT_WARMED_BUFFERS: set[int] = set()


def _prepare_routed_weight(weight, scale, outer_scale):
    import deep_gemm

    experts, rows, packed_cols = weight.shape
    if weight.dtype != torch.uint8 or scale.dtype != torch.float8_e4m3fn:
        raise ValueError("Expected packed ModelOpt NVFP4 weights and E4M3 block scales")
    scale = deep_gemm.transform_sf_into_required_layout(
        scale.contiguous().view(torch.int32), rows, 2 * packed_cols, (1, 16), experts
    )
    return weight.contiguous().view(torch.int8), scale, outer_scale.contiguous()


class _GLM53SharedMegaMoE(GLM5MegaMoE):
    def _get_group(self):
        cfg = self.cfg
        if not dist.is_initialized():
            raise RuntimeError("GLM53 fused MegaMoE requires an initialized EP group")
        group = dist.group.WORLD
        if group.size() != cfg.ep_size or group.rank() != cfg.ep_rank:
            raise ValueError(
                "GLM53 fused MegaMoE requires WORLD to match the local EP group"
            )
        return group

    def _load_shared_weights(self, weights):
        cfg = self.cfg
        shared_l1 = weights.pop(W.ffn_w13).T.contiguous()
        shared_l2 = weights.pop(W.ffn_w2).T.contiguous()
        if (
            shared_l1.dtype != torch.bfloat16
            or shared_l2.dtype != torch.bfloat16
            or shared_l1.shape != (2 * cfg.moe_inter_dim, cfg.dim)
            or shared_l2.shape != (cfg.dim, cfg.moe_inter_dim)
        ):
            raise ValueError(
                "GLM53 NVFP4 MegaMoE requires full BF16 weights for one shared expert"
            )
        return shared_l1, shared_l2

    def _finish_setup(self):
        import deep_gemm

        self._mega_buf, self._mega_y = self._get_runtime(graph=False)
        if (
            mega_moe_jit_warmup_enabled()
            and id(self._mega_buf) not in _JIT_WARMED_BUFFERS
        ):
            self._warmup_jit(
                self._resolve_jit_warmup_token_counts(deep_gemm.get_num_sms())
            )
            _JIT_WARMED_BUFFERS.add(id(self._mega_buf))

    def _get_runtime(self, *, graph: bool):
        cfg = self.cfg
        dimensions = (
            self._mega_group,
            cfg.n_routed_experts,
            cfg.max_tokens_per_rank,
            cfg.n_activated_experts,
            cfg.dim,
            cfg.moe_inter_dim,
        )
        key = (type(self), *dimensions, graph)
        if key not in _RUNTIME_CACHE:
            buffer = self._allocate_buffer(dimensions)
            output = torch.empty(
                (buffer.num_max_tokens_per_rank, cfg.dim),
                device=self._mega_l1_w.device,
                dtype=torch.bfloat16,
            )
            _RUNTIME_CACHE[key] = (buffer, output)
            logger.info(
                "GLM53 native NVFP4 MegaMoE with BF16 shared expert, EP=%d, capacity=%d, "
                "symmetric buffer %.3f GiB, graph=%s",
                cfg.ep_size,
                cfg.max_tokens_per_rank,
                buffer.buffer.nbytes / (1024**3),
                graph,
            )
        return _RUNTIME_CACHE[key]

    def clone_for_cuda_graph(self) -> "_GLM53SharedMegaMoE":
        clone = type(self)(self.cfg)
        clone._routed_l1 = self._routed_l1
        clone._routed_l2 = self._routed_l2
        clone._mega_l1_w = self._mega_l1_w
        clone._mega_group = self._mega_group
        clone._activation_scales = self._activation_scales
        clone._shared_l1 = self._shared_l1
        clone._shared_l2 = self._shared_l2
        clone._mega_buf, output = clone._get_runtime(graph=True)
        clone._mega_y = torch.empty_like(output)
        return clone


class GLM53Nvfp4MegaMoE(_GLM53SharedMegaMoE):
    def setup_native_weights(self, weights: dict[str, torch.Tensor]) -> None:
        import deep_gemm

        cfg = self.cfg
        group = self._mega_group = self._get_group()
        routed_w1 = weights.pop(W.moe_w1)
        routed_s1 = weights.pop(W.moe_s1)
        w1_outer = weights.pop(W.moe_w1_s2)
        # RTP loads the fused L1 tensor in [up, gate] order.
        up_w, gate_w = routed_w1.chunk(2, dim=1)
        up_s, gate_s = routed_s1.view(torch.uint8).chunk(2, dim=1)
        l1 = _prepare_routed_weight(
            torch.cat((gate_w, up_w), dim=1),
            torch.cat((gate_s, up_s), dim=1).view(torch.float8_e4m3fn),
            w1_outer.reshape(cfg.n_local_experts, 2).flip(1),
        )
        l2 = _prepare_routed_weight(
            weights.pop(W.moe_w2),
            weights.pop(W.moe_s2),
            weights.pop(W.moe_w2_s2).reshape(cfg.n_local_experts),
        )
        self._routed_l1, self._routed_l2 = (
            deep_gemm.transform_weights_for_mega_moe_nvfp4(l1, l2)
        )
        self._mega_l1_w = self._routed_l1[0]
        device = self._mega_l1_w.device

        l1_scales = weights.pop(W.moe_w1_i_s).float()
        l2_scale = (
            weights.pop(W.moe_w2_i_s).reshape(cfg.n_local_experts).float().contiguous()
        )
        valid = (
            (l1_scales.isfinite() & (l1_scales > 0)).all()
            & (l2_scale.isfinite() & (l2_scale > 0)).all()
        ).to(torch.int32)
        dist.all_reduce(valid, op=dist.ReduceOp.MIN, group=group)
        if not bool(valid):
            raise ValueError(
                "ModelOpt NVFP4 activation scales must be finite and positive"
            )
        # The routed L1 input is packed once before dispatch across the EP group.
        l1_scale = l1_scales.amax().reshape(1)
        dist.all_reduce(l1_scale, op=dist.ReduceOp.MAX, group=group)
        self._activation_scales = (l1_scale, l1_scale.reciprocal(), l2_scale)

        shared_l1, shared_l2 = self._load_shared_weights(weights)
        empty_scales = torch.empty(0, device=device, dtype=torch.int32)
        self._shared_l1, self._shared_l2 = (
            deep_gemm.transform_shared_weights_for_mega_moe_nvfp4(
                (shared_l1, empty_scales), (shared_l2, empty_scales)
            )
        )

        self._finish_setup()

    def _allocate_buffer(self, dimensions):
        import deep_gemm

        return deep_gemm.get_symm_buffer_for_mega_moe_nvfp4(
            *dimensions, activation=_ACTIVATION, num_shared_experts=1, shared_bf16=True
        )

    def forward(self, x: torch.Tensor, weights: torch.Tensor, indices: torch.Tensor):
        import deep_gemm
        from flashinfer import fp4_quantize

        tokens = x.size(0)
        buf = self._mega_buf
        l1_scale, l1_scale_inv, l2_scale = self._activation_scales
        if tokens > buf.num_max_tokens_per_rank:
            raise ValueError(
                f"NVFP4 MegaMoE capacity {buf.num_max_tokens_per_rank} < {tokens}"
            )
        if tokens:
            packed, scales = fp4_quantize(
                x.contiguous(),
                l1_scale_inv,
                is_sf_swizzled_layout=False,
                backend="cute-dsl",
            )
            buf.x[:tokens].copy_(packed.view(torch.int8))
            buf.x_sf[:tokens].copy_(scales.view(torch.int32))
            buf.x_gsf[:tokens].copy_(l1_scale)
            buf.topk_idx[:tokens].copy_(indices)
            buf.topk_weights[:tokens].copy_(weights)
            buf.shared_l1_acts[:tokens].copy_(x)

        self._maybe_pre_kernel_barrier(tokens)
        _sync_cuda_graph_warmup_ranks(
            f"glm53.nvfp4_mega_moe.layer{self.cfg.layer_id}.before_deepgemm", x.device
        )
        output = self._mega_y[:tokens]
        deep_gemm.nvfp4_nvfp4_mega_moe(
            output,
            self._routed_l1,
            self._routed_l2,
            buf,
            shared_l1_weights=self._shared_l1,
            shared_l2_weights=self._shared_l2,
            activation=_ACTIVATION,
            activation_clamp=_activation_clamp(self.cfg),
            fast_math=False,
            l2_input_scales=l2_scale,
        )
        return output


class GLM53Bf16DraftMegaMoE(_GLM53SharedMegaMoE):
    def setup_native_weights(self, weights: dict[str, torch.Tensor]) -> None:
        import deep_gemm

        self._mega_group = self._get_group()
        up, gate = weights.pop(W.moe_w1).chunk(2, dim=1)
        down = weights.pop(W.moe_w2)
        if gate.dtype != torch.bfloat16 or down.dtype != torch.bfloat16:
            raise ValueError(
                "The ModelOpt checkpoint's GLM53 draft experts must be BF16"
            )
        self._routed_l1, self._routed_l2 = deep_gemm.transform_weights_for_mega_moe(
            torch.cat((gate, up), dim=1), down.contiguous()
        )
        self._mega_l1_w = self._routed_l1
        self._shared_l1, self._shared_l2 = deep_gemm.transform_weights_for_mega_moe(
            *self._load_shared_weights(weights)
        )
        self._activation_scales = None
        self._finish_setup()

    def _allocate_buffer(self, dimensions):
        import deep_gemm

        return deep_gemm.get_symm_buffer_for_mega_moe(
            *dimensions, mma_type="bf16xbf16", num_shared_experts=1
        )

    def forward(self, x: torch.Tensor, weights: torch.Tensor, indices: torch.Tensor):
        import deep_gemm

        tokens = x.size(0)
        buf = self._mega_buf
        if tokens > buf.num_max_tokens_per_rank:
            raise ValueError(
                f"BF16 draft MegaMoE capacity {buf.num_max_tokens_per_rank} < {tokens}"
            )
        buf.x[:tokens].copy_(x)
        buf.topk_idx[:tokens].copy_(indices)
        buf.topk_weights[:tokens].copy_(weights)
        self._maybe_pre_kernel_barrier(tokens)
        _sync_cuda_graph_warmup_ranks(
            f"glm53.bf16_draft_mega_moe.layer{self.cfg.layer_id}.before_deepgemm",
            x.device,
        )
        output = self._mega_y[:tokens]
        deep_gemm.bf16_mega_moe(
            output,
            self._routed_l1,
            self._routed_l2,
            buf,
            shared_l1_weights=self._shared_l1,
            shared_l2_weights=self._shared_l2,
            activation_clamp=_activation_clamp(self.cfg),
            fast_math=False,
        )
        return output


class MegaMoeNvfp4SEWrapper(MegaMoeWrapper):
    # The upstream prepacker emits an FP8 activation layout.
    prepacked_input_views = None
    forward_prepacked = None

    def __init__(self, config, parallelism_config, weights, *args, **kwargs):
        # Prefill token sharding replicates the shared expert even when FFN
        # tensor parallelism still shards the dense layers.
        if (
            parallelism_config.get_ffn_tp_size() != 1
            and not shared_expert_local_enabled(
                config.model_type,
                parallelism_config.role_type,
                is_glm53_mtp=getattr(config, "is_glm53_mtp", False),
            )
        ):
            raise ValueError(
                "NVFP4 MegaMoE requires a full shared expert on each EP rank"
            )
        self._bf16_draft = getattr(config, "glm53_nvfp4_bf16_mtp", False)
        self._draft_prefill = (
            self._bf16_draft and parallelism_config.role_type == RoleType.PREFILL
        )
        super().__init__(config, parallelism_config, weights, *args, **kwargs)

    def _get_mega_moe_cls(self):
        return GLM53Bf16DraftMegaMoE if self._bf16_draft else GLM53Nvfp4MegaMoE

    def _setup_expert_weights(self, weights: dict[str, torch.Tensor]) -> None:
        if self._draft_prefill:
            # The draft has its own symmetric buffer. Bound it independently
            # of the target's 1M-context budget and use its TP-padded chunks.
            cfg = self.mega_moe.cfg
            self.mega_moe.cfg = replace(
                cfg, max_tokens_per_rank=min(4096, cfg.max_tokens_per_rank)
            )
        self.mega_moe.setup_native_weights(weights)
