"""Opt-in SM90 FP8 MegaMoE routed path with the existing TP contract.

The selected backend must provide the Hopper FP8 API. The regular DeepEP path
and its weights remain available for decode and unsupported input sizes.
"""

import importlib
import logging
import os
from functools import lru_cache

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)


@lru_cache(maxsize=None)
def _workspace(module_name, capacity, device_index):
    backend = importlib.import_module(module_name)
    if not dist.is_initialized():
        raise RuntimeError("SM90 MegaMoE requires an initialized EP process group")
    group = dist.group.WORLD
    if group.size() != 8:
        raise ValueError("This Qwen SM90 MegaMoE adapter requires WORLD=EP=TP=8")
    with torch.cuda.device(device_index):
        buffer = backend.get_symm_buffer_for_sm90_mega_moe(
            group, 512, capacity, 10, 2560, 640, True, "swiglu"
        )
        sf = buffer.l2_acts_sf
        if not (
            buffer.buffer.data_ptr() % 16 == 0
            and sf.data_ptr() % 16 == 0
            and sf.shape[1] == 10
            and sf.stride(0) == 1
            and sf.stride(1) * sf.element_size() % 16 == 0
        ):
            raise ValueError("SM90 MegaMoE L2 scale layout is not TMA aligned")
        logger.info(
            "[QwenSM90MegaMoE] L2 SF shape=%s stride=%s outer_bytes=%d aligned=True",
            tuple(sf.shape), tuple(sf.stride()), sf.stride(1) * sf.element_size(),
        )
        return buffer


class QwenSm90Fp8MegaMoe:
    """A separate routed backend; shared-expert math is owned by the caller."""

    def __init__(self, layer, max_decode_batch):
        from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.deepgemm_masked_executor import (
            DeepGemmMaskedExecutor,
        )
        from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.routers.deepep_low_latency_router import (
            DeepEpLowLatencyRouter,
        )

        pc = layer.parallelism_config
        executor = layer.fused_moe.fused_experts
        router = layer.fused_moe.router
        if not (
            pc.tp_size == pc.ep_size == 8
            and pc.dp_size == 1
            and pc.ffn_sp_size == 1
            and layer.ffn_tp_size == 8
            and type(executor) is DeepGemmMaskedExecutor
            and type(router) is DeepEpLowLatencyRouter
            and router.config.tp_size == 8
            and router.config.ep_size == 8
            and layer.correction_bias is None
            and layer.fake_balance_expert is None
            and not layer.config.eplb_config.enable_eplb()
            and layer.use_ep_shared_allreduce
            and not layer.use_unified_tp_allreduce
            and layer.top_k == 10
            and tuple(executor._w1.shape) == (64, 1280, 2560)
            and tuple(executor._w2.shape) == (64, 2560, 640)
            and executor._w1.dtype == torch.float8_e4m3fn
            and executor._w2.dtype == torch.float8_e4m3fn
            and executor._w1_scale.dtype == torch.float32
            and executor._w2_scale.dtype == torch.float32
            and tuple(executor._w1_scale.shape) == (64, 10, 20)
            and tuple(executor._w2_scale.shape) == (64, 20, 5)
            and getattr(layer.config, "moe_w1_layout", "up_gate") == "up_gate"
            and torch.version.hip is None
            and torch.cuda.get_device_capability(executor._w1.device) == (9, 0)
            and max_decode_batch > 0
        ):
            raise ValueError("Qwen SM90 FP8 MegaMoE was requested for an unsupported configuration")

        self.module_name = os.environ.get(
            "QWEN_MOE_SM90_MEGA_MODULE", "rtp_megagemm_precision_fma_v1"
        )
        self.backend = importlib.import_module(self.module_name)
        self.min_tokens = max_decode_batch + 1
        self.router = router
        self.buffer = _workspace(
            self.module_name,
            int(os.environ.get("QWEN_MOE_SM90_MEGA_MAX_TOKENS_PER_RANK", "1024")),
            executor._w1.device.index,
        )
        # Canonical RTP FP8 W13 is up|gate. Swap the two 640-row weight
        # halves and the corresponding five block-scale rows exactly once.
        # The backend then interleaves gate/up rows without requantization.
        w13 = torch.cat((executor._w1[:, 640:], executor._w1[:, :640]), dim=1)
        s13 = torch.cat(
            (executor._w1_scale[:, 5:], executor._w1_scale[:, :5]), dim=1
        ).contiguous()
        self.l1, self.l2 = self.backend.transform_weights_for_mega_moe_sm90(
            (w13, s13), (executor._w2, executor._w2_scale)
        )
        self._logged_shapes = set()
        logger.info(
            "[QwenSM90MegaMoE] prepared module=%s rank=%d min_tokens=%d capacity_per_rank=%d",
            self.module_name, dist.get_rank(), self.min_tokens,
            self.buffer.num_max_tokens_per_rank,
        )

    def supports(self, x):
        return (
            not torch.cuda.is_current_stream_capturing()
            and x.is_cuda
            and x.dtype == torch.bfloat16
            and x.ndim == 2
            and x.shape[1] == 2560
            and x.is_contiguous()
            and self.min_tokens <= x.shape[0]
            <= self.buffer.num_max_tokens_per_rank * 8
        )

    def forward(self, x, topk_ids, topk_weights):
        tokens = x.shape[0]
        local_x, local_ids, local_weights = self.router._prepare_pre_tp_slice(
            x, topk_ids, topk_weights
        )
        # DeepEP uses int64 indices; MegaMoE pack reads int32 and stores int64.
        local_ids = local_ids.to(dtype=torch.int32).contiguous()
        local_weights = local_weights.contiguous()
        count = local_x.shape[0]
        output = torch.empty_like(local_x)
        self.backend.mega_moe_pre_dispatch_sm90(
            local_x, local_ids, local_weights,
            self.buffer.x, self.buffer.x_sf,
            self.buffer.topk_idx, self.buffer.topk_weights,
            count, 128, 1.0,
        )
        self.backend.fp8_mega_moe(
            output, self.l1, self.l2, self.buffer,
            recipe=(128, 128, 128), activation="swiglu",
            activation_clamp=None, fast_math=False,
        )
        if tokens not in self._logged_shapes:
            self._logged_shapes.add(tokens)
            logger.info(
                "[QwenSM90MegaMoE] forward rank=%d tokens=%d local_tokens=%d module=%s",
                dist.get_rank(), tokens, count, self.module_name,
            )
        return self.router._finalize_post_tp_gather(
            output, {"original_num_tokens": tokens}
        )
