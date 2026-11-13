"""Dimension-derived adapter for FlashInfer static b12x NVFP4 MoE."""

import inspect
from types import FunctionType
from weakref import WeakValueDictionary

import torch

_WORKSPACE_POOLS = WeakValueDictionary()


class _WorkspacePool:
    def __init__(self, model_config):
        self.model_config = model_config
        self.states = {}


class B12xNvfp4MoE:
    """Adapter for the FlashInfer b12x static NVFP4 ABI at 7fa5c3b909a7."""

    def __init__(self, w1, w2, scales, ep_rank, model_config, top_k):
        from flashinfer.cute_dsl.utils import convert_sf_to_mma_layout
        from flashinfer.fused_moe.cute_dsl.blackwell_sm12x import moe_dispatch

        launch = moe_dispatch.launch_sm120_static_moe
        # The pinned launcher selects static via a module flag. Bind it in a
        # private namespace so unrelated FlashInfer users keep their dispatch.
        if (
            "_FORCED_BACKEND" not in launch.__code__.co_names
            or "quant_mode" not in inspect.signature(launch).parameters
            or "branches"
            not in inspect.signature(moe_dispatch._get_weight_views).parameters
        ):
            raise RuntimeError(
                "b12x NVFP4 requires FlashInfer's pinned static NVFP4 ABI"
            )
        self.launch = FunctionType(
            launch.__code__,
            dict(launch.__globals__, _FORCED_BACKEND="static"),
            launch.__name__,
            launch.__defaults__,
            launch.__closure__,
        )
        self.launch.__kwdefaults__ = launch.__kwdefaults__
        self.dispatch = moe_dispatch
        self.device = w1.device
        self.ep_rank = ep_rank
        self.num_experts = w1.shape[0]
        self.hidden_size = w2.shape[1]
        self.intermediate_size = w2.shape[2] * 16
        self.top_k = top_k
        if w2.shape[0] != self.num_experts or tuple(w1.shape) != (
            self.num_experts,
            2 * self.intermediate_size,
            self.hidden_size // 16,
        ):
            raise ValueError(
                "b12x requires packed gated W1 [E,2N,K/16] and W2 [E,K,N/16]"
            )
        key = (
            id(model_config),
            w1.device.index,
            self.num_experts,
            self.hidden_size,
            self.intermediate_size,
            top_k,
        )
        pool = _WORKSPACE_POOLS.get(key)
        if pool is None:
            pool = _WorkspacePool(model_config)
            _WORKSPACE_POOLS[key] = pool
        self.workspace_pool = pool
        self.states = pool.states

        def mma_scale(scale, rows, cols):
            return convert_sf_to_mma_layout(
                scale.contiguous().view(torch.float8_e4m3fn),
                m=rows,
                k=cols,
                num_groups=self.num_experts,
                sf_vec_size=16,
            )

        self.weights = moe_dispatch._get_weight_views(
            w1_fp4=w1.view(torch.uint8),
            w1_blockscale=mma_scale(
                scales[1], 2 * self.intermediate_size, self.hidden_size
            ),
            w2_fp4=w2.view(torch.uint8),
            w2_blockscale=mma_scale(
                scales[4], self.hidden_size, self.intermediate_size
            ),
            w1_alphas=scales[2],
            w2_alphas=scales[5],
            n=self.intermediate_size,
            k=self.hidden_size,
            quant_mode="nvfp4",
            branches=2,
        )
        self.input_scale = scales[0].expand(self.num_experts).contiguous()
        self.down_scale = scales[3].expand(self.num_experts).contiguous()

    def __call__(self, x, ids, weights):
        import triton

        from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.nvfp4_routes import (
            map_local_routes,
        )

        rows = x.shape[0]
        state = self.states.get(rows)
        if state is None:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("Warm up b12x row buckets before CUDA Graph capture")
            state = self.dispatch.allocate_sm120_moe_workspace(
                state_E=self.num_experts,
                weight_E=self.num_experts,
                routed_rows=rows * self.top_k,
                k=self.hidden_size,
                n=self.intermediate_size,
                num_topk=self.top_k,
                device=self.device,
                quant_mode="nvfp4",
                backend="static",
            )
            self.states[rows] = state
        local_ids = torch.empty(
            (rows, self.top_k), device=self.device, dtype=torch.int32
        )
        output = torch.empty_like(x)
        map_local_routes[(triton.cdiv(rows * self.top_k, 128),)](
            ids,
            local_ids,
            rows * self.top_k,
            self.ep_rank * self.num_experts,
            self.num_experts,
            128,
        )
        return self.launch(
            workspace=state,
            weights=self.weights,
            a=x,
            topk_ids=local_ids,
            topk_weights=weights,
            input_gs=self.input_scale,
            down_input_scale=self.down_scale,
            scatter_output=output,
            num_experts=self.num_experts,
            num_tokens=rows,
            k=self.hidden_size,
            n=self.intermediate_size,
            top_k=self.top_k,
            quant_mode="nvfp4",
            fast_math=False,
            input_scales_are_reciprocal=True,
        )
