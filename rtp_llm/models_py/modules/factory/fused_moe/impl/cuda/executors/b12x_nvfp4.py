"""Dimension-derived adapter for FlashInfer static b12x NVFP4 MoE."""

from dis import get_instructions
from functools import cache
from hashlib import sha256
import logging
from pathlib import Path
from types import FunctionType
from weakref import WeakValueDictionary

import torch

_WORKSPACE_POOLS = WeakValueDictionary()
_STATIC_BACKEND = "static"
logger = logging.getLogger(__name__)

# Reference sources from the SM120-tested FlashInfer 0.7.3 wheel. Static
# dispatch, workspace reset and -1 route handling depend on private code.
_DISPATCH_SHA256 = "67c3cb8482e0a6ee7e936e54a02f9f68635c36c7683c9e92dfd2794b049db602"
_STATIC_KERNEL_SHA256 = (
    "bd1b3cd2cfcc76c84a15c4198df6783b422090c8202c11ffed2312f61ef6209d"
)
_SCALE_UTILS_SHA256 = "df35130f8c8c867a440d7ff3392ac23b19c8628429a7b65177c80bd0410c75cf"


@cache
def _matches_static_sources(dispatch_path, scale_utils_path):
    dispatch = Path(dispatch_path)
    return all(
        sha256(path.read_bytes()).hexdigest() == expected
        for path, expected in (
            (dispatch, _DISPATCH_SHA256),
            (dispatch.with_name("moe_static_kernel.py"), _STATIC_KERNEL_SHA256),
            (Path(scale_utils_path), _SCALE_UTILS_SHA256),
        )
    )


class _WorkspacePool:
    def __init__(self, model_config):
        self.model_config = model_config
        self.states = {}


class B12xNvfp4MoE:
    """Adapter for the static NVFP4 ABI in FlashInfer 0.7.3 (7fa5c3b909a7)."""

    @staticmethod
    def _runtime():
        from flashinfer.cute_dsl import utils
        from flashinfer.fused_moe.cute_dsl.blackwell_sm12x import moe_dispatch

        if not _matches_static_sources(moe_dispatch.__file__, utils.__file__):
            raise RuntimeError(
                "b12x NVFP4 requires the verified FlashInfer static sources "
                "from 0.7.3 (7fa5c3b909a7)."
            )
        launch = moe_dispatch.launch_sm120_static_moe
        # In the pinned dispatcher this function reads the flag directly in
        # both micro-selection branches. Its callees do not arbitrate either
        # branch again. Reject wrappers with a different binding contract.
        if not isinstance(launch, FunctionType) or not any(
            op.opname == "LOAD_GLOBAL" and op.argval == "_FORCED_BACKEND"
            for op in get_instructions(launch)
        ):
            raise RuntimeError("Unrecognized FlashInfer static MoE launcher")
        return utils.convert_sf_to_mma_layout, moe_dispatch

    @classmethod
    @cache
    def is_available(cls):
        try:
            cls._runtime()
        except (ImportError, RuntimeError, OSError, AttributeError, TypeError) as error:
            logger.info("SM120 b12x unavailable: %s", error)
            return False
        return True

    def __init__(self, w1, w2, scales, ep_rank, model_config, top_k):
        convert_sf_to_mma_layout, moe_dispatch = self._runtime()
        launch = moe_dispatch.launch_sm120_static_moe
        # This launcher uses the flag locally for both micro branches. Keep
        # the override private so other FlashInfer callers keep their dispatch.
        self.launch = FunctionType(
            launch.__code__,
            dict(launch.__globals__, _FORCED_BACKEND=_STATIC_BACKEND),
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
        # Each static launch restores its scratch on the worker's serialized
        # Decode stream. Layers and successive graph replays can reuse a bucket.
        self.states = pool.states

        def mma_scale(scale, rows, cols):
            # FlashInfer 7fa5c3b909a7, cute_dsl/utils.py:
            # convert_sf_to_mma_layout consumes already-swizzled block16
            # storage and returns a strided view over those bytes.
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
                backend=_STATIC_BACKEND,
            )
            self.states[rows] = state
        local_ids = torch.empty(
            (rows, self.top_k), device=self.device, dtype=torch.int32
        )
        # A rank can own none of a token's routes. Its contribution starts
        # at zero even if the external launcher elides that output row.
        output = torch.zeros_like(x)
        map_local_routes[(triton.cdiv(rows * self.top_k, 128),)](
            ids,
            local_ids,
            rows * self.top_k,
            self.ep_rank * self.num_experts,
            self.num_experts,
            128,
        )
        # The verified static finalize writes every output row, including
        # zeros for rows whose route IDs are all -1.
        self.launch(
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
        return output
