"""Dimension-derived adapter for FlashInfer static b12x NVFP4 MoE."""

from functools import cache
from hashlib import sha256
import logging
from pathlib import Path
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
# Official PyPI wheel sha256 548a3ea1f9057025ace0a21b298824f8de049eb4b6c5c458f763309509e2ae40.
# Cover the dispatcher callees, all CuTe DSL helpers and the JIT build code.
_STATIC_SOURCES_SHA256 = (
    "14f7183343168b25899d866914710f7502e9c2ad30196c142e2449c2db738f5b"
)


@cache
def _matches_static_sources(dispatch_path, scale_utils_path):
    dispatch = Path(dispatch_path)
    if not all(
        sha256(path.read_bytes()).hexdigest() == expected
        for path, expected in (
            (dispatch, _DISPATCH_SHA256),
            (dispatch.with_name("moe_static_kernel.py"), _STATIC_KERNEL_SHA256),
            (Path(scale_utils_path), _SCALE_UTILS_SHA256),
        )
    ):
        return False
    root = dispatch.parents[3]
    paths = []
    for directory in ("fused_moe/cute_dsl/blackwell_sm12x", "cute_dsl", "jit"):
        paths.extend((root / directory).rglob("*.py"))
    paths.extend(
        root / name
        for name in (
            "utils.py",
            "api_logging.py",
            "artifacts.py",
            "compilation_context.py",
        )
    )
    digest = sha256()
    for path in sorted(paths):
        digest.update(
            ("flashinfer/" + path.relative_to(root).as_posix()).encode() + b"\0"
        )
        digest.update(sha256(path.read_bytes()).digest())
    return digest.hexdigest() == _STATIC_SOURCES_SHA256


def _launch_static(
    dispatch,
    *,
    workspace,
    weights,
    a,
    topk_ids,
    topk_weights,
    input_gs,
    down_input_scale,
    scatter_output,
    num_experts,
    num_tokens,
    k,
    n,
    top_k,
    quant_mode="nvfp4",
    fast_math=False,
    input_scales_are_reciprocal=True
):
    """Call the pinned static compiler directly, with no backend arbitration.

    This is the NVFP4/SiLU static ABI from FlashInfer 0.7.3. In particular it
    never calls launch_sm120_static_moe, which can also select micro kernels.
    Source hashes above cover the private compiler, schedule helpers and ABI.
    """
    if quant_mode != "nvfp4":
        raise ValueError("b12x static adapter requires NVFP4")
    static_n = weights.static_intermediate_size or weights.intermediate_size or n
    static_n = max(n, dispatch._align_up(int(static_n), dispatch._LEVEL_TILE_N))
    pairs = num_tokens * top_k
    base_mac = min(dispatch.get_max_active_clusters(1), dispatch.get_num_sm(a.device))
    tuned = dispatch._lookup_mac_ladder(dispatch._STATIC_MAC_LADDER, pairs)
    mac = min(tuned or base_mac, base_mac)
    if pairs < 40:
        mac = min(mac, 64)
    merged = top_k > 1 and dispatch._static_merged_groups(static_n, pairs)
    compiled, _ = dispatch._get_static_kernel(
        int(workspace.row_counts.shape[0]),
        num_experts,
        num_tokens,
        k,
        static_n,
        top_k,
        int(workspace.token_map.shape[1]),
        topk_ids_dtype=torch.int32,
        input_scales_are_reciprocal=input_scales_are_reciprocal,
        fast_math=fast_math,
        mac_override=mac,
        activation="silu",
        activation_precision="fp4",
        quant_mode="nvfp4",
        weight_n=weights.static_intermediate_size or weights.intermediate_size,
        route_rows=workspace.max_rows,
        merged_groups=merged,
        deferred_init=dispatch._static_deferred_init(pairs),
        source_scales=bool(weights.source_scales),
    )
    route_scratch = workspace.route_output_scratch.view(-1)
    if merged:
        route_scratch = route_scratch[: workspace.max_rows * k]
    w13_sf, down_sf = weights._w13_sf_storage, weights._down_sf_storage
    if weights.static_w13_fp4 is None:
        weights.ensure_legacy()
        w13, down = weights.w13_fp4, weights.down_fp4
    else:
        w13, down = weights.static_w13_fp4, weights.static_down_fp4
        if weights.static_w13_sf_storage is not None:
            w13_sf, down_sf = (
                weights.static_w13_sf_storage,
                weights.static_down_sf_storage,
            )
    compiled(
        a,
        topk_ids.view(-1).to(torch.int32),
        topk_weights.view(-1).to(torch.float32),
        workspace.packed_a_view,
        workspace.packed_input_scale.data_ptr(),
        workspace.packed_a_flat,
        route_scratch,
        workspace.scale_flat,
        workspace.barrier_count,
        workspace.barrier_epoch,
        workspace.route_state,
        w13,
        w13_sf.data_ptr(),
        down,
        down_sf.data_ptr(),
        workspace.row_counts,
        workspace.active_expert_count,
        workspace.weight_expert_ids,
        workspace.global_to_local_expert,
        workspace.virt_route_scratch,
        input_gs,
        weights.w1_alpha,
        weights.w2_alpha,
        down_input_scale,
        scatter_output,
        workspace.token_map,
        workspace.token_weights,
    )
    return scatter_output


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
        return utils.convert_sf_to_mma_layout, moe_dispatch

    @classmethod
    @cache
    def is_available(cls):
        try:
            cls._runtime()
        except (ImportError, RuntimeError, OSError, AttributeError, TypeError) as error:
            logger.info("SM120 b12x unavailable: %s", error)
            return False
        from importlib.metadata import PackageNotFoundError, version

        try:
            runtime_version = version("flashinfer-python")
        except PackageNotFoundError:
            # The source digest is the admission check. Some source installs
            # do not expose wheel metadata.
            runtime_version = "unknown"

        logger.info(
            "SM120 b12x verified: FlashInfer=%s static_sources=%s",
            runtime_version,
            _STATIC_SOURCES_SHA256,
        )
        return True

    def __init__(self, w1, w2, scales, ep_rank, model_config, top_k):
        convert_sf_to_mma_layout, moe_dispatch = self._runtime()
        self.dispatch = moe_dispatch
        self.launch = lambda **kwargs: _launch_static(moe_dispatch, **kwargs)
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
            # https://github.com/flashinfer-ai/flashinfer/blob/7fa5c3b909a7/flashinfer/cute_dsl/utils.py#L367
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
