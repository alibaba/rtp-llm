"""SM120 Qwen decode with vLLM's NVFP4 weight-only Marlin kernel.

Weight/scale conversion follows vLLM v0.31.0 marlin_utils_fp4.py (Apache-2.0).
"""

import functools
import logging
import math
from pathlib import Path

import torch
import triton
import triton.language as tl

from rtp_llm.models_py.modules.factory.fused_moe.defs.fused_moe import (
    CombineForwardPayload,
    FusedMoeExpertExecutor,
)


@functools.lru_cache(maxsize=1)
def load_library():
    from torch.utils.cpp_extension import load

    root = Path(__file__).resolve().parent / "marlin_kernels"
    kernels = root / "libtorch_stable/moe/marlin_moe_wna16"
    sources = [
        root / "schema.cpp",
        kernels / "ops.cu",
        kernels / "sm80_kernel_bfloat16_fe2m1f_bfloat16.cu",
        root / "libtorch_stable/quantization/marlin/gptq_marlin_repack.cu",
    ]
    return load(
        name="rtp_qwen4_nvfp4_marlin",
        sources=[str(p) for p in sources],
        extra_include_paths=[str(root)],
        extra_cflags=["-O3", "-std=c++17", "-DUSE_CUDA"],
        extra_cuda_cflags=[
            "-O3",
            "-std=c++17",
            "-DUSE_CUDA",
            "--expt-relaxed-constexpr",
            "-static-global-template-stub=false",
            "-gencode=arch=compute_120,code=sm_120",
        ],
        is_python_module=False,
    )


@functools.lru_cache(maxsize=1)
def _library_available():
    try:
        load_library()
        return True
    except (RuntimeError, OSError) as error:
        logging.getLogger(__name__).warning(
            "Qwen decode Marlin is unavailable; retaining CUTLASS: %s", error
        )
        return False


def prepare_weight(q, sf, global_sf):
    experts, n, packed_k = q.shape
    k = packed_k * 2
    assert n % 64 == 0 and k % 128 == 0 and sf.shape == (experts, n, k // 16)
    out = torch.empty((experts, k // 16, n * 2), device=q.device, dtype=torch.int32)
    for expert in range(experts):
        out[expert] = torch.ops.rtp_qwen4_marlin.gptq_marlin_repack(
            q[expert].view(torch.int32).T.contiguous(), k, n, 4, False
        )
    perm = [i + 8 * j for i in range(8) for j in range(8)]
    scales = sf.to(torch.bfloat16).transpose(1, 2).contiguous()
    max_sf = scales.float().max().item()
    factor = (
        2.0 ** int(math.floor(math.log2(448.0 / max_sf))) if 0 < max_sf < 448 else 1.0
    )
    scales = scales.reshape(-1, 64)[:, perm].reshape(experts, k // 16, n)
    scales = scales.half().reshape(-1, 4)[:, [0, 2, 1, 3]].reshape(experts * k // 16, n)
    scales = (scales.float() * factor).half() * 128
    scales[scales < 2] = 0
    scales = (
        (scales.view(torch.int16) << 1).view(torch.float8_e4m3fn)[:, 1::2].contiguous()
    )
    return (
        out,
        scales.reshape(experts, k // 16, n),
        global_sf.float() * (2.0**119 / factor),
    )


@triton.jit
def _count(
    IDS,
    OFFSETS,
    SORTED,
    EXPERT_IDS,
    PADDED,
    ROUTES: tl.constexpr,
    EXPERTS: tl.constexpr,
    FIRST: tl.constexpr,
    RTILE: tl.constexpr,
    SLOTS: tl.constexpr,
    STILE: tl.constexpr,
    BLOCKS: tl.constexpr,
    BTILE: tl.constexpr,
):
    r = tl.arange(0, RTILE)
    ids = tl.load(IDS + r, r < ROUTES, other=-1) - FIRST
    valid = (r < ROUTES) & (ids >= 0) & (ids < EXPERTS)
    counts = tl.histogram(ids, EXPERTS, mask=valid)
    padded = tl.cdiv(counts, 8) * 8
    inclusive = tl.cumsum(padded)
    e = tl.arange(0, EXPERTS)
    tl.store(OFFSETS + e, inclusive - padded)
    tl.store(PADDED, tl.sum(padded))
    s = tl.arange(0, STILE)
    tl.store(SORTED + s, ROUTES, s < SLOTS)
    b = tl.arange(0, BTILE)
    tl.store(EXPERT_IDS + b, 0, b < BLOCKS)


@triton.jit
def _scatter(
    IDS,
    OFFSETS,
    SORTED,
    EXPERT_IDS,
    ROUTES: tl.constexpr,
    EXPERTS: tl.constexpr,
    FIRST: tl.constexpr,
    RTILE: tl.constexpr,
):
    route = tl.program_id(0)
    expert = tl.load(IDS + route)
    local = expert - FIRST
    if local >= 0 and local < EXPERTS:
        r = tl.arange(0, RTILE)
        earlier = tl.load(IDS + r, r < route, other=-1)
        ordinal = tl.sum(((r < route) & (earlier == expert)).to(tl.int32))
        dst = tl.load(OFFSETS + local) + ordinal
        tl.store(SORTED + dst, route)
        if ordinal % 8 == 0:
            tl.store(EXPERT_IDS + dst // 8, local)


def align(ids, experts, first):
    routes = ids.numel()
    active = min(routes, experts)
    blocks = active + (routes - active) // 8
    slots = blocks * 8
    offsets = torch.empty(experts, device=ids.device, dtype=torch.int32)
    sorted_ids = torch.empty(slots, device=ids.device, dtype=torch.int32)
    expert_ids = torch.empty(blocks, device=ids.device, dtype=torch.int32)
    padded = torch.empty(1, device=ids.device, dtype=torch.int32)
    _count[(1,)](
        ids,
        offsets,
        sorted_ids,
        expert_ids,
        padded,
        ROUTES=routes,
        EXPERTS=experts,
        FIRST=first,
        RTILE=triton.next_power_of_2(routes),
        SLOTS=slots,
        STILE=triton.next_power_of_2(slots),
        BLOCKS=blocks,
        BTILE=triton.next_power_of_2(blocks),
        num_warps=4,
    )
    if routes <= 320:
        _scatter[(routes,)](
            ids,
            offsets,
            sorted_ids,
            expert_ids,
            ROUTES=routes,
            EXPERTS=experts,
            FIRST=first,
            RTILE=triton.next_power_of_2(routes),
            num_warps=4,
        )
    else:
        _scatter_by_expert[(experts,)](
            ids,
            offsets,
            sorted_ids,
            expert_ids,
            ROUTES=routes,
            FIRST=first,
            RTILE=triton.next_power_of_2(routes),
            num_warps=8,
        )
    return sorted_ids, expert_ids, padded


@triton.jit
def _scatter_by_expert(
    IDS,
    OFFSETS,
    SORTED,
    EXPERT_IDS,
    ROUTES: tl.constexpr,
    FIRST: tl.constexpr,
    RTILE: tl.constexpr,
):
    local = tl.program_id(0)
    r = tl.arange(0, RTILE)
    ids = tl.load(IDS + r, r < ROUTES, other=-1)
    valid = (r < ROUTES) & (ids == local + FIRST)
    ordinal = tl.cumsum(valid.to(tl.int32), 0) - 1
    dst = tl.load(OFFSETS + local) + ordinal
    tl.store(SORTED + dst, r, valid)
    tl.store(EXPERT_IDS + dst // 8, local, valid & (ordinal % 8 == 0))


@triton.jit
def _activate(
    X,
    Y,
    IDS,
    N: tl.constexpr,
    FIRST: tl.constexpr,
    EXPERTS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    route = tl.program_id(0)
    col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    expert = tl.load(IDS + route)
    valid = (expert >= FIRST) & (expert < FIRST + EXPERTS)
    gate = tl.load(X + route * (2 * N) + col, (col < N) & valid, other=0).to(tl.float32)
    up = tl.load(X + route * (2 * N) + N + col, (col < N) & valid, other=0).to(
        tl.float32
    )
    silu = (gate / (1.0 + tl.exp(-gate))).to(tl.bfloat16).to(tl.float32)
    tl.store(Y + route * N + col, silu * up, col < N)


@triton.jit
def _sum(
    X,
    OUT,
    IDS,
    K: tl.constexpr,
    TOPK: tl.constexpr,
    FIRST: tl.constexpr,
    EXPERTS: tl.constexpr,
    BLOCK: tl.constexpr,
    KTILE: tl.constexpr,
):
    row = tl.program_id(0)
    col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    t = tl.arange(0, KTILE)
    ids = tl.load(IDS + row * TOPK + t, t < TOPK, other=-1)
    valid = (t < TOPK) & (ids >= FIRST) & (ids < FIRST + EXPERTS)
    values = tl.load(
        X + (row * TOPK + t[:, None]) * K + col[None, :],
        valid[:, None] & (col[None, :] < K),
        other=0,
    ).to(tl.float32)
    tl.store(OUT + row * K + col, tl.sum(values, 0), col < K)


class Nvfp4MarlinMoE:
    def __init__(self, q1, s1, g1, q2, s2, g2, ep_size, ep_rank=0):
        load_library()
        self.experts, both_n, packed_k = q1.shape
        self.n, self.k = both_n // 2, packed_k * 2
        self._one_row_sm120 = (self.k, self.n) == (
            2560,
            640,
        ) and torch.cuda.get_device_capability(q1.device) == (12, 0)
        self.first = self.experts * ep_rank
        self.w1, self.s1, self.g1 = prepare_weight(
            torch.cat((q1[:, self.n :], q1[:, : self.n]), 1),
            torch.cat((s1[:, self.n :], s1[:, : self.n]), 1),
            g1,
        )
        self.w2, self.s2, self.g2 = prepare_weight(q2, s2, g2)
        self.workspaces = {}

    def __call__(self, x, ids, weights):
        m, k = x.shape
        assert (
            0 < m
            and k == self.k
            and x.dtype == torch.bfloat16
            and ids.dtype == torch.int32
        )
        topk = ids.shape[1]
        stream = torch.cuda.current_stream(x.device).cuda_stream
        if stream not in self.workspaces:
            self.workspaces[stream] = torch.zeros(
                torch.cuda.get_device_properties(x.device).multi_processor_count * 4,
                device=x.device,
                dtype=torch.int32,
            )
        workspace = self.workspaces[stream]
        sorted_ids, expert_ids, padded = align(ids, self.experts, self.first)
        upgate = torch.empty((m * topk, self.n * 2), device=x.device, dtype=x.dtype)
        activated = torch.empty((m * topk, self.n), device=x.device, dtype=x.dtype)
        down = torch.empty((m * topk, k), device=x.device, dtype=x.dtype)
        out = torch.empty_like(x)
        for a, c, w, sf, glob, size_m, n, kk, kt, weighted in [
            (x, upgate, self.w1, self.s1, self.g1, m, 2 * self.n, k, topk, False),
            (activated, down, self.w2, self.s2, self.g2, m * topk, k, self.n, 1, True),
        ]:
            thread_config = (
                (128, 64, 2 if weighted else 1)
                if m == 1 and self._one_row_sm120
                else (-1, -1, -1)
            )
            torch.ops.rtp_qwen4_marlin.moe_wna16_marlin_gemm(
                a,
                c,
                w,
                None,
                sf,
                None,
                glob,
                None,
                workspace,
                sorted_ids,
                expert_ids,
                padded,
                weights,
                8,
                kt,
                weighted,
                562949953487106,
                size_m,
                n,
                kk,
                False,
                True,
                False,
                *thread_config,
            )
            if not weighted:
                _activate[(m * topk, triton.cdiv(self.n, 256))](
                    upgate,
                    activated,
                    ids,
                    N=self.n,
                    FIRST=self.first,
                    EXPERTS=self.experts,
                    BLOCK=256,
                )
        _sum[(m, triton.cdiv(k, 256))](
            down,
            out,
            ids,
            K=k,
            TOPK=topk,
            FIRST=self.first,
            EXPERTS=self.experts,
            BLOCK=256,
            KTILE=triton.next_power_of_2(topk),
            num_warps=4,
        )
        return out


def _linear_scales(swizzled, experts, rows, cols):
    return (
        swizzled.view(torch.float8_e4m3fn)
        .reshape(experts, rows // 128, cols // 4, 32, 4, 4)
        .permute(0, 1, 4, 3, 2, 5)
        .contiguous()
        .reshape(experts, rows, cols)
    )


def make_decode_marlin_executor(native, role, max_generate_batch_size):
    """Replace only the measured EP2 SM120 Qwen decode geometry.

    Repacking reuses the original weight/scale storages. Keeping both layouts
    for every layer would exceed device memory. Prefill and larger serving
    capacities retain the factory-selected executor.
    """
    if (
        str(role).upper().rsplit(".", 1)[-1] != "DECODE"
        or not 0 < max_generate_batch_size <= 8
        or type(native).__name__ != "CutlassNvfp4Executor"
        or torch.version.hip is not None
    ):
        return native

    from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.cutlass_nvfp4_executor import (
        CutlassNvfp4Executor,
    )
    from rtp_llm.utils.model_weight import W

    if not (
        isinstance(native, CutlassNvfp4Executor)
        and str(role).upper().rsplit(".", 1)[-1] == "DECODE"
        and 0 < max_generate_batch_size <= 8
        and native.ep_size == 2
        and native.tp_size == 1
        and native.w1.shape == (256, 1280, 160)
        and native.w2.shape == (256, 2560, 40)
        and native.w1.is_cuda
        and torch.version.hip is None
        and torch.cuda.get_device_capability(native.w1.device) == (12, 0)
        and not native.config.model_config.eplb_config.enable_eplb()
        and all(
            native.weights[name].is_contiguous()
            and native.weights[name].data_ptr() == value.data_ptr()
            for name, value in (
                (W.moe_w1, native.w1),
                (W.moe_w2, native.w2),
                (W.moe_s1, native.quant_scales[1]),
                (W.moe_s2, native.quant_scales[4]),
            )
        )
    ):
        return native
    return Qwen4DecodeMarlinExecutor(native) if _library_available() else native


class Qwen4DecodeMarlinExecutor(FusedMoeExpertExecutor):
    @classmethod
    def executor_type(cls):
        from rtp_llm.models_py.modules.factory.fused_moe.defs.type import ExecutorType

        return ExecutorType.CUTLASS_NVFP4

    @property
    def topk_ids_dtype(self):
        return torch.int32

    @property
    def local_num_experts(self):
        return self.moe.experts

    def __init__(self, native):
        from rtp_llm.utils.model_weight import W

        super().__init__(native.config, native.quant_config, native.weights)
        self.ep_size, self.ep_rank = native.ep_size, native.ep_rank
        self.tp_size, self.tp_rank = native.tp_size, native.tp_rank
        q1, q2 = native.w1.view(torch.uint8), native.w2.view(torch.uint8)
        experts, both_n, packed_k = q1.shape
        n, k = both_n // 2, packed_k * 2
        owner = self.weights[W.moe_w1]
        identity = (
            self.ep_size,
            self.ep_rank,
            q2.data_ptr(),
            native.quant_scales[1].data_ptr(),
            native.quant_scales[4].data_ptr(),
        )
        prior = getattr(owner, "_qwen4_marlin_layout", None)
        if prior is not None:
            if prior[:5] != identity:
                raise ValueError("Changing packed Qwen MoE weights requires restart")
            moe = object.__new__(Nvfp4MarlinMoE)
            moe.experts, moe.n, moe.k = experts, n, k
            moe._one_row_sm120 = (k, n) == (
                2560,
                640,
            ) and torch.cuda.get_device_capability(q1.device) == (12, 0)
            moe.first = experts * self.ep_rank
            moe.w1 = q1.view(torch.int32).reshape(experts, k // 16, both_n * 2)
            moe.w2 = q2.view(torch.int32).reshape(experts, n // 16, k * 2)
            moe.s1 = (
                native.quant_scales[1]
                .view(torch.float8_e4m3fn)
                .reshape(experts, k // 16, both_n)
            )
            moe.s2 = (
                native.quant_scales[4]
                .view(torch.float8_e4m3fn)
                .reshape(experts, n // 16, k)
            )
            moe.g1, moe.g2 = prior[5:]
            moe.workspaces = {}
            self.moe = moe
            self._record_packed_consumers()
            return
        sf1 = _linear_scales(native.quant_scales[1], experts, both_n, k // 16)
        sf2 = _linear_scales(native.quant_scales[4], experts, k, n // 16)
        g1, g2 = native.weights[W.moe_w1_s2], native.weights[W.moe_w2_s2]
        for scale in (sf1, sf2, g1, g2):
            if not bool(torch.isfinite(scale.float()).all()) or bool(
                (scale.float() < 0).any()
            ):
                raise ValueError("Marlin NVFP4 requires finite nonnegative scales")
        moe = Nvfp4MarlinMoE(q1, sf1, g1, q2, sf2, g2, self.ep_size, self.ep_rank)
        destinations = (q1, q2, native.quant_scales[1], native.quant_scales[4])
        converted = (moe.w1, moe.w2, moe.s1, moe.s2)
        for target, source in zip(destinations, converted):
            if (
                target.numel() * target.element_size()
                != source.numel() * source.element_size()
            ):
                raise ValueError("Marlin repack must preserve weight storage size")
        packed = []
        for target, source in zip(destinations, converted):
            target.view(torch.uint8).reshape(-1).copy_(
                source.view(torch.uint8).reshape(-1)
            )
            packed.append(target.view(source.dtype).reshape(source.shape))
        moe.w1, moe.w2, moe.s1, moe.s2 = packed
        # Keep only tiny global scales on the source Tensor. Holding the MoE
        # here would create a cycle through its packed views and their owner.
        owner._qwen4_marlin_layout = (*identity, moe.g1, moe.g2)
        self.moe = moe
        self._record_packed_consumers()

    def _record_packed_consumers(self):
        from rtp_llm.models_py.modules.qwen4_exp.weight_views import WeightView
        from rtp_llm.utils.model_weight import W

        views = getattr(self.weights, "consumer_views", None)
        if views is None:
            return
        for name, compute in (
            (W.moe_w1, self.moe.w1),
            (W.moe_w2, self.moe.w2),
            (W.moe_s1, self.moe.s1),
            (W.moe_s2, self.moe.s2),
            (W.moe_w1_s2, self.moe.g1),
            (W.moe_w2_s2, self.moe.g2),
        ):
            owner = self.weights[name]
            views[name] = WeightView(name, owner, None, compute, True, owner)

    def execute(
        self,
        payload,
        activation,
        expert_map,
        a2_scale,
        apply_router_weight_on_input,
        extra_expert_args,
    ):
        if activation.lower() not in ("silu", "swiglu", "siglu"):
            raise ValueError("Marlin NVFP4 requires SwiGLU")
        if (
            apply_router_weight_on_input
            or payload.expert_ids_are_local
            or expert_map is not None
        ):
            raise ValueError(
                "Marlin NVFP4 requires global IDs and output router weighting"
            )
        x = payload.expert_x
        if x.numel() == 0:
            output = torch.empty_like(x)
        else:
            output = self.moe(
                x,
                payload.expert_topk_ids.to(torch.int32),
                payload.expert_topk_weights.to(torch.float32),
            )
        return CombineForwardPayload(fused_expert_output=output)
