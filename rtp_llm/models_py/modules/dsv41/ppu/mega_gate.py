# SPDX-License-Identifier: Apache-2.0
"""Opt-in PPU DeepGEMM MegaGate adapters with an explicit finite-domain contract.

``try_mega_gate`` retains the original eager Prefill experiment API.
``PPUMegaGate`` adds model-owned dispatch for Prefill, Decode and Verify,
including warmed Decode/Verify graph capture. Both default to declining.
Metadata admission does not establish numerical equivalence: the caller must
explicitly accept the finite-domain contract before selecting this path.
Native sqrtsoftplus uses sum + 1e-20, whereas the existing router uses 1e-12;
near ties, underflow and output slot order still require device qualification.
The selected native provider does not uniformly reject nonfinite logits,
bias or projection overflow and does not promise the old router's uniform
fallback. The finite-domain opt-in is a caller precondition, not a validation
result. This adapter adds no host finite scan or device validation kernel.

The native ABI follows sglang-for-sail's ``bf16_mega_gate`` wrapper. Logical
expert IDs leave this adapter; the existing MoE engine applies EPLB once.
No native failure is retried through the old router. Capture owns its output
and native scratch allocations; the model adapter caches no tensor buffers
that could be shared accidentally by independent graphs.
"""

import math
from functools import lru_cache
from typing import Callable, Optional, Tuple

import torch


HIDDEN_SIZE = 5120
NUM_EXPERTS = 384
TOP_K = 6
COMM_SLOTS = 8
MAX_PREFILL_ROWS = 16384
MAX_DECODE_ROWS = 2048
_FP32_MAX = 3.4028234663852886e38


def _hardware_supported(hidden: torch.Tensor, *, allow_capture: bool = False) -> bool:
    if not hidden.is_cuda:
        return False
    from rtp_llm.models_py.modules.dsv41.ppu._quant import is_ppu

    if not is_ppu():
        return False
    with torch.cuda.device(hidden.device):
        return (
            torch.cuda.get_device_capability(hidden.device) == (8, 9)
            and (allow_capture or not torch.cuda.is_current_stream_capturing())
        )


def _dense(value, shape, dtype, device) -> bool:
    return (
        isinstance(value, torch.Tensor)
        and tuple(value.shape) == shape
        and value.dtype == dtype
        and value.device == device
        and value.layout == torch.strided
        and value.is_contiguous()
        and not value.is_neg()
        and not value.is_conj()
        and not value.requires_grad
    )


def eligibility_reason(
    hidden,
    router_weight,
    router_bias,
    *,
    top_k=TOP_K,
    routed_scaling_factor=1.5,
    image_rows=None,
    router_bias_vl=None,
    graph_routes=None,
    ep_rank=0,
    finite_domain_qualified=False,
    allow_capture=False,
):
    """Return a metadata exclusion reason; this never inspects tensor values."""
    if finite_domain_qualified is not True:
        return "unqualified_numerics"
    if torch.is_grad_enabled():
        return "autograd"
    if (
        not isinstance(hidden, torch.Tensor)
        or hidden.ndim != 2
        or hidden.shape[1] != HIDDEN_SIZE
        or not _dense(hidden, tuple(hidden.shape), torch.bfloat16, hidden.device)
    ):
        return "hidden"
    rows, device = hidden.shape[0], hidden.device
    if not _dense(
        router_weight, (NUM_EXPERTS, HIDDEN_SIZE), torch.bfloat16, device
    ):
        return "weight"
    if router_bias is not None and not _dense(
        router_bias, (NUM_EXPERTS,), torch.float32, device
    ):
        return "bias"
    if type(top_k) is not int or top_k != TOP_K:
        return "top_k"
    if (
        type(routed_scaling_factor) not in (int, float)
        or not math.isfinite(routed_scaling_factor)
        or routed_scaling_factor <= 0
        or routed_scaling_factor > _FP32_MAX
    ):
        return "route_scale"
    if type(ep_rank) is not int or not 0 <= ep_rank < 8:
        return "ep_rank"
    if (image_rows is None) != (router_bias_vl is None):
        return "visual_pair"
    if image_rows is not None and (
        not _dense(image_rows, (rows,), torch.bool, device)
        or not _dense(router_bias_vl, (NUM_EXPERTS,), torch.float32, device)
    ):
        return "visual"
    if graph_routes is not None:
        if not isinstance(graph_routes, tuple) or len(graph_routes) != 2:
            return "routes"
        valid, count = graph_routes
        if (
            not _dense(valid, (rows,), torch.bool, device)
            or not isinstance(count, torch.Tensor)
            or count.numel() != 1
            or not _dense(count, tuple(count.shape), torch.int64, device)
        ):
            return "routes"
    if not _hardware_supported(hidden, allow_capture=allow_capture):
        return "hardware_or_capture"
    return None


def _native_gate() -> Optional[Callable]:
    try:
        import deep_gemm
    except (ImportError, OSError):
        deep_gemm = None
    operation = getattr(deep_gemm, "bf16_mega_gate", None)
    if callable(operation):
        return operation
    # The separately packaged provider supports the same finite-input ABI
    # when the older dense/MegaMoE package does not export MegaGate.
    try:
        import rtp_ppu_mega_gate_native
    except (ImportError, OSError):
        return None
    operation = getattr(rtp_ppu_mega_gate_native, "bf16_mega_gate", None)
    return operation if callable(operation) else None


@lru_cache(maxsize=1)
def _pack_kernel():
    # Keep CPU adapter contracts independent of the PPU Triton installation.
    import triton
    import triton.language as tl

    @triton.jit
    def pack_dispatch(
        native_ids,
        native_weights,
        valid_mask,
        packed_ids,
        packed_weights,
        routed_count,
        M,
        COUNT_ROWS: tl.constexpr,
        BLOCK_ROWS: tl.constexpr,
    ):
        pid = tl.program_id(0)
        rows = pid * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
        slots = tl.arange(0, 8)
        real = tl.load(valid_mask + rows, rows < M, other=False)
        live = (rows[:, None] < M) & real[:, None] & (slots[None, :] < 6)
        ids = tl.load(
            native_ids + rows[:, None] * 6 + slots[None, :], live, other=-1
        )
        weights = tl.load(
            native_weights + rows[:, None] * 6 + slots[None, :], live, other=0.0
        )
        destination = rows[:, None] * 8 + slots[None, :]
        tl.store(packed_ids + destination, ids, rows[:, None] < M)
        tl.store(packed_weights + destination, weights, rows[:, None] < M)
        if pid == 0:
            # As in the existing fused router, the native contract gives every
            # valid row six nonnegative IDs. Counting valid rows therefore
            # equals the previous any(ids >= 0) reduction. COUNT_ROWS only pads
            # this reduction; M and all pack tiles retain runtime row bounds.
            count_rows = tl.arange(0, COUNT_ROWS)
            count_valid = tl.load(
                valid_mask + count_rows, count_rows < M, other=False
            )
            tl.store(routed_count, tl.sum(count_valid.to(tl.int32), 0))

    return triton, pack_dispatch


def _pack_dispatch_triton(ids, weights, valid, count):
    triton, operation = _pack_kernel()
    rows = ids.shape[0]
    packed_ids = ids.new_empty((rows, COMM_SLOTS))
    packed_weights = weights.new_empty((rows, COMM_SLOTS))
    operation[(triton.cdiv(rows, 128),)](
        ids, weights, valid, packed_ids, packed_weights, count, rows,
        COUNT_ROWS=triton.next_power_of_2(rows), BLOCK_ROWS=128, num_warps=4,
    )
    return packed_ids, packed_weights


def _use_triton_pack(ids):
    return ids.is_cuda


def _pack_dispatch(ids, weights, valid, count):
    if _use_triton_pack(ids):
        return _pack_dispatch_triton(ids, weights, valid, count)
    # CPU native doubles retain the independent portable adapter contract.
    rows = ids.shape[0]
    ids = torch.cat((ids, ids.new_full((rows, COMM_SLOTS - TOP_K), -1)), dim=1)
    weights = torch.cat(
        (weights, weights.new_zeros((rows, COMM_SLOTS - TOP_K))), dim=1
    )
    ids.masked_fill_(~valid[:, None], -1)
    weights.masked_fill_(~valid[:, None], 0)
    count.copy_((ids >= 0).any(-1).sum())
    return ids, weights


def _execute_gate(
    hidden,
    router_weight,
    router_bias,
    *,
    top_k,
    routed_scaling_factor,
    image_rows,
    router_bias_vl,
    graph_routes,
    ep_rank,
    operation,
    valid_mask=None,
):
    """Dispatch after admission; errors propagate, including device finite checks."""
    rows = hidden.shape[0]
    if rows == 0:
        slots = COMM_SLOTS if graph_routes is not None else top_k
        if graph_routes is not None:
            graph_routes[1].zero_()
        return (
            hidden.new_empty((0, slots), dtype=torch.int64),
            hidden.new_empty((0, slots), dtype=torch.float32),
        )
    ids = torch.empty((rows, top_k), dtype=torch.int64, device=hidden.device)
    weights = torch.empty_like(ids, dtype=torch.float32)
    valid = valid_mask if graph_routes is None else graph_routes[0]
    operation(
        hidden,
        router_weight,
        top_k,
        use_shared_as_routed=False,
        num_shared_experts=0,
        routed_scaling_factor=float(routed_scaling_factor),
        ep_rank=ep_rank,
        scoring_func="sqrtsoftplus",
        mask=valid,
        bias=router_bias,
        image_bias=router_bias_vl,
        image_token_mask=image_rows,
        out=(ids, weights),
    )
    if graph_routes is not None:
        ids, weights = _pack_dispatch(ids, weights, valid, graph_routes[1])
    return ids, weights


def try_mega_gate(
    hidden: torch.Tensor,
    router_weight: torch.Tensor,
    router_bias: Optional[torch.Tensor],
    *,
    top_k: int = TOP_K,
    routed_scaling_factor: float = 1.5,
    image_rows: Optional[torch.Tensor] = None,
    router_bias_vl: Optional[torch.Tensor] = None,
    graph_routes: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ep_rank: int = 0,
    finite_domain_qualified: bool = False,
    native: Optional[Callable] = None,
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """Return owned logical ``(ids, weights)`` or decline before native launch.

    ``finite_domain_qualified=True`` is an explicit experiment contract, not a
    runtime assertion about tensor contents. It must not be supplied by an
    automatic production path before numerical and input-domain qualification.
    ``native`` permits test/candidate injection without a persistent selector.
    Once dispatch starts, errors propagate and no possibly-written output is
    retried. ``graph_routes`` only borrows eager Prefill valid-row metadata;
    CUDA graph capture itself is outside this adapter's admitted domain.
    """
    reason = eligibility_reason(
        hidden,
        router_weight,
        router_bias,
        top_k=top_k,
        routed_scaling_factor=routed_scaling_factor,
        image_rows=image_rows,
        router_bias_vl=router_bias_vl,
        graph_routes=graph_routes,
        ep_rank=ep_rank,
        finite_domain_qualified=finite_domain_qualified,
    )
    if reason is not None:
        return None
    # Preserve the original empty-input contract: no provider import is needed.
    operation = None if hidden.shape[0] == 0 else (
        _native_gate() if native is None else native
    )
    if hidden.shape[0] and not callable(operation):
        return None
    return _execute_gate(
        hidden,
        router_weight,
        router_bias,
        top_k=top_k,
        routed_scaling_factor=routed_scaling_factor,
        image_rows=image_rows,
        router_bias_vl=router_bias_vl,
        graph_routes=graph_routes,
        ep_rank=ep_rank,
        operation=operation,
    )


class PPUMegaGate:
    """Model-owned optional router; ``None`` means no native dispatch occurred.

    Construct once per gate with explicit ``enabled`` and
    ``finite_domain_qualified`` flags, both False by default. The latter accepts
    the native finite-domain/epsilon semantics; it is not a numerical PASS.
    Forward returns logical ``(ids int64, weights float32)``. With graph_routes
    it also pads 6 routes to 8, applies the live mask and writes the row count.
    A separate valid_mask admits non-direct callers without changing their
    six-slot output or moving their existing output masking/packing.

    Graph warmup must call ``forward(..., is_graph=True)`` outside capture with
    the same phase, row bucket, optional operands and stream as capture. This
    prepares both native JIT and the pack kernel. During capture a cold shape
    declines before importing/building anything. Inputs, masks and count remain
    caller-owned live graph operands; outputs/scratch belong to the graph pool.
    No data-dependent host branch, finite scan, synchronization or fallback
    GEMM runs in this adapter. Native checks catch nonfinite/overflow failures;
    they deliberately do not emulate the original router's recovery policy.

    All new runtime paths are unvalidated until the deferred device/Graph and
    model tests run. Prefill capture and speculative proposal/top-3 are outside
    the admitted domain; the existing router keeps those calls.
    """

    def __init__(self, *, enabled=False, finite_domain_qualified=False):
        if type(enabled) is not bool or type(finite_domain_qualified) is not bool:
            raise TypeError("MegaGate flags must be explicit booleans")
        self.enabled = enabled
        self.finite_domain_qualified = finite_domain_qualified
        self._operation = None
        self._warmed = set()

    def forward(
        self,
        hidden,
        router_weight,
        router_bias,
        *,
        phase,
        is_graph=False,
        top_k=TOP_K,
        routed_scaling_factor=1.5,
        image_rows=None,
        router_bias_vl=None,
        graph_routes=None,
        ep_rank=0,
        valid_mask=None,
    ):
        if not self.enabled or not self.finite_domain_qualified:
            return None
        if phase not in ("prefill", "decode", "verify") or type(is_graph) is not bool:
            return None
        if is_graph and phase == "prefill":
            return None
        if torch.are_deterministic_algorithms_enabled():
            return None
        reason = eligibility_reason(
            hidden,
            router_weight,
            router_bias,
            top_k=top_k,
            routed_scaling_factor=routed_scaling_factor,
            image_rows=image_rows,
            router_bias_vl=router_bias_vl,
            graph_routes=graph_routes,
            ep_rank=ep_rank,
            finite_domain_qualified=self.finite_domain_qualified,
            allow_capture=is_graph,
        )
        if reason is not None:
            return None
        rows = hidden.shape[0]
        if valid_mask is not None and (
            graph_routes is not None
            or not _dense(valid_mask, (rows,), torch.bool, hidden.device)
        ):
            return None
        if rows > (MAX_PREFILL_ROWS if phase == "prefill" else MAX_DECODE_ROWS):
            return None
        if hidden.data_ptr() % 16 or router_weight.data_ptr() % 16:
            return None
        with torch.cuda.device(hidden.device):
            capturing = torch.cuda.is_current_stream_capturing()
            stream = torch.cuda.current_stream().cuda_stream
        signature = (
            hidden.device, phase, rows, top_k, router_bias is not None,
            image_rows is not None, graph_routes is not None,
            valid_mask is not None, stream,
        )
        if capturing and signature not in self._warmed:
            return None
        operation = self._operation
        if operation is None:
            # A capture can reach here only after a successful matching warmup.
            # A missing provider keeps the existing router, without a partial
            # write to its count or route buffers.
            if capturing:
                return None
            operation = _native_gate()
            if not callable(operation):
                return None
            self._operation = operation
        result = _execute_gate(
            hidden,
            router_weight,
            router_bias,
            top_k=top_k,
            routed_scaling_factor=routed_scaling_factor,
            image_rows=image_rows,
            router_bias_vl=router_bias_vl,
            graph_routes=graph_routes,
            ep_rank=ep_rank,
            operation=operation,
            valid_mask=valid_mask,
        )
        if is_graph and not capturing:
            self._warmed.add(signature)
        return result
