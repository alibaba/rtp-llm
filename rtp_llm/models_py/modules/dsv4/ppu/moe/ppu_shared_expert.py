"""PPU shared expert; the loader's merged FP8 weights remain unchanged."""

import torch
from rtp_llm.models_py.modules.dsv4._profiler import record_function_range
from rtp_llm.models_py.modules.factory.fused_moe.utils.fp8_fp4.shared_expert import (
    SharedExpertExecutor,
    W13SharedExpert,
)
from rtp_llm.models_py.modules.factory.linear.ppu_impl.fp8_linear import PpuFp8Linear
from torch import nn


class PpuSharedExpert(W13SharedExpert):
    def __init__(
        self,
        dim,
        inter_dim,
        expert_weights,
        swiglu_limit=0.0,
        *,
        sglang_moe=False,
        platform_provider,
    ):
        from rtp_llm.models_py.modules.dsv4.ppu.utils import _v4_fp8_linear

        torch.nn.Module.__init__(self)
        w13 = expert_weights["w13_w"]
        if w13.shape != (2 * inter_dim, dim):
            raise ValueError(
                f"PPU shared w13 shape {tuple(w13.shape)} != {(2 * inter_dim, dim)}"
            )
        self.w13 = _v4_fp8_linear(
            w13, expert_weights["w13_s"], platform_provider=platform_provider
        )
        self.w2 = _v4_fp8_linear(
            expert_weights["w2_w"],
            expert_weights["w2_s"],
            platform_provider=platform_provider,
        )
        self.swiglu_limit = swiglu_limit
        self.preserve_output_dtype = bool(sglang_moe)

    def forward(self, x, weights=None, *, input_quantized=None):
        if input_quantized is not None and (
            not self.preserve_output_dtype or weights is not None
        ):
            raise ValueError("Quantized shared inputs require the PPU BF16 contract")
        if not self.preserve_output_dtype or weights is not None:
            return super().forward(x, weights)
        return self.finish_projection(self.project(x, input_quantized=input_quantized))

    def project(self, x, *, input_quantized=None):
        """First shared GEMM, retaining its BF16 output boundary."""
        with record_function_range("dsv4.shared_expert.w13"):
            return (
                self._apply_layer(self.w13, x)
                if input_quantized is None
                else self.w13.forward_quantized(*input_quantized).view(
                    *x.shape[:-1], self.w13.n
                )
            )

    def finish_projection(self, gate_up):
        """Activation, quantization and second GEMM with unchanged arithmetic."""
        from rtp_llm.models_py.kernels.ppu.moe.ppu_shared_swiglu import (
            silu_mul_merged_bf16,
        )

        if getattr(self, "fuse_activation_quant", False):
            from rtp_llm.models_py.kernels.ppu.attention.ppu_decode_fused_quant import (
                supports_swiglu_fp8,
                swiglu_fp8,
            )

            if type(self.w2) is PpuFp8Linear and supports_swiglu_fp8(
                gate_up, self.w2.quantization
            ):
                with record_function_range("dsv4.shared_expert.silu_mul_quant"):
                    payload, scale = swiglu_fp8(
                        gate_up, self.swiglu_limit, quantization=self.w2.quantization
                    )
                with record_function_range("dsv4.shared_expert.w2"):
                    result = self.w2.forward_quantized(payload, scale)
                    return result.view(*gate_up.shape[:-1], self.w2.n)
        with record_function_range("dsv4.shared_expert.silu_mul"):
            hidden = silu_mul_merged_bf16(gate_up, self.swiglu_limit)
        with record_function_range("dsv4.shared_expert.w2"):
            return self._apply_layer(self.w2, hidden)


class _LocalMegaMoEGroup:
    """Single-rank group surface used by the replicated shared expert."""

    @staticmethod
    def size():
        return 1

    @staticmethod
    def rank():
        return 0

    @staticmethod
    def barrier():
        return None


class PpuTPMegaMoEFP8SharedResources:
    """One serial TP-local FP8 workspace shared by every model layer."""

    def __init__(self):
        self._key = None
        self._buffer = None
        self._group = None

    def bind(self, cfg, module):
        import torch.distributed as dist

        key = (cfg.max_tokens_per_rank, cfg.dim, cfg.moe_inter_dim)
        if self._buffer is not None:
            if self._key != key:
                raise RuntimeError(
                    "TP shared MegaMoE layers require one workspace layout"
                )
            return self._buffer
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("TP shared MegaMoE requires eager workspace creation")
        world = dist.get_world_size()
        rank = dist.get_rank()
        groups = [dist.new_group([member]) for member in range(world)]
        self._group = groups[rank]
        self._buffer = module.get_symm_buffer_for_mega_moe(
            self._group,
            8,
            cfg.max_tokens_per_rank,
            1,
            cfg.dim,
            cfg.moe_inter_dim,
            mma_type="fp8xfp8",
            quantization="blockwise",
        )
        self._key = key
        return self._buffer


class PpuMegaMoEFP8SharedExpert(nn.Module):
    """Route A shared expert with exact UE8M0/BF16 boundaries."""

    preserve_output_dtype = True
    routea_exact = True

    def __init__(
        self,
        cfg,
        expert_weights,
        swiglu_limit=0.0,
        *,
        module,
        resources=None,
        scale_is_prepared=False,
    ):
        super().__init__()
        w13, s13 = expert_weights["w13_w"], expert_weights["w13_s"]
        w2, s2 = expert_weights["w2_w"], expert_weights["w2_s"]
        expected = ((2 * cfg.moe_inter_dim, cfg.dim), (cfg.dim, cfg.moe_inter_dim))
        e8m0 = getattr(torch, "float8_e8m0fnu", None)
        for name, weight, scale, shape in (
            ("w13", w13, s13, expected[0]),
            ("w2", w2, s2, expected[1]),
        ):
            if (
                tuple(weight.shape) != shape
                or weight.dtype != torch.float8_e4m3fn
                or tuple(scale.shape) != (shape[0] // 128, shape[1] // 128)
                or e8m0 is None
                or scale.dtype != (torch.float32 if scale_is_prepared else e8m0)
                or scale.device != weight.device
            ):
                raise ValueError(
                    f"Route A shared {name} requires canonical V4 FP8 weights"
                )
        # Route A's scheduler requires eight EP1 expert slots.  Slot zero is
        # the real shared expert; the remaining slots are unreachable zeros.
        w13_8 = torch.cat(
            (w13.unsqueeze(0), torch.zeros_like(w13).unsqueeze(0).expand(7, -1, -1))
        ).contiguous()
        s13_8 = torch.cat(
            (
                s13.float().unsqueeze(0),
                torch.ones_like(s13, dtype=torch.float32)
                .unsqueeze(0)
                .expand(7, -1, -1),
            )
        ).contiguous()
        w2_8 = torch.cat(
            (w2.unsqueeze(0), torch.zeros_like(w2).unsqueeze(0).expand(7, -1, -1))
        ).contiguous()
        s2_8 = torch.cat(
            (
                s2.float().unsqueeze(0),
                torch.ones_like(s2, dtype=torch.float32).unsqueeze(0).expand(7, -1, -1),
            )
        ).contiguous()
        pair13, pair2 = module.transform_weights_for_fp8_mega_moe(
            (w13_8, s13_8), (w2_8, s2_8), quantization="blockwise"
        )
        self.register_buffer("_w13", pair13[0], persistent=False)
        self.register_buffer("_s13", pair13[1], persistent=False)
        self.register_buffer("_w2", pair2[0], persistent=False)
        self.register_buffer("_s2", pair2[1], persistent=False)
        self.register_buffer(
            "_route_ids",
            torch.zeros(
                (cfg.max_tokens_per_rank, 1), dtype=torch.int64, device=w13.device
            ),
            persistent=False,
        )
        self.register_buffer(
            "_route_weights",
            torch.ones(
                (cfg.max_tokens_per_rank, 1), dtype=torch.float32, device=w13.device
            ),
            persistent=False,
        )
        self._mega = module
        self._cfg = cfg
        self._resources = resources
        self._sym = (
            module.get_symm_buffer_for_mega_moe(
                _LocalMegaMoEGroup(),
                8,
                cfg.max_tokens_per_rank,
                1,
                cfg.dim,
                cfg.moe_inter_dim,
                mma_type="fp8xfp8",
                quantization="blockwise",
            )
            if resources is None
            else None
        )
        self.swiglu_limit = float(swiglu_limit)

    def forward(self, x, weights=None, *, input_quantized=None):
        if weights is not None or input_quantized is not None:
            raise ValueError("Route A shared expert consumes BF16 input directly")
        shape = x.shape
        flat = x.reshape(-1, shape[-1]).contiguous()
        tokens = flat.shape[0]
        if tokens > self._route_ids.shape[0]:
            raise ValueError("Route A shared input exceeds its fixed capacity")
        if self._sym is None:
            self._sym = self._resources.bind(self._cfg, self._mega)
        self._mega.mega_moe_pre_dispatch_ue8m0(
            flat,
            self._route_ids[:tokens],
            self._route_weights[:tokens],
            self._sym.x,
            self._sym.x_sf,
            self._sym.topk_idx,
            self._sym.topk_weights,
            tokens,
        )
        out = torch.empty_like(flat, dtype=torch.float32)
        self._mega.fp8_fp8_mega_moe_fp32(
            out,
            (self._w13, self._s13),
            (self._w2, self._s2),
            self._sym,
            activation_clamp=self.swiglu_limit if self.swiglu_limit > 0 else None,
            fast_math=True,
        )
        return out.view(*shape)


class PpuTPMegaMoEFP8SharedExpert(PpuMegaMoEFP8SharedExpert):
    """TP-local shared partial with the official UE8M0 MegaMoE arithmetic."""


class PpuSharedExpertExecutor(SharedExpertExecutor):
    """Execute the selected PPU FP8 path without CUDA packed-scale dispatch.

    PPU linears own row-major FP32 scales. The CUDA fused executor accepts a
    different scale ABI, so this executor validates its module at preparation
    and preserves the SGLang BF16 shared result for the FP32 MoE epilogue.
    Errors propagate directly; there is no generic expert fallback.
    """

    name = "ppu_sequential"

    def __init__(self, stream_pool=None, *, start_before_routing=False, staged=False):
        if start_before_routing and stream_pool is None:
            raise ValueError("Early shared execution requires a PPU overlap stream")
        if staged and (stream_pool is None or not start_before_routing):
            raise ValueError("Staged shared execution requires early overlap")
        self.staged = bool(staged)
        self._gate_up = None
        self.start_before_routing = bool(start_before_routing)
        self._shared = None
        self._out = None
        self._stream_pool = stream_pool
        self._stream = None
        self.name = "ppu_overlap" if stream_pool is not None else "ppu_sequential"

    def prepare(self, shared_experts):
        from rtp_llm.models_py.modules.factory.linear.ppu_impl.fp8_linear import (
            PpuFp8Linear,
        )

        if self._out is not None or self._gate_up is not None:
            raise RuntimeError("PPU shared executor still has a pending output")
        exact = isinstance(shared_experts, PpuMegaMoEFP8SharedExpert)
        if not exact and (
            not isinstance(shared_experts, PpuSharedExpert)
            or not all(
                isinstance(getattr(shared_experts, name, None), PpuFp8Linear)
                for name in ("w13", "w2")
            )
        ):
            raise TypeError("PPU shared executor requires PPU FP8 w13/w2 linears")
        if not shared_experts.preserve_output_dtype:
            raise ValueError("PPU shared executor requires the SGLang BF16 contract")
        if self._stream_pool is not None:
            device = (
                shared_experts._w13.device
                if exact
                else shared_experts.w13.weight.device
            )
            self._stream = self._stream_pool.get("shared_expert", device)
        self._shared = shared_experts

    def start(self, shared_experts, x, *, input_quantized=None):
        if self._shared is None or shared_experts is not self._shared:
            raise RuntimeError("PPU shared executor must use its prepared module")
        if self._out is not None or self._gate_up is not None:
            raise RuntimeError("PPU shared executor still has a pending output")
        if self._stream is None:
            with record_function_range("dsv4.moe.shared_expert"):
                self._out = (
                    shared_experts(x)
                    if input_quantized is None
                    else shared_experts(x, input_quantized=input_quantized)
                )
            return
        if x.device != self._stream.device:
            raise ValueError("PPU shared input must use the prepared stream device")
        current = torch.cuda.current_stream(x.device)
        if not torch.cuda.is_current_stream_capturing():
            x.record_stream(self._stream)
            if input_quantized is not None:
                for tensor in input_quantized:
                    tensor.record_stream(self._stream)
        self._stream.wait_stream(current)
        try:
            with torch.cuda.stream(self._stream):
                with record_function_range("dsv4.moe.shared_expert"):
                    if self.staged:
                        self._gate_up = shared_experts.project(
                            x, input_quantized=input_quantized
                        )
                    else:
                        self._out = (
                            shared_experts(x)
                            if input_quantized is None
                            else shared_experts(x, input_quantized=input_quantized)
                        )
        except Exception:
            current.wait_stream(self._stream)
            self._out = self._gate_up = None
            raise

    def after_routed_gemm1(self):
        """Enqueue shared G2 alongside routed G2 after the producer boundary."""
        if not self.staged or self._gate_up is None or self._out is not None:
            raise RuntimeError("Staged shared executor has no pending projection")
        self._stream.wait_stream(torch.cuda.current_stream(self._gate_up.device))
        with torch.cuda.stream(self._stream):
            self._out = self._shared.finish_projection(self._gate_up)
        self._gate_up = None

    def abort(self):
        """Join already-enqueued work and release a failed invocation's storage."""
        if self._stream is not None:
            torch.cuda.current_stream(self._stream.device).wait_stream(self._stream)
        self._out = self._gate_up = None

    def finish(self):
        if self._gate_up is not None:
            raise RuntimeError("Routed G1 did not complete the staged shared schedule")
        if self._out is None:
            raise RuntimeError("PPU shared executor has no pending output")
        if self._stream is not None:
            current = torch.cuda.current_stream(self._out.device)
            current.wait_stream(self._stream)
            if not torch.cuda.is_current_stream_capturing():
                self._out.record_stream(current)
        out, self._out = self._out, None
        return out


class PpuTPSharedExpert(PpuSharedExpert):
    def __init__(
        self,
        dim,
        inter_dim,
        expert_weights,
        *,
        tp_size,
        tp_rank,
        swiglu_limit=0.0,
        platform_provider,
    ):
        nn.Module.__init__(self)
        if tp_size <= 0 or not 0 <= tp_rank < tp_size:
            raise ValueError(
                "PPU shared expert requires a positive TP size and valid rank"
            )
        if inter_dim <= 0 or inter_dim % (128 * tp_size) or dim <= 0 or dim % 128:
            raise ValueError("Shared expert dimensions must preserve block-128 scales")
        local = inter_dim // tp_size
        for name, shape in (("w13", (2 * local, dim)), ("w2", (dim, local))):
            weight, scale = expert_weights[name + "_w"], expert_weights[name + "_s"]
            if tuple(weight.shape) != shape:
                raise ValueError(
                    f"Shared {name} requires loader-prepared shape {shape}"
                )
            setattr(
                self,
                name,
                PpuFp8Linear(
                    weight,
                    scale,
                    scale_is_prepared=tp_size > 1,
                    share_input_quantization=platform_provider._bool(
                        "DSV4_PPU_SHARED_QKV_QUANT", False
                    ),
                ),
            )
        self.swiglu_limit = swiglu_limit
        self.preserve_output_dtype = True
        self.tp_size, self.tp_rank = tp_size, tp_rank
