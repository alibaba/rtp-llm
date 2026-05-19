"""Online MXFP4 linear layer.

Uses MXFP4 (block_size=32, UE8M0 scales) packed weights produced at weight-load
time by :func:`mxfp4_quantize_linear_weight` (centralised in
``rtp_llm.model_loader.online_modelopt_fp4_quant_weight``). Inference uses
``flashinfer.mm_fp4`` with the cute-dsl backend.

Activated by env ``USE_ONLINE_FP4GEMM=1``.

Constraint: K must be divisible by 128 (required by swizzled scale factor layout).
N has no alignment requirement.
"""

from typing import Optional

import torch
from flashinfer import autotune, mm_fp4, mxfp4_quantize

from rtp_llm.model_loader.online_modelopt_fp4_quant_weight import (
    is_online_fp4gemm_enabled,
    mxfp4_quantize_linear_weight,
)
from rtp_llm.models_py.modules.factory.linear import LinearBase

_ENABLED = is_online_fp4gemm_enabled()


class CudaOnlineMxfp4Linear(LinearBase):
    """Online MXFP4 quantized linear using flashinfer mm_fp4 (cute-dsl backend)."""

    @classmethod
    def can_handle(
        cls,
        quant_config: object,
        weight: torch.Tensor,
        weight_scales: Optional[torch.Tensor],
        hw_kernel_config=None,
        weight_scale_2: Optional[torch.Tensor] = None,
        input_scale: Optional[torch.Tensor] = None,
    ) -> bool:
        if not _ENABLED or weight_scales is not None:
            return False
        if weight.dtype not in (torch.bfloat16, torch.float16):
            return False
        if weight.dim() != 2:
            return False
        K, N = weight.shape
        return K % 128 == 0

    @torch.inference_mode()
    def __init__(
        self,
        weight: torch.Tensor,
        weight_scales: Optional[torch.Tensor] = None,
        input_scales: Optional[torch.Tensor] = None,
        bias: Optional[torch.Tensor] = None,
        quant_config: object = None,
        weight_scale_2: Optional[torch.Tensor] = None,
    ):
        super().__init__(
            weight, weight_scales, input_scales, bias, quant_config, weight_scale_2
        )

        K, N = weight.shape
        # Centralised in online_modelopt_fp4_quant_weight: produces the same
        # transposed FP4 packed weight + scale layout that mm_fp4 expects.
        # Releases the BF16 source internally.
        w_fp4_t, w_sf_t = mxfp4_quantize_linear_weight(weight)
        del weight

        self.register_buffer("w_fp4_t", w_fp4_t)
        self.register_buffer("w_sf_t", w_sf_t)
        self.N = N
        self.bias = bias

        # JIT-compile the cute-dsl mm_fp4 kernel NOW so the first real forward
        # call (which may happen inside CUDA graph capture) does not have to
        # invoke the cutlass JIT compiler — that compilation issues stream-side
        # operations cuda-graph-capture forbids (observed: SIGABRT at
        # CudaGraphRunner::initCapture). One dummy forward in eager mode is
        # enough to populate the cache.
        try:
            self._prime_kernel_cache(K, device=w_fp4_t.device)
        except Exception:
            # If the prime fails (e.g., GPU OOM at a pathological shape) fall
            # through; the first inference call will then JIT and crash, which
            # is the pre-fix behaviour. We do NOT want this prime to mask real
            # bugs — surface them.
            raise

    @torch.inference_mode()
    def _prime_kernel_cache(self, K: int, device) -> None:
        from flashinfer import mxfp4_quantize as _mxfp4_quantize

        # Use a multi-row dummy so the cute-dsl wrapper picks the
        # ``stride[1] == 1`` template (single-row inputs can land on a
        # squeezed layout that the wrapper rejects).
        dummy = torch.zeros((16, K), device=device, dtype=torch.bfloat16)
        a_fp4, a_sf = _mxfp4_quantize(dummy, backend="cute-dsl")
        with autotune(tune_mode=False):
            mm_fp4(
                a_fp4,
                self.w_fp4_t,
                a_sf,
                self.w_sf_t,
                None,
                torch.bfloat16,
                block_size=32,
                use_nvfp4=False,
                backend="cute-dsl",
            )
        torch.cuda.synchronize()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        shape = input.shape
        x = input.reshape(-1, shape[-1]) if input.dim() > 2 else input

        a_fp4, a_sf = mxfp4_quantize(x, backend="cute-dsl")

        # The flashinfer autotuner allocates workspaces, JIT-compiles candidate
        # kernels, and runs benchmark sweeps. Even when the FIRST forward
        # (warmup) is allowed to tune, the resulting cache state breaks cuda
        # graph capture: the per-call check `tune_mode and not _tuned` flips to
        # False at capture time but the underlying autotuner stream / workspace
        # state from the warmup pass still aborts in
        # `CudaGraphRunner::initCapture()` (observed: SIGABRT immediately after
        # `[Autotuner]: Autotuning process ends`).
        #
        # Use the default (non-tuned) kernel always: deterministic across
        # warmup / capture / replay, and works under cuda graph. Performance
        # parity with the previous code path because that path required tuning
        # *every* forward anyway when warmup was disabled (`--warm_up 0`).
        with autotune(tune_mode=False):
            out = mm_fp4(
                a_fp4,
                self.w_fp4_t,
                a_sf,
                self.w_sf_t,
                None,
                torch.bfloat16,
                block_size=32,
                use_nvfp4=False,
                backend="cute-dsl",
            )

        if self.bias is not None:
            out = out + self.bias

        if input.dim() > 2:
            out = out.reshape(*shape[:-1], self.N)

        return out


# Backward-compatible alias
CudaOnlineNvfp4Linear = CudaOnlineMxfp4Linear
