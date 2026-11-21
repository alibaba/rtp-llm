"""Qwen4 role-specific dense storage; execution stays in the main runtime.

The official FP8 source quantizes only experts and host PLE tables offline.
Attention matrices become block FP8, while the remaining eligible 2-D matrices
use a single storage scale after layout conversion and rank selection.
"""
import torch

from rtp_llm.models_py.new_models.qwen4_exp.hyper_connection import (
    Qwen4ProjectionWeight, Qwen4TensorWeight,
)


class Qwen4BlockProjection(Qwen4TensorWeight):
    def __init__(self, shape, *, dtype=torch.bfloat16):
        super().__init__(shape, dtype=dtype)
        self.register_buffer("weight_scale", None)
        self._storage_ready = False

    def process_weights_after_loading(self):
        if self._storage_ready:
            return
        from rtp_llm.models_py.quant_methods.fp8 import per_block_quant_like_legacy
        from rtp_llm.models_py.kernels.cuda.deepgemm_wrapper import is_deep_gemm_e8m0_used
        if self.weight.is_cuda and is_deep_gemm_e8m0_used():
            # Main's SM100/SM120 online path quantizes the original BF16 values
            # directly into the power-of-two domain. Quantize-then-requantize
            # would introduce a second FP8 rounding and break golden parity.
            from rtp_llm.models_py.kernels.cuda.fp8_kernel import quant_weight_ue8m0_packed
            weight, scale = quant_weight_ue8m0_packed(self.weight.contiguous())
        else:
            weight, scale = per_block_quant_like_legacy(self.weight, 128)
        self.weight.data = weight
        self.weight_scale = scale
        self.weight._rtp_requires_restart_for_update = True
        self.weight_scale._rtp_requires_restart_for_update = True
        self._storage_ready = True


class Qwen4ShardedProjection(Qwen4ProjectionWeight):
    """Shared FFN rank selection precedes scalar storage quantization."""
    def __init__(self, shape, *, axis, tp_size, tp_rank, dtype, fp8_storage=False):
        if not 0 <= tp_rank < tp_size or shape[axis] % tp_size:
            raise ValueError("Qwen4 shared projection shard must divide FFN TP")
        self.source_shape = tuple(shape)
        self.axis, self.tp_size, self.tp_rank = axis, tp_size, tp_rank
        local = list(shape)
        local[axis] //= tp_size
        super().__init__(tuple(local), dtype=dtype, fp8_storage=fp8_storage)

    def load_weights(self, weights):
        for name, tensor in weights.items():
            if name != "weight" or tuple(tensor.shape) != self.source_shape:
                raise ValueError("Invalid Qwen4 shared projection checkpoint")
            count = self.source_shape[self.axis] // self.tp_size
            super().load_weights({"weight": tensor.narrow(self.axis, self.tp_rank * count, count)})


def bind_projection(owner, *, hw_kernel_config=None, role="PDFUSION"):
    """Construct the original runtime from typed owned tensors, without W.*."""
    from rtp_llm.models_py.modules.qwen4_exp.weight_views import Qwen4LinearFactory
    scale = getattr(owner, "storage_scale", None)
    if scale is not None:
        from rtp_llm.models_py.modules.base.common.fp8_linear import Fp8Linear
        return Fp8Linear(owner.weight, scale, role=role)
    block_scale = getattr(owner, "weight_scale", None)
    if block_scale is not None:
        from rtp_llm.config.quant_config import Fp8BlockWiseQuantConfig
        from rtp_llm.models_py.kernels.cuda.deepgemm_wrapper import is_deep_gemm_e8m0_used
        # The existing LinearFactory consumes the legacy flat KN representation
        # for ordinary scales and the NK representation for packed UE8M0.
        weight = owner.weight
        if is_deep_gemm_e8m0_used():
            if block_scale.dtype != torch.int32:
                raise RuntimeError("Qwen4 packed block runtime requires direct BF16-to-UE8M0 preparation on its execution device; reload the original checkpoint")
        else:
            weight = weight.reshape(weight.shape[1], weight.shape[0])
            block_scale = block_scale.reshape(block_scale.shape[1], block_scale.shape[0])
        return Qwen4LinearFactory.create_linear(weight, None, block_scale,
            Fp8BlockWiseQuantConfig(is_quanted=True), hw_kernel_config=hw_kernel_config)
    return Qwen4LinearFactory.create_linear(owner.weight.t(), None, None, None,
        hw_kernel_config=hw_kernel_config)
