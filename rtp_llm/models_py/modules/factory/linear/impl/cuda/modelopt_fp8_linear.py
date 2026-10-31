"""Static FP8 projections with the checkpoint's per-projection scales intact."""

import os

import torch

from rtp_llm.models_py.kernels.cuda.fp8_kernel import scaled_fp8_per_tensor_quant
from rtp_llm.models_py.kernels.cuda.modelopt_fp8_quant import (
    gated_fp8_quant_rows,
    static_fp8_quant_rows,
)
from rtp_llm.models_py.modules.factory.linear.impl.cuda.fp8_per_tensor_linear import (
    CudaFp8PerTensorLinear,
)
from rtp_llm.models_py.utils.arch import get_sm


class CudaModelOptFp8Linear(CudaFp8PerTensorLinear):
    # Keep multi-scale projection/concatenation workspaces bounded for prefill.
    # Decode and single-scale GEMMs retain their existing execution path.
    _PREFILL_CHUNK_ROWS = 4096

    @classmethod
    def can_handle(cls, quant_config, weight, weight_scales, *args, **kwargs):
        return (
            quant_config is not None
            and quant_config.get_method() == "modelopt_mixed"
            and weight.dtype == torch.float8_e4m3fn
            and weight_scales is not None
            and weight_scales.shape == (2, weight.shape[0])
        )

    def __init__(
        self,
        weight,
        weight_scales=None,
        input_scales=None,
        bias=None,
        quant_config=None,
        weight_scale_2=None,
    ):
        super().__init__(
            weight, weight_scales, input_scales, bias, quant_config, weight_scale_2
        )
        if weight.is_cuda:
            get_sm(weight.device)  # Prime device eligibility before graph/forward.
        self.weight_scale = weight_scales[:1].contiguous()
        # A fused QKV can still have distinct static activation scales. Partition
        # only at their boundaries; identical scales use one GEMM, without
        # re-quantizing checkpoint weights to a shared global weight scale.
        values = weight_scales[1].cpu().tolist()
        boundaries = (
            [0]
            + [i for i in range(1, len(values)) if values[i] != values[i - 1]]
            + [len(values)]
        )
        self.parts = []
        for start, end in zip(boundaries, boundaries[1:]):
            self.parts.append(
                (
                    weight[start:end].contiguous().T,
                    self.weight_scale[:, start:end].contiguous(),
                    weight_scales[1, start].reshape(()),
                )
            )

        self._use_skinny = (
            os.getenv("RTP_LLM_SM120_FUSION", "0") == "1"
            and weight.is_cuda
            and get_sm(weight.device) == (12, 0)
        )

    def _scaled_projection(self, qinput, weight, scale_a, scale_b, dtype):
        if getattr(self, "_use_skinny", False):
            from rtp_llm.models_py.kernels.cuda.sm120_fp8_skinny import skinny_fp8_mm

            output = skinny_fp8_mm(qinput, weight, scale_a, scale_b)
            if output is not None:
                return output
        return torch._scaled_mm(
            qinput, weight, scale_a=scale_a, scale_b=scale_b, out_dtype=dtype
        )

    def forward(self, input):
        if input.dtype != torch.bfloat16:
            raise ValueError("ModelOpt mixed FP8 linear requires BF16 activations")
        x = input.reshape(-1, input.shape[-1])
        if x.shape[0] == 0:
            return x.new_empty((*input.shape[:-1], self.weight.shape[1]))
        if len(self.parts) == 1 or x.shape[0] <= self._PREFILL_CHUNK_ROWS:
            output = self._forward_rows(x)
        else:
            rows = self._PREFILL_CHUNK_ROWS
            first = self._forward_rows(x[:rows])
            output = first.new_empty((x.shape[0], self.weight.shape[1]))
            output[:rows].copy_(first)
            del first
            for start in range(rows, x.shape[0], rows):
                end = min(start + rows, x.shape[0])
                output[start:end].copy_(self._forward_rows(x[start:end]))
        return output.reshape(*input.shape[:-1], self.weight.shape[1])

    def _forward_rows(self, x):
        outputs = []
        for weight, scale, input_scale in self.parts:
            fused = static_fp8_quant_rows(x, input_scale)
            if fused is None:
                qinput, scale_a = scaled_fp8_per_tensor_quant(
                    x.contiguous(), input_scale
                )
                scale_a = scale_a.reshape(1, 1).expand(qinput.shape[0], 1).contiguous()
            else:
                qinput, scale_a = fused
            output = self._scaled_projection(qinput, weight, scale_a, scale, x.dtype)
            if isinstance(output, tuple):
                output = output[0]
            outputs.append(output[: x.shape[0]])
        output = outputs[0] if len(outputs) == 1 else torch.cat(outputs, dim=-1)
        if self.bias is not None:
            output = output + self.bias
        return output

    def input_scale_for_fusion(self):
        if len(self.parts) != 1 or self.parts[0][0].shape[0] != 5120:
            return None
        return self.parts[0][2]

    def forward_with_prequantized(self, input, quantized):
        qinput, scale_a = quantized
        if (
            len(self.parts) != 1
            or input.ndim != 2
            or input.dtype != torch.bfloat16
            or qinput.shape != input.shape
            or qinput.device != input.device
            or qinput.dtype != torch.float8_e4m3fn
            or not qinput.is_contiguous()
            or scale_a.shape != (input.shape[0], 1)
            or scale_a.dtype != torch.float32
            or scale_a.device != input.device
            or not scale_a.is_contiguous()
        ):
            return None
        weight, scale, _ = self.parts[0]
        output = self._scaled_projection(qinput, weight, scale_a, scale, input.dtype)
        if isinstance(output, tuple):
            output = output[0]
        output = output[: input.shape[0]]
        if self.bias is not None:
            output = output + self.bias
        return output

    def forward_with_gate(self, input, gate, *, norm_weight=None, eps=1e-6):
        """Consume a private gate/norm producer without materializing BF16 output."""
        if len(self.parts) != 1:
            return None
        weight, scale, input_scale = self.parts[0]
        if input.shape[0] == 0:
            return None
        fused = gated_fp8_quant_rows(
            input, gate, input_scale, norm_weight=norm_weight, eps=eps
        )
        if fused is None:
            return None
        qinput, scale_a = fused
        output = self._scaled_projection(qinput, weight, scale_a, scale, input.dtype)
        if isinstance(output, tuple):
            output = output[0]
        output = output[: input.shape[0]]
        if self.bias is not None:
            output = output + self.bias
        return output
