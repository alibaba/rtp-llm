"""AWQ (w4a16) Linear 量化方法（新 loader），参照 vLLM ``AWQLinearMethod`` 移植。

权重布局(AutoAWQ/vLLM 约定,注意 qweight 是 ``[in, out]`` 而非标准 ``[out, in]``):
  - ``qweight``: [input_size, output_size // 8]  int32（8 个 4-bit 打包进一个 int32）
  - ``qzeros`` : [input_size // group_size, output_size // 8]  int32
  - ``scales`` : [input_size // group_size, output_size]  fp16/bf16

前向:``awq_dequantize_triton`` 反量化成 [input, output] 权重 → ``torch.matmul(x, w)``。
group_size 来自上游 ``config/quant_config.AWQConfig``（经 QuantizationConfig.source_config
透传,见 [[newloader-quant-dispatch-refactor]] 走法1）。
TP 切分在 ``ColumnParallelLinear/RowParallelLinear.load_weights`` 里对 qweight/qzeros/scales
处理（Column 切 dim1=输出、Row 切 dim0=输入）。

注意：本 reference 实现不注册为生产 quant method。它每次 forward 都会全量反量化，
只能用于后续 fused/cache-backed AWQ GEMM 接入前的格式参考，不能进入 decode 热路径。
"""

from typing import Any, Optional

import torch
import torch.nn as nn

from rtp_llm.models_py.quant_methods.base import LinearMethodBase


class AWQLinearMethod(LinearMethodBase):
    PACK_FACTOR = 8  # 32 // 4-bit

    def __init__(self, quant_config: Any = None):
        self.quant_config = quant_config
        # 上游旧 AWQConfig（带 group_size/bits），经 source_config 透传。
        self.source_config = getattr(quant_config, "source_config", None)

    def _group_size(self, input_size: int) -> int:
        src = self.source_config
        if src is None:
            return input_size
        # 旧 QuantizationConfig 的 group_size 是「方法」(返回 self._group_size),
        # 也兼容当作属性的情况。
        gs = getattr(src, "group_size", None)
        if callable(gs):
            gs = gs()
        if gs in (-1, 0, None):
            return input_size  # 整个输入维一个 group
        return int(gs)

    def create_weights(
        self,
        layer,
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **kwargs,
    ):
        pf = self.PACK_FACTOR
        gs = self._group_size(input_size)
        if input_size % gs != 0:
            raise ValueError(
                f"[AWQ] input_size {input_size} 不能被 group_size {gs} 整除"
                f"（可能 TP 切分过大）。"
            )
        if output_size % pf != 0:
            raise ValueError(
                f"[AWQ] output_size {output_size} 不能被 pack_factor {pf} 整除"
                f"（可能 TP 切分过大）。"
            )
        num_groups = input_size // gs

        layer.register_parameter(
            "qweight",
            nn.Parameter(
                torch.empty(input_size, output_size // pf, dtype=torch.int32),
                requires_grad=False,
            ),
        )
        layer.register_parameter(
            "qzeros",
            nn.Parameter(
                torch.empty(num_groups, output_size // pf, dtype=torch.int32),
                requires_grad=False,
            ),
        )
        layer.register_parameter(
            "scales",
            nn.Parameter(
                torch.empty(num_groups, output_size, dtype=params_dtype),
                requires_grad=False,
            ),
        )
        layer._awq_group_size = gs

    def process_weights_after_loading(self, layer):
        # 权重已是目标布局,无需后处理（反量化在 apply 里按需做）。
        return None

    def apply(
        self, layer, x: torch.Tensor, bias: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        from rtp_llm.models_py.quant_methods.awq_triton import awq_dequantize_triton

        # Reference path: dequantize to [input, output] on every forward, then
        # run a standard matmul. This keeps first-phase weight loading correct;
        # a fused/cache-backed w4a16 GEMM should replace it before this AWQ path
        # is used on decode hot paths.
        weight = awq_dequantize_triton(layer.qweight, layer.scales, layer.qzeros)
        out_dim = weight.shape[1]
        reshaped_x = x.reshape(-1, x.shape[-1])
        out = torch.matmul(reshaped_x, weight)
        if bias is not None:
            out = out + bias
        return out.reshape(*x.shape[:-1], out_dim)
