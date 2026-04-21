import logging
import re
from typing import List

import torch

from rtp_llm.device.cuda_device import GpuImpl
from rtp_llm.device.device_base import MemInfo
from rtp_llm.ops.compute_ops import ExecCtxExporter
from rtp_llm.utils.model_weight import W
from rtp_llm.utils.swizzle_utils import swizzle_tensor


class RocmImpl(GpuImpl):
    def __init__(self, exported_device: ExecCtxExporter):
        super().__init__(exported_device)
        try:
            from pyrsmi import rocml

            self.rocml = rocml
            rocml.smi_initialize()
        except Exception as e:
            self.rocml = None
            logging.warn(f"no rocm smi found: " + str(e))

    # ===== Attention 优先级路由 =====

    def get_prefill_mha_priorities(self):
        from rtp_llm.models_py.modules.factory.attention.rocm_impl.aiter import (
            AiterPrefillImplAsm,
            AiterPrefillImplNonAsm,
            AiterPrefillImplPaged,
        )

        return [AiterPrefillImplPaged, AiterPrefillImplAsm, AiterPrefillImplNonAsm]

    def get_decode_mha_priorities(self):
        from rtp_llm.models_py.modules.factory.attention.rocm_impl.aiter import (
            AiterDecodeImplAsm,
            AiterDecodeImplNonAsm,
            AiterDecodeImplTriton,
        )

        return [AiterDecodeImplAsm, AiterDecodeImplNonAsm, AiterDecodeImplTriton]

    # ===== Base Ops 分派 =====

    def get_base_ops(self):
        from rtp_llm.device.base_ops import BaseOps
        from rtp_llm.models_py.modules.base.rocm.activation import FusedSiluAndMul
        from rtp_llm.models_py.modules.base.rocm.moe_gating import SigmoidGateScaleAdd
        from rtp_llm.models_py.modules.base.rocm.norm import (
            AddBiasResLayerNorm,
            FusedQKRMSNorm,
            QKRMSNorm,
            RMSNorm,
            RMSResNorm,
        )
        from rtp_llm.models_py.modules.base.rocm.not_implemented_ops import (
            FakeBalanceExpert,
            GroupTopK,
            IndexerOp,
        )
        from rtp_llm.models_py.modules.base.rocm.select_topk import SelectTopk

        return BaseOps(
            FusedSiluAndMul=FusedSiluAndMul,
            RMSNorm=RMSNorm,
            RMSResNorm=RMSResNorm,
            AddBiasResLayerNorm=AddBiasResLayerNorm,
            FusedQKRMSNorm=FusedQKRMSNorm,
            QKRMSNorm=QKRMSNorm,
            SelectTopk=SelectTopk,
            GroupTopK=GroupTopK,
            FakeBalanceExpert=FakeBalanceExpert,
            IndexerOp=IndexerOp,
            SigmoidGateScaleAdd=SigmoidGateScaleAdd,
        )

    # ===== Linear 分派 =====

    def register_linear_impl(self):
        import rtp_llm.models_py.modules.factory.linear.impl.rocm  # noqa: F401

    # ===== MoE 策略路由 =====

    def get_moe_strategy_candidates(self):
        from rtp_llm.models_py.modules.factory.fused_moe.impl.common.strategy.batched_triton_strategy import (
            BatchedTritonStrategy,
        )
        from rtp_llm.models_py.modules.factory.fused_moe.impl.rocm.strategy import (
            RocmBf16PureTPStrategy,
            RocmEpLowLatencyStrategy,
            RocmEpNormalStrategy,
            RocmFp8PerChannelPureTPStrategy,
        )

        return [
            RocmEpLowLatencyStrategy,
            RocmEpNormalStrategy,
            RocmFp8PerChannelPureTPStrategy,
            RocmBf16PureTPStrategy,
            BatchedTritonStrategy,
        ]

    @staticmethod
    def cat_0(ts: List[torch.Tensor], dim: int = 0) -> torch.Tensor:
        if len(ts) == 1:
            return ts[0]
        # torch.cat() does not support fp8 in current rocm torch version
        if ts[0].dtype in [
            torch.float8_e4m3fn,
            torch.float8_e4m3fnuz,
            torch.float8_e5m2,
            torch.float8_e5m2fnuz,
        ]:
            dtype = ts[0].dtype
            out_u8 = torch.cat([x.view(torch.uint8) for x in ts], dim=dim).contiguous()
            return out_u8.view(dtype)
        else:
            return torch.cat(ts, dim=dim).contiguous()

    # column major
    def pack_int8_tensor_to_packed_int4(self, tensor: torch.Tensor):
        assert tensor.dtype == torch.int8
        tensor -= (tensor >> 4) << 4
        tensor = tensor.view(torch.uint8)
        tensor = tensor[::2, :] * 16 + tensor[1::2, :]
        tensor ^= 1 << 3
        tensor ^= 1 << 7
        shape = tensor.shape
        return tensor.view(torch.int8).transpose(0, 1).contiguous().view(shape)

    def preprocess_weights_for_mixed_gemm(
        self,
        tensor: torch.Tensor,
        quant_mode: torch.dtype,
        arch: str = "",
    ) -> torch.Tensor:
        assert tensor.dtype == torch.int8, "Weight must be torch.int8"
        assert tensor.numel() % 4 == 0, "Total elements must be divisible by 4"

        # 1. 展平并转为 uint8，便于位操作
        x = tensor.view(-1).to(torch.uint8)

        # 2. 解包：每个 byte 拆为 [high, low] → 得到 (N*2,) 的 4-bit 数列
        nibbles = torch.stack([x >> 4, x & 0x0F], dim=1).view(-1)  # shape: [numel * 2]

        # 3. 每 8 个 4-bit 数一组，应用 permute: [0,1,2,3,4,5,6,7] -> [2,0,6,4,3,1,7,5]
        n_groups = nibbles.numel() // 8
        permute_idx = torch.tensor([2, 0, 6, 4, 3, 1, 7, 5], device=x.device)

        # 重塑为 (n_groups, 8)，应用索引，展平
        reordered = nibbles.view(n_groups, 8)[:, permute_idx].view(-1)

        # 4. 重新打包：每两个 4-bit 数合并为一个 byte
        hi = reordered[::2] << 4
        lo = reordered[1::2]
        packed = (hi | lo).to(torch.int8)  # shape: (n_groups * 4,)

        # 5. reshape 为 reversed(input.shape)，然后 transpose(0,1)
        orig_shape = tensor.shape
        new_shape = list(reversed(orig_shape))
        permuted_weight = packed.view(new_shape)

        return permuted_weight.transpose(0, 1)

    def _get_mem_info(self) -> MemInfo:
        from pyrsmi import rocml

        id = self.get_device_id()
        used = rocml.smi_get_device_memory_used(id)
        total = rocml.smi_get_device_memory_total(id)
        return MemInfo(free=total - used, used=used)

    @property
    def arch(self) -> str:
        if self.rocml:
            try:
                id = self.get_device_id()
                device_name = self.rocml.smi_get_device_name(id)
                # 从设备名称中提取架构信息（假设名称包含 gfx 版本）
                gfx_match = re.search(r"gfx(\d+)", device_name)
                if gfx_match:
                    return gfx_match.group(1)
            except Exception as e:
                logging.warn(f"Cannot get ROCm device gfx version: {e}")
        # 如果无法获取，则使用环境变量或默认值
        specify_gpu_arch = self.py_env_configs.runtime_config.specify_gpu_arch
        return "900" if specify_gpu_arch == "" else specify_gpu_arch

    def preprocess_groupwise_weight_params(
        self,
        qweight_int32,
        qzeros_int32,
        scales_fp16,
        device: str,
        gptq: bool,
        awq: bool,
        weight_bits: int,
    ):
        GPTQ_FLAG = 1 if gptq == True else 0
        qweight = qweight_int32.reshape(qweight_int32.shape[0], -1)
        qzeros = qzeros_int32.reshape(qzeros_int32.shape[0], -1)
        scales_fp16 = scales_fp16.reshape(scales_fp16.shape[0], -1)
        packer = self.pack_int8_tensor_to_packed_int4
        preprocessor = self.preprocess_weights_for_mixed_gemm
        is_int8 = weight_bits == 8
        if is_int8:
            zero_shift = 128
            quant_type = torch.int8
        else:
            zero_shift = 8
            quant_type = torch.quint4x2

        if awq:
            qweight = (
                self.unpack_int32_into_int16(qweight, is_int8).contiguous() - zero_shift
            )
            qweight = self.reverse_awq_order(qweight)
        elif gptq:
            qweight = (
                self.unpack_int32_into_int16(qweight.T, is_int8).T.contiguous()
                - zero_shift
            )

        qweight = qweight.to(torch.int8)
        if not is_int8:
            qweight = packer(qweight)
        qweight_interleaved = preprocessor(qweight, quant_type, self.specify_gpu_arch)

        # zero = 0 if qzeros_int32 = -2004318072 torch.int32 for awq
        # zero = 0 if qzeros_int32 = 2004318071  torch.int32 for gptq
        qzeros = self.unpack_int32_into_int16(qzeros, is_int8)
        if awq:
            qzeros = self.reverse_awq_order(qzeros)

        # zeros = zeros * scales
        UINT_TO_INT_FLAG = 1
        zeros_x_scales_fp16 = (
            -qzeros + zero_shift * UINT_TO_INT_FLAG - GPTQ_FLAG
        ) * scales_fp16
        zeros_x_scales_fp16 = zeros_x_scales_fp16.half()

        ###########################################################
        # scales row major -> scales column major layout to match CK kernel layout
        # TODO: need add device infomation for selection
        scales_fp16_t = scales_fp16.transpose(0, 1).contiguous()
        scales_fp16 = scales_fp16_t.transpose(1, 0).cpu()

        # zeros_x_scales row major -> zeros_x_scales column major layout to match CK kernel layout
        zeros_x_scales_fp16_t = zeros_x_scales_fp16.transpose(0, 1).contiguous()
        zeros_x_scales_fp16 = zeros_x_scales_fp16_t.transpose(1, 0).cpu()
        ###########################################################

        # return processed interleaved weight, original scales and zeros * scales
        # return qweight_interleaved.contiguous().to(device),  zeros_x_scales_fp16.contiguous().to(device), scales_fp16.contiguous().to(device)
        # kernel, scales, zeros all need for column major layout
        return (
            qweight_interleaved.to(device),
            zeros_x_scales_fp16.to(device),
            scales_fp16.to(device),
        )

    def shuffle_moe_weight(
        self, x: torch.Tensor, datatype: torch.dtype, name: str
    ) -> torch.Tensor:
        def _padding_to_multiply_512(x_, is_gate):
            align = [0, 512, 0] if is_gate else [0, 0, 512]
            shape_tmp = list(
                x_.shape
            )  # due to gate+up, need temporarily seperate them for padding
            if is_gate:
                shape_tmp[1] = shape_tmp[1] // 2
            # align and padding to multiply of 512
            padding = [0 for i in range(len(align) * 2)]
            for i in range(len(align)):
                if (align[i] > 0) and (shape_tmp[i] % align[i] > 0):
                    padding[-(i * 2 + 1)] = align[i] - (shape_tmp[i] % align[i])
            if sum(padding):
                if is_gate:
                    x_ = torch.cat(
                        [
                            torch.nn.functional.pad(
                                x_[:, : x_.shape[1] // 2, :],
                                padding,
                                mode="constant",
                                value=0,
                            ),
                            torch.nn.functional.pad(
                                x_[:, x_.shape[1] // 2 :, :],
                                padding,
                                mode="constant",
                                value=0,
                            ),
                        ],
                        dim=1,
                    )
                else:
                    x_ = torch.nn.functional.pad(
                        x_, tuple(padding), mode="constant", value=0
                    )
                # logging.info(f'Moe padding shape {[ele for ele in x.shape]} with {padding} to {[ele for ele in x_.shape]}')
            return x_

        def _shuffle_weight(x_, layout=(16, 16), use_int4=False):
            # Hardcode BLOCK_K and BLOCK_N
            IN, IK = layout
            BK = IK * 2
            K = 16 // x_.element_size() if not use_int4 else 32
            BN = IN
            assert (
                x_.shape[-2] % BN == 0
            ), f"{x_.shape[-2]} % {BN} == {x_.shape[-2] % BN }"
            assert (
                x_.shape[-1] % BK == 0
            ), f"{x_.shape[-1]} % {BK} == {x_.shape[-1] % BK }"
            x__ = x_.view(-1, x_.shape[-2] // BN, BN, x_.shape[-1] // BK, BK // K, K)
            x__ = x__.permute(0, 1, 3, 4, 2, 5)
            x__ = x__.contiguous()
            x__ = x__.view(*x_.shape)
            return x__

        is_gate = name in [W.moe_w1, W.moe_s1]
        do_shuffle = name in [W.moe_w1, W.moe_w2]
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        x_ = (
            self.cat_0([x[:, x.shape[1] // 2 :, :], x[:, : x.shape[1] // 2, :]], dim=1)
            if is_gate
            else x
        )  # swap from [up, gate] to [gate, up]
        if do_shuffle:
            # for now we use ck_moe for dtype is not fp8, so we need to pad to multiply of 512
            if x_.dtype not in [torch.float8_e4m3fn, torch.float8_e4m3fnuz]:
                x_ = _padding_to_multiply_512(x_, is_gate)
            x_ = _shuffle_weight(x_)
        return x_

    def maybe_rewrite_weight_by_key(
        self, key: str, weight: torch.Tensor
    ) -> torch.Tensor:
        if key == "weight":
            assert weight.dtype == torch.float8_e4m3fn
            weight_as_int8 = weight.view(torch.int8)
            ROCM_FP8_NAN_AS_INT = -128
            weight_as_int8[weight_as_int8 == ROCM_FP8_NAN_AS_INT] = 0
            weight = weight_as_int8.view(torch.float8_e4m3fnuz)
        elif key == "scale":
            weight = weight * 2.0

        if key in [
            W.attn_qkv_w,
            W.attn_o_w,
            W.ffn_w2,
            W.ffn_w13,
            W.ffn_w3,
            W.moe_gate,
            W.multi_tokens_predict_eh_proj,
        ]:
            if self.py_env_configs.py_hw_kernel_config.use_swizzleA:
                if (
                    self.py_env_configs.model_specific_config.load_python_model
                    and weight.dtype != torch.float8_e4m3fn
                ):
                    weight = swizzle_tensor(weight.t(), False).t()
                else:
                    weight = swizzle_tensor(weight, weight.dtype != torch.float8_e4m3fn)
            elif weight.dtype == torch.float8_e4m3fn:
                weight = self.shuffle_gemm_weight(weight)

        return weight

    # def convert_fp8_weight_params(self, weight: torch.Tensor, weight_scale: torch.Tensor):
    #   assert weight.dtype == torch.float8_e4m3fn
    #   # The bits pattern 10000000(-128) represents zero in e4m3fn
    #   # but NaN in e4m3fnuz. So here we set it to 0.
    #   # https://onnx.ai/onnx/technical/float8.html
    #   weight_as_int8 = weight.view(torch.int8)
    #   ROCM_FP8_NAN_AS_INT = -128
    #   weight_as_int8[weight_as_int8 == ROCM_FP8_NAN_AS_INT] = 0
    #   weight = weight_as_int8.view(torch.float8_e4m3fnuz)
    #   # For the same bits representation, e4m3fnuz value is half of
    #   # the e4m3fn value, so we should double the scaling factor to
    #   # get the same dequantized value.
    #   # https://onnx.ai/onnx/technical/float8.html
    #   weight_scale = weight_scale * 2.0
    #   return weight, weight_scale
    def shuffle_gemm_weight(self, x: torch.Tensor) -> torch.Tensor:
        # Hardcode BLOCK_K and BLOCK_N
        layout = (16, 16)
        use_int4 = False
        IN, IK = layout
        BK = IK * 2
        K = 16 // x.element_size() if not use_int4 else 32
        BN = IN
        assert x.shape[-2] % BN == 0, f"{x.shape[-2]} % {BN} == {x.shape[-2] % BN }"
        assert x.shape[-1] % BK == 0, f"{x.shape[-1]} % {BK} == {x.shape[-1] % BK }"

        x_ = x
        x_ = x_.view(-1, x.shape[-2] // BN, BN, x.shape[-1] // BK, BK // K, K)
        x_ = x_.permute(0, 1, 3, 4, 2, 5)
        x_ = x_.contiguous()
        x_ = x_.view(*x.shape)
        return x_

    def convert_fp8_weight_params(
        self, weight: torch.Tensor, weight_scale: torch.Tensor
    ):
        assert weight.dtype == torch.float8_e4m3fn
        # The bits pattern 10000000(-128) represents zero in e4m3fn
        # but NaN in e4m3fnuz. So here we set it to 0.
        # https://onnx.ai/onnx/technical/float8.html
        weight_as_int8 = weight.view(torch.int8)
        ROCM_FP8_NAN_AS_INT = -128
        weight_as_int8[weight_as_int8 == ROCM_FP8_NAN_AS_INT] = 0
        weight = weight_as_int8.view(torch.float8_e4m3fnuz)
        # For the same bits representation, e4m3fnuz value is half of
        # the e4m3fn value, so we should double the scaling factor to
        # get the same dequantized value.
        # https://onnx.ai/onnx/technical/float8.html
        weight_scale = weight_scale * 2.0
        return weight, weight_scale
