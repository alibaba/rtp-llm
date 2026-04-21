import psutil
import torch

from rtp_llm.device.device_base import DeviceBase, MemInfo
from rtp_llm.ops.compute_ops import ExecCtxExporter
from rtp_llm.utils.model_weight import W


class CpuImpl(DeviceBase):
    def __init__(self, exported_device: ExecCtxExporter):
        super().__init__(exported_device)

    def _get_mem_info(self) -> MemInfo:
        vmem = psutil.virtual_memory()
        return MemInfo(vmem.used, vmem.free)


class ArmCpuImpl(CpuImpl):
    def __init__(self, exported_device: ExecCtxExporter):
        super().__init__(exported_device)
        self.gemm_rewrite_list = [
            W.attn_qkv_w,
            W.attn_o_w,
            W.ffn_w1,
            W.ffn_w2,
            W.ffn_w3,
        ]

    def maybe_rewrite_weight_by_key(
        self, key: str, weight: torch.Tensor
    ) -> torch.Tensor:
        return self.exported_device.preprocess_gemm_weight_by_key(
            key, weight, self.py_env_configs.py_hw_kernel_config.arm_gemm_use_kai
        )

    def unpack_int32_into_int16(self, w_packed: torch.Tensor, int8: bool):
        if int8:
            return w_packed.contiguous().view(torch.uint8).to(torch.int16)
        # unpack inputs packed in int32/float32 into uint4 and store them in int8 format
        w_packed_int4x2 = w_packed.contiguous().view(torch.uint8)
        w_unpacked = torch.zeros(
            w_packed_int4x2.shape[0], w_packed_int4x2.shape[1] * 2, dtype=torch.int8
        )
        w_unpacked[:, ::2] = w_packed_int4x2 % 16
        w_unpacked[:, 1::2] = w_packed_int4x2 // 16
        return w_unpacked.to(torch.int16).contiguous()

    def pack_int8_tensor_to_packed_int4(self, tensor: torch.Tensor):
        assert tensor.dtype == torch.int8
        tensor -= (tensor >> 4) << 4
        tensor = tensor.view(torch.uint8)
        tensor = (tensor[:, 1::2] * 16 + tensor[:, ::2]).view(torch.int8)
        return tensor

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
        qweight = qweight_int32.reshape(qweight_int32.shape[0], -1).cpu()
        qzeros = qzeros_int32.reshape(qzeros_int32.shape[0], -1).cpu()
        scales_fp16 = scales_fp16.reshape(scales_fp16.shape[0], -1).cpu()
        packer = self.pack_int8_tensor_to_packed_int4
        preprocess_weight_scale = self.exported_device.preprocess_weight_scale
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
        qweight_interleaved = preprocess_weight_scale(qweight, scales_fp16)

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

        # return processed interleaved weight, original scales and zeros * scales
        return (
            qweight_interleaved.contiguous().to(device),
            zeros_x_scales_fp16.contiguous().to(device),
            scales_fp16.contiguous().to(device),
        )
