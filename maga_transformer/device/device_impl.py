from maga_transformer.device.device_base import DeviceBase, DeviceType, MemInfo
from maga_transformer.ops import DeviceType, DeviceExporter
from maga_transformer.utils.model_weight import W

import torch
import psutil

class CpuImpl(DeviceBase):
    def __init__(self, exported_device: DeviceExporter):
        super().__init__(exported_device)

    def get_mem_info(self) -> MemInfo:
        vmem = psutil.virtual_memory()
        return MemInfo(vmem.used, vmem.free)

class ArmCpuImpl(CpuImpl):
    def __init__(self, exported_device: DeviceExporter):
        super().__init__(exported_device)
        self.gemm_rewrite_list = [
            W.attn_qkv_w,
            W.attn_o_w,
            W.ffn_w1,
            W.ffn_w2,
            W.ffn_w3,
        ]

    def maybe_rewrite_weight_by_key(self, key: str, weight: torch.Tensor) -> torch.Tensor:
        return self.exported_device.preprocess_gemm_weight_by_key(key, weight)

class GpuImpl(DeviceBase):
    def __init__(self, exported_device: DeviceExporter):
        super().__init__(exported_device)

    def get_device_id(self) -> int:
        return torch.cuda.current_device()

    def unpack_int32_into_int16(self, w_packed: torch.Tensor, int8: bool):
        if int8:
            return w_packed.contiguous().view(torch.uint8).to(torch.int16)
        # unpack inputs packed in int32/float32 into uint4 and store them in int8 format
        w_packed_int4x2 = w_packed.contiguous().view(torch.uint8)
        w_unpacked = torch.zeros(w_packed_int4x2.shape[0],
                                 w_packed_int4x2.shape[1] * 2,
                                 dtype=torch.int8)
        w_unpacked[:, ::2] = w_packed_int4x2 % 16
        w_unpacked[:, 1::2] = w_packed_int4x2 // 16
        return w_unpacked.to(torch.int16).contiguous()

    def reverse_awq_order(self, ori_tensor: torch.Tensor):
        # AWQ_REVERSE_ORDER = [0, 4, 1, 5, 2, 6, 3, 7]

        assert ori_tensor.shape[-1] % 8 == 0
        reorder_tensor = ori_tensor.reshape(-1, 2,4).transpose(2,1).reshape(ori_tensor.shape)

        return reorder_tensor

    def apply_int8(self, tensor: torch.Tensor, device: str):
        shape = tensor.shape
        int8_weight, int8_scale = self.exported_device.symmetric_quantize_last_axis_of_batched_matrix( # type: ignore
            tensor.reshape([shape[0], -1]).cpu(), torch.int8)
        int8_weight = int8_weight.reshape(shape)
        return int8_weight.to(device), int8_scale.to(device)

    def moe_apply_int8(self, tensor: torch.Tensor, device: str):
        assert tensor.dim() == 3
        tensor_list = torch.chunk(tensor, tensor.shape[0], dim=0)
        int8_weights = []
        int8_scales = []
        for t in tensor_list:
            t = torch.squeeze(t).transpose(1,0).contiguous()
            shape = t.shape
            weight, scale = self.exported_device.symmetric_quantize_last_axis_of_batched_matrix( # type: ignore
                t.reshape([shape[0], -1]).cpu(), torch.int8)
            int8_weights.append(weight)
            int8_scales.append(scale)
        int8_weight = torch.stack(int8_weights, dim=0)
        int8_scale = torch.stack(int8_scales, dim=0)
        return int8_weight.to(device), int8_scale.to(device)

    def preprocess_groupwise_weight_params(self, qweight_int32, qzeros_int32, scales_fp16, device: str,
                                           gptq: bool, awq: bool, weight_bits: int):
        GPTQ_FLAG = 1 if gptq == True else 0
        qweight = qweight_int32.reshape(qweight_int32.shape[0], -1).cpu()
        qzeros = qzeros_int32.reshape(qzeros_int32.shape[0], -1).cpu()
        scales_fp16 = scales_fp16.reshape(scales_fp16.shape[0], -1).cpu()
        packer = self.exported_device.pack_int8_tensor_to_packed_int4
        preprocessor = self.exported_device.preprocess_weights_for_mixed_gemm
        is_int8 = weight_bits == 8
        if is_int8:
            zero_shift = 128
            quant_type = torch.int8
        else:
            zero_shift = 8
            quant_type = torch.quint4x2

        if awq:
            qweight = self.unpack_int32_into_int16(qweight, is_int8).contiguous() - zero_shift
            qweight = self.reverse_awq_order(qweight)
        elif gptq:
            qweight = self.unpack_int32_into_int16(qweight.T, is_int8).T.contiguous() - zero_shift

        qweight = qweight.to(torch.int8)
        if not is_int8:
            qweight = packer(qweight)
        qweight_interleaved = preprocessor(qweight, quant_type)

        # zero = 0 if qzeros_int32 = -2004318072 torch.int32 for awq
        # zero = 0 if qzeros_int32 = 2004318071  torch.int32 for gptq
        qzeros = self.unpack_int32_into_int16(qzeros, is_int8)
        if awq:
            qzeros = self.reverse_awq_order(qzeros)

        # zeros = zeros * scales
        UINT_TO_INT_FLAG = 1
        zeros_x_scales_fp16 = (-qzeros + zero_shift * UINT_TO_INT_FLAG -
                               GPTQ_FLAG) * scales_fp16
        zeros_x_scales_fp16 = zeros_x_scales_fp16.half()

        # return processed interleaved weight, original scales and zeros * scales
        return qweight_interleaved.contiguous().to(device),  zeros_x_scales_fp16.contiguous().to(device), scales_fp16.contiguous().to(device)

    def preprocess_moe_groupwise_weight_params(self, qweight_int32, qzeros_int32, scales_fp16, device: str, gptq: bool, awq: bool, weight_bits: int):
        assert qweight_int32.dim() == 3

        qweight_list = torch.chunk(qweight_int32, qweight_int32.shape[0], dim=0)
        qzeros_list = torch.chunk(qzeros_int32, qzeros_int32.shape[0], dim=0)
        scales_list = torch.chunk(scales_fp16, scales_fp16.shape[0], dim=0)
        processed_weights = []
        processed_zeros = []
        processed_scalses = []
        for w, z, s in zip(qweight_list, qzeros_list, scales_list):
            w = torch.squeeze(w).transpose(1, 0).contiguous()
            z = torch.squeeze(z).transpose(1, 0).contiguous()
            s = torch.squeeze(s).transpose(1, 0).contiguous()
            p_w, p_z, p_s = self.preprocess_groupwise_weight_params(w, z, s, device, gptq, awq, weight_bits)
            processed_weights.append(p_w)
            processed_zeros.append(p_z)
            processed_scalses.append(p_s)
        processed_weights = torch.stack(processed_weights, dim=0)
        processed_zeros = torch.stack(processed_zeros, dim=0)
        processed_scalses = torch.stack(processed_scalses, dim=0)
        return processed_weights, processed_zeros, processed_scalses

class CudaImpl(GpuImpl):
    def __init__(self, exported_device: DeviceExporter):
        super().__init__(exported_device)
        import pynvml
        pynvml.nvmlInit()

    def get_mem_info(self) -> MemInfo:
        import pynvml
        handle = pynvml.nvmlDeviceGetHandleByIndex(torch.cuda._parse_visible_devices()[0])
        meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
        return MemInfo(meminfo.used, meminfo.free)

class RocmImpl(GpuImpl):
    def __init__(self, exported_device: DeviceExporter):
        super().__init__(exported_device)
        from pyrsmi import rocml
        rocml.smi_initialize()

    def get_mem_info(self) -> MemInfo:
        from pyrsmi import rocml
        id = self.get_device_id()
        used = rocml.smi_get_device_memory_used(id)
        total = rocml.smi_get_device_memory_total(id)
        return MemInfo(total - used, used)
