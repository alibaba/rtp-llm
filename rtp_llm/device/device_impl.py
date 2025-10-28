import logging
import re

import psutil
import torch

from rtp_llm.device.device_base import DeviceBase, MemInfo
from rtp_llm.ops import DeviceExporter
from rtp_llm.utils.model_weight import W
from rtp_llm.utils.swizzle_utils import swizzle_tensor

class CpuImpl(DeviceBase):
    def __init__(self, exported_device: DeviceExporter):
        super().__init__(exported_device)

    def _get_mem_info(self) -> MemInfo:
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
        packer = self.exported_device.pack_int8_tensor_to_packed_int4
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
        w_unpacked = torch.zeros(
            w_packed_int4x2.shape[0], w_packed_int4x2.shape[1] * 2, dtype=torch.int8
        )
        w_unpacked[:, ::2] = w_packed_int4x2 % 16
        w_unpacked[:, 1::2] = w_packed_int4x2 // 16
        return w_unpacked.to(torch.int16).contiguous()

    def reverse_awq_order(self, ori_tensor: torch.Tensor):
        # AWQ_REVERSE_ORDER = [0, 4, 1, 5, 2, 6, 3, 7]

        assert ori_tensor.shape[-1] % 8 == 0
        reorder_tensor = (
            ori_tensor.reshape(-1, 2, 4).transpose(2, 1).reshape(ori_tensor.shape)
        )

        return reorder_tensor

    @property
    def specify_gpu_arch(self):
        return self.py_env_configs.py_device_resource_config.specify_gpu_arch

    def apply_int8(self, tensor: torch.Tensor, device: str):
        shape = tensor.shape
        int8_weight, int8_scale = self.exported_device.symmetric_quantize_last_axis_of_batched_matrix(  # type: ignore
            tensor.reshape([shape[0], -1]).cpu(), torch.int8, self.specify_gpu_arch
        )
        int8_weight = int8_weight.reshape(shape)
        return int8_weight.to(device), int8_scale.to(device)

    def moe_apply_int8(self, tensor: torch.Tensor, device: str):
        assert tensor.dim() == 3
        tensor_list = torch.chunk(tensor, tensor.shape[0], dim=0)
        int8_weights = []
        int8_scales = []
        for t in tensor_list:
            t = torch.squeeze(t).transpose(1, 0).contiguous()
            shape = t.shape
            weight, scale = self.exported_device.symmetric_quantize_last_axis_of_batched_matrix(  # type: ignore
                t.reshape([shape[0], -1]).cpu(), torch.int8, self.specify_gpu_arch
            )
            int8_weights.append(weight)
            int8_scales.append(scale)
        int8_weight = torch.stack(int8_weights, dim=0)
        int8_scale = torch.stack(int8_scales, dim=0)
        return int8_weight.to(device), int8_scale.to(device)

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

        # return processed interleaved weight, original scales and zeros * scales
        return (
            qweight_interleaved.contiguous().to(device),
            zeros_x_scales_fp16.contiguous().to(device),
            scales_fp16.contiguous().to(device),
        )

    def preprocess_moe_groupwise_weight_params(
        self,
        qweight_int32,
        qzeros_int32,
        scales_fp16,
        device: str,
        gptq: bool,
        awq: bool,
        weight_bits: int,
    ):
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
            p_w, p_z, p_s = self.preprocess_groupwise_weight_params(
                w, z, s, device, gptq, awq, weight_bits
            )
            processed_weights.append(p_w)
            processed_zeros.append(p_z)
            processed_scalses.append(p_s)
        processed_weights = torch.stack(processed_weights, dim=0)
        processed_zeros = torch.stack(processed_zeros, dim=0)
        processed_scalses = torch.stack(processed_scalses, dim=0)
        return processed_weights, processed_zeros, processed_scalses

    def shuffle_moe_weight(
        self, x: torch.Tensor, datatype: torch.dtype, name: str
    ) -> torch.Tensor:
        return x

    def shuffle_gemm_weight(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def convert_fp8_weight_params(
        self, weight: torch.Tensor, weight_scale: torch.Tensor
    ):
        return [weight, weight_scale]


class CudaImpl(GpuImpl):
    def __init__(self, exported_device: DeviceExporter):
        super().__init__(exported_device)
        try:
            import pynvml

            pynvml.nvmlInit()
        except Exception as e:
            logging.warn(f"no nvml found: " + str(e))

    def _get_mem_info(self) -> MemInfo:
        import pynvml

        handle = pynvml.nvmlDeviceGetHandleByIndex(
            torch.cuda._parse_visible_devices()[0]
        )
        meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
        return MemInfo(meminfo.used, meminfo.free)

    @property
    def arch(self) -> int:
        try:
            device = self.get_device_id()
            major, minor = torch.cuda.get_device_capability(device)
            arch = major * 10 + minor
            return arch
        except Exception as e:
            logging.warn(f"Cannot get CUDA device capability: {e}")
            return super().arch  # 使用父类的实现

    @property
    def support_dio_load(self) -> bool:
        return True


class PpuImpl(CudaImpl):
    @property
    def support_dio_load(self) -> bool:
        return False


class RocmImpl(GpuImpl):
    def __init__(self, exported_device: DeviceExporter):
        super().__init__(exported_device)
        try:
            from pyrsmi import rocml

            rocml.smi_initialize()
        except Exception as e:
            logging.warn(f"no rocm smi found: " + str(e))

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
        specify_gpu_arch = (
            self.py_env_configs.py_device_resource_config.specify_gpu_arch
        )
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
        specify_gpu_arch = (
            self.py_env_configs.py_device_resource_config.specify_gpu_arch
        )
        return "900" if specify_gpu_arch == "" else specify_gpu_arch

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
        x_ = (
            torch.cat([x[:, x.shape[1] // 2 :, :], x[:, : x.shape[1] // 2, :]], dim=1)
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

        if key in [W.attn_qkv_w, W.attn_o_w, W.ffn_w2, W.ffn_w13]:
            if self.py_env_configs.py_hw_kernel_config.use_swizzleA:
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
