import logging
import os

import torch

from rtp_llm.device.device_base import DeviceBase, MemInfo
from rtp_llm.ops.compute_ops import ExecCtxExporter
from rtp_llm.utils.model_weight import W


class GpuImpl(DeviceBase):
    def __init__(self, exported_device: ExecCtxExporter):
        super().__init__(exported_device)

    def get_device_id(self) -> int:
        return torch.cuda.current_device()

    def unpack_int32_into_int16(self, w_packed: torch.Tensor, int8: bool):
        if int8:
            return w_packed.contiguous().view(torch.uint8).to(torch.int16)
        # unpack inputs packed in int32/float32 into uint4 and store them in int8 format
        w_packed_int4x2 = w_packed.contiguous().view(torch.uint8)
        w_unpacked = torch.zeros(
            w_packed_int4x2.shape[0],
            w_packed_int4x2.shape[1] * 2,
            dtype=torch.int8,
            device=w_packed_int4x2.device,
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

    def preprocess_weights_for_mixed_gemm(
        self,
        tensor: torch.Tensor,
        quant_mode: torch.dtype,
        arch: str = "",
    ) -> torch.Tensor:
        raise NotImplementedError(
            "preprocess_weights_for_mixed_gemm is not implemented"
        )

    def symmetric_quantize_last_axis_of_batched_matrix(
        self, weight, quant_mode, return_unprocessed_weight=False
    ):
        amax = weight.abs().max(dim=0)[0]
        amax = torch.clamp(amax, min=1e-8)
        if quant_mode == torch.int8:
            scale = amax / 128.0
            qweight = torch.clamp((weight / scale).round(), -128, 127).char()
        else:
            scale = amax / 8.0
            qweight = torch.clamp((weight / scale).round(), -8, 7).char()
            qweight[qweight < 0] += 16
            qweight = qweight.T.view(torch.uint8)
            qweight = (qweight[:, 1::2] * 16 + qweight[:, ::2]).view(torch.int8)
            qweight = qweight.reshape(weight.shape[0], weight.shape[1] // 2)
        processed_qweight = self.preprocess_weights_for_mixed_gemm(qweight, quant_mode)
        if return_unprocessed_weight:
            return qweight, processed_qweight, scale
        else:
            return processed_qweight, scale

    def pack_int8_tensor_to_packed_int4(self, tensor: torch.Tensor):
        assert tensor.dtype == torch.int8
        tensor -= (tensor >> 4) << 4
        tensor = tensor.view(torch.uint8)
        tensor = (tensor[:, 1::2] * 16 + tensor[:, ::2]).view(torch.int8)
        return tensor

    @property
    def specify_gpu_arch(self):
        return self.py_env_configs.runtime_config.specify_gpu_arch

    def apply_int8(self, tensor: torch.Tensor, device: str):
        shape = tensor.shape
        int8_weight, int8_scale = self.symmetric_quantize_last_axis_of_batched_matrix(  # type: ignore
            tensor.reshape([shape[0], -1]),
            torch.int8,
        )
        int8_weight = int8_weight.reshape(shape)
        return int8_weight, int8_scale

    def moe_apply_int8(self, tensor: torch.Tensor, device: str):
        assert tensor.dim() == 3
        tensor_list = torch.chunk(tensor, tensor.shape[0], dim=0)
        int8_weights = []
        int8_scales = []
        for t in tensor_list:
            t = torch.squeeze(t).transpose(1, 0).contiguous()
            shape = t.shape
            weight, scale = self.symmetric_quantize_last_axis_of_batched_matrix(  # type: ignore
                t.reshape([shape[0], -1]),
                torch.int8,
            )
            int8_weights.append(weight)
            int8_scales.append(scale)
        int8_weight = torch.stack(int8_weights, dim=0)
        int8_scale = torch.stack(int8_scales, dim=0)
        return int8_weight, int8_scale

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
            qweight = self.pack_int8_tensor_to_packed_int4(qweight)
        qweight_interleaved = self.preprocess_weights_for_mixed_gemm(
            qweight, quant_type, self.specify_gpu_arch
        )

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
    def __init__(self, exported_device: ExecCtxExporter):
        super().__init__(exported_device)
        try:
            import pynvml

            pynvml.nvmlInit()
        except Exception as e:
            logging.warn(f"no nvml found: " + str(e))

        self._cache_permute_indices: dict[torch.Size, torch.Tensor] = {}
        if self.py_env_configs.moe_config.fp4_moe_op == "auto":
            self.py_env_configs.moe_config.fp4_moe_op = "trtllm"
            if (
                self.py_env_configs.moe_config.use_deepep_moe
                and self.py_env_configs.moe_config.use_deepep_low_latency
            ):
                self.py_env_configs.moe_config.fp4_moe_op = "cutedsl"

    def _cuda_version_ge(self, major: int, minor: int) -> bool:
        try:
            if torch.version.cuda:
                cur_major, cur_minor = map(int, torch.version.cuda.split(".")[:2])
                return (cur_major, cur_minor) >= (major, minor)
        except Exception:
            pass
        return False

    # ===== Attention 优先级路由 =====

    def get_prefill_mha_priorities(self):
        from rtp_llm.models_py.modules.factory.attention.cuda_cp_impl.prefill_cp_flashinfer import (
            CPFlashInferImpl,
        )
        from rtp_llm.models_py.modules.factory.attention.cuda_headwise_impl.headwise import (
            HeadWisePrefillImpl,
        )
        from rtp_llm.models_py.modules.factory.attention.cuda_headwise_impl.headwise_fp8 import (
            HeadWiseFP8PrefillImpl,
        )
        from rtp_llm.models_py.modules.factory.attention.cuda_impl.flash_infer import (
            FlashInferPrefillImpl,
        )
        from rtp_llm.models_py.modules.factory.attention.cuda_impl.py_flashinfer_mha import (
            PyFlashinferPagedPrefillImpl,
            PyFlashinferPrefillImpl,
        )
        from rtp_llm.models_py.modules.factory.attention.cuda_impl.trt import (
            TRTMHAImpl,
            TRTPagedMHAImpl,
        )
        from rtp_llm.models_py.modules.factory.attention.cuda_impl.trtllm_gen import (
            FlashInferTRTLLMPrefillImpl,
            FlashInferTRTLLMSpecDecodeImpl,
        )

        return [
            HeadWiseFP8PrefillImpl,
            HeadWisePrefillImpl,
            FlashInferTRTLLMSpecDecodeImpl,
            FlashInferTRTLLMPrefillImpl,
            TRTMHAImpl,
            PyFlashinferPrefillImpl,
            PyFlashinferPagedPrefillImpl,
            TRTPagedMHAImpl,
            FlashInferPrefillImpl,
            PyFlashinferPrefillImpl,  # fallback
            CPFlashInferImpl,
        ]

    def get_decode_mha_priorities(self):
        from rtp_llm.models_py.modules.factory.attention.cuda_impl.flash_infer import (
            FlashInferDecodeImpl,
        )
        from rtp_llm.models_py.modules.factory.attention.cuda_impl.py_flashinfer_mha import (
            PyFlashinferDecodeImpl,
        )
        from rtp_llm.models_py.modules.factory.attention.cuda_impl.trtllm_gen import (
            FlashInferTRTLLMDecodeImpl,
        )
        from rtp_llm.models_py.modules.factory.attention.cuda_impl.xqa import XQAImpl

        return [
            FlashInferTRTLLMDecodeImpl,
            XQAImpl,
            FlashInferDecodeImpl,
            PyFlashinferDecodeImpl,  # fallback
        ]

    def get_prefill_mla_priorities(self):
        from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.flashinfer_mla_wrapper import (
            MlaFlashInferPrefillImpl,
        )

        priorities = [MlaFlashInferPrefillImpl]
        if self._cuda_version_ge(12, 9):
            from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.flashmla_sparse_cp_impl import (
                SparseMlaCpImpl,
            )
            from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.flashmla_sparse_impl import (
                SparseMlaImpl,
            )

            priorities.append(SparseMlaImpl)
            priorities.append(SparseMlaCpImpl)
        return priorities

    def get_decode_mla_priorities(self):
        from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.flashinfer_mla_wrapper import (
            MlaFlashInferDecodeImpl,
        )

        priorities = [MlaFlashInferDecodeImpl]
        if self._cuda_version_ge(12, 9):
            from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.flashmla_sparse_impl import (
                SparseMlaImpl,
            )

            priorities.append(SparseMlaImpl)
        return priorities

    # ===== Base Ops 分派 =====

    def get_base_ops(self):
        from rtp_llm.device.base_ops import BaseOps
        from rtp_llm.models_py.modules.base.cuda.activation import FusedSiluAndMul
        from rtp_llm.models_py.modules.base.cuda.indexer_op import IndexerOp
        from rtp_llm.models_py.modules.base.cuda.moe_gating import SigmoidGateScaleAdd
        from rtp_llm.models_py.modules.base.cuda.norm import (
            AddBiasResLayerNorm,
            FusedQKRMSNorm,
            QKRMSNorm,
            RMSNorm,
            RMSResNorm,
        )
        from rtp_llm.models_py.modules.base.cuda.select_topk import (
            FakeBalanceExpert,
            GroupTopK,
            SelectTopk,
        )

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
        import rtp_llm.models_py.modules.factory.linear.impl.cuda  # noqa: F401

    # ===== MoE 策略路由 =====

    def get_moe_strategy_candidates(self):
        from rtp_llm.models_py.modules.factory.fused_moe.impl.common.strategy.batched_triton_strategy import (
            BatchedTritonStrategy,
        )
        from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.strategy import (
            CudaFp8PerBlockEpLowLatencyStrategy,
            CudaFp8PerBlockEpNormalStrategy,
            CudaFp8PerBlockNoDPMaskedStrategy,
            CudaFp8PerBlockNoDPStrategy,
            CudaFp8PerTensorEpLowLatencyStrategy,
            CudaFp8PerTensorEpNormalStrategy,
            CudaFp8PerTensorNoDPStrategy,
            CudaNoQuantCppStrategy,
            CudaNoQuantDpNormalStrategy,
            CudaNoQuantEpLowLatencyStrategy,
            CudaW4a8Int4PerChannelEpLowLatencyStrategy,
            CudaW4a8Int4PerChannelEpNormalStrategy,
            CudaW4a8Int4PerChannelNoDPStrategy,
        )

        candidates = [
            CudaFp8PerTensorEpLowLatencyStrategy,
            CudaFp8PerTensorEpNormalStrategy,
            CudaFp8PerBlockEpLowLatencyStrategy,
            CudaFp8PerBlockEpNormalStrategy,
            CudaFp8PerBlockNoDPMaskedStrategy,
            CudaFp8PerBlockNoDPStrategy,
            CudaFp8PerTensorNoDPStrategy,
            CudaNoQuantEpLowLatencyStrategy,
            CudaNoQuantDpNormalStrategy,
            CudaNoQuantCppStrategy,
            BatchedTritonStrategy,
            CudaW4a8Int4PerChannelEpLowLatencyStrategy,
            CudaW4a8Int4PerChannelEpNormalStrategy,
            CudaW4a8Int4PerChannelNoDPStrategy,
        ]
        if self.supports_fp4:
            from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.strategy import (
                CudaFp4EpLowLatencyStrategy,
                CudaFp4EpNormalStrategy,
                CudaFp4NoDPStrategy,
            )

            candidates.extend(
                [
                    CudaFp4EpLowLatencyStrategy,
                    CudaFp4EpNormalStrategy,
                    CudaFp4NoDPStrategy,
                ]
            )
        return candidates

    # ===== 能力查询 =====

    @property
    def supports_fp4(self) -> bool:
        try:
            return torch.cuda.is_available() and self.arch >= 100
        except Exception:
            return False

    @property
    def supports_fp8(self) -> bool:
        try:
            return torch.cuda.is_available() and self.arch >= 89
        except Exception:
            return False

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

    def preprocess_weights_for_mixed_gemm(
        self,
        tensor: torch.Tensor,
        quant_mode: torch.dtype,
        arch: str = "",
    ) -> torch.Tensor:
        arch_int = int(arch) if arch else self.arch

        if len(tensor.shape) == 2:
            tensor = tensor.unsqueeze(0)

        num_experts = tensor.shape[0]
        num_rows = tensor.shape[1]
        num_cols = tensor.shape[2]

        # permute_B_rows_for_mixed_gemm
        BITS_PER_ELT = 4 if quant_mode == torch.quint4x2 else 8
        MMA_SHAPE_N = 8
        B_ROWS_PER_MMA = 8 * 16 // BITS_PER_ELT
        ELTS_PER_REG = 32 // BITS_PER_ELT

        # assert self.arch >= 75
        assert num_rows % B_ROWS_PER_MMA == 0
        assert num_cols % MMA_SHAPE_N == 0

        if arch_int >= 75:
            # permute_B_rows_for_mixed_gemm
            permutation_map = [
                8 * ((i % ELTS_PER_REG) // 2) + i % 2 + 2 * (i // ELTS_PER_REG)
                for i in range(B_ROWS_PER_MMA)
            ]
            row_idx_list = [
                (row_idx // B_ROWS_PER_MMA) * B_ROWS_PER_MMA
                + permutation_map[row_idx % B_ROWS_PER_MMA]
                for row_idx in range(num_rows)
            ]
            tensor = tensor[:, row_idx_list, :]

            # subbyte_transpose
            original_shape = tensor.shape
            if BITS_PER_ELT == 4:
                tensor = tensor.view(torch.uint8)
                high_tensor = (tensor >> 4).permute(0, 2, 1).unsqueeze(2)
                low_tensor = ((tensor << 4) >> 4).permute(0, 2, 1).unsqueeze(2)
                new_tensor = torch.cat([low_tensor, high_tensor], dim=2).reshape(
                    tensor.shape[0], -1, tensor.shape[1]
                )
                new_tensor = new_tensor[:, :, 0::2] + new_tensor[:, :, 1::2] * 16
                tensor = new_tensor.view(torch.int8).reshape(original_shape)
            else:
                tensor = tensor.permute(0, 2, 1).reshape(original_shape)

            # interleave_column_major_tensor\
            interleave = 16 // BITS_PER_ELT
            if interleave > 1:
                rows_per_tile = 64
                assert num_rows % ELTS_PER_REG == 0
                assert num_rows % rows_per_tile == 0
                tensor = tensor.reshape(
                    num_experts,
                    -1,
                    interleave,
                    num_rows // rows_per_tile,
                    rows_per_tile * 4 // ELTS_PER_REG,
                )
                tensor = tensor.permute(0, 1, 3, 2, 4).reshape(original_shape)

        # add_bias_and_interleave_quantized_tensor_inplace
        if BITS_PER_ELT == 8:
            tensor = (tensor.int() + 128).char()
            tensor = tensor.reshape(-1, 4)[:, [0, 2, 1, 3]].reshape(tensor.shape)
        elif BITS_PER_ELT == 4:
            tensor = tensor.view(torch.uint8)
            high_tensor = (tensor >> 4).unsqueeze(-1)
            low_tensor = ((tensor << 4) >> 4).unsqueeze(-1)
            new_tensor = torch.cat([low_tensor, high_tensor], dim=-1).reshape(
                tensor.shape[0], tensor.shape[1], -1
            )
            new_tensor = new_tensor.reshape(-1, 8)[:, [0, 2, 4, 6, 1, 3, 5, 7]].reshape(
                new_tensor.shape
            )
            new_tensor += -16 * (new_tensor > 7).byte() + 8
            new_tensor = new_tensor[:, :, 0::2] + new_tensor[:, :, 1::2] * 16
            tensor = new_tensor.view(torch.int8)
        else:
            raise NotImplementedError

        return tensor.squeeze(0).contiguous()

    @staticmethod
    def swizzle_blockscale(scale: torch.Tensor):
        """
        Swizzle the scale tensor into a blockwise interleaved format for NVFP4 quantization.
        """
        assert scale.dtype == torch.float8_e4m3fn
        # Pad and blockwise interleave weight_scale
        scale_ndim = scale.ndim
        if scale.ndim == 2:
            scale = scale.unsqueeze(0)
        assert scale.ndim == 3
        B, M, K = scale.shape
        round_up_multiple = lambda x, m: (x + m - 1) // m * m
        M_padded = round_up_multiple(M, 128)
        K_padded = round_up_multiple(K, 4)
        padded_scale = torch.zeros((B, M_padded, K_padded), dtype=scale.dtype)
        padded_scale[:B, :M, :K] = scale
        batches, rows, cols = padded_scale.shape
        assert rows % 128 == 0
        assert cols % 4 == 0
        padded_scale = padded_scale.reshape(batches, rows // 128, 4, 32, cols // 4, 4)
        swizzled_scale = padded_scale.permute((0, 1, 4, 3, 2, 5))
        swizzled_scale = swizzled_scale.contiguous().cuda()
        return (
            swizzled_scale.reshape(M_padded, K_padded)
            if scale_ndim == 2
            else swizzled_scale.reshape(B, M_padded, K_padded)
        )

    @staticmethod
    def convert_fp4_gemm_weight_params(
        weight: torch.Tensor, weight_scale: torch.Tensor
    ):
        backend = os.getenv("RTP_LLM_FP4_GEMM_BACKEND", "cutlass")
        if backend == "trtllm":
            from flashinfer import shuffle_matrix_a, shuffle_matrix_sf_a

            epilogue_tile_m = 128
            processed_weight = shuffle_matrix_a(
                weight.view(torch.uint8), epilogue_tile_m
            )
            processed_weight_scale = (
                shuffle_matrix_sf_a(weight_scale.view(torch.uint8), epilogue_tile_m)
                .reshape(weight_scale.shape)
                .view(torch.float8_e4m3fn)
            )
        else:
            # Pad and blockwise interleave weight_scales
            scales = weight_scale
            scale_ndim = scales.ndim
            if scale_ndim == 2:
                scales = scales.unsqueeze(0)
            assert scales.ndim == 3
            B, M, K = scales.shape
            round_up_multiple = lambda x, m: (x + m - 1) // m * m
            M_padded = round_up_multiple(M, 128)
            K_padded = round_up_multiple(K, 4)
            padded_scales = torch.zeros((B, M_padded, K_padded), dtype=scales.dtype)
            padded_scales[:B, :M, :K] = scales
            batches, rows, cols = padded_scales.shape
            assert rows % 128 == 0
            assert cols % 4 == 0
            padded_scales = padded_scales.reshape(
                batches, rows // 128, 4, 32, cols // 4, 4
            )
            padded_scales = padded_scales.permute((0, 1, 4, 3, 2, 5))
            padded_scales = padded_scales.contiguous().cuda()
            padded_scales = (
                padded_scales.reshape(M_padded, K_padded)
                if scale_ndim == 2
                else padded_scales.reshape(B, M_padded, K_padded)
            )
            processed_weight = weight
            processed_weight_scale = padded_scales
        return [processed_weight, processed_weight_scale]

    @staticmethod
    def prepare_static_weights_for_trtllm_fp4_moe(
        weight,
        scale,
        shape,
        findices,
        cache,
    ):
        from flashinfer.fused_moe.core import nvfp4_block_scale_interleave

        epilogue_tile_m = 128  # FIXME: this depends on the kernel internals

        weight_fp4 = weight.view(torch.float8_e4m3fn).reshape(
            *shape[:-1], shape[-1] // 2
        )  # packed fp4
        scale_linear_fp4 = scale.view(torch.float8_e4m3fn).reshape(
            *shape[:-1], shape[-1] // 16
        )  # fp8 scaling factors

        weight_fp4_shuffled = []
        scale_fp4_shuffled = []
        for i in range(shape[0]):
            permute_indices = findices(
                cache, weight_fp4[i].view(torch.uint8), epilogue_tile_m
            )
            weight_fp4_shuffled.append(
                weight_fp4[i]
                .view(torch.uint8)[permute_indices.to(weight_fp4.device)]
                .contiguous()
            )

            permute_sf_indices = findices(
                cache,
                scale_linear_fp4[i].view(torch.uint8),
                epilogue_tile_m,
                num_elts_per_sf=16,
            )
            scale_fp4_shuffled.append(
                nvfp4_block_scale_interleave(
                    scale_linear_fp4[i]
                    .view(torch.uint8)[permute_sf_indices.to(scale_linear_fp4.device)]
                    .contiguous()
                )
            )

        weight_fp4_shuffled = torch.stack(weight_fp4_shuffled)
        scale_fp4_shuffled = (
            torch.stack(scale_fp4_shuffled)
            .view(torch.float8_e4m3fn)
            .reshape(*shape[:-1], shape[-1] // 16)
        )
        return weight_fp4_shuffled, scale_fp4_shuffled

    def maybe_prepare_static_weights_for_fp4_moe(
        self,
        kernel_name: str,
        scale_name: str,
        kernel: torch.Tensor,
        scale: torch.Tensor,
    ):
        if kernel_name not in [W.moe_w2, W.moe_w1]:
            return kernel, scale

        if self.py_env_configs.moe_config.fp4_moe_op == "cutedsl":
            # cutedsl moe needs gate+up format for w13
            if kernel_name == W.moe_w1:
                kernel = torch.cat(
                    [
                        kernel[:, kernel.shape[1] // 2 :, :],
                        kernel[:, : kernel.shape[1] // 2, :],
                    ],
                    dim=1,
                )
                scale = torch.cat(
                    [
                        scale[:, scale.shape[1] // 2 :, :],
                        scale[:, : scale.shape[1] // 2, :],
                    ],
                    dim=1,
                )
            swizzled_scale = self.swizzle_blockscale(scale)
            return kernel, swizzled_scale

        if self.py_env_configs.moe_config.fp4_moe_op != "trtllm":
            return kernel, scale

        from flashinfer.fused_moe.core import (
            _maybe_get_cached_w3_w1_permute_indices,
            get_w2_permute_indices_with_cache,
        )

        return CudaImpl.prepare_static_weights_for_trtllm_fp4_moe(
            kernel,
            scale,
            [*kernel.shape[:-1], kernel.shape[-1] * 2],
            (
                _maybe_get_cached_w3_w1_permute_indices
                if kernel_name == W.moe_w1
                else get_w2_permute_indices_with_cache
            ),
            self._cache_permute_indices,
        )


class PpuImpl(CudaImpl):
    @property
    def support_dio_load(self) -> bool:
        return False
