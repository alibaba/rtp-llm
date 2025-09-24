#include "rtp_llm/cpp/devices/rocm_impl/ROCmDevice.h"
#include "rtp_llm/cpp/core/Dispatch.h"
#include "rtp_llm/cpp/rocm/quantizePreprocessors.h"
#include "rtp_llm/cpp/kernels/rocm/quantization_rocm.h"
#include "rtp_llm/cpp/core/torch_utils/BufferTorchUtils.h"
#include "quant.h"
// #include "aiter_meta/csrc/include/quant.h"

namespace rtp_llm {
using namespace rocm;

BufferPtr ROCmDevice::quantize(const QuantizeParams& params) {
    ROCM_CHECK_VALUE((params.input.dim() == 2), "quantize only support 2D.");
    ROCM_CHECK_VALUE((params.input.type() == DataType::TYPE_FP16 || params.input.type() == DataType::TYPE_FP32
                      || params.input.type() == DataType::TYPE_BF16),
                     "quantize only support half or float quantize. but get %d.",
                     params.input.type());

    BufferPtr kernel, scales, zeros;
    if (params.input.where() == MemoryType::MEMORY_GPU) {
        if (params.qtype == DataType::TYPE_QINT4X2) {
            ROCM_CHECK_VALUE((params.input.dim() == 2), "quantize only support 2D input.");
            size_t groupSize   = params.groupSize;
            size_t scales_dim0 = params.input.shape()[0] / groupSize;

            kernel = allocateBuffer({QBufferDtype2BufferDtype(params.qtype),
                                     params.input.shape(),
                                     getMemAllocationType(params.input.where())},
                                    {"kernel"});
            scales = allocateBuffer({DataType::TYPE_FP16,
                                     {scales_dim0, params.input.shape()[1]},
                                     getMemAllocationType(params.input.where())},
                                    {"scales"});
            zeros  = allocateBuffer({DataType::TYPE_FP16,
                                     {scales_dim0, params.input.shape()[1]},
                                     getMemAllocationType(params.input.where())},
                                    {"zeros"});
            DISPATCH_CUDA_FUNCTION_DATA_TYPE(params.input.type(),
                                             invokePerColQuantizationInt4x2,
                                             params.input.data(),
                                             params.input.shape()[0],
                                             params.input.shape()[1],
                                             groupSize,
                                             (uint8_t*)(kernel->data()),
                                             scales->data<half>(),
                                             zeros->data<half>(),
                                             stream_);
        } else if (params.qscheme == QScheme::Qfp8PerToken) {
            ROCM_CHECK_VALUE((params.qtype == DataType::TYPE_QFP8_E4M3),
                             "Qfp8PerToken only support qtype = TYPE_QFP8_E4M3");
            ROCM_CHECK_VALUE((params.axis == 1), "Qfp8PerToken only support axis = 1");
            size_t num_token = params.input.shape()[0];
            size_t model_dim = params.input.shape()[1];
            kernel           = allocateBuffer({DataType::TYPE_FP8_E4M3, params.input.shape()}, {"quant_kernel"});
            scales           = allocateBuffer({DataType::TYPE_FP32, {num_token, 1}}, {"quant_scale"});
            zeros            = BufferPtr(new Buffer(MemoryType::MEMORY_GPU, DataType::TYPE_INVALID, {0}, nullptr));
            if (num_token > 0) {
                torch::Tensor input_tensor  = Buffer2torchTensor(params.input, false);
                torch::Tensor kernel_tensor = Buffer2torchTensor(kernel, false);
                torch::Tensor scales_tensor = Buffer2torchTensor(scales, false);

                // invoke aiter quant kernel
                aiter::dynamic_per_token_scaled_quant(
                    /*out=*/kernel_tensor,
                    /*input=*/input_tensor,
                    /*scales=*/scales_tensor,
                    /*scale_ub=*/std::nullopt);
            }
        } else if (params.qscheme == QScheme::Qfp8PerTokenBlock) {
            ROCM_CHECK_VALUE((params.groupSize == 32 || params.groupSize == 64 || params.groupSize == 128),
                             "Qfp8PerTokenBlock only support groupSize = 32,64 or 128");
            ROCM_CHECK_VALUE((params.qtype == DataType::TYPE_QFP8_E4M3),
                             "Qfp8PerTokenBlock only support qtype = TYPE_QFP8_E4M3");
            ROCM_CHECK_VALUE((params.axis == 1), "Qfp8PerTokenBlock only support axis = 1");
            size_t num_token     = params.input.shape()[0];
            size_t model_dim     = params.input.shape()[1];
            size_t block_scale_k = params.groupSize;
            kernel               = allocateBuffer({DataType::TYPE_FP8_E4M3, params.input.shape()}, {"quant_kernel"});
            scales = allocateBuffer({DataType::TYPE_FP32, {num_token, model_dim / block_scale_k}}, {"quant_scale"});
            zeros  = BufferPtr(new Buffer(MemoryType::MEMORY_GPU, DataType::TYPE_INVALID, {0}, nullptr));
            if (num_token > 0) {
                torch::Tensor input_tensor  = Buffer2torchTensor(params.input, false);
                torch::Tensor kernel_tensor = Buffer2torchTensor(kernel, false);
                torch::Tensor scales_tensor = Buffer2torchTensor(scales, false);

                input_tensor =
                    input_tensor.view({(int)num_token, (int)model_dim / (int)block_scale_k, (int)block_scale_k});

                // invoke aiter quant kernel
                aiter::dynamic_per_token_scaled_quant(
                    /*out=*/kernel_tensor,
                    /*input=*/input_tensor,
                    /*scales=*/scales_tensor,
                    /*scale_ub=*/std::nullopt);
            }
        } else {
            ROCM_FAIL("other quantize not implemented");
        }

        ROCM_CHECK_ERROR();
        return BufferPtr(new QBuffer(std::move(kernel), std::move(scales), std::move(zeros)));
    } else {
        ROCM_FAIL("cpu quantize not implemented");
    }
}

BufferPtr ROCmDevice::dequantize(const QuantizeParams& params) {
    if (params.input.where() == MemoryType::MEMORY_GPU) {
        if (params.qtype == DataType::TYPE_QINT4X2) {
            const QBuffer& QB          = reinterpret_cast<const QBuffer&>(params.input);
            size_t         kernel_dim0 = QB.kernel().shape()[0];
            size_t         scales_dim0 = QB.scales().shape()[0];
            size_t         group_size  = (kernel_dim0 / scales_dim0);

            BufferPtr fpB =
                allocateBuffer({QB.scales().type(), {QB.kernel().shape()}, AllocationType::DEVICE}, {"fpB"});

            DISPATCH_CUDA_FUNCTION_DATA_TYPE(fpB.get()->type(),
                                             invokePerColDequantizationInt4x2,
                                             fpB.get()->data(),
                                             (size_t)(QB.kernel().shape()[0]),
                                             (size_t)(QB.kernel().shape()[1]),
                                             group_size,
                                             (int8_t*)(QB.kernel().data()),
                                             QB.scales().data<half>(),
                                             QB.zeros().data<half>(),
                                             stream_);

            ROCM_CHECK_ERROR();
            return fpB;
        } else {
            ROCM_FAIL("other dequantize not implemented");
        }
    } else {
        ROCM_FAIL("cpu dequantize not implemented");
    }
}

}  // namespace rtp_llm
