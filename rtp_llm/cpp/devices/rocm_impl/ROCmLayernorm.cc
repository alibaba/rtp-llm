#include "norm.h"
#include "rmsnorm.h"
#include "rtp_llm/cpp/devices/rocm_impl/ROCmDevice.h"
#include "rtp_llm/cpp/devices/rocm_impl/ROCmAllocator.h"
#include "rtp_llm/cpp/core/TrackerAllocator.h"
#include "rtp_llm/cpp/devices/DeviceFactory.h"
#include "rtp_llm/cpp/kernels/add_residual_kernels.h"
#include "rtp_llm/cpp/devices/ShapeCheck.h"
#include "rtp_llm/cpp/core/Dispatch.h"
#include <cstring>

#include "rtp_llm/cpp/kernels/rmsnormKernels.h"
#include "rtp_llm/cpp/kernels/activation_kernels.h"

// layerNorm
#include "rtp_llm/cpp/kernels/rocm/layernorm_kernels.h"
#include "rtp_llm/cpp/kernels/add_residual_kernels.h"
#include "rtp_llm/cpp/kernels/rmsnormKernels.h"
#include "rtp_llm/cpp/kernels/rocm/fused_qk_rmsnorm.h"
#include "rtp_llm/cpp/core/torch_utils/BufferTorchUtils.h"

namespace rtp_llm {
using namespace rocm;

LayernormOutput ROCmDevice::layernormWithStride(const LayernormWithStrideParams& params) {
    RTP_LLM_CHECK_WITH_INFO(params.qscheme == QScheme::NoQuantize, "qscheme must be NoQuantize in layernormWithStride");
    const auto  data_type   = params.input->type();
    const auto  m           = params.input->shape()[0];
    const auto  in_stride   = params.input->shape()[1];
    const auto  norm_weight = params.norm_weight;
    const auto& gamma       = norm_weight ? norm_weight->get().gamma.get()->data() : nullptr;
    const auto& beta = (norm_weight && norm_weight->get().beta) ? norm_weight->get().beta.get()->data() : nullptr;
    const auto  eps  = params.eps;

    int       out_stride;
    int       out_offset;
    BufferPtr norm_output;

    // if not in_place, we hope that the output is contiguous
    if (params.in_place) {
        norm_output = params.input;
        out_stride  = in_stride;
        out_offset  = params.offset;
    } else {
        norm_output = allocateBuffer({data_type, {m, params.norm_group_size}, AllocationType::DEVICE},
                                     {"norm_with_stride_output"});
        out_stride  = params.norm_group_size;
        out_offset  = 0;
    }

    if (params.norm_type == NormType::layernorm) {
        DISPATCH_CUDA_FUNCTION_DATA_TYPE(data_type,
                                         invokeLayerNormWithStride,
                                         norm_output->dataWithOffset(out_offset),
                                         out_stride,
                                         params.input->dataWithOffset(params.offset),
                                         in_stride,
                                         gamma,
                                         beta,
                                         eps,
                                         m,
                                         params.norm_group_size,
                                         norm_weight->get().gamma.get()->shape()[0],
                                         stream_);
        ROCM_CHECK_ERROR();
        return LayernormOutput({norm_output, nullptr});
    } else if (params.norm_type == NormType::rmsnorm) {
        DISPATCH_CUDA_FUNCTION_DATA_TYPE(data_type,
                                         invokeRmsNormWithStride,
                                         norm_output->dataWithOffset(out_offset),
                                         out_stride,
                                         params.input->dataWithOffset(params.offset),
                                         in_stride,
                                         gamma,
                                         beta,
                                         eps,
                                         m,
                                         params.norm_group_size,
                                         norm_weight->get().gamma.get()->shape()[0],
                                         stream_);
        ROCM_CHECK_ERROR();
        return LayernormOutput({norm_output, nullptr});
    } else {
        throw std::runtime_error(autil::StringUtil::formatString(
            "unsupported layernorm type for layernormWithStride: %d", int(params.norm_type)));
    }
}

QkRmsNormOutput ROCmDevice::qkRmsNorm(const QkRmsNormParams& params) {
    const auto  data_type = params.input->type();
    const auto  m         = params.input->shape()[0];
    const auto  n         = params.input->shape()[1];
    const auto& q_gamma   = params.q_norm_weight->get().gamma.get()->data();
    const auto& q_beta    = (params.q_norm_weight && params.q_norm_weight->get().beta) ?
                                params.q_norm_weight->get().beta.get()->data() :
                                nullptr;
    const auto& k_gamma   = params.k_norm_weight->get().gamma.get()->data();
    const auto& k_beta    = (params.k_norm_weight && params.k_norm_weight->get().beta) ?
                                params.k_norm_weight->get().beta.get()->data() :
                                nullptr;
    RTP_LLM_CHECK_WITH_INFO((q_beta != nullptr && k_beta != nullptr) || (q_beta == nullptr && k_beta == nullptr),
                            "q_gamma and k_gamma should both nullptr or not nullptr");
    const auto eps         = params.eps;
    const auto q_group_num = params.q_group_num;
    const auto k_group_num = params.k_group_num;
    const auto norm_size   = params.norm_size;

    DISPATCH_CUDA_FUNCTION_DATA_TYPE(data_type,
                                     invokeFusedQkRmsNorm,
                                     params.input->data(),
                                     q_gamma,
                                     q_beta,
                                     k_gamma,
                                     k_beta,
                                     eps,
                                     q_group_num,
                                     k_group_num,
                                     m,
                                     n,
                                     norm_size,
                                     stream_);
    return params.input;
}

LayernormOutput ROCmDevice::layernorm(const LayernormParams& params) {
    BufferPtr   input        = params.input;
    BufferPtr   norm_output  = input;
    BufferPtr   output       = params.before_norm_output;
    float*      scales_ptr   = nullptr;
    int8_t*     quant_output = nullptr;
    const auto  data_type    = input->type();
    const auto  m            = input->shape()[0];
    const auto  n            = input->shape()[1];
    auto        norm_weight  = params.norm_weight;
    const auto& gamma        = norm_weight ? norm_weight->get().gamma.get()->data() : nullptr;
    const auto& beta = (norm_weight && norm_weight->get().beta) ? norm_weight->get().beta.get()->data() : nullptr;
    const auto& static_scale = (norm_weight && norm_weight->get().static_scale) ?
                                   norm_weight->get().static_scale.get()->data<float>() :
                                   nullptr;
    const auto  norm_type    = params.norm_type;
    const auto  eps          = params.eps;
    const auto& weights      = params.norm_weight;

    if ((!params.is_inplace && (params.qscheme == QScheme::NoQuantize || params.qscheme == QScheme::Qfp8PerTokenBlock))
        || (params.qscheme == QScheme::Qfp8PerToken && params.norm_type == NormType::layernorm)
        || params.qscheme == QScheme::Qint8PerTensor) {
        norm_output = allocateBufferLike(*params.input);
    } else if (params.qscheme == QScheme::Qint8PerToken) {
        auto kernel  = allocateBuffer({DataType::TYPE_INT8, {input->shape()}, AllocationType::DEVICE}, {"kernel"});
        auto scales  = allocateBuffer({DataType::TYPE_FP32, {input->shape()[1]}, AllocationType::DEVICE}, {"scales"});
        norm_output  = BufferPtr(new QBuffer(
            std::move(kernel),
            std::move(scales),
            std::move(BufferPtr(new Buffer(MemoryType::MEMORY_GPU, DataType::TYPE_INVALID, {0}, nullptr)))));
        quant_output = std::dynamic_pointer_cast<QBuffer>(norm_output)->kernel().data<int8_t>();
        scales_ptr   = std::dynamic_pointer_cast<QBuffer>(norm_output)->scalesData<float>();
    } else if (params.qscheme == QScheme::Qfp8PerToken && params.norm_type == NormType::rmsnorm) {
        auto scale_shape   = params.input->shape();
        scale_shape.back() = 1;
        auto kernel = allocateBuffer({DataType::TYPE_FP8_E4M3, {input->shape()}, AllocationType::DEVICE}, {"kernel"});
        auto scales = allocateBuffer({DataType::TYPE_FP32, {scale_shape}, AllocationType::DEVICE}, {"scales"});
        norm_output = BufferPtr(new QBuffer(
            std::move(kernel),
            std::move(scales),
            std::move(BufferPtr(new Buffer(MemoryType::MEMORY_GPU, DataType::TYPE_INVALID, {0}, nullptr)))));
    } else if (params.qscheme == QScheme::Qfp8PerTensor) {
        RTP_LLM_LOG_ERROR("Qfp8PerTensor not implemented!!!");
        throw OpException(OpErrorType::ERROR_UNIMPLEMENTED);
    }

    if (params.norm_type == NormType::alphanorm || !norm_weight.has_value()) {
        if (params.alpha == 0.f || params.bias.has_value() || params.residual1.has_value()
            || params.residual2.has_value()) {
            DISPATCH_CUDA_FUNCTION_DATA_TYPE(data_type,
                                             invokeAddBiasResidual,
                                             norm_output->data(),
                                             input->data(),
                                             params.residual1 ? params.residual1.value().get().data() : nullptr,
                                             params.residual2 ? params.residual2.value().get().data() : nullptr,
                                             params.bias.has_value() ? params.bias.value().get().data() : nullptr,
                                             nullptr,  // scale_inter
                                             nullptr,  // scale_out
                                             m,
                                             n,
                                             stream_);
            ROCM_CHECK_ERROR();
            return LayernormOutput({std::move(norm_output), nullptr});
        } else if (params.alpha != 0.f) {
            DISPATCH_CUDA_FUNCTION_DATA_TYPE(data_type,
                                             invokeAddBiasResidual,
                                             output->data(),
                                             input->data(),
                                             params.residual1 ? params.residual1.value().get().data() : nullptr,
                                             params.residual2 ? params.residual2.value().get().data() : nullptr,
                                             params.bias.has_value() ? params.bias.value().get().data() : nullptr,
                                             nullptr,  // scale_inter
                                             nullptr,  // scale_out
                                             m,
                                             n,
                                             stream_);
            ROCM_CHECK_ERROR();
            return LayernormOutput({std::move(norm_output), nullptr});
        } else {
            throw OpException(OpErrorType::ERROR_UNIMPLEMENTED);
        }
    }

    if ((params.qscheme == QScheme::NoQuantize || params.qscheme == QScheme::Qfp8PerToken
         || params.qscheme == QScheme::Qint8PerTensor)
        && data_type != DataType::TYPE_FP32
        && ((params.norm_type == NormType::layernorm && beta) || (params.norm_type == NormType::rmsnorm))) {
        int fused_add = params.residual1 ? 1 : 0;
        int xbias     = params.bias ? 1 : 0;

        auto input_tensor  = Buffer2torchTensor(input, false);
        auto weight_tensor = Buffer2torchTensor(*norm_weight->get().gamma.get(), false);

        if (params.norm_type == NormType::layernorm) {
            auto                         out_tensor = Buffer2torchTensor(norm_output, false);
            std::optional<torch::Tensor> bias_tensor;
            if (xbias)
                bias_tensor = Buffer2torchTensor(params.bias.value().get(), false);

            auto beta_tensor = Buffer2torchTensor(*norm_weight->get().beta.get(), false);
            if (fused_add) {
                auto residual_in_tensor = Buffer2torchTensor(params.residual1.value().get(), false);
                if (params.is_inplace && !params.return_normed_output) {
                    RTP_LLM_CHECK_WITH_INFO(params.before_norm_output != norm_output,
                                            "input, output before/after norm cannot be the same");
                }
                auto residual_out_tensor = Buffer2torchTensor(
                    (params.before_norm_output == nullptr) ? params.input : params.before_norm_output, false);
                layernorm2d_with_add(out_tensor,
                                     input_tensor,
                                     residual_in_tensor,
                                     residual_out_tensor,
                                     weight_tensor,
                                     beta_tensor,
                                     static_cast<double>(eps),
                                     bias_tensor);
                if (params.return_normed_output) {
                    auto residual_out_buffer = torchTensor2Buffer(residual_out_tensor);
                    norm_output->swap(*residual_out_buffer);
                }
            } else {
                auto res_tensor =
                    layernorm2d(input_tensor, weight_tensor, beta_tensor, static_cast<double>(eps), bias_tensor);
                auto res_buffer = torchTensor2Buffer(res_tensor);
                res_buffer->swap(*norm_output);
                if (params.return_normed_output) {
                    auto res_buffer = torchTensor2Buffer(res_tensor);
                    res_buffer->swap(*params.before_norm_output);
                }
            }
        } else if (params.norm_type == NormType::rmsnorm) {
            if (params.qscheme == QScheme::Qfp8PerToken /* Do fuse fp8 pertoken*/) {
                auto qout              = std::dynamic_pointer_cast<QBuffer>(norm_output);
                auto out_kernel_tensor = Buffer2torchTensor(qout->kernelPtr(), /*copyData=*/false);  // [m,n], FP8/Byte
                auto out_scale_tensor  = Buffer2torchTensor(qout->scalesPtr(), /*copyData=*/false);  // [m,1], FP32

                if (fused_add) {
                    auto residual_in_tensor  = Buffer2torchTensor(params.residual1.value().get(), false);
                    auto residual_out_tensor = Buffer2torchTensor(
                        (params.before_norm_output == nullptr) ? params.input : params.before_norm_output, false);
                    rmsnorm2d_with_add_dynamicquant(
                        /*out=*/out_kernel_tensor,
                        /*input=*/input_tensor,
                        /*residual_in=*/residual_in_tensor,
                        /*residual_out=*/residual_out_tensor,
                        /*yscale=*/out_scale_tensor,
                        /*weight=*/weight_tensor,
                        /*epsilon=*/static_cast<double>(eps),
                        /*use_model_sensitive_rmsnorm=*/0);
                } else {
                    rmsnorm2d_with_dynamicquant(
                        /*out=*/out_kernel_tensor,
                        /*input=*/input_tensor,
                        /*yscale=*/out_scale_tensor,
                        /*weight=*/weight_tensor,
                        /*epsilon=*/static_cast<double>(eps),
                        /*use_model_sensitive_rmsnorm=*/0);
                }
            } else {
                auto out_tensor = Buffer2torchTensor(norm_output, false);
                if (fused_add) {
                    auto residual_in_tensor  = Buffer2torchTensor(params.residual1.value().get(), false);
                    auto residual_out_tensor = Buffer2torchTensor(
                        (params.before_norm_output == nullptr) ? params.input : params.before_norm_output, false);
                    rmsnorm2d_with_add(out_tensor,
                                       input_tensor,
                                       residual_in_tensor,
                                       residual_out_tensor,
                                       weight_tensor,
                                       static_cast<double>(eps),
                                       0);
                } else {
                    auto res_tensor = rmsnorm2d(input_tensor, weight_tensor, static_cast<double>(eps), 0);
                    auto res_buffer = torchTensor2Buffer(res_tensor);
                    res_buffer->swap(*norm_output);
                }
            }
        } else {
            throw OpException(OpErrorType::ERROR_UNIMPLEMENTED);
        }
    } else {
        auto quant_data_type =
            (params.qscheme == QScheme::Qfp8PerTensor) ? DataType::TYPE_FP8_E4M3 : DataType::TYPE_INT8;
        if (params.norm_type == NormType::layernorm) {
            if (params.residual1.has_value() || params.bias.has_value()) {
                DISPATCH_CUDA_FUNCTION_DATA_TYPE(
                    data_type,
                    invokeGeneralAddBiasResidualLayerNorm,
                    // add_bias_output.data(),
                    (params.before_norm_output == nullptr) ? input->data() : params.before_norm_output->data(),
                    norm_output->data(),
                    input->data(),
                    params.bias ? params.bias.value().get().data() : nullptr,
                    params.residual1 ? params.residual1.value().get().data() : nullptr,
                    gamma,
                    beta,
                    eps,
                    m,
                    n,
                    stream_,
                    true,          // use_diff_of_squares
                    nullptr,       // scale
                    scales_ptr,    // dynamic_scale
                    quant_output,  // out_quant
                    params.return_normed_output);
            } else {
                DISPATCH_CUDA_FUNCTION_DATA_TYPE(
                    data_type,
                    invokeGeneralLayerNorm,
                    params.before_norm_output == nullptr ? nullptr : params.before_norm_output->data(),
                    norm_output->data(),
                    input->data(),
                    gamma,
                    beta,
                    eps,
                    m,
                    n,
                    stream_,
                    true,          // use_diff_of_squares
                    nullptr,       // scale
                    scales_ptr,    // dynamic_scale
                    quant_output,  // out_quant
                    params.return_normed_output);
            }
        } else if (params.norm_type == NormType::rmsnorm) {
            if (params.residual1.has_value() || params.bias.has_value()) {
                DISPATCH_CUDA_FUNCTION_COMPUTE_QUANT_TYPES(
                    data_type,
                    quant_data_type,
                    invokeAddBiasResidualRmsNorm,
                    params.before_norm_output->data(),  // or null
                    norm_output->data(),
                    input->data(),
                    params.bias ? params.bias.value().get().data() : nullptr,
                    params.residual1 ? params.residual1.value().get().data() : nullptr,
                    params.residual2 ? params.residual2.value().get().data() : nullptr,
                    gamma,
                    beta,
                    eps,
                    m,
                    n,
                    stream_,
                    nullptr,        // scale
                    scales_ptr,     // dynamic_scale
                    quant_output);  // out_quant
            } else {
                DISPATCH_CUDA_FUNCTION_COMPUTE_QUANT_TYPES(data_type,
                                                           quant_data_type,
                                                           invokeGeneralRmsNorm,
                                                           norm_output->data(),
                                                           input->data(),
                                                           gamma,
                                                           beta,
                                                           eps,
                                                           m,
                                                           n,
                                                           stream_,
                                                           nullptr,        // scale
                                                           scales_ptr,     // dynamic_scale
                                                           quant_output);  // out_quant
            }
        } else {
            throw OpException(OpErrorType::ERROR_UNIMPLEMENTED);
        }
    }
    ROCM_CHECK_ERROR();
    return LayernormOutput({norm_output, params.before_norm_output});
}

#define ARGS_DISPATCH(Atype, Dtype, out, bias, gate, gate_bias, m, n, stream)                                          \
    do {                                                                                                               \
        invokeGenericActivation<Atype>((Dtype*)out,                                                                    \
                                       (const Dtype*)bias,                                                             \
                                       (const Dtype*)gate,                                                             \
                                       (const Dtype*)gate_bias,                                                        \
                                       (const int*)nullptr,                                                            \
                                       (const Dtype*)nullptr,                                                          \
                                       (int)m,                                                                         \
                                       (int)n,                                                                         \
                                       0,                                                                              \
                                       (const float*)nullptr,                                                          \
                                       (const float*)nullptr,                                                          \
                                       (const Dtype*)nullptr,                                                          \
                                       stream);                                                                        \
    } while (0)

#define ATYPE_DISPATCH(Dtype, Atype, KERNEL, ...)                                                                      \
    do {                                                                                                               \
        if (Atype == ActivationType::Silu) {                                                                           \
            KERNEL(SiluActivation, Dtype, __VA_ARGS__);                                                                \
        } else if (Atype == ActivationType::Gelu) {                                                                    \
            KERNEL(GeluActivation, Dtype, __VA_ARGS__);                                                                \
        } else if (Atype == ActivationType::Swiglu) {                                                                  \
            KERNEL(SiluActivation, Dtype, __VA_ARGS__);                                                                \
        } else {                                                                                                       \
            throw OpException(OpErrorType::ERROR_UNIMPLEMENTED);                                                       \
        }                                                                                                              \
    } while (0)

#define DTYPE_DISPATCH(Dtype, ...)                                                                                     \
    do {                                                                                                               \
        if (Dtype == DataType::TYPE_FP16) {                                                                            \
            ATYPE_DISPATCH(half, __VA_ARGS__);                                                                         \
        } else {                                                                                                       \
            throw OpException(OpErrorType::ERROR_UNIMPLEMENTED);                                                       \
        }                                                                                                              \
    } while (0)

#define DISPATCH(Dtype, Atype, ...)                                                                                    \
    do {                                                                                                               \
        DTYPE_DISPATCH(Dtype, Atype, ARGS_DISPATCH, __VA_ARGS__);                                                      \
    } while (0)

}  // namespace rtp_llm
