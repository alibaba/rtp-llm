#include "layernorm2d_fwd.hpp"
#include "rmsnorm2d_fwd.hpp"
// #include "aiter_meta/3rdparty/composable_kernel/example/ck_tile/10_rmsnorm2d/rmsnorm2d_fwd.hpp"
// #include "aiter_meta/3rdparty/composable_kernel/example/ck_tile/02_layernorm2d/layernorm2d_fwd.hpp"
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
    const auto& beta      = (norm_weight && norm_weight->get().beta) ? norm_weight->get().beta.get()->data() : nullptr;
    const auto  norm_type = params.norm_type;
    const auto  eps       = params.eps;
    const auto& weights   = params.norm_weight;

    if (!params.is_inplace
        && (params.qscheme == QScheme::NoQuantize || params.qscheme == QScheme::Qfp8PerTokenBlock
            || params.qscheme == QScheme::Qfp8PerToken)) {
        norm_output = allocateBufferLike(*params.input);
    } else if (params.qscheme == Qint8PerToken) {
        auto kernel  = allocateBuffer({DataType::TYPE_INT8, {input->shape()}, AllocationType::DEVICE}, {"kernel"});
        auto scales  = allocateBuffer({DataType::TYPE_FP32, {input->shape()[1]}, AllocationType::DEVICE}, {"scales"});
        norm_output  = BufferPtr(new QBuffer(
            std::move(kernel),
            std::move(scales),
            std::move(BufferPtr(new Buffer(MemoryType::MEMORY_GPU, DataType::TYPE_INVALID, {0}, nullptr)))));
        quant_output = std::dynamic_pointer_cast<QBuffer>(norm_output)->kernel().data<int8_t>();
        scales_ptr   = std::dynamic_pointer_cast<QBuffer>(norm_output)->scalesData<float>();
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
            check_cuda_error();
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
            check_cuda_error();
            return LayernormOutput({std::move(norm_output), nullptr});
        } else {
            throw OpException(OpErrorType::ERROR_UNIMPLEMENTED);
        }
    }

    if (!(norm_type == NormType::layernorm || norm_type == NormType::rmsnorm)) {
        throw OpException(OpErrorType::ERROR_UNIMPLEMENTED);
    }

    auto quant_data_type = (params.qscheme == QScheme::Qfp8PerTensor) ? DataType::TYPE_FP8_E4M3 : DataType::TYPE_INT8;

    if (params.residual1.has_value() || params.bias.has_value()) {
        if (params.norm_type == NormType::layernorm) {
            if ((!params.bias.has_value()) && (data_type == DataType::TYPE_FP16 && m > 32 && n <= 768)) {
                layernorm2d_fwd_traits traits{"fp16", "fp16", "fp32", "fp32", 0, 1, 0};
                layernorm2d_fwd_args   args{input->data(),
                                          params.residual1.value().get().data(),
                                          nullptr,
                                          nullptr,
                                          gamma,
                                          beta,

                                          norm_output->data(),
                                          (params.before_norm_output == nullptr) ? input->data() :
                                                                                     params.before_norm_output->data(),
                                          nullptr,
                                          nullptr,  // p_mean, unsupported yet
                                          nullptr,  // p_invStd, unsupported yet

                                          static_cast<float>(eps),
                                          static_cast<int32_t>(m),
                                          static_cast<int32_t>(n),
                                          static_cast<int32_t>(n),   // x row_stride
                                          static_cast<int32_t>(n),   // x residule row stride
                                          static_cast<int32_t>(n),   // y row stride
                                          static_cast<int32_t>(n)};  // y residule row stride

                layernorm2d_fwd(traits, args, {stream_, false, 0, 0, 1});
            } else {
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
            }
            check_cuda_error();
            return LayernormOutput({norm_output, params.before_norm_output});
        } else if (params.norm_type == NormType::rmsnorm) {
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
                nullptr,      // scale
                scales_ptr,   // dynamic_scale
                quant_output  // out_quant
            );
            check_cuda_error();
            return LayernormOutput({norm_output, params.before_norm_output});
        }
    } else {
        if (params.norm_type == NormType::layernorm) {
            if (data_type == DataType::TYPE_FP16 && m > 32 && n <= 768) {
                layernorm2d_fwd_traits traits{"fp16", "fp16", "fp32", "fp32", 0, 0, 0};
                layernorm2d_fwd_args   args{input->data(),
                                          nullptr,
                                          nullptr,
                                          nullptr,
                                          gamma,
                                          beta,

                                          norm_output->data(),
                                          nullptr,
                                          nullptr,
                                          nullptr,  // p_mean, unsupported yet
                                          nullptr,  // p_invStd, unsupported yet

                                          static_cast<float>(eps),
                                          static_cast<int32_t>(m),
                                          static_cast<int32_t>(n),
                                          static_cast<int32_t>(n),   // x row_stride
                                          static_cast<int32_t>(n),   // x residule row stride
                                          static_cast<int32_t>(n),   // y row stride
                                          static_cast<int32_t>(n)};  // y residule row stride

                layernorm2d_fwd(traits, args, {stream_, false, 0, 0, 1});
            } else {
                DISPATCH_CUDA_FUNCTION_DATA_TYPE(data_type,
                                                 invokeGeneralLayerNorm,
                                                 nullptr,
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
            check_cuda_error();
            return LayernormOutput({norm_output, params.before_norm_output});
        } else if (params.norm_type == NormType::rmsnorm) {
            std::string prec_i;
            if (data_type == DataType::TYPE_FP16)
                prec_i = "fp16";
            else if (data_type == DataType::TYPE_BF16)
                prec_i = "bf16";
            else
                throw OpException(OpErrorType::ERROR_UNIMPLEMENTED);

            rmsnorm2d_fwd_traits traits{prec_i, prec_i, "fp32", "fp32", 0, 0, 0, 0};

            rmsnorm2d_fwd_args args{input->data(),
                                    nullptr,
                                    nullptr,
                                    gamma,
                                    norm_output->data(),
                                    nullptr,
                                    nullptr,
                                    nullptr,
                                    nullptr,
                                    static_cast<float>(eps),
                                    static_cast<int32_t>(m),
                                    static_cast<int32_t>(n),
                                    static_cast<int32_t>(n),
                                    static_cast<int32_t>(n),
                                    static_cast<int32_t>(n),
                                    static_cast<int32_t>(n)};

            float run_time = rmsnorm2d_fwd(traits, args, {stream_, false, 0, 0, 1});

            // std::cout << "rmsnorm2d_fwd run_time: " << run_time * 1.E3 << " us"<< std::endl;

            check_cuda_error();
            return LayernormOutput({norm_output, params.before_norm_output});
        }
    }
    throw OpException(OpErrorType::ERROR_UNIMPLEMENTED);
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
