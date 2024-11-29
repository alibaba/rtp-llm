#include "src/fastertransformer/rocm/rocmLayernorm2d.h"
#include "src/fastertransformer/devices/rocm_impl/ROCmDevice.h"
#include "src/fastertransformer/devices/rocm_impl/ROCmAllocator.h"
#include "src/fastertransformer/core/TrackerAllocator.h"
#include "src/fastertransformer/devices/DeviceFactory.h"
#include "src/fastertransformer/kernels/gpt_kernels.h"
#include "src/fastertransformer/kernels/add_residual_kernels.h"
#include "src/fastertransformer/utils/ShapeCheck.h"
#include "src/fastertransformer/cuda/Dispatch.h"
#include <cstring>

#include "src/fastertransformer/kernels/hello_world.h"
#include "src/fastertransformer/kernels/rmsnormKernels.h"
#include "src/fastertransformer/kernels/activation_kernels.h"

// layerNorm
#include "src/fastertransformer/kernels/rocm/layernorm_kernels.h"
#include "src/fastertransformer/kernels/add_residual_kernels.h"
#include "src/fastertransformer/kernels/alpha_layernorm_kernels.h"
#include "src/fastertransformer/kernels/rmsnormKernels.h"
namespace fastertransformer {
using namespace rocm;

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

    if (!params.is_inplace && params.qscheme == QScheme::NoQuantize) {
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

    if (!weights.has_value()) {
        if (params.alpha != 0 || (norm_type == NormType::alphanorm)) {
            const auto alpha = params.alpha;
            DISPATCH_CUDA_FUNCTION_DATA_TYPE(data_type,
                                             invokeAlphaAddBiasResidual,
                                             norm_output->data(),
                                             input->data(),
                                             params.residual1 ? params.residual1.value().get().data() : nullptr,
                                             params.bias ? params.bias.value().get().data() : nullptr,
                                             params.alpha,
                                             m,
                                             n,
                                             stream_);
            sync_check_cuda_error();
            return LayernormOutput({std::move(norm_output), nullptr});
        } else if (params.bias.has_value() || params.residual1.has_value() || params.residual2.has_value()) {
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
            sync_check_cuda_error();
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
            sync_check_cuda_error();
            return LayernormOutput({norm_output, params.before_norm_output});
        } else if (params.norm_type == NormType::rmsnorm) {
            DISPATCH_CUDA_FUNCTION_COMPUTE_QUANT_TYPES(
                data_type,
                quant_data_type,
                invokeAddBiasResidualRmsNorm,
                (params.before_norm_output == nullptr) ? input->data() : params.before_norm_output->data(),  // or null
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
                nullptr,      // scale
                scales_ptr,   // dynamic_scale
                quant_output  // out_quant
            );
            sync_check_cuda_error();
            return LayernormOutput({norm_output, params.before_norm_output});
        } else {
            throw OpException(OpErrorType::ERROR_UNIMPLEMENTED);
        }
    } else {
        if (params.norm_type == NormType::layernorm) {
            if (data_type == DataType::TYPE_FP16 && m > 32 && n <= 768) {
                layernorm2d_fwd_traits traits{"fp16", "fp16", "fp32", "fp32", 0, 0, 0};
                layernorm2d_fwd_args   args{input->data(),
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
            sync_check_cuda_error();
            return LayernormOutput({norm_output, params.before_norm_output});
        } else if (params.norm_type == NormType::rmsnorm) {
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
                                                       nullptr,      // scale
                                                       scales_ptr,   // dynamic_scale
                                                       quant_output  // out_quant
            );
            sync_check_cuda_error();
            return LayernormOutput({norm_output, params.before_norm_output});
        } else {
            throw OpException(OpErrorType::ERROR_UNIMPLEMENTED);
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

}  // namespace fastertransformer
