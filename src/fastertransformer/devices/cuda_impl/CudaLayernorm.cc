#include "src/fastertransformer/devices/cuda_impl/CudaDevice.h"
#include "src/fastertransformer/cuda/Dispatch.h"
#include "src/fastertransformer/devices/CommonDefines.h"
#include "src/fastertransformer/devices/utils/DebugUtils.h"
#include "src/fastertransformer/kernels/rmsnormKernels.h"
#include "src/fastertransformer/kernels/layernorm_kernels.h"
#include "src/fastertransformer/kernels/add_residual_kernels.h"
#include "src/fastertransformer/kernels/alpha_layernorm_kernels.h"

using namespace std;

namespace fastertransformer {

LayernormOutput CudaDevice::layernorm(const LayernormParams& params) {
    BufferPtr input = params.input;
    BufferPtr norm_output = input;
    float* scales_ptr = nullptr;
    void* quant_output = nullptr;
    const auto data_type = input->type();
    const auto m = input->shape()[0];
    const auto n = input->shape()[1];
    auto norm_weight = params.norm_weight;
    const auto& gamma = norm_weight ? norm_weight->get().gamma.get()->data() : nullptr;
    const auto& beta = (norm_weight && norm_weight->get().beta) ? norm_weight->get().beta.get()->data() : nullptr;

    const auto& static_scale = (norm_weight && norm_weight->get().static_scale) ? norm_weight->get().static_scale.get()->data<float>() : nullptr;
    const auto eps = params.eps;

    if (!params.is_inplace && params.qscheme == QScheme::NoQuantize) {
        norm_output = allocateBufferLike(*params.input);
    } else if (params.qscheme != QScheme::NoQuantize) {
        auto quant_data_type = (params.qscheme == QScheme::Qfp8PerTensor) ? DataType::TYPE_FP8_E4M3 : DataType::TYPE_INT8;
        auto kernel = allocateBuffer({quant_data_type,
                                            {input->shape()},
                                            AllocationType::DEVICE},
                                            {"kernel"});
        BufferPtr scales;
        // when QScheme::Qint8PerToken the scale is created by layernorm kernel
        if (params.qscheme == QScheme::Qint8PerToken) {
            scales = allocateBuffer({DataType::TYPE_FP32,
                                            {input->shape()[0]},
                                            AllocationType::DEVICE},
                                            {"scales"});
        // when QScheme::Qint8PerTensor, the scale is from ckpt
        } else if (params.qscheme == QScheme::Qint8PerTensor || params.qscheme == QScheme::Qfp8PerTensor){
            FT_CHECK_WITH_INFO(norm_weight && norm_weight->get().static_scale_reciprocal, "static_scale_reciprocal should not be None");
            scales = BufferPtr(new Buffer(
                norm_weight->get().static_scale_reciprocal->where(),
                norm_weight->get().static_scale_reciprocal->type(),
                norm_weight->get().static_scale_reciprocal->shape(),
                norm_weight->get().static_scale_reciprocal->data()
            ));
        } else {
            FT_CHECK_WITH_INFO(false, "unknown qscheme type : %d", int(params.qscheme));
        }
        norm_output = BufferPtr(new QBuffer(std::move(kernel),
                                            std::move(scales),
                                            std::move(BufferPtr(
                                                new Buffer(MemoryType::MEMORY_GPU,
                                                DataType::TYPE_INVALID,
                                                {0},
                                                nullptr)))));
        quant_output = std::dynamic_pointer_cast<QBuffer>(norm_output)->kernel().data();
        if (params.qscheme == QScheme::Qint8PerToken) {
            scales_ptr = std::dynamic_pointer_cast<QBuffer>(norm_output)->scalesData<float>();
        }
    }

    if (params.stride != 0) {
        FT_CHECK_WITH_INFO(params.bias == std::nullopt && params.residual1 == std::nullopt && params.residual2 == std::nullopt && params.is_inplace == true, "check error with stride");
        if (params.norm_type == NormType::layernorm) {
            DISPATCH_CUDA_FUNCTION_DATA_TYPE(data_type,
                                            invokeLayerNormWithStride,
                                            input->data(),
                                            gamma,
                                            beta,
                                            eps,
                                            m,
                                            params.hidden_size,
                                            norm_weight->get().gamma.get()->shape()[0],
                                            params.stride,
                                            params.offset,
                                            stream_);
            sync_check_cuda_error();
            return LayernormOutput({norm_output, nullptr});
        } else if (params.norm_type == NormType::rmsnorm) {
            DISPATCH_CUDA_FUNCTION_DATA_TYPE(data_type,
                                            invokeRmsNormWithStride,
                                            input->data(),
                                            gamma,
                                            beta,
                                            eps,
                                            m,
                                            params.hidden_size,
                                            norm_weight->get().gamma.get()->shape()[0],
                                            params.stride,
                                            params.offset,
                                            stream_);
            sync_check_cuda_error();
            return LayernormOutput({norm_output, nullptr});
        }
    }
    FT_LOG_DEBUG("params.norm_type: params.norm_type");
    if (params.norm_type == NormType::alphanorm || !norm_weight.has_value()) {
        // TODO(lidongjin)
        // we can merge invokeAddBiasResidual and invokeAlphaAddBiasResidual into a singel func.
        if (params.alpha == 0.f || params.bias.has_value() || params.residual1.has_value() || params.residual2.has_value()) {
            DISPATCH_CUDA_FUNCTION_DATA_TYPE(data_type, invokeAddBiasResidual,
                norm_output->data(),
                input->data(),
                params.residual1 ? params.residual1.value().get().data() : nullptr,
                params.residual2 ? params.residual2.value().get().data() : nullptr,
                params.bias.has_value() ? params.bias.value().get().data() : nullptr,
                nullptr, // scale_inter
                nullptr, // scale_out
                m,
                n,
                stream_
            );
            sync_check_cuda_error();
            return LayernormOutput({std::move(norm_output), nullptr});
        } else if (params.alpha != 0.f) {
            DISPATCH_CUDA_FUNCTION_DATA_TYPE(data_type, invokeAlphaAddBiasResidual,
                norm_output->data(),
                input->data(),
                params.residual1 ? params.residual1.value().get().data() : nullptr,
                params.bias ? params.bias.value().get().data() : nullptr,
                params.alpha,
                m,
                n,
                stream_);
            sync_check_cuda_error();
            return LayernormOutput({norm_output, nullptr});
        }
    }

    auto quant_data_type = (params.qscheme == QScheme::Qfp8PerTensor) ? DataType::TYPE_FP8_E4M3 : DataType::TYPE_INT8;
    FT_LOG_DEBUG("quant_data_type: %d, params.norm_type is layernorm\n", quant_data_type);
    if (params.norm_type == NormType::layernorm) {
        if (params.residual1.has_value() || params.bias.has_value()) {
            DISPATCH_CUDA_FUNCTION_COMPUTE_QUANT_TYPES(data_type, quant_data_type, invokeGeneralAddBiasResidualLayerNorm,
                (params.before_norm_output == nullptr) ? norm_output->data() : params.before_norm_output->data(),
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
                true, // use_diff_of_squares
                static_scale,
                scales_ptr, // dynamic_scale
                quant_output, // out_quant
                params.return_normed_output
            );
            sync_check_cuda_error();
            return LayernormOutput({norm_output, params.before_norm_output});
        } else {
	        FT_LOG_DEBUG("quant_data_type: %d, params.norm_type is layernorm\n", quant_data_type);
            sync_check_cuda_error();
            DISPATCH_CUDA_FUNCTION_COMPUTE_QUANT_TYPES(data_type, quant_data_type, invokeGeneralLayerNorm,
                params.before_norm_output == nullptr ? nullptr: params.before_norm_output->data(),
                norm_output->data(),
                input->data(),
                gamma,
                beta,
                eps,
                m,
                n,
                stream_,
                true, // use_diff_of_squares
                static_scale,
                scales_ptr, // dynamic_scale
                quant_output, // out_quant
                params.return_normed_output
            );
            sync_check_cuda_error();
            return LayernormOutput({norm_output, params.before_norm_output});
        }
    }

    if (params.norm_type == NormType::rmsnorm) {
        if (params.residual1.has_value() || params.bias.has_value()) {
           DISPATCH_CUDA_FUNCTION_COMPUTE_QUANT_TYPES(data_type, quant_data_type, invokeAddBiasResidualRmsNorm,
                                            params.before_norm_output->data(),
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
                                            static_scale, // scale
                                            scales_ptr, // dynamic_scale
                                            quant_output // out_quant
                                        );
            sync_check_cuda_error();
            return LayernormOutput({norm_output, params.before_norm_output});
        } else {
            DISPATCH_CUDA_FUNCTION_COMPUTE_QUANT_TYPES(data_type, quant_data_type, invokeGeneralRmsNorm,
                norm_output->data(),
                input->data(),
                gamma,
                beta,
                eps,
                m,
                n,
                stream_,
                (float*)static_scale, // scale
                scales_ptr, // dynamic_scale
                quant_output // out_quant
            );
            sync_check_cuda_error();
            return LayernormOutput({norm_output, params.before_norm_output});
        }
    }

    throw OpException(OpErrorType::ERROR_UNIMPLEMENTED);
}


} // namespace fastertransformer
