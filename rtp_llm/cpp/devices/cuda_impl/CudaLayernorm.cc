#include "rtp_llm/cpp/devices/cuda_impl/CudaDevice.h"
#include "rtp_llm/cpp/core/Dispatch.h"
#include "rtp_llm/cpp/devices/CommonDefines.h"
#include "rtp_llm/cpp/devices/utils/DebugUtils.h"
#include "rtp_llm/cpp/kernels/rmsnormKernels.h"
#include "rtp_llm/cpp/kernels/layernorm_kernels.h"
#include "rtp_llm/cpp/kernels/add_residual_kernels.h"
#include "rtp_llm/cpp/kernels/fused_qk_rmsnorm.h"

using namespace std;

namespace rtp_llm {

LayernormOutput CudaDevice::layernormWithStride(const LayernormWithStrideParams& params) {
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
        check_cuda_error();
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
        check_cuda_error();
        return LayernormOutput({norm_output, nullptr});
    } else {
        throw std::runtime_error(autil::StringUtil::formatString(
            "unsupported layernorm type for layernormWithStride: %d", int(params.norm_type)));
    }
}

QkRmsNormOutput CudaDevice::qkRmsNorm(const QkRmsNormParams& params) {
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

    if (initParams().profile_debug_logging_config.check_nan) {
        if (params.input->isQBuffer()) {
            const auto& qbuffer = reinterpret_cast<const QBuffer&>(*params.input);
            checkNAN(qbuffer.kernel(), "qkRmsNorm_input_kernel_dump", nullptr, true);
            checkNAN(qbuffer.scales(), "qkRmsNorm_input_scales_dump", nullptr, true);
        } else {
            checkNAN(*params.input, "qkRmsNorm_input_dump", nullptr, true);
        }
        checkNAN(*params.q_norm_weight->get().gamma, "qkRmsNorm_q_gamma_dump", nullptr, true);
        if (params.q_norm_weight->get().beta) {
            checkNAN(*params.q_norm_weight->get().beta, "qkRmsNorm_q_beta_dump", nullptr, true);
        }
        checkNAN(*params.k_norm_weight->get().gamma, "qkRmsNorm_k_gamma_dump", nullptr, true);
        if (params.k_norm_weight->get().beta) {
            checkNAN(*params.k_norm_weight->get().beta, "qkRmsNorm_k_beta_dump", nullptr, true);
        }
    }
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
    if (initParams().profile_debug_logging_config.check_nan) {
        if (params.input->isQBuffer()) {
            const auto& qbuffer = reinterpret_cast<const QBuffer&>(*params.input);
            checkNAN(qbuffer.kernel(), "qkRmsNorm_output_kernel_dump", nullptr, true);
            checkNAN(qbuffer.scales(), "qkRmsNorm_output_scales_dump", nullptr, true);
        } else {
            checkNAN(*params.input, "qkRmsNorm_output_dump", nullptr, true);
        }
    }
    return params.input;
}

LayernormOutput CudaDevice::layernorm(const LayernormParams& params) {
    BufferPtr   input        = params.input;
    BufferPtr   norm_output  = input;
    float*      scales_ptr   = nullptr;
    void*       quant_output = nullptr;
    const auto  data_type    = input->type();
    const auto  m            = input->shape()[0];
    const auto  n            = input->shape()[1];
    auto        norm_weight  = params.norm_weight;
    const auto& gamma        = norm_weight ? norm_weight->get().gamma.get()->data() : nullptr;
    if (gamma != nullptr) {
        printBufferData(*norm_weight->get().gamma, "norm_weight->get().gamma.get()");
    }
    const auto& beta = (norm_weight && norm_weight->get().beta) ? norm_weight->get().beta.get()->data() : nullptr;
    if (beta != nullptr) {
        printBufferData(*norm_weight->get().beta, "norm_weight->get().beat.get()");
    }

    const auto& static_scale = (norm_weight && norm_weight->get().static_scale) ?
                                   norm_weight->get().static_scale.get()->data<float>() :
                                   nullptr;
    const auto  eps          = params.eps;

    if (!params.is_inplace && (params.qscheme == QScheme::NoQuantize || params.qscheme == QScheme::Qfp8PerTokenBlock)) {
        RTP_LLM_LOG_DEBUG("allocate norm_output");
        if (params.attn_swap_comm_buffer && attn_ag_comm_buffer_) {
            norm_output = BufferPtr(
                new Buffer(MemoryType::MEMORY_GPU,
                           params.input->type(),
                           {input->shape()},
                           (char*)attn_ag_comm_buffer_->_ubuf + init_params_.tp_rank * params.input->sizeBytes()));
        } else if (params.ffn_swap_comm_buffer && ffn_ag_comm_buffer_) {
            norm_output = BufferPtr(
                new Buffer(MemoryType::MEMORY_GPU,
                           params.input->type(),
                           {input->shape()},
                           (char*)ffn_ag_comm_buffer_->_ubuf + init_params_.ffn_tp_rank * params.input->sizeBytes()));
        } else {
            norm_output = allocateBufferLike(*params.input);
        }
    } else if (params.qscheme != QScheme::NoQuantize && params.qscheme != QScheme::Qfp8PerTokenBlock) {
        auto quant_data_type =
            (params.qscheme == QScheme::Qfp8PerTensor) ? DataType::TYPE_FP8_E4M3 : DataType::TYPE_INT8;
        BufferPtr kernel = nullptr;
        if (params.attn_swap_comm_buffer && attn_ag_comm_buffer_) {
            kernel =
                BufferPtr(new Buffer(MemoryType::MEMORY_GPU,
                                     quant_data_type,
                                     {input->shape()},
                                     (char*)attn_ag_comm_buffer_->_ubuf
                                         + init_params_.tp_rank * params.input->size() * getTypeSize(quant_data_type)));
        } else if (params.ffn_swap_comm_buffer && ffn_ag_comm_buffer_) {
            kernel = BufferPtr(
                new Buffer(MemoryType::MEMORY_GPU,
                           quant_data_type,
                           {input->shape()},
                           (char*)ffn_ag_comm_buffer_->_ubuf
                               + init_params_.ffn_tp_rank * params.input->size() * getTypeSize(quant_data_type)));
        } else {
            kernel = allocateBuffer({quant_data_type, {input->shape()}, AllocationType::DEVICE}, {"kernel"});
        }
        BufferPtr scales = nullptr;
        // when QScheme::Qint8PerToken the scale is created by layernorm kernel
        if (params.qscheme == QScheme::Qint8PerToken) {
            RTP_LLM_LOG_DEBUG("Qint8PerToken");
            if (params.attn_swap_comm_buffer && attn_ag_scale_comm_buffer_) {
                scales = BufferPtr(
                    new Buffer(MemoryType::MEMORY_GPU,
                               DataType::TYPE_FP32,
                               {input->shape()[0]},
                               (char*)attn_ag_scale_comm_buffer_->_ubuf
                                   + init_params_.tp_rank * input->shape()[0] * getTypeSize(DataType::TYPE_FP32)));
            } else if (params.ffn_swap_comm_buffer && ffn_ag_scale_comm_buffer_) {
                scales = BufferPtr(
                    new Buffer(MemoryType::MEMORY_GPU,
                               DataType::TYPE_FP32,
                               {input->shape()[0]},
                               (char*)ffn_ag_scale_comm_buffer_->_ubuf
                                   + init_params_.ffn_tp_rank * input->shape()[0] * getTypeSize(DataType::TYPE_FP32)));
            } else {
                scales = allocateBuffer({DataType::TYPE_FP32, {input->shape()[0]}, AllocationType::DEVICE}, {"scales"});
            }
            // when QScheme::Qint8PerTensor, the scale is from ckpt
        } else if (params.qscheme == QScheme::Qint8PerTensor || params.qscheme == QScheme::Qfp8PerTensor) {
            RTP_LLM_LOG_DEBUG("QScheme::Qint8PerTensor");
            RTP_LLM_CHECK_WITH_INFO(norm_weight && norm_weight->get().static_scale_reciprocal,
                                    "static_scale_reciprocal should not be None");
            scales = BufferPtr(new Buffer(norm_weight->get().static_scale_reciprocal->where(),
                                          norm_weight->get().static_scale_reciprocal->type(),
                                          norm_weight->get().static_scale_reciprocal->shape(),
                                          norm_weight->get().static_scale_reciprocal->data()));
        } else if (params.qscheme == QScheme::Qfp8PerTokenBlock) {
            RTP_LLM_LOG_DEBUG("now fp8 per token block not set scale");
        } else {
            RTP_LLM_CHECK_WITH_INFO(false, "unknown qscheme type : %d", int(params.qscheme));
        }
        norm_output  = BufferPtr(new QBuffer(
            std::move(kernel),
            std::move(scales),
            std::move(BufferPtr(new Buffer(MemoryType::MEMORY_GPU, DataType::TYPE_INVALID, {0}, nullptr)))));
        quant_output = std::dynamic_pointer_cast<QBuffer>(norm_output)->kernel().data();
        if (params.qscheme == QScheme::Qint8PerToken) {
            scales_ptr = std::dynamic_pointer_cast<QBuffer>(norm_output)->scalesData<float>();
        }
    }

    RTP_LLM_LOG_DEBUG("params.norm_type: %d", params.norm_type);
    if (params.norm_type == NormType::alphanorm || !norm_weight.has_value()) {
        // TODO(lidongjin)
        // we can merge invokeAddBiasResidual and invokeAlphaAddBiasResidual into a singel func.
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
                                             invokeAlphaAddBiasResidual,
                                             norm_output->data(),
                                             input->data(),
                                             params.residual1 ? params.residual1.value().get().data() : nullptr,
                                             params.bias ? params.bias.value().get().data() : nullptr,
                                             params.alpha,
                                             m,
                                             n,
                                             stream_);
            check_cuda_error();
            return LayernormOutput({norm_output, nullptr});
        }
    }

    auto quant_data_type = (params.qscheme == QScheme::Qfp8PerTensor || params.qscheme == QScheme::Qfp8PerTokenBlock) ?
                               DataType::TYPE_FP8_E4M3 :
                               DataType::TYPE_INT8;
    RTP_LLM_LOG_DEBUG("quant_data_type: %d, params.norm_type is %d\n", quant_data_type, params.norm_type);
    if (params.norm_type == NormType::layernorm) {
        if (params.residual1.has_value() || params.bias.has_value()) {
            DISPATCH_CUDA_FUNCTION_COMPUTE_QUANT_TYPES(
                data_type,
                quant_data_type,
                invokeGeneralAddBiasResidualLayerNorm,
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
                true,  // use_diff_of_squares
                static_scale,
                scales_ptr,    // dynamic_scale
                quant_output,  // out_quant
                params.return_normed_output);
            check_cuda_error();
            return LayernormOutput({norm_output, params.before_norm_output});
        } else {
            RTP_LLM_LOG_DEBUG("quant_data_type: %d, params.norm_type is layernorm\n", quant_data_type);
            check_cuda_error();
            DISPATCH_CUDA_FUNCTION_COMPUTE_QUANT_TYPES(
                data_type,
                quant_data_type,
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
                true,  // use_diff_of_squares
                static_scale,
                scales_ptr,    // dynamic_scale
                quant_output,  // out_quant
                params.return_normed_output);
            check_cuda_error();
            return LayernormOutput({norm_output, params.before_norm_output});
        }
    }

    if (params.norm_type == NormType::rmsnorm) {
        if (params.residual1.has_value() || params.bias.has_value()) {
            RUNTIME_ASSERT_OP_ARG(params.before_norm_output != nullptr,
                                  "before_norm_output should not be null when residual1 or bias is set");
            DISPATCH_CUDA_FUNCTION_COMPUTE_QUANT_TYPES(
                data_type,
                quant_data_type,
                invokeAddBiasResidualRmsNorm,
                params.before_norm_output->data(),
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
                static_scale,  // scale
                scales_ptr,    // dynamic_scale
                quant_output   // out_quant
            );
            check_cuda_error();
            return LayernormOutput({norm_output, params.before_norm_output});
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
                                                       (float*)static_scale,  // scale
                                                       scales_ptr,            // dynamic_scale
                                                       quant_output           // out_quant
            );
            check_cuda_error();
            return LayernormOutput({norm_output, params.before_norm_output});
        }
    }

    throw OpException(OpErrorType::ERROR_UNIMPLEMENTED);
}

}  // namespace rtp_llm