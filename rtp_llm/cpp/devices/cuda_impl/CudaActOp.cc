#include "rtp_llm/cpp/devices/cuda_impl/CudaDevice.h"
#include "rtp_llm/cpp/devices/CommonDefines.h"
#include "rtp_llm/cpp/kernels/activation_kernels.h"
#include "rtp_llm/cpp/core/Dispatch.h"
#include "rtp_llm/cpp/core/torch_utils/BufferTorchUtils.h"
#include "rtp_llm/cpp/kernels/scaled_fp8_quant.h"
#include "3rdparty/flashinfer/flashinfer.h"

using namespace std;

namespace rtp_llm {

#define ARGS_DISPATCH(Atype, Dtype, out, bias, gate, gate_bias, m, n, act_scale, stream)                               \
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
                                       (const Dtype*)act_scale,                                                        \
                                       stream);                                                                        \
    } while (0)

#define ATYPE_DISPATCH(Dtype, cpp_type, Atype, ...)                                                                    \
    case Dtype:                                                                                                        \
        if (Atype == ActivationType::Silu) {                                                                           \
            ARGS_DISPATCH(SiluActivation, cpp_type, __VA_ARGS__);                                                      \
        } else if (Atype == ActivationType::Gelu) {                                                                    \
            ARGS_DISPATCH(GeluActivation, cpp_type, __VA_ARGS__);                                                      \
        } else if (Atype == ActivationType::Geglu) {                                                                   \
            ARGS_DISPATCH(GeluActivation, cpp_type, __VA_ARGS__);                                                      \
        } else if (Atype == ActivationType::Swiglu) {                                                                  \
            ARGS_DISPATCH(SiluActivation, cpp_type, __VA_ARGS__);                                                      \
        } else if (Atype == ActivationType::Identity) {                                                                \
            ARGS_DISPATCH(IdentityActivation, cpp_type, __VA_ARGS__);                                                  \
        } else {                                                                                                       \
            throw OpException(OpErrorType::ERROR_UNIMPLEMENTED);                                                       \
        }                                                                                                              \
        continue;

#define DTYPE_DISPATCH(Dtype, ...)                                                                                     \
    do {                                                                                                               \
        switch (Dtype) { DISPATCH_FOR_EACH_COMPUTE_TYPE(ATYPE_DISPATCH, __VA_ARGS__); }                                \
    } while (0);

BufferPtr CudaDevice::activation(const ActivationParams& params) {
    auto states = params.states;
    RTP_LLM_CHECK_WITH_INFO(states != nullptr, "state should not be nullptr in activation");
    const auto data_type = params.states->type();

    if (initParams().profile_debug_logging_config.check_nan) {
        checkNAN(*states, "activation_states_dump", nullptr, true);
        if (params.bias.has_value()) {
            checkNAN(params.bias.value().get(), "activation_bias_dump", nullptr, true);
        }
        if (params.gate.has_value()) {
            checkNAN(params.gate.value().get(), "activation_gate_dump", nullptr, true);
        }
        if (params.gate_bias.has_value()) {
            checkNAN(params.gate_bias.value().get(), "activation_gate_bias_dump", nullptr, true);
        }
        if (params.act_scale.has_value()) {
            checkNAN(params.act_scale.value().get(), "activation_act_scale_dump", nullptr, true);
        }
    }

    if (params.atype == ActivationType::Sigmoid) {
        RUNTIME_ASSERT_OP_ARG(!params.bias, "Sigmoid does not support bias");
        RUNTIME_ASSERT_OP_ARG(!params.gate, "Sigmoid does not support gate");
        RUNTIME_ASSERT_OP_ARG(!params.gate_bias, "Sigmoid does not support gate_bias");
        RUNTIME_ASSERT_OP_ARG(!params.act_scale, "Sigmoid does not support act_scale");
        DISPATCH_CUDA_FUNCTION_DATA_TYPE(data_type, invokeSigmoid, states->data(), states->size(), 1.0f, stream_);
        return states;
    }

    RUNTIME_ASSERT_OP_ARG(
        states->shape().size() == 2, "activation states must be 2D, but got %ld", states->shape().size());
    size_t m = states->shape()[0];
    size_t n = states->shape()[1];

    auto bias      = params.bias ? params.bias.value().get().data() : nullptr;
    auto gate      = params.gate ? params.gate.value().get().data() : nullptr;
    auto gate_bias = params.gate_bias ? params.gate_bias.value().get().data() : nullptr;
    auto act_scale = params.act_scale ? params.act_scale.value().get().data() : nullptr;
    if (params.fuse_gate_up) {
        RTP_LLM_CHECK_WITH_INFO(params.output_buffer != nullptr,
                                "when fuse gate and up, output buffer must not be nullptr");

#ifdef ENABLE_FP8
        if (params.qscheme == QScheme::Qfp8PerTokenBlock) {
            auto padded_shape_m = (params.output_buffer->shape()[0] + 63) / 64 * 64;
            auto act_output     = allocateBuffer(
                {DataType::TYPE_FP8_E4M3, {padded_shape_m, params.output_buffer->shape()[1]}, AllocationType::DEVICE});
            auto act_scale = allocateBuffer({DataType::TYPE_FP32,
                                             {params.output_buffer->shape()[1] / 128, padded_shape_m},
                                             AllocationType::DEVICE});
            auto act_zeros = BufferPtr(new Buffer(act_scale->where(), DataType::TYPE_INVALID, {0}, nullptr));
            if (params.atype == ActivationType::Swiglu || params.atype == ActivationType::Silu) {
                rtp_llm::computeFP8ActivationAndQuantize(act_output->data<__nv_fp8_e4m3>(),
                                                         act_scale->data<float>(),
                                                         params.states->data<__nv_bfloat16>(),
                                                         params.output_buffer->shape()[0],
                                                         params.output_buffer->shape()[1],
                                                         stream_);
            } else {
                throw OpException(OpErrorType::ERROR_UNIMPLEMENTED);
            }
            check_cuda_error();
            auto out_qbuffer =
                std::make_shared<QBuffer>(std::move(act_output), std::move(act_scale), std::move(act_zeros));
            return std::move(out_qbuffer);
        } else
#endif
        {
            torch::Tensor gate_up_tensor = Buffer2torchTensor(*params.states, false);
            torch::Tensor output_tensor  = Buffer2torchTensor(*params.output_buffer, false);
            if (params.atype == ActivationType::Swiglu || params.atype == ActivationType::Silu) {
                silu_and_mul(output_tensor, gate_up_tensor, (int64_t)stream_);
            } else if (params.atype == ActivationType::Gelu) {
                gelu_and_mul(output_tensor, gate_up_tensor, (int64_t)stream_);
            } else {
                throw OpException(OpErrorType::ERROR_UNIMPLEMENTED);
            }
            check_cuda_error();
            return params.output_buffer;
        }
    } else {
        DTYPE_DISPATCH(states->type(), params.atype, states->data(), bias, gate, gate_bias, m, n, act_scale, stream_);
        check_cuda_error();
        if (initParams().profile_debug_logging_config.check_nan) {
            checkNAN(*states, "activation_output_dump", nullptr, true);
        }
        return states;
    }
}

}  // namespace rtp_llm
