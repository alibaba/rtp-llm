#include "rtp_llm/cpp/devices/rocm_impl/ROCmDevice.h"
#include "rtp_llm/cpp/kernels/activation_kernels.h"
#include "rtp_llm/cpp/kernels/activation_kernels.h"
#include "rtp_llm/cpp/core/Dispatch.h"
#include "rtp_llm/cpp/core/torch_utils/BufferTorchUtils.h"

// aiter kernels
#include "activation.h"
// #include "aiter_meta/csrc/include/activation.h"

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

BufferPtr ROCmDevice::activation(const ActivationParams& params) {
    auto states = params.states;
    ROCM_CHECK_VALUE(states != nullptr, "state should not be nullptr in activation");
    const auto data_type = params.states->type();

    if (params.atype == ActivationType::Sigmoid) {
        RUNTIME_ASSERT_OP_ARG(!params.bias, "Sigmoid does not support bias");
        RUNTIME_ASSERT_OP_ARG(!params.gate, "Sigmoid does not support gate");
        RUNTIME_ASSERT_OP_ARG(!params.gate_bias, "Sigmoid does not support gate_bias");
        RUNTIME_ASSERT_OP_ARG(!params.act_scale, "Sigmoid does not support act_scale");
        DISPATCH_CUDA_FUNCTION_DATA_TYPE(data_type, invokeSigmoid, states->data(), states->size(), 1.0f, stream_);
        return states;
    }

    RUNTIME_ASSERT_OP_ARG(
        states->shape().size() == 2, "activation states must be 2D, but got %zu", states->shape().size());
    size_t m = states->shape()[0];
    size_t n = states->shape()[1];

    auto bias      = params.bias ? params.bias.value().get().data() : nullptr;
    auto gate      = params.gate ? params.gate.value().get().data() : nullptr;
    auto gate_bias = params.gate_bias ? params.gate_bias.value().get().data() : nullptr;
    auto act_scale = params.act_scale ? params.act_scale.value().get().data() : nullptr;

    if (params.fuse_gate_up) {
        torch::Tensor gate_up_tensor = Buffer2torchTensor(*params.states, false);
        torch::Tensor output_tensor  = Buffer2torchTensor(*params.output_buffer, false);
        if (params.atype == ActivationType::Swiglu || params.atype == ActivationType::Silu) {
            aiter::silu_and_mul(output_tensor, gate_up_tensor);
        } else if (params.atype == ActivationType::Gelu) {
            aiter::gelu_and_mul(output_tensor, gate_up_tensor);
        } else {
            throw OpException(OpErrorType::ERROR_UNIMPLEMENTED);
        }
        return params.output_buffer;
    } else {
        DTYPE_DISPATCH(states->type(), params.atype, states->data(), bias, gate, gate_bias, m, n, act_scale, stream_);
        return states;
    }
}

}  // namespace rtp_llm