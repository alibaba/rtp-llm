#include "src/fastertransformer/devices/cuda_impl/CudaDevice.h"
#include "src/fastertransformer/devices/CommonDefines.h"
#include "src/fastertransformer/kernels/layernorm_kernels.h"
#include "src/fastertransformer/kernels/activation_kernels.h"
#include "src/fastertransformer/cutlass/interface.h"
#include "src/fastertransformer/utils/compiler_config.h"
#include "src/fastertransformer/cuda/Dispatch.h"


using namespace std;

namespace fastertransformer {


#define ARGS_DISPATCH(Atype,        \
                      Dtype,        \
                      out,          \
                      bias,         \
                      gate,         \
                      gate_bias,    \
                      m,            \
                      n,            \
                      stream) do {  \
    invokeGenericActivation<Atype>( \
        (Dtype*) out,               \
        (const Dtype*) bias,        \
        (const Dtype*) gate,        \
        (const Dtype*) gate_bias,   \
        (const int*)  nullptr,      \
        (const Dtype*) nullptr,     \
        (int)m,                     \
        (int)n,                     \
        0,                          \
        (const float*) nullptr,     \
        (const float*) nullptr,     \
        (const Dtype*) nullptr,     \
        stream);                    \
} while (0)


#define ATYPE_DISPATCH(Dtype, cpp_type, Atype, ...) case Dtype: \
    if (Atype == ActivationType::Silu) {                        \
        ARGS_DISPATCH(SiluActivation, cpp_type, __VA_ARGS__);   \
    } else if (Atype == ActivationType::Gelu) {                 \
        ARGS_DISPATCH(GeluActivation, cpp_type, __VA_ARGS__);   \
    } else if (Atype == ActivationType::Geglu) {                \
        ARGS_DISPATCH(GeluActivation, cpp_type, __VA_ARGS__);   \
    } else if (Atype == ActivationType::Swiglu) {               \
        ARGS_DISPATCH(SiluActivation, cpp_type, __VA_ARGS__);   \
    } else {                                                    \
        throw OpException(OpErrorType::ERROR_UNIMPLEMENTED);    \
    }                                                           \
    continue;

#define DTYPE_DISPATCH(Dtype, ...) do {                              \
    switch (Dtype) {                                                 \
        DISPATCH_FOR_EACH_COMPUTE_TYPE(ATYPE_DISPATCH, __VA_ARGS__); \
    }                                                                \
} while(0);

void CudaDevice::activation(const ActivationParams& params) {
    const auto& states = params.states;
    const auto data_type = states.type();

    if (params.atype == ActivationType::Sigmoid) {
        RUNTIME_ASSERT_OP_ARG(!params.bias, "Sigmoid does not support bias");
        RUNTIME_ASSERT_OP_ARG(!params.gate, "Sigmoid does not support gate");
        RUNTIME_ASSERT_OP_ARG(!params.gate_bias, "Sigmoid does not support gate_bias");
        DISPATCH_CUDA_FUNCTION_DATA_TYPE(
            data_type, invokeSigmoid,
            states.data(), states.size(), 1.0f, stream_
        );
        return;
    }

    RUNTIME_ASSERT_OP_ARG(states.shape().size() == 2,
                          "activation states must be 2D, but got %d", states.shape().size());
    size_t m = states.shape()[0];
    size_t n = states.shape()[1];

    auto bias = params.bias ? params.bias.value().get().data() : nullptr;
    auto gate = params.gate ? params.gate.value().get().data() : nullptr;
    auto gate_bias = params.gate_bias ? params.gate_bias.value().get().data() : nullptr;

    DTYPE_DISPATCH(
        states.type(), params.atype,
        states.data(), bias,
        gate, gate_bias, m, n, stream_
    );
}

} // namespace fastertransformer